//! Server-side catalog bookkeeping: views that follow their upstream
//! relations, and a DDL journal that lets the catalog survive restarts.
//!
//! **Views follow their upstream.** DataFusion's `ViewTable` keeps the plan
//! resolved when the view was created, including the providers it reads. If
//! an upstream table or view is replaced, existing downstream views keep
//! reading the old one. `CatalogManager` records each view's SQL and the
//! relations it reads; after a DDL statement changes a relation, it re-creates
//! every view that depends on it, transitively. A view whose upstream is
//! dropped is deregistered (queries fail with "not found" rather than read
//! stale data) and comes back when the upstream is created again. Views stay
//! ordinary `ViewTable`s, so filter pushdown through them is unchanged.
//!
//! **Journal.** With a journal file, every successful DDL statement is appended
//! (JSON lines, one `{"sql": ...}` per statement, fsynced). On startup the
//! journal is replayed, retrying statements that fail because something they
//! need is created later, then rewritten to just the live objects. In-memory
//! tables (`CREATE TABLE AS`) are recomputed on replay; their data is not
//! stored. Rows added with `INSERT`, `SET` options, and views whose upstream
//! is missing at restart are not kept.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::{SchemaReference, TableReference};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{DdlStatement, LogicalPlan};
use datafusion::prelude::SessionContext;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

/// Runs SQL against a shared `SessionContext`, keeping views current and
/// (optionally) journaling DDL. DDL statements are serialized; queries and
/// other statements run without the lock.
pub struct CatalogManager {
    ctx: SessionContext,
    state: Mutex<CatalogState>,
}

#[derive(Default)]
struct CatalogState {
    /// Live catalog objects created through this manager, in the order they
    /// were last (re)created.
    entries: Vec<Entry>,
    /// Views whose upstream is missing; retried after every catalog change.
    broken: Vec<Entry>,
    journal: Option<Journal>,
}

#[derive(Clone, Debug)]
struct Entry {
    /// `catalog`, `catalog.schema`, or `catalog.schema.table`.
    key: String,
    kind: EntryKind,
    /// The statement that created it, as the client sent it.
    sql: String,
    /// Keys of the relations a view reads (empty for other kinds).
    deps: HashSet<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryKind {
    Catalog,
    Schema,
    Table,
    View,
}

/// What a DDL statement does to the catalog.
enum Change {
    Create {
        key: String,
        kind: EntryKind,
        deps: HashSet<String>,
        if_not_exists: bool,
    },
    Drop {
        key: String,
    },
    DropSchema {
        key: String,
    },
    /// Not a catalog object we track (e.g. `CREATE FUNCTION`), or not DDL.
    None,
}

impl CatalogManager {
    /// A manager without a journal: views follow their upstream, nothing is
    /// persisted.
    pub fn new(ctx: SessionContext) -> Self {
        Self {
            ctx,
            state: Mutex::new(CatalogState::default()),
        }
    }

    /// A manager that replays `path` (if it exists), rewrites it to the live
    /// objects, and appends every later DDL statement to it.
    pub async fn open(ctx: SessionContext, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let manager = Self::new(ctx);
        {
            let mut state = manager.state.lock().await;
            let mut pending = read_journal(path)?;
            let replayed = pending.len();
            // A statement can fail because what it reads is created later in
            // the journal (e.g. an upstream view replaced after its
            // dependents were defined). Retry until nothing more succeeds.
            loop {
                let before = pending.len();
                let mut failed = Vec::new();
                for sql in pending {
                    if let Err(e) = manager.execute_ddl(&mut state, &sql).await {
                        failed.push((sql, e));
                    }
                }
                if failed.is_empty() || failed.len() == before {
                    for (sql, e) in &failed {
                        tracing::warn!(%sql, error = %e, "dropping journal entry that no longer applies");
                    }
                    break;
                }
                pending = failed.into_iter().map(|(sql, _)| sql).collect();
            }
            state.journal = Some(Journal::rewrite(path, &state.entries)?);
            tracing::info!(
                path = %path.display(),
                replayed,
                live = state.entries.len(),
                "replayed catalog journal"
            );
        }
        Ok(manager)
    }

    pub fn ctx(&self) -> &SessionContext {
        &self.ctx
    }

    /// Run a statement to completion and return its schema and result.
    pub async fn execute(&self, sql: &str) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        let plan = self.ctx.state().create_logical_plan(sql).await?;
        if matches!(plan, LogicalPlan::Ddl(_)) {
            // Plan again under the lock so the statement sees the catalog as
            // left by the previous DDL statement.
            let mut state = self.state.lock().await;
            let result = self.execute_ddl(&mut state, sql).await?;
            return Ok(result);
        }
        let df = self.ctx.execute_logical_plan(plan).await?;
        let schema = Arc::new(df.schema().as_arrow().clone());
        Ok((schema, df.collect().await?))
    }

    async fn execute_ddl(
        &self,
        state: &mut CatalogState,
        sql: &str,
    ) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        let plan = self.ctx.state().create_logical_plan(sql).await?;
        let change = self.classify(&plan, sql)?;
        // `IF NOT EXISTS` on an existing object changes nothing; keep the
        // existing definition rather than recording the new text.
        let no_op = match &change {
            Change::Create {
                key,
                kind,
                if_not_exists: true,
                ..
            } => self.exists(*kind, key)?,
            _ => false,
        };

        let df = self.ctx.execute_logical_plan(plan).await?;
        let schema = Arc::new(df.schema().as_arrow().clone());
        let batches = df.collect().await?;

        if !no_op && !matches!(change, Change::None) {
            if let Some(journal) = &mut state.journal {
                journal.append(sql).map_err(|e| {
                    DataFusionError::External(
                        format!("statement ran but could not be journaled: {e}").into(),
                    )
                })?;
            }
            self.apply(state, change, sql).await?;
        }
        Ok((schema, batches))
    }

    /// Record `change` and re-create the views it affects.
    async fn apply(&self, state: &mut CatalogState, change: Change, sql: &str) -> Result<()> {
        let changed = match change {
            Change::Create {
                key, kind, deps, ..
            } => {
                state.entries.retain(|e| e.key != key);
                state.broken.retain(|e| e.key != key);
                state.entries.push(Entry {
                    key: key.clone(),
                    kind,
                    sql: sql.to_string(),
                    deps,
                });
                vec![key]
            }
            Change::Drop { key } => {
                state.entries.retain(|e| e.key != key);
                state.broken.retain(|e| e.key != key);
                vec![key]
            }
            Change::DropSchema { key } => {
                let prefix = format!("{key}.");
                let in_schema = |e: &Entry| e.key == key || e.key.starts_with(&prefix);
                let removed = state
                    .entries
                    .iter()
                    .filter(|e| in_schema(e))
                    .map(|e| e.key.clone())
                    .collect();
                state.entries.retain(|e| !in_schema(e));
                state.broken.retain(|e| !in_schema(e));
                removed
            }
            Change::None => return Ok(()),
        };
        self.refresh_views(state, changed).await
    }

    /// Re-create every live view that (transitively) reads a relation in
    /// `changed`, and retry broken views. Each view is re-created at most once.
    async fn refresh_views(&self, state: &mut CatalogState, changed: Vec<String>) -> Result<()> {
        let mut changed: HashSet<String> = changed.into_iter().collect();
        loop {
            let mut progressed = false;

            let stale: Vec<String> = state
                .entries
                .iter()
                .filter(|e| {
                    e.kind == EntryKind::View
                        && !changed.contains(&e.key)
                        && !e.deps.is_disjoint(&changed)
                })
                .map(|e| e.key.clone())
                .collect();
            for key in stale {
                let pos = state.entries.iter().position(|e| e.key == key).unwrap();
                match self.recreate_view(&state.entries[pos].sql).await {
                    Ok(deps) => state.entries[pos].deps = deps,
                    Err(e) => {
                        tracing::warn!(view = %key, error = %e, "view's upstream is gone; deregistering it until it returns");
                        self.ctx.deregister_table(key.as_str())?;
                        let entry = state.entries.remove(pos);
                        state.broken.push(entry);
                    }
                }
                changed.insert(key);
                progressed = true;
            }

            let mut still_broken = Vec::new();
            for mut entry in std::mem::take(&mut state.broken) {
                match self.recreate_view(&entry.sql).await {
                    Ok(deps) => {
                        tracing::info!(view = %entry.key, "view's upstream is back; re-created it");
                        entry.deps = deps;
                        changed.insert(entry.key.clone());
                        state.entries.push(entry);
                        progressed = true;
                    }
                    Err(_) => still_broken.push(entry),
                }
            }
            state.broken = still_broken;

            if !progressed {
                return Ok(());
            }
        }
    }

    /// Plan a stored `CREATE [OR REPLACE] VIEW` against the current catalog
    /// and register it, replacing any existing view. Returns its dependencies.
    async fn recreate_view(&self, sql: &str) -> Result<HashSet<String>> {
        let plan = self.ctx.state().create_logical_plan(sql).await?;
        let LogicalPlan::Ddl(DdlStatement::CreateView(mut view)) = plan else {
            return Err(DataFusionError::Internal(format!(
                "stored view definition is not CREATE VIEW: {sql}"
            )));
        };
        view.or_replace = true;
        let deps = self.relations_read(sql, &view.name)?;
        self.ctx
            .execute_logical_plan(LogicalPlan::Ddl(DdlStatement::CreateView(view)))
            .await?
            .collect()
            .await?;
        Ok(deps)
    }

    fn classify(&self, plan: &LogicalPlan, sql: &str) -> Result<Change> {
        let LogicalPlan::Ddl(ddl) = plan else {
            return Ok(Change::None);
        };
        Ok(match ddl {
            DdlStatement::CreateExternalTable(t) => Change::Create {
                key: self.table_key(&t.name),
                kind: EntryKind::Table,
                deps: HashSet::new(),
                if_not_exists: t.if_not_exists,
            },
            DdlStatement::CreateMemoryTable(t) => Change::Create {
                key: self.table_key(&t.name),
                kind: EntryKind::Table,
                deps: HashSet::new(),
                if_not_exists: t.if_not_exists,
            },
            DdlStatement::CreateView(v) => Change::Create {
                key: self.table_key(&v.name),
                kind: EntryKind::View,
                deps: self.relations_read(sql, &v.name)?,
                if_not_exists: false,
            },
            DdlStatement::CreateCatalogSchema(s) => Change::Create {
                key: self.schema_key_from_name(&s.schema_name),
                kind: EntryKind::Schema,
                deps: HashSet::new(),
                if_not_exists: s.if_not_exists,
            },
            DdlStatement::CreateCatalog(c) => Change::Create {
                key: c.catalog_name.clone(),
                kind: EntryKind::Catalog,
                deps: HashSet::new(),
                if_not_exists: c.if_not_exists,
            },
            DdlStatement::DropTable(t) => Change::Drop {
                key: self.table_key(&t.name),
            },
            DdlStatement::DropView(v) => Change::Drop {
                key: self.table_key(&v.name),
            },
            DdlStatement::DropCatalogSchema(s) => Change::DropSchema {
                key: self.schema_key(&s.name),
            },
            _ => Change::None,
        })
    }

    /// Keys of the relations a `CREATE VIEW` statement names (CTEs
    /// excluded), other than the view itself. Taken from the SQL, not the
    /// plan: DataFusion expands referenced views while planning, so the
    /// plan only shows the base tables underneath.
    fn relations_read(&self, sql: &str, view: &TableReference) -> Result<HashSet<String>> {
        let state = self.ctx.state();
        let dialect = state.config_options().sql_parser.dialect;
        let statement = state.sql_to_statement(sql, &dialect)?;
        let own_key = self.table_key(view);
        Ok(state
            .resolve_table_references(&statement)?
            .iter()
            .map(|r| self.table_key(r))
            .filter(|key| *key != own_key)
            .collect())
    }

    fn exists(&self, kind: EntryKind, key: &str) -> Result<bool> {
        Ok(match kind {
            EntryKind::Table | EntryKind::View => self.ctx.table_exist(key)?,
            EntryKind::Schema => {
                let (catalog, schema) = key.split_once('.').unwrap_or(("", key));
                self.ctx
                    .catalog(catalog)
                    .is_some_and(|c| c.schema(schema).is_some())
            }
            EntryKind::Catalog => self.ctx.catalog(key).is_some(),
        })
    }

    fn defaults(&self) -> (String, String) {
        let options = self.ctx.state().config_options().catalog.clone();
        (options.default_catalog, options.default_schema)
    }

    fn table_key(&self, name: &TableReference) -> String {
        let (catalog, schema) = self.defaults();
        let resolved = name.clone().resolve(&catalog, &schema);
        format!(
            "{}.{}.{}",
            resolved.catalog, resolved.schema, resolved.table
        )
    }

    fn schema_key(&self, name: &SchemaReference) -> String {
        match name {
            SchemaReference::Bare { schema } => format!("{}.{schema}", self.defaults().0),
            SchemaReference::Full { schema, catalog } => format!("{catalog}.{schema}"),
        }
    }

    /// `CREATE SCHEMA` takes `schema` or `catalog.schema`, as DataFusion does.
    fn schema_key_from_name(&self, name: &str) -> String {
        if name.contains('.') {
            name.to_string()
        } else {
            format!("{}.{name}", self.defaults().0)
        }
    }
}

#[derive(Serialize, Deserialize)]
struct JournalRecord {
    sql: String,
}

/// Append-only file of DDL statements, one JSON record per line.
struct Journal {
    file: File,
}

impl Journal {
    fn append(&mut self, sql: &str) -> std::io::Result<()> {
        let line = serde_json::to_string(&JournalRecord {
            sql: sql.to_string(),
        })?;
        writeln!(self.file, "{line}")?;
        self.file.sync_data()
    }

    /// Atomically replace `path` with `entries` and open it for appending.
    fn rewrite(path: &Path, entries: &[Entry]) -> Result<Self> {
        let tmp = tmp_path(path);
        {
            let mut file = File::create(&tmp)?;
            for entry in entries {
                let line = serde_json::to_string(&JournalRecord {
                    sql: entry.sql.clone(),
                })
                .map_err(|e| DataFusionError::External(e.into()))?;
                writeln!(file, "{line}")?;
            }
            file.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        let file = OpenOptions::new().append(true).open(path)?;
        Ok(Self { file })
    }
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Statements in the journal at `path`, in order. A missing file is an empty
/// journal; an unreadable line (e.g. torn by a crash mid-write) is skipped.
fn read_journal(path: &Path) -> Result<Vec<String>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut statements = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<JournalRecord>(&line) {
            Ok(record) => statements.push(record.sql),
            Err(e) => tracing::warn!(line = n + 1, error = %e, "skipping unreadable journal line"),
        }
    }
    Ok(statements)
}
