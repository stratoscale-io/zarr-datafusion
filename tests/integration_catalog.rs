//! Tests for `CatalogManager`: views that follow their upstream relations,
//! and the DDL journal that lets the catalog survive restarts.

mod common;

use std::path::{Path, PathBuf};

use arrow::array::Int64Array;
use arrow::record_batch::RecordBatch;
use common::*;
use zarr_datafusion::flightsql::CatalogManager;
use zarr_datafusion::session::{build_session_context, zarr_session_config};

/// A session with `weather` (synthetic v3) registered directly, the way
/// `--init-sql` registers tables, then a manager over it.
async fn manager(journal: Option<&Path>) -> CatalogManager {
    let ctx = build_session_context(zarr_session_config()).unwrap();
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION '{SYNTHETIC_V3}'"
    ))
    .await
    .unwrap();
    match journal {
        Some(path) => CatalogManager::open(ctx, path).await.unwrap(),
        None => CatalogManager::new(ctx),
    }
}

async fn exec(m: &CatalogManager, sql: &str) -> Vec<RecordBatch> {
    m.execute(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .1
}

/// The single Int64 value of a one-row, one-column result.
async fn scalar(m: &CatalogManager, sql: &str) -> i64 {
    let batches = exec(m, sql).await;
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 result")
        .value(0)
}

/// A fresh journal path for one test.
fn journal_path(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("zarr-catalog-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("catalog.jsonl")
}

fn journal_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(String::from)
        .collect()
}

/// Phase 0's V5, fixed: replacing an upstream view is seen downstream,
/// transitively.
#[tokio::test]
async fn test_replaced_upstream_is_seen_downstream() {
    let m = manager(None).await;
    exec(
        &m,
        "CREATE VIEW up AS SELECT time FROM weather WHERE time = 1",
    )
    .await;
    exec(
        &m,
        "CREATE VIEW down AS SELECT DISTINCT CAST(time AS BIGINT) AS t FROM up",
    )
    .await;
    exec(&m, "CREATE VIEW down2 AS SELECT t FROM down").await;
    exec(
        &m,
        "CREATE OR REPLACE VIEW up AS SELECT time FROM weather WHERE time = 2",
    )
    .await;

    assert_eq!(scalar(&m, "SELECT t FROM down").await, 2);
    assert_eq!(scalar(&m, "SELECT t FROM down2").await, 2);
}

/// Re-created views are ordinary views: filters still reach ZarrExec.
#[tokio::test]
async fn test_refreshed_view_keeps_pushdown() {
    let m = manager(None).await;
    exec(&m, "CREATE VIEW base AS SELECT * FROM weather").await;
    exec(
        &m,
        "CREATE VIEW agg AS SELECT time, lat, AVG(temperature) AS a FROM base GROUP BY time, lat",
    )
    .await;
    exec(
        &m,
        "CREATE OR REPLACE VIEW base AS SELECT * FROM weather WHERE lat >= 0",
    )
    .await;

    let plan = get_physical_plan(m.ctx(), "SELECT * FROM agg WHERE time = 3").await;
    let filters = find_zarr_exec(&plan)
        .expect("ZarrExec")
        .coord_filters()
        .expect("coordinate filters");
    assert!(
        filters.get("time").is_some(),
        "time filter lost: {filters:?}"
    );
    assert!(filters.get("lat").is_some(), "lat filter lost: {filters:?}");
}

/// A view whose upstream is dropped stops resolving (no stale reads) and
/// comes back when the upstream is created again.
#[tokio::test]
async fn test_dropped_upstream_breaks_then_restores_downstream() {
    let m = manager(None).await;
    exec(
        &m,
        "CREATE VIEW up AS SELECT time FROM weather WHERE time = 1",
    )
    .await;
    exec(
        &m,
        "CREATE VIEW down AS SELECT DISTINCT CAST(time AS BIGINT) AS t FROM up",
    )
    .await;

    exec(&m, "DROP VIEW up").await;
    let err = m.execute("SELECT t FROM down").await.unwrap_err();
    assert!(err.to_string().contains("not found"), "{err}");

    exec(
        &m,
        "CREATE VIEW up AS SELECT time FROM weather WHERE time = 2",
    )
    .await;
    assert_eq!(scalar(&m, "SELECT t FROM down").await, 2);
}

/// The journal brings back schemas, views and in-memory tables, and is
/// compacted to the live objects on open.
#[tokio::test]
async fn test_journal_replays_and_compacts() {
    let path = journal_path("replay");
    {
        let m = manager(Some(&path)).await;
        exec(&m, "CREATE SCHEMA analytics").await;
        exec(
            &m,
            "CREATE VIEW analytics.v1 AS SELECT time, temperature FROM weather WHERE time = 1",
        )
        .await;
        exec(
            &m,
            "CREATE VIEW analytics.v2 AS SELECT CAST(time AS BIGINT) AS t, COUNT(*) AS n \
             FROM analytics.v1 GROUP BY time",
        )
        .await;
        exec(
            &m,
            "CREATE OR REPLACE VIEW analytics.v1 AS \
             SELECT time, temperature FROM weather WHERE time = 4",
        )
        .await;
        exec(
            &m,
            "CREATE TABLE analytics.counts AS SELECT CAST(COUNT(*) AS BIGINT) AS n FROM weather",
        )
        .await;
        exec(&m, "CREATE VIEW analytics.scratch AS SELECT 1 AS x").await;
        exec(&m, "DROP VIEW analytics.scratch").await;
        assert_eq!(journal_lines(&path).len(), 7);
    }

    let m = manager(Some(&path)).await;
    assert_eq!(scalar(&m, "SELECT t FROM analytics.v2").await, 4);
    assert_eq!(scalar(&m, "SELECT n FROM analytics.counts").await, 700);
    assert!(m.execute("SELECT * FROM analytics.scratch").await.is_err());
    // schema, v1, v2, counts
    assert_eq!(journal_lines(&path).len(), 4, "{:#?}", journal_lines(&path));
}

/// Compaction keeps creation order, which can put a view before an upstream
/// that was replaced later; replay retries until it resolves.
#[tokio::test]
async fn test_journal_replay_resolves_out_of_order_definitions() {
    let path = journal_path("order");
    {
        let m = manager(Some(&path)).await;
        exec(
            &m,
            "CREATE VIEW u AS SELECT time FROM weather WHERE time = 1",
        )
        .await;
        exec(
            &m,
            "CREATE VIEW d AS SELECT DISTINCT CAST(time AS BIGINT) AS t FROM u",
        )
        .await;
        exec(
            &m,
            "CREATE VIEW w AS SELECT time FROM weather WHERE time = 5",
        )
        .await;
        exec(&m, "CREATE OR REPLACE VIEW u AS SELECT time FROM w").await;
    }
    // Compacted order is now d, w, u: d comes before the u it reads.
    let m = manager(Some(&path)).await;
    assert_eq!(scalar(&m, "SELECT t FROM d").await, 5);
    let lines = journal_lines(&path);
    assert_eq!(lines.len(), 3, "{lines:#?}");
}

/// `IF NOT EXISTS` on an existing table changes nothing and is not recorded.
#[tokio::test]
async fn test_if_not_exists_keeps_original_definition() {
    let path = journal_path("ifnotexists");
    let m = manager(Some(&path)).await;
    exec(
        &m,
        &format!("CREATE EXTERNAL TABLE w2 STORED AS ZARR LOCATION '{SYNTHETIC_V3}'"),
    )
    .await;
    exec(
        &m,
        &format!("CREATE EXTERNAL TABLE IF NOT EXISTS w2 STORED AS ZARR LOCATION '{SYNTHETIC_V2}'"),
    )
    .await;
    let lines = journal_lines(&path);
    assert_eq!(lines.len(), 1, "{lines:#?}");
    assert!(lines[0].contains(SYNTHETIC_V3));
}

/// `DROP SCHEMA ... CASCADE` forgets everything in the schema.
#[tokio::test]
async fn test_drop_schema_cascade_forgets_objects() {
    let path = journal_path("cascade");
    {
        let m = manager(Some(&path)).await;
        exec(&m, "CREATE SCHEMA s").await;
        exec(&m, "CREATE VIEW s.v AS SELECT * FROM weather").await;
        exec(&m, "DROP SCHEMA s CASCADE").await;
    }
    let m = manager(Some(&path)).await;
    assert!(m.execute("SELECT * FROM s.v").await.is_err());
    assert!(journal_lines(&path).is_empty());
}
