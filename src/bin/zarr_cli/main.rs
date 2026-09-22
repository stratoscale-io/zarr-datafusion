mod highlight;

use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::util::pretty::print_batches;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::{SessionConfig, SessionContext};
use highlight::SqlHelper;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use tracing_subscriber::EnvFilter;
use zarr_datafusion::datasource::factory::ZarrTableFactory;
use zarr_datafusion::datasource::object_store_registry::RemoteObjectStoreRegistry;
use zarr_datafusion::optimizer::{
    CardinalityRule, CountStatisticsRule, MinMaxStatisticsRule, ZarrLimitPushdownRule,
};
use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;
use zarr_datafusion::reader::stats::{format_bytes, SharedIoStats};
use zarr_datafusion::udfs::register_metric_udfs;
use zarr_datafusion::udtf::register_zarr_functions;

const HISTORY_FILE: &str = ".zarr_cli_history";

/// Get the path to the history file (~/.zarr_cli_history)
fn get_history_path() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(HISTORY_FILE)
}

/// Parsed command-line arguments.
struct CliArgs {
    /// `.sql` files to execute (positional args or `-f`/`--file`), in order.
    files: Vec<String>,
    /// Inline statements (`-c`/`--command`) run before any files.
    commands: Vec<String>,
    /// Show usage and exit.
    help: bool,
    /// Show version and exit.
    version: bool,
}

/// Parse argv into [`CliArgs`]. Bare positional args are treated as file paths.
fn parse_cli_args() -> Result<CliArgs, String> {
    let mut files = Vec::new();
    let mut commands = Vec::new();
    let mut help = false;
    let mut version = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => help = true,
            "-V" | "--version" => version = true,
            "-f" | "--file" => {
                let p = it
                    .next()
                    .ok_or_else(|| format!("{arg} requires a path argument"))?;
                files.push(p);
            }
            "-c" | "--command" => {
                let s = it
                    .next()
                    .ok_or_else(|| format!("{arg} requires a SQL argument"))?;
                commands.push(s);
            }
            s if s.starts_with('-') && s.len() > 1 => {
                return Err(format!("Unknown option: {s} (try --help)"));
            }
            // Anything else is a positional file path.
            _ => files.push(arg),
        }
    }

    Ok(CliArgs {
        files,
        commands,
        help,
        version,
    })
}

fn print_usage() {
    println!(
        r#"Zarr-DataFusion CLI

USAGE:
    zarr-cli                       Start the interactive SQL REPL
    zarr-cli [OPTIONS] [FILE...]   Run SQL from files / commands, then exit
    zarr-cli < script.sql          Run SQL piped from stdin, then exit

OPTIONS:
    -f, --file <PATH>     Execute statements from a .sql file (repeatable)
    -c, --command <SQL>   Run an inline SQL statement before any files (repeatable)
    -h, --help            Show this help
    -V, --version         Show version and exit

NOTES:
    Files and stdin may hold multiple `;`-separated statements that span
    several lines and contain `--` line comments. Tables must be registered
    before they are queried, e.g.:

        zarr-cli -c "CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION 'data/era5_sst_local.zarr';" \
                 sql/oni_djf2025_extract.sql"#
    );
}

/// Split a SQL script into individual statements.
///
/// Handles the cases that tripped up a naive `readline`-per-line loop:
/// strips `--` line comments, collapses newlines, and splits on `;` while
/// respecting single-quoted strings and double-quoted identifiers (so a `;`
/// or `--` inside a literal does not split or truncate a statement).
fn split_sql_statements(input: &str) -> Vec<String> {
    let mut stmts = Vec::new();
    let mut cur = String::new();
    let mut in_squote = false; // '...' string literal
    let mut in_dquote = false; // "..." quoted identifier

    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if in_squote {
            cur.push(c);
            if c == '\'' {
                in_squote = false;
            }
            continue;
        }
        if in_dquote {
            cur.push(c);
            if c == '"' {
                in_dquote = false;
            }
            continue;
        }

        match c {
            '\'' => {
                in_squote = true;
                cur.push(c);
            }
            '"' => {
                in_dquote = true;
                cur.push(c);
            }
            // `--` line comment: skip to end of line.
            '-' if chars.peek() == Some(&'-') => {
                while let Some(&nc) = chars.peek() {
                    if nc == '\n' {
                        break;
                    }
                    chars.next();
                }
                cur.push(' '); // keep tokens on either side separated
            }
            ';' => {
                let s = cur.trim();
                if !s.is_empty() {
                    stmts.push(s.to_string());
                }
                cur.clear();
            }
            '\n' | '\r' | '\t' => cur.push(' '),
            _ => cur.push(c),
        }
    }

    let s = cur.trim();
    if !s.is_empty() {
        stmts.push(s.to_string());
    }
    stmts
}

// Why `Send + Sync` in the error type?
//
// Even though we don't explicitly spawn threads, errors must be `Send + Sync` because:
// 1. Tokio runtime is multi-threaded by default - tasks may move between threads at .await points
// 2. DataFusion uses parallelism internally for query execution
// 3. Any error held across an .await must be Send to satisfy the async runtime
//
// References:
// - Tokio multi-threaded runtime: https://tokio.rs/tokio/tutorial/spawning#concurrency
// - Rust Send/Sync traits: https://doc.rust-lang.org/nomicon/send-and-sync.html
// - DataFusion async execution: https://docs.rs/datafusion/latest/datafusion/execution/context/struct.SessionContext.html

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing subscriber (controlled via RUST_LOG env var)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(true)
        .with_line_number(true)
        .init();

    let args = match parse_cli_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    if args.version {
        println!("zarr-cli {}", env!("ZARR_BUILD_VERSION"));
        return Ok(());
    }

    if args.help {
        print_usage();
        return Ok(());
    }

    let config = SessionConfig::new().with_information_schema(true);
    // Lets Parquet/CSV/JSON tables use http(s)://, gs:// and s3:// locations
    // (Zarr tables open their own stores).
    let runtime = RuntimeEnvBuilder::new()
        .with_object_store_registry(Arc::new(RemoteObjectStoreRegistry::new()))
        .build_arc()?;
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_runtime_env(runtime)
        .with_table_factory("ZARR".to_string(), Arc::new(ZarrTableFactory) as _)
        .with_optimizer_rule(Arc::new(CountStatisticsRule::new()))
        .with_optimizer_rule(Arc::new(MinMaxStatisticsRule::new()))
        .with_physical_optimizer_rule(Arc::new(ZarrLimitPushdownRule::new()))
        .with_physical_optimizer_rule(Arc::new(CardinalityRule::new()))
        .build();
    let ctx = SessionContext::new_with_state(state);

    // Register `COPY TO ... STORED AS ZARR` (the write verb).
    if let Err(e) = zarr_datafusion::writer::register_zarr_write_format(&ctx) {
        eprintln!("warning: could not register Zarr write format: {e}");
    }

    // Register Zarr-specific table functions
    register_zarr_functions(&ctx);

    // Register metric UDFs for weather evaluation
    register_metric_udfs(&ctx);

    // Batch mode: run when files/commands are given, or when stdin is piped.
    // Otherwise fall through to the interactive REPL.
    let stdin_piped = !io::stdin().is_terminal();
    let batch = !args.files.is_empty() || !args.commands.is_empty() || stdin_piped;
    if batch {
        run_batch(&ctx, &args, stdin_piped).await;
        return Ok(());
    }

    run_repl(&ctx).await
}

/// Execute inline commands, then files, then piped stdin (when no files given).
async fn run_batch(ctx: &SessionContext, args: &CliArgs, stdin_piped: bool) {
    for cmd in &args.commands {
        for stmt in split_sql_statements(cmd) {
            execute_statement(ctx, &stmt).await;
        }
    }

    for path in &args.files {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                for stmt in split_sql_statements(&content) {
                    execute_statement(ctx, &stmt).await;
                }
            }
            Err(e) => eprintln!("Error reading {path}: {e}"),
        }
    }

    // Read piped stdin only when no explicit files were given (so `-f a.sql`
    // plus a stray pipe doesn't double-run something unexpected).
    if args.files.is_empty() && stdin_piped {
        let mut buf = String::new();
        if let Err(e) = io::stdin().read_to_string(&mut buf) {
            eprintln!("Error reading stdin: {e}");
        } else {
            for stmt in split_sql_statements(&buf) {
                execute_statement(ctx, &stmt).await;
            }
        }
    }
}

/// Interactive REPL with history and syntax highlighting.
async fn run_repl(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Zarr-DataFusion CLI");
    println!("\nType SQL queries or 'help' for commands.\n");

    let helper = SqlHelper::new();
    let mut rl = Editor::new()?;
    rl.set_helper(Some(helper));

    // Load command history (ignore error if file doesn't exist)
    let history_path = get_history_path();
    let _ = rl.load_history(&history_path);

    loop {
        match rl.readline("zarr> ") {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(line);

                if line.eq_ignore_ascii_case("quit") || line.eq_ignore_ascii_case("exit") {
                    break;
                }

                if line.eq_ignore_ascii_case("help") {
                    print_help();
                    continue;
                }

                execute_statement(ctx, line).await;
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("Error: {err}");
                break;
            }
        }
    }

    // Save command history
    if let Err(e) = rl.save_history(&history_path) {
        eprintln!("Warning: Could not save history: {e}");
    }

    println!("Goodbye!");
    Ok(())
}

/// Execute one statement (a meta-command, DDL, or query) and print results.
///
/// Shared by both the interactive REPL and batch mode so behavior is identical.
async fn execute_statement(ctx: &SessionContext, line: &str) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }

    // Meta-command: list tables.
    if line.starts_with("\\d") || line.eq_ignore_ascii_case("show tables") {
        match ctx.sql("SHOW TABLES").await {
            Ok(df) => {
                if let Err(e) = df.show().await {
                    eprintln!("Error: {e}");
                }
            }
            Err(e) => eprintln!("Error: {e}"),
        }
        return;
    }

    // Custom DESCRIBE with extended Zarr metadata.
    if let Some(table_name) = parse_describe_query(line) {
        let query = format!("SELECT * FROM zarr_describe('{}')", table_name);
        match ctx.sql(&query).await {
            Ok(df) => {
                if let Err(e) = df.show().await {
                    eprintln!("Error: {e}");
                }
            }
            Err(e) => eprintln!("Error: {e}"),
        }
        return;
    }

    // Execute SQL with timing.
    let start = Instant::now();
    match ctx.sql(line).await {
        Ok(df) => {
            // DDL statements return empty results - don't show the empty table.
            let line_upper = line.to_uppercase();
            let is_ddl = line_upper.starts_with("CREATE ")
                || line_upper.starts_with("DROP ")
                || line_upper.starts_with("ALTER ");

            if is_ddl {
                // Execute DDL silently - only show errors.
                if let Err(e) = df.collect().await {
                    eprintln!("Error: {e}");
                } else {
                    let elapsed = start.elapsed();
                    println!("OK ({:.3}s)", elapsed.as_secs_f64());
                }
            } else {
                // Create physical plan to access ZarrExec for I/O stats.
                match df.create_physical_plan().await {
                    Ok(plan) => {
                        // Find ZarrExec in the plan tree to get I/O stats.
                        let io_stats = find_zarr_exec_stats(&plan);

                        // Start live stats display if we have ZarrExec stats.
                        // Only show live updates when stdout is a terminal.
                        let stop_flag = Arc::new(AtomicBool::new(false));
                        let is_tty = io::stdout().is_terminal();
                        let live_task = if is_tty {
                            io_stats
                                .as_ref()
                                .map(|stats| spawn_live_stats(stats.clone(), stop_flag.clone()))
                        } else {
                            None
                        };

                        // Execute using the same plan (so stats are populated).
                        let task_ctx = ctx.task_ctx();
                        let result = collect(plan, task_ctx).await;

                        // Stop live stats display.
                        stop_flag.store(true, Ordering::Relaxed);
                        if let Some(task) = live_task {
                            let _ = task.await;
                        }

                        match result {
                            Ok(batches) => {
                                let elapsed = start.elapsed();
                                let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();

                                // Clear the live stats line if we were showing it.
                                if is_tty && io_stats.is_some() {
                                    print!("\r\x1b[K");
                                    let _ = io::stdout().flush();
                                }

                                // Print results table.
                                if let Err(e) = print_batches(&batches) {
                                    eprintln!("Error displaying results: {e}");
                                } else {
                                    // Print compact stats line.
                                    print_stats_line(
                                        row_count,
                                        elapsed.as_secs_f64(),
                                        io_stats.as_ref(),
                                    );
                                }
                            }
                            Err(e) => eprintln!("Error executing query: {e}"),
                        }
                    }
                    Err(e) => eprintln!("Error creating plan: {e}"),
                }
            }
        }
        Err(e) => eprintln!("SQL Error: {e}"),
    }
}

fn print_help() {
    println!(
        r#"
  Zarr-DataFusion CLI Commands:
    <SQL>           Execute a SQL query
    show tables     List registered tables
    \d              List registered tables
    help            Show this help
    quit/exit       Exit the CLI

  Loading data:
    CREATE EXTERNAL TABLE <name> STORED AS ZARR LOCATION '<path>';
    DROP TABLE <name>;

  Example:
    CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION 'data/synthetic.zarr';
    SELECT * FROM weather LIMIT 10;
    SELECT AVG(temperature) FROM weather GROUP BY lat, lon;
    DROP TABLE weather;
  "#
    );
}

/// Recursively search the execution plan tree for ZarrExec and return its I/O stats
fn find_zarr_exec_stats(plan: &Arc<dyn ExecutionPlan>) -> Option<SharedIoStats> {
    // Check if this node is ZarrExec
    if let Some(zarr_exec) = plan.downcast_ref::<ZarrExec>() {
        return Some(zarr_exec.io_stats());
    }

    // Recursively check children
    for child in plan.children() {
        if let Some(stats) = find_zarr_exec_stats(child) {
            return Some(stats);
        }
    }

    None
}

/// Print compact stats line: "5 rows · 3 arrays · 6.70 KB disk · 13.92 KB mem · 0.013s"
fn print_stats_line(row_count: usize, elapsed_secs: f64, io_stats: Option<&SharedIoStats>) {
    let mut parts = vec![format!(
        "{} row{}",
        row_count,
        if row_count == 1 { "" } else { "s" }
    )];

    if let Some(stats) = io_stats {
        let total_arrays =
            stats.coord_arrays.load(Ordering::Relaxed) + stats.data_arrays.load(Ordering::Relaxed);
        let mem_bytes = stats.total_bytes();

        parts.push(format!(
            "{} array{}",
            total_arrays,
            if total_arrays == 1 { "" } else { "s" }
        ));
        // "n/a" where nothing counted (icechunk and VirtualiZarr do their own object
        // I/O) — reporting those as `0 B` would read as "this query fetched nothing".
        parts.push(match stats.disk_bytes_tracked() {
            Some(bytes) => format!("{} disk", format_bytes(bytes)),
            None => "n/a disk".to_string(),
        });
        parts.push(format!("{} mem", format_bytes(mem_bytes)));
    }

    parts.push(format!("{:.3}s", elapsed_secs));

    println!("\n{}", parts.join(" · "));
}

/// Spawn a background task that displays live I/O stats
fn spawn_live_stats(stats: SharedIoStats, stop: Arc<AtomicBool>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while !stop.load(Ordering::Relaxed) {
            let arrays = stats.coord_arrays.load(Ordering::Relaxed)
                + stats.data_arrays.load(Ordering::Relaxed);
            let disk = match stats.disk_bytes_tracked() {
                Some(bytes) => format_bytes(bytes),
                None => "n/a".to_string(),
            };

            // Use \r to overwrite line, \x1b[K to clear to end of line
            print!(
                "\r{} array{} · {} disk...\x1b[K",
                arrays,
                if arrays == 1 { "" } else { "s" },
                disk
            );
            let _ = io::stdout().flush();

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
}

/// Parse DESCRIBE query and extract table name
/// Matches: DESCRIBE table, DESCRIBE table;, DESC table
fn parse_describe_query(line: &str) -> Option<String> {
    let line = line.trim().trim_end_matches(';').trim();
    let upper = line.to_uppercase();

    if upper.starts_with("DESCRIBE ") || upper.starts_with("DESC ") {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            return Some(parts[1].to_string());
        }
    }
    None
}
