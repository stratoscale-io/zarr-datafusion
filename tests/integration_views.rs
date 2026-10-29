//! Phase 0 of docs/dbt-flightsql-implementation-plan.md: does coordinate
//! filter pushdown survive dbt-style stacks of views?
//!
//! Every case compares a query written against a view chain with the
//! equivalent hand-written query, by what actually reached `ZarrExec`
//! (`CoordFilters`) and how many bytes it read (`ZarrIoStats`). Result values
//! alone prove nothing here: most coord filters are `Inexact`, so DataFusion
//! re-applies them above the scan and results stay correct even when
//! pushdown is lost.
//!
//! Synthetic data: time(7) × lat(10) × lon(10); temperature is chunked
//! [1, 10, 10], i.e. one chunk per time step.

mod common;

use std::sync::atomic::Ordering;
use std::sync::Arc;

use common::*;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::SessionContext;
use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;

/// What the Zarr scans of one executed query did.
#[derive(Debug)]
#[allow(dead_code)] // `filters` is only read through Debug in printed reports
struct ScanReport {
    /// `CoordFilters` per `ZarrExec`, Debug-formatted ("none" if absent).
    filters: Vec<String>,
    /// Uncompressed data-variable bytes read, summed over all `ZarrExec`s.
    data_bytes: u64,
    rows: usize,
}

fn collect_zarr_execs<'a>(plan: &'a Arc<dyn ExecutionPlan>, out: &mut Vec<&'a ZarrExec>) {
    if let Some(exec) = plan.downcast_ref::<ZarrExec>() {
        out.push(exec);
    }
    for child in plan.children() {
        collect_zarr_execs(child, out);
    }
}

async fn scan_report(ctx: &SessionContext, sql: &str) -> ScanReport {
    let df = ctx.sql(sql).await.expect("Query failed");
    let plan = df
        .create_physical_plan()
        .await
        .expect("Failed to create physical plan");
    let batches = collect(plan.clone(), ctx.task_ctx())
        .await
        .expect("Failed to execute");

    let mut execs = Vec::new();
    collect_zarr_execs(&plan, &mut execs);
    assert!(
        !execs.is_empty(),
        "Expected at least one ZarrExec in:\n{sql}"
    );

    ScanReport {
        filters: execs
            .iter()
            .map(|e| match e.coord_filters() {
                Some(f) if !f.is_empty() => format!("{:?}", f.filters),
                _ => "none".to_string(),
            })
            .collect(),
        data_bytes: execs
            .iter()
            .map(|e| e.io_stats().data_bytes.load(Ordering::Relaxed))
            .sum(),
        rows: batches.iter().map(|b| b.num_rows()).sum(),
    }
}

async fn ddl(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.expect("DDL failed");
}

/// Context with `data` registered plus a full-scan byte count for reference.
async fn setup() -> (SessionContext, u64) {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);
    let full = scan_report(&ctx, "SELECT time, temperature FROM data").await;
    (ctx, full.data_bytes)
}

/// Print both reports and assert the view query read exactly what the
/// hand-written query read, and less than a full scan.
fn assert_same_pruning(case: &str, view: &ScanReport, direct: &ScanReport, full: u64) {
    println!("{case}\n  view:   {view:?}\n  direct: {direct:?}\n  full scan bytes: {full}");
    assert_eq!(view.rows, direct.rows, "{case}: row counts differ");
    assert_eq!(
        view.data_bytes, direct.data_bytes,
        "{case}: view read {} bytes, hand-written query read {}",
        view.data_bytes, direct.data_bytes
    );
    assert!(
        view.data_bytes < full,
        "{case}: view read {} bytes, same as a full scan ({full})",
        view.data_bytes
    );
}

/// V1: a plain `SELECT *` view.
#[tokio::test]
async fn test_view_plain_select_pushes_filter() {
    let (ctx, full) = setup().await;
    ddl(&ctx, "CREATE VIEW v1 AS SELECT * FROM data").await;

    let view = scan_report(&ctx, "SELECT time, temperature FROM v1 WHERE time = 3").await;
    let direct = scan_report(&ctx, "SELECT time, temperature FROM data WHERE time = 3").await;
    assert_same_pruning("V1 plain view", &view, &direct, full);
}

/// V2: filter on a GROUP BY key must pass below the Aggregate.
#[tokio::test]
async fn test_view_aggregate_pushes_group_key_filter() {
    let (ctx, full) = setup().await;
    ddl(
        &ctx,
        "CREATE VIEW v2 AS SELECT time, lat, AVG(temperature) AS avg_t FROM data GROUP BY time, lat",
    )
    .await;

    let view = scan_report(&ctx, "SELECT * FROM v2 WHERE time = 3").await;
    let direct = scan_report(
        &ctx,
        "SELECT time, lat, AVG(temperature) AS avg_t FROM data WHERE time = 3 GROUP BY time, lat",
    )
    .await;
    assert_same_pruning("V2 aggregate view", &view, &direct, full);
}

/// V3: three stacked views (stg → int → fct), with renames and arithmetic,
/// as a dbt project would have. The filter is written on the top view.
#[tokio::test]
async fn test_view_stack_pushes_filter() {
    let (ctx, full) = setup().await;
    ddl(
        &ctx,
        "CREATE VIEW stg AS SELECT time AS t, lat, lon, temperature - 273.15 AS temp_c \
         FROM data WHERE lat BETWEEN 2 AND 5",
    )
    .await;
    ddl(
        &ctx,
        "CREATE VIEW int_v AS SELECT t, AVG(temp_c) AS a FROM stg GROUP BY t",
    )
    .await;
    ddl(&ctx, "CREATE VIEW fct AS SELECT t, a * 2 AS x FROM int_v").await;

    let view = scan_report(&ctx, "SELECT * FROM fct WHERE t = 3").await;
    let direct = scan_report(
        &ctx,
        "SELECT time AS t, AVG(temperature - 273.15) * 2 AS x FROM data \
         WHERE lat BETWEEN 2 AND 5 AND time = 3 GROUP BY time",
    )
    .await;
    assert_same_pruning("V3 stacked views", &view, &direct, full);
}

/// V3b: the staging model wraps the coordinate in a cast (the ERA5 recipe
/// must `arrow_cast` dictionary-encoded `time`). The downstream filter then
/// reaches the scan as `arrow_cast(time, ..) = 3`.
#[tokio::test]
async fn test_view_cast_coordinate_pushes_filter() {
    let (ctx, full) = setup().await;
    ddl(
        &ctx,
        "CREATE VIEW stg_cast AS SELECT arrow_cast(time, 'Int64') AS t, temperature FROM data",
    )
    .await;

    let view = scan_report(&ctx, "SELECT * FROM stg_cast WHERE t = 3").await;
    let direct = scan_report(
        &ctx,
        "SELECT arrow_cast(time, 'Int64') AS t, temperature FROM data WHERE time = 3",
    )
    .await;
    assert_same_pruning("V3b cast coordinate", &view, &direct, full);
}

/// V3c: a join with a climatology-style view, as in ONI's int_anomaly. Only
/// the side the filter applies to can prune; the climatology side needs all
/// time steps either way.
#[tokio::test]
async fn test_view_join_pushes_filter_to_matching_side() {
    let (ctx, full) = setup().await;
    ddl(
        &ctx,
        "CREATE VIEW per_t AS SELECT time, lat, AVG(temperature) AS avg_t FROM data GROUP BY time, lat",
    )
    .await;
    ddl(
        &ctx,
        "CREATE VIEW clim AS SELECT lat, AVG(temperature) AS clim_t FROM data GROUP BY lat",
    )
    .await;
    ddl(
        &ctx,
        "CREATE VIEW anom AS SELECT p.time, p.lat, p.avg_t - c.clim_t AS anomaly \
         FROM per_t p JOIN clim c ON p.lat = c.lat",
    )
    .await;

    let view = scan_report(&ctx, "SELECT * FROM anom WHERE time = 3").await;
    let direct = scan_report(
        &ctx,
        "SELECT p.time, p.lat, p.avg_t - c.clim_t AS anomaly \
         FROM (SELECT time, lat, AVG(temperature) AS avg_t FROM data WHERE time = 3 GROUP BY time, lat) p \
         JOIN (SELECT lat, AVG(temperature) AS clim_t FROM data GROUP BY lat) c ON p.lat = c.lat",
    )
    .await;
    println!("V3c join\n  view:   {view:?}\n  direct: {direct:?}\n  full scan bytes: {full}");
    assert_eq!(view.rows, direct.rows);
    assert_eq!(view.data_bytes, direct.data_bytes);
}

/// V4: a rolling window over time (ONI's 3-month mean). A filter on the
/// window's ORDER BY column cannot move below the window without changing
/// its result, so this is expected to read every time step.
#[tokio::test]
async fn test_view_window_blocks_filter() {
    let (ctx, full) = setup().await;
    ddl(
        &ctx,
        "CREATE VIEW per_t AS SELECT time, AVG(temperature) AS a FROM data GROUP BY time",
    )
    .await;
    ddl(
        &ctx,
        "CREATE VIEW rolling AS SELECT time, \
         AVG(a) OVER (ORDER BY time ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS r FROM per_t",
    )
    .await;

    let view = scan_report(&ctx, "SELECT * FROM rolling WHERE time = 3").await;
    println!("V4 window\n  view: {view:?}\n  full scan bytes: {full}");
    assert_eq!(view.rows, 1);
    assert_eq!(
        view.data_bytes, full,
        "V4: window view no longer reads everything; update the plan's V4 note"
    );
}

/// V5: replacing an upstream view. DataFusion's ViewTable keeps the plan
/// resolved at creation, so a downstream view keeps the old definition.
#[tokio::test]
#[ignore = "documents DataFusion: stale downstream views; zarr-flightsql fixes it in CatalogManager (tests/integration_catalog.rs)"]
async fn test_view_replace_upstream_is_seen_downstream() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);
    ddl(
        &ctx,
        "CREATE VIEW up AS SELECT time FROM data WHERE time = 1",
    )
    .await;
    ddl(&ctx, "CREATE VIEW down AS SELECT DISTINCT time FROM up").await;
    ddl(
        &ctx,
        "CREATE OR REPLACE VIEW up AS SELECT time FROM data WHERE time = 2",
    )
    .await;

    let batch = execute_query_single(&ctx, "SELECT CAST(time AS BIGINT) AS t FROM down").await;
    let seen = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("Int64")
        .value(0);
    println!("V5 stale view: downstream sees time = {seen} (2 = fresh, 1 = stale)");
    assert_eq!(
        seen, 2,
        "downstream view still uses the replaced upstream view"
    );
}

// ---------------------------------------------------------------------------
// V3b on real ERA5 metadata: `time` is Dictionary(Int16, Timestamp(µs, "UTC")),
// so date functions need the cookbook's
// `arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))')` first. These cases
// check which shapes of downstream filter on that cast column still reach
// ZarrExec. era5_v3.zarr has 3 hourly steps: 2025-01-01T00/01/02Z.
// ---------------------------------------------------------------------------

/// Context with `era5` registered, a cookbook-style staging view, and the
/// bytes a hand-written `time = 01:00` query reads (one time step).
async fn era5_setup() -> (SessionContext, u64, u64) {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "era5", ERA5_V3);
    ddl(
        &ctx,
        "CREATE VIEW stg_era5 AS SELECT \
           arrow_cast(time, 'Timestamp(Microsecond, Some(\"UTC\"))') AS ts, \
           CAST(extract(hour FROM arrow_cast(time, 'Timestamp(Microsecond, Some(\"UTC\"))')) AS INT) AS hr, \
           temperature - 273.15 AS temp_c \
         FROM era5",
    )
    .await;
    let full = scan_report(&ctx, "SELECT time, temperature FROM era5").await;
    let one = scan_report(
        &ctx,
        "SELECT time, temperature FROM era5 WHERE time = TIMESTAMP '2025-01-01T01:00:00Z'",
    )
    .await;
    println!(
        "ERA5 baseline\n  one step: {one:?}\n  full scan bytes: {}",
        full.data_bytes
    );
    assert!(
        one.data_bytes < full.data_bytes,
        "baseline itself is not pruned"
    );
    (ctx, one.data_bytes, full.data_bytes)
}

fn assert_era5_pruned(case: &str, view: &ScanReport, expected_bytes: u64, full: u64) {
    println!("{case}\n  view: {view:?}\n  expected bytes: {expected_bytes}, full: {full}");
    assert_eq!(
        view.data_bytes, expected_bytes,
        "{case}: read {} bytes, expected {expected_bytes} (full scan = {full})",
        view.data_bytes
    );
}

/// Equality on the cast timestamp column.
#[tokio::test]
async fn test_era5_view_cast_timestamp_eq_pushes() {
    let (ctx, one, full) = era5_setup().await;
    let view = scan_report(
        &ctx,
        "SELECT ts, temp_c FROM stg_era5 WHERE ts = TIMESTAMP '2025-01-01T01:00:00Z'",
    )
    .await;
    assert_era5_pruned("E1 ts = literal", &view, one, full);
}

/// Range on the cast timestamp column (two of the three steps).
#[tokio::test]
async fn test_era5_view_cast_timestamp_range_pushes() {
    let (ctx, one, full) = era5_setup().await;
    let view = scan_report(
        &ctx,
        "SELECT ts, temp_c FROM stg_era5 \
         WHERE ts BETWEEN TIMESTAMP '2025-01-01T01:00:00Z' AND TIMESTAMP '2025-01-01T02:00:00Z'",
    )
    .await;
    assert_era5_pruned("E2 ts BETWEEN", &view, 2 * one, full);
}

/// The cookbook pattern: extract() on the cast column, written downstream.
#[tokio::test]
async fn test_era5_view_extract_on_cast_pushes() {
    let (ctx, one, full) = era5_setup().await;
    let view = scan_report(
        &ctx,
        "SELECT ts, temp_c FROM stg_era5 WHERE extract(hour FROM ts) = 1",
    )
    .await;
    assert_era5_pruned("E3 extract(hour FROM ts) = 1", &view, one, full);
}

/// The dbt pattern: staging exposes `hr = CAST(extract(hour ...) AS INT)` and
/// a downstream model filters on `hr`.
#[tokio::test]
async fn test_era5_view_derived_date_part_column_pushes() {
    let (ctx, one, full) = era5_setup().await;
    let view = scan_report(&ctx, "SELECT ts, temp_c FROM stg_era5 WHERE hr = 1").await;
    assert_era5_pruned("E4 hr = 1 (CAST(extract) column)", &view, one, full);
}

/// A cast that changes unit and drops the timezone.
#[tokio::test]
async fn test_era5_view_cast_to_naive_nanos_pushes() {
    let (ctx, one, full) = era5_setup().await;
    ddl(
        &ctx,
        "CREATE VIEW stg_naive AS SELECT \
           arrow_cast(time, 'Timestamp(Nanosecond, None)') AS ts, temperature FROM era5",
    )
    .await;
    let view = scan_report(
        &ctx,
        "SELECT * FROM stg_naive WHERE ts = TIMESTAMP '2025-01-01T01:00:00'",
    )
    .await;
    assert_era5_pruned("E5 naive-ns ts = literal", &view, one, full);
}
