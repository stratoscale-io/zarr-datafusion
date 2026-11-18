//! Multi-axis (N-D box) partition fan-out — end-to-end.
//!
//! When the outer axis alone under-parallelizes (fewer chunks than
//! `target_partitions`), the planner also splits the best inner axis so the scan
//! fans out across the machine. These tests use `era5_v3` — `temperature` is
//! `[time=3, hybrid=2, latitude=721, longitude=1440]` chunked `[1, 1, 181, 720]`,
//! so the outer time axis has 3 chunks while latitude has 4 — a real inner axis to
//! split. They assert the fan-out happens *and* leaves results unchanged.

mod common;

use common::*;

use datafusion::prelude::{SessionConfig, SessionContext};

/// A context whose scans target `n`-way parallelism.
fn ctx_with_target(n: usize) -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(n);
    SessionContext::new_with_config(config)
}

#[tokio::test]
async fn fans_out_onto_inner_axis_when_outer_underfills() {
    let ctx = ctx_with_target(8);
    register_zarr_table(&ctx, "era5", ERA5_V3);
    let plan = get_physical_plan(&ctx, "SELECT temperature FROM era5").await;
    let zarr = find_zarr_exec(&plan).expect("plan contains a ZarrExec");
    // outer time = 3 chunks; inner_budget = 8 / 3 = 2; split latitude (4 chunks)
    // into 2 => 3 × 2 = 6 box partitions.
    assert_eq!(
        zarr.partitions().len(),
        6,
        "expected 3 outer × 2 inner = 6 partitions"
    );
    // Every partition is a box: outer time slice + one inner (latitude) restriction.
    assert!(
        zarr.partitions().iter().all(|p| p.extra.len() == 1),
        "each partition should restrict exactly one inner axis"
    );
}

#[tokio::test]
async fn single_partition_when_target_matches_outer() {
    // target 3 == outer chunks => no inner fan-out (plain outer-only partitions).
    let ctx = ctx_with_target(3);
    register_zarr_table(&ctx, "era5", ERA5_V3);
    let plan = get_physical_plan(&ctx, "SELECT temperature FROM era5").await;
    let zarr = find_zarr_exec(&plan).expect("plan contains a ZarrExec");
    assert_eq!(zarr.partitions().len(), 3);
    assert!(zarr.partitions().iter().all(|p| p.extra.is_empty()));
}

#[tokio::test]
async fn fans_out_across_multiple_inner_axes() {
    // With a high enough target the fan-out spans several inner axes (N-D boxes).
    // temperature[time=3, hybrid=2, latitude=721, longitude=1440] chunked
    // [1,1,181,720] => outer time=3; inner_budget=24/3=8; latitude(4)×hybrid(2)=8.
    let ctx = ctx_with_target(24);
    register_zarr_table(&ctx, "era5", ERA5_V3);
    let plan = get_physical_plan(&ctx, "SELECT temperature FROM era5").await;
    let zarr = find_zarr_exec(&plan).expect("plan contains a ZarrExec");
    assert_eq!(
        zarr.partitions().len(),
        24,
        "expected 3 outer × (4 lat × 2 hybrid) = 24 partitions"
    );
    assert!(
        zarr.partitions().iter().all(|p| p.extra.len() == 2),
        "each N-D box restricts two inner axes"
    );
}

#[tokio::test]
async fn nd_fan_out_is_value_transparent() {
    // A multi-inner-axis fan-out must still tile the cube exactly.
    let sql = "SELECT CAST(SUM(temperature) AS DOUBLE) AS s, COUNT(*) AS c \
               FROM era5 WHERE longitude = 0.0";

    let single = ctx_with_target(1);
    register_zarr_table(&single, "era5", ERA5_V3);
    let base = execute_query(&single, sql).await;

    let fanned = ctx_with_target(24);
    register_zarr_table(&fanned, "era5", ERA5_V3);
    let plan = get_physical_plan(
        &fanned,
        "SELECT temperature FROM era5 WHERE longitude = 0.0",
    )
    .await;
    let boxes = find_zarr_exec(&plan).unwrap();
    assert!(
        boxes.partitions().iter().any(|p| p.extra.len() >= 2),
        "expected N-D boxes (≥2 inner axes) under target 24"
    );

    let got = execute_query(&fanned, sql).await;
    assert_eq!(format!("{:?}", base), format!("{:?}", got));
}

#[tokio::test]
async fn fan_out_is_value_transparent() {
    // The same aggregate must be identical whether the scan runs single-partition
    // or fanned out into inner-axis boxes — proving the boxes tile the cube exactly
    // (no gaps, no double reads). Filter longitude to keep the read small while
    // leaving latitude (the fan-out axis) unrestricted.
    let sql = "SELECT CAST(SUM(temperature) AS DOUBLE) AS s, COUNT(*) AS c \
               FROM era5 WHERE longitude = 0.0";

    let single = ctx_with_target(1);
    register_zarr_table(&single, "era5", ERA5_V3);
    let base = execute_query(&single, sql).await;

    let fanned = ctx_with_target(8);
    register_zarr_table(&fanned, "era5", ERA5_V3);
    // Confirm the fanned plan really is multi-partition on this query.
    let plan = get_physical_plan(
        &fanned,
        "SELECT temperature FROM era5 WHERE longitude = 0.0",
    )
    .await;
    let n = find_zarr_exec(&plan).unwrap().partitions().len();
    assert!(n > 3, "expected inner fan-out (>3 partitions), got {n}");

    let got = execute_query(&fanned, sql).await;
    assert_eq!(
        format!("{:?}", base),
        format!("{:?}", got),
        "fanned-out aggregate must equal single-partition aggregate"
    );
}
