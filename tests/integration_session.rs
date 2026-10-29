//! Tests for the shared session builder (`zarr_datafusion::session`), which
//! `zarr-cli` and the Flight SQL server both use.

mod common;

use common::*;
use zarr_datafusion::session::{build_session_context, zarr_session_config};

#[tokio::test]
async fn test_session_registers_zarr_rules_and_functions() {
    let ctx = build_session_context(zarr_session_config()).unwrap();
    let state = ctx.state();

    let logical: Vec<&str> = state.optimizers().iter().map(|r| r.name()).collect();
    for name in ["count_statistics", "minmax_statistics"] {
        assert!(logical.contains(&name), "missing optimizer rule {name}");
    }
    let physical: Vec<&str> = state
        .physical_optimizers()
        .iter()
        .map(|r| r.name())
        .collect();
    for name in ["zarr_limit_pushdown", "cardinality"] {
        assert!(physical.contains(&name), "missing physical rule {name}");
    }

    assert!(state.scalar_functions().contains_key("mae"));
    assert!(state.aggregate_functions().contains_key("rmse"));
    assert!(ctx.table_function("zarr_describe").is_ok());
    assert!(
        state.get_file_format_factory("zarr").is_some(),
        "COPY TO ... STORED AS ZARR not registered"
    );
}

#[tokio::test]
async fn test_session_create_external_table_and_information_schema() {
    let ctx = build_session_context(zarr_session_config()).unwrap();
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION '{SYNTHETIC_V3}'"
    ))
    .await
    .unwrap();

    let batch = execute_query_single(
        &ctx,
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'weather'",
    )
    .await;
    assert_eq!(get_scalar_i64(&batch), 1);

    // time(7) × lat(10) × lon(10)
    let batch = execute_query_single(&ctx, "SELECT COUNT(*) FROM weather").await;
    assert_eq!(get_scalar_i64(&batch), 700);
}

#[tokio::test]
async fn test_session_keeps_caller_config() {
    let ctx = build_session_context(zarr_session_config().with_batch_size(1234)).unwrap();
    let config = ctx.copied_config();
    assert_eq!(config.batch_size(), 1234);
    assert!(config.information_schema());
}
