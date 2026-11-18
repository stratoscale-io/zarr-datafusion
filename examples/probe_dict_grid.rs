//! Probe: can the target grid be recovered from a RecordBatch's dictionaries?
//!
//! docs/zarr-write-roundtrip-plan.md §5 proposes taking coordinate arrays from
//! "the dict values, not the rows", and flags it as needing empirical checking
//! before anything is designed around it. If it holds, a skeleton needs no
//! `like` option and no source store: the first batch's dictionary *is* the
//! grid, because our scan builds dict values from the full (filtered) axis
//! rather than from the values present in the batch.
//!
//! For each scenario this prints, per batch, every dictionary column's number of
//! dict VALUES (the candidate axis length) alongside the batch's row count. The
//! question is whether dict-values stays equal to the true axis length
//! regardless of projection, filtering, slicing, batch size, and partitioning.
//!
//! Run: cargo run --example probe_dict_grid

use std::sync::Arc;

use arrow::array::Array;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{SessionConfig, SessionContext};
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::schema_inference::infer_schema_with_meta;

const STORE: &str = "data/synthetic_rt_v3.zarr";

/// Report each dictionary column's dict-value count for one batch.
fn describe(batch: &RecordBatch) -> String {
    let mut parts = Vec::new();
    for (i, field) in batch.schema().fields().iter().enumerate() {
        if let DataType::Dictionary(_, _) = field.data_type() {
            let col = batch.column(i);
            // Downcast via the Array trait object to read the values length
            // without caring about the key width.
            let n_values = match col.data_type() {
                DataType::Dictionary(k, _) => match **k {
                    DataType::Int16 => col
                        .as_any()
                        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int16Type>>(
                        )
                        .map(|d| d.values().len()),
                    DataType::Int32 => col
                        .as_any()
                        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int32Type>>(
                        )
                        .map(|d| d.values().len()),
                    DataType::Int64 => col
                        .as_any()
                        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int64Type>>(
                        )
                        .map(|d| d.values().len()),
                    _ => None,
                },
                _ => None,
            };
            match n_values {
                Some(n) => parts.push(format!("{}={}", field.name(), n)),
                None => parts.push(format!("{}=?", field.name())),
            }
        }
    }
    if parts.is_empty() {
        "(no dictionary columns)".to_string()
    } else {
        parts.join(" ")
    }
}

async fn scenario(ctx: &SessionContext, label: &str, sql: &str) {
    println!("\n--- {label}");
    println!("    {sql}");
    match ctx.sql(sql).await {
        Ok(df) => match df.collect().await {
            Ok(batches) => {
                if batches.is_empty() {
                    println!("    (no batches)");
                }
                for (i, b) in batches.iter().enumerate() {
                    println!(
                        "    batch {i}: rows={:<5} dict_values: {}",
                        b.num_rows(),
                        describe(b)
                    );
                }
            }
            Err(e) => println!("    collect failed: {e}"),
        },
        Err(e) => println!("    plan failed: {e}"),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // True grid: time(7) x lat(10) x lon(12).
    println!("Store: {STORE}   true axes: time=7 lat=10 lon=12");

    let register = |ctx: &SessionContext| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (schema, meta) = infer_schema_with_meta(STORE)?;
        let table = ZarrTable::with_metadata(Arc::new(schema), STORE, meta);
        ctx.register_table("rt", Arc::new(table))?;
        Ok(())
    };

    let ctx = SessionContext::new();
    register(&ctx)?;

    scenario(&ctx, "baseline: all columns", "SELECT * FROM rt").await;

    scenario(
        &ctx,
        "projection: one coord dropped (lon absent from output)",
        "SELECT time, lat, temperature FROM rt",
    )
    .await;

    scenario(
        &ctx,
        "projection: data var only, no coords",
        "SELECT temperature FROM rt",
    )
    .await;

    scenario(
        &ctx,
        "filter on one coord (pushed down): grid should shrink to the region",
        "SELECT time, lat, lon, temperature FROM rt WHERE time = 3",
    )
    .await;

    scenario(
        &ctx,
        "filter on a data var (NOT pushed down; FilterExec slices above the scan)",
        "SELECT time, lat, lon, temperature FROM rt WHERE temperature > 0",
    )
    .await;

    scenario(
        &ctx,
        "limit: slices the batch",
        "SELECT time, lat, lon FROM rt LIMIT 5",
    )
    .await;

    scenario(
        &ctx,
        "projection through an expression",
        "SELECT time, lat, lon, temperature * 2 AS t2 FROM rt",
    )
    .await;

    // Small batch size: does an axis survive being split across many batches?
    let ctx_small = SessionContext::new_with_config(SessionConfig::new().with_batch_size(32));
    register(&ctx_small)?;
    scenario(
        &ctx_small,
        "batch_size=32: axis split across batches",
        "SELECT time, lat, lon FROM rt",
    )
    .await;

    // Multiple partitions: each partition owns an outer-axis range.
    let ctx_par = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    register(&ctx_par)?;
    scenario(
        &ctx_par,
        "target_partitions=4: per-partition grids",
        "SELECT time, lat, lon FROM rt",
    )
    .await;

    // The Phase 3 hop: does the grid survive a Parquet round trip?
    let out = std::env::temp_dir().join("probe_dict_grid.parquet");
    let _ = std::fs::remove_file(&out);
    let copy = format!(
        "COPY (SELECT time, lat, lon, temperature FROM rt) TO '{}' STORED AS PARQUET",
        out.display()
    );
    ctx.sql(&copy).await?.collect().await?;
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE viapq STORED AS PARQUET LOCATION '{}'",
        out.display()
    ))
    .await?
    .collect()
    .await?;
    scenario(
        &ctx,
        "AFTER a parquet round trip (Phase 3's middle hop)",
        "SELECT time, lat, lon FROM viapq",
    )
    .await;

    Ok(())
}
