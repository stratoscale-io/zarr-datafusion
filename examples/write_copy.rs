//! Copy a Zarr store to another, driven entirely by a `SELECT` (Phase 4 round
//! trip, docs/zarr-write-roundtrip-plan.md).
//!
//! `SELECT * FROM src` -> derive the target grid -> create the skeleton -> execute
//! -> write. The target is always Zarr v3 (reading v2 stays supported), so copying
//! a v2 store *upgrades* it to v3 and adds `dimension_names`.
//!
//! ```bash
//! cargo run --example write_copy -- data/synthetic_rt_v3.zarr /tmp/copy_v3.zarr
//! # verify against the original with the external oracle:
//! uv run --with 'zarr>=3' --with numpy scripts/compare_zarr.py \
//!     data/synthetic_rt_v3.zarr /tmp/copy_v3.zarr
//!
//! # a v2 source upgrades to v3 (dimension names added) -> allow that enrichment:
//! cargo run --example write_copy -- data/synthetic_rt_v2.zarr /tmp/copy_v2.zarr
//! uv run --with 'zarr>=3' --with numpy scripts/compare_zarr.py \
//!     --allow-added-dim-names data/synthetic_rt_v2.zarr /tmp/copy_v2.zarr
//! ```
//!
//! An optional third argument re-chunks the target (`--example write_copy -- src
//! dst 7,10,12`); by default it keeps the source's chunk shape.

use std::sync::Arc;

use datafusion::physical_plan::collect;
use datafusion::prelude::SessionContext;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::schema_inference::infer_schema_with_meta;
use zarr_datafusion::writer::{create_skeleton, derive_skeleton_spec, write_batches};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: write_copy <source.zarr> <target.zarr> [chunks e.g. 1,4,5]");
        std::process::exit(2);
    }
    let (source, target) = (&args[1], &args[2]);
    let _ = std::fs::remove_dir_all(target);

    let (schema, meta) = infer_schema_with_meta(source)?;

    // Chunks: explicit third arg, else keep the source's data-variable chunking.
    let chunks: Vec<u64> = match args.get(3) {
        Some(s) => s.split(',').map(|p| p.trim().parse()).collect::<Result<_, _>>()?,
        None => meta
            .data_vars
            .first()
            .and_then(|v| v.chunks.clone())
            .ok_or("source data variable has no chunk shape; pass chunks explicitly")?,
    };

    let ctx = SessionContext::new();
    let table = ZarrTable::with_metadata(Arc::new(schema), source, meta);
    ctx.register_table("src", Arc::new(table))?;

    let plan = ctx
        .sql("SELECT * FROM src")
        .await?
        .create_physical_plan()
        .await?;

    let spec = derive_skeleton_spec(&plan, chunks.clone()).await?;
    create_skeleton(target, &spec)?;
    let batches = collect(plan, ctx.task_ctx()).await?;
    let rows = write_batches(target, &spec, batches)?;

    println!("Copied {source} -> {target}");
    println!("  {rows} rows, chunks {chunks:?}, target is Zarr v3");
    println!("\nVerify against the original:");
    println!("  uv run --with 'zarr>=3' --with numpy scripts/compare_zarr.py {source} {target}");
    println!("  (add --allow-added-dim-names if the source is v2)");
    Ok(())
}
