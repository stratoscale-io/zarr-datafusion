//! Milestone 1: read a native, local icechunk store from Rust.
//!
//! Proves the end-to-end path (icechunk repo -> readonly session ->
//! AsyncIcechunkStore -> zarrs::Array -> values) on the simplest store: fully
//! local, all-native chunks, no virtual references, no network. Isolates
//! icechunk-format reading from virtual-HDF5 (M2) and remote GCS (M3).
//!
//! Generate the store first:
//!   uv run --with 'icechunk==2.1.0' --with 'zarr>=3' --with numpy \
//!     scripts/gen_synthetic_icechunk.py data/synthetic.icechunk
//! Then:
//!   cargo run --features icechunk --example icechunk_read -- data/synthetic.icechunk
use std::sync::Arc;

use icechunk::repository::VersionInfo;
use icechunk::{new_local_filesystem_storage, Repository, RepositoryConfig};
use zarrs::array::Array;
use zarrs::array::ArraySubset;
use zarrs_icechunk::AsyncIcechunkStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/synthetic.icechunk".to_string());
    println!("Opening local icechunk repo at {path}");

    let storage = new_local_filesystem_storage(std::path::Path::new(&path)).await?;
    let repo = Repository::open(storage)
        .config(RepositoryConfig::default())
        .execute()
        .await?;
    let session = repo
        .readonly_session(&VersionInfo::BranchTipRef("main".to_string()))
        .await?;
    let store = Arc::new(AsyncIcechunkStore::new(session));
    println!("Opened readonly session at branch tip 'main'\n");

    // Read every array's metadata through the icechunk store.
    for name in ["time", "lat", "lon", "temperature"] {
        let arr = Array::async_open(store.clone(), &format!("/{name}")).await?;
        println!(
            "  /{name:<12} shape={:?} dtype={}",
            arr.shape(),
            arr.data_type()
        );
    }
    println!();

    // Read the full temperature array and verify the deterministic values
    // written by the Python generator: temperature[t,y,x] = t*100 + y*10 + x.
    let temp = Array::async_open(store.clone(), "/temperature").await?;
    let subset = ArraySubset::new_with_shape(temp.shape().to_vec());
    let data = temp
        .async_retrieve_array_subset::<ndarray::ArrayD<f64>>(&subset)
        .await?;
    println!("temperature ndim={} shape={:?}", data.ndim(), data.shape());

    let checks = [([0, 0, 0], 0.0), ([3, 4, 5], 345.0), ([6, 9, 9], 699.0)];
    let mut all_ok = true;
    for (idx, expected) in checks {
        let got = data[[idx[0], idx[1], idx[2]]];
        let ok = (got - expected).abs() < 1e-9;
        all_ok &= ok;
        println!(
            "  temperature{idx:?} = {got}  (expect {expected})  {}",
            if ok { "OK" } else { "MISMATCH" }
        );
    }

    // Also read a coordinate array to confirm 1-D reads.
    let lat = Array::async_open(store.clone(), "/lat").await?;
    let lat_data = lat
        .async_retrieve_array_subset::<ndarray::ArrayD<f64>>(&ArraySubset::new_with_shape(
            lat.shape().to_vec(),
        ))
        .await?;
    println!("\n  lat = {:?}", lat_data.as_slice().unwrap());

    println!(
        "\n{}",
        if all_ok {
            "MILESTONE 1 PASS: native local icechunk store read correctly."
        } else {
            "MILESTONE 1 FAIL: value mismatch."
        }
    );
    Ok(())
}
