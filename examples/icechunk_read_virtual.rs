//! Milestone 2: read an icechunk store whose chunks are VIRTUAL references into
//! a local NetCDF4/HDF5 file. Proves the virtual-HDF5 decode path (icechunk
//! resolves each chunk to a byte-range in the .nc and decodes it) before the
//! 7.5 GB remote CIRA files in M3. Everything is local — no network.
//!
//! Generate the store first:
//!   uv run --with 'icechunk==2.1.0' --with 'virtualizarr[hdf,icechunk]==2.7.0' \
//!     --with xarray --with h5netcdf --with obstore scripts/gen_virtual_icechunk.py
//! Then:
//!   cargo run --features icechunk --example icechunk_read_virtual
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use icechunk::repository::VersionInfo;
use icechunk::virtual_chunks::VirtualChunkContainer;
use icechunk::{ObjectStoreConfig, Repository, RepositoryConfig};
use zarrs::array::Array;
use zarrs::array::ArraySubset;
use zarrs_icechunk::AsyncIcechunkStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store_path = "data/m2.icechunk";

    // The virtual refs are absolute file:// URLs into data/. Reconstruct the
    // container prefix the generator used from the canonical data dir.
    let data_dir = std::fs::canonicalize("data")?;
    let prefix = format!("file://{}/", data_dir.display());
    println!("virtual chunk container prefix: {prefix}");

    let storage = icechunk::new_local_filesystem_storage(Path::new(store_path)).await?;
    let mut config = RepositoryConfig::default();
    config.set_virtual_chunk_container(VirtualChunkContainer::new(
        prefix.clone(),
        ObjectStoreConfig::LocalFileSystem(PathBuf::from("/")),
    )?)?;
    // Local filesystem needs no credentials (None), keyed by the container prefix.
    let repo = Repository::open(
        Some(config),
        storage,
        HashMap::from([(prefix.clone(), None)]),
    )
    .await?;
    let session = repo
        .readonly_session(&VersionInfo::BranchTipRef("main".to_string()))
        .await?;
    let store = Arc::new(AsyncIcechunkStore::new(session));
    println!("Opened readonly session at 'main'\n");

    let arr = Array::async_open(store.clone(), "/data").await?;
    println!("/data shape={:?} dtype={}", arr.shape(), arr.data_type());

    // Reading this subset forces icechunk to fetch the virtual chunk(s) from the
    // .nc file and decode the HDF5 chunk payload.
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let data = arr
        .async_retrieve_array_subset::<ndarray::ArrayD<f64>>(&subset)
        .await?;
    println!("read ndim={} shape={:?}", data.ndim(), data.shape());

    // Verify the deterministic values: data[z,y,x] = z*100 + y*10 + x.
    let checks = [([0, 0, 0], 0.0), ([2, 3, 4], 234.0), ([3, 4, 5], 345.0)];
    let mut all_ok = true;
    for (idx, expected) in checks {
        let got = data[[idx[0], idx[1], idx[2]]];
        let ok = (got - expected).abs() < 1e-9;
        all_ok &= ok;
        println!(
            "  data{idx:?} = {got}  (expect {expected})  {}",
            if ok { "OK" } else { "MISMATCH" }
        );
    }

    println!(
        "\n{}",
        if all_ok {
            "MILESTONE 2 PASS: virtual HDF5-backed icechunk chunks decoded correctly."
        } else {
            "MILESTONE 2 FAIL: value mismatch."
        }
    );
    Ok(())
}
