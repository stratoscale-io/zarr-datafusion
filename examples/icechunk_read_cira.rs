//! Milestone 3: read the REAL CIRA FourCastNetv2 forecast from the
//! ExtremeWeatherBench icechunk repo, anonymously over the network. This is the
//! end-to-end proof of the icechunk path against production data:
//!
//!   * the icechunk repo lives at `gs://extremeweatherbench/cira-icechunk/`
//!     (public GCS bucket, read with ANONYMOUS GCS credentials),
//!   * its coordinate arrays are native icechunk chunks, but the data-variable
//!     chunks are VIRTUAL references into ~7.5 GB NetCDF-4/HDF5 files on
//!     `s3://noaa-oar-mlwp-data/` (anonymous S3, us-east-1),
//!   * icechunk resolves each virtual chunk to a byte-range in the remote HDF5
//!     and decodes it, so we only pull a few KB even though the files are huge.
//!
//! We open the `FOUR_v200_GFS` model group, print its hierarchy, then read a
//! tiny lat/lon window of `t2` (2 m / surface air temperature, Kelvin) at the
//! first init_time/lead_time and sanity-check the values are ~230-320 K.
//!
//! Run (needs network):
//!   cargo run --features icechunk --example icechunk_read_cira
use std::collections::HashMap;
use std::sync::Arc;

use icechunk::config::{Credentials, S3Options};
use icechunk::repository::VersionInfo;
use icechunk::storage::{new_gcs_storage, GcsCredentials, S3Credentials};
use icechunk::virtual_chunks::VirtualChunkContainer;
use icechunk::{ObjectStoreConfig, Repository, RepositoryConfig};
use zarrs::array::{data_type, Array, ArraySubset};
use zarrs::node::Node;
use zarrs_icechunk::AsyncIcechunkStore;

// The CIRA icechunk repo (public) and the archival HDF5 bucket its data-variable
// chunks virtually reference (public, anonymous S3).
const GCS_BUCKET: &str = "extremeweatherbench";
const GCS_PREFIX: &str = "cira-icechunk";
const S3_VIRTUAL_CONTAINER: &str = "s3://noaa-oar-mlwp-data/";
const MODEL_GROUP: &str = "/FOUR_v200_GFS";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Storage: public GCS bucket, anonymous (skip signature) ---
    let storage = new_gcs_storage(
        GCS_BUCKET.to_string(),
        Some(GCS_PREFIX.to_string()),
        Some(GcsCredentials::Anonymous),
        None,
    )?;
    println!("Opened anonymous GCS storage gs://{GCS_BUCKET}/{GCS_PREFIX}/");

    // --- Repository config: OVERRIDE the persisted S3 virtual-chunk container.
    //
    // The persisted container declares `anonymous: false`. icechunk 2.1.0's
    // object-store S3 backend has an ordering bug: passing
    // `Credentials::S3(Anonymous)` sets `skip_signature=true`, but the following
    // config block unconditionally re-applies `with_skip_signature(config.anonymous)`,
    // clobbering it back to false — so the request gets signed and the anonymous
    // fetch from the public NOAA bucket fails. We work around it by replacing the
    // container with one whose S3Options has `anonymous: true`, so skip_signature
    // stays true. (Config merge REPLACES virtual_chunk_containers, not extends.)
    let mut config = RepositoryConfig::default();
    config.set_virtual_chunk_container(VirtualChunkContainer::new(
        S3_VIRTUAL_CONTAINER.to_string(),
        ObjectStoreConfig::S3(
            S3Options::default()
                .with_region("us-east-1")
                .with_anonymous(true),
        ),
    )?)?;

    // Still authorize the container with anonymous S3 credentials.
    let authorize = HashMap::from([(
        S3_VIRTUAL_CONTAINER.to_string(),
        Some(Credentials::S3(S3Credentials::Anonymous)),
    )]);
    let repo = Repository::open(Some(config), storage, authorize).await?;
    let session = repo
        .readonly_session(&VersionInfo::BranchTipRef("main".to_string()))
        .await?;
    let store = Arc::new(AsyncIcechunkStore::new(session));
    println!("Opened readonly session at branch 'main'\n");

    // --- Discover the model group's hierarchy (arrays, shapes, dtypes) ---
    let group = Node::async_open(store.clone(), MODEL_GROUP).await?;
    println!("Hierarchy under {MODEL_GROUP}:\n{}", group.hierarchy_tree());

    // --- Read a tiny window of t2 (surface air temperature, Kelvin) ---
    let t2_path = format!("{MODEL_GROUP}/t2");
    let arr = Array::async_open(store.clone(), &t2_path).await?;
    let shape = arr.shape().to_vec();
    println!(
        "{t2_path}\n  shape={:?}\n  dtype={}\n  dims={:?}",
        shape,
        arr.data_type(),
        arr.dimension_names()
    );

    // Build a subset: index 0 on every outer dim, a small window on the last two
    // (assumed lat, lon). This forces exactly the virtual HDF5 chunk(s) covering
    // that corner to be fetched and decoded — a few byte-range reads, not the
    // whole 7.5 GB file.
    let ndim = shape.len();
    let win: u64 = 4;
    let start = vec![0u64; ndim]; // first corner (init_time=0, lead_time=0, lat=0, lon=0)
    let mut sub_shape = vec![1u64; ndim];
    if ndim >= 2 {
        sub_shape[ndim - 1] = win.min(shape[ndim - 1]);
        sub_shape[ndim - 2] = win.min(shape[ndim - 2]);
    } else if ndim == 1 {
        sub_shape[0] = win.min(shape[0]);
    }
    let subset = ArraySubset::new_with_start_shape(start, sub_shape)?;
    println!(
        "\nReading subset start={:?} shape={:?} ...",
        subset.start(),
        subset.shape()
    );

    // t2 is very likely float32; handle both f32/f64 so a dtype surprise doesn't
    // force a re-run over the network.
    let dtype = arr.data_type();
    let values: Vec<f64> = if *dtype == data_type::float32() {
        let v = arr.async_retrieve_array_subset::<Vec<f32>>(&subset).await?;
        v.into_iter().map(f64::from).collect()
    } else if *dtype == data_type::float64() {
        arr.async_retrieve_array_subset::<Vec<f64>>(&subset).await?
    } else {
        return Err(format!("unexpected t2 dtype: {dtype}").into());
    };
    let (min, max) = min_max(&values);

    println!("Read {} values.", values.len());
    let preview: Vec<f64> = values.iter().take(8).copied().collect();
    println!("  first values (K): {preview:?}");
    println!("  window min/max (K): {min:.2} / {max:.2}");

    // Sanity: surface air temperature over any realistic box sits ~230-320 K.
    let sane = values.iter().all(|&k| (200.0..=340.0).contains(&k));
    println!(
        "\n{}",
        if sane {
            "MILESTONE 3 PASS: remote CIRA t2 read anonymously (GCS metadata + S3 virtual HDF5 decode), values are sane Kelvin."
        } else {
            "MILESTONE 3 WARN: read succeeded but values fall outside the expected Kelvin range — inspect above."
        }
    );
    Ok(())
}

fn min_max(v: &[f64]) -> (f64, f64) {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &x in v {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    (mn, mx)
}
