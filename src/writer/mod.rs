//! Zarr write path.
//!
//! Phase 1 of `docs/zarr-write-roundtrip-plan.md`: create a target store's
//! metadata and coordinate arrays, but no data-variable chunks. This mirrors the
//! `da.empty(...)` + `to_zarr(compute=False)` step xarray users already do by
//! hand before filling regions.
//!
//! Writing is v3-only by design; reading v2 stays supported.

pub mod copy_format;
pub mod data_sink;
pub mod materialize;
pub mod plan;
pub mod skeleton;
pub mod sink;
mod store;

pub use copy_format::{register_zarr_write_format, ZarrFileFormat, ZarrFormatFactory};
pub use data_sink::{zarr_write_exec, ZarrDataSink};
pub use materialize::{derive_skeleton_spec, materialize_spec};
pub use plan::{derive_write_shape, RejectReason, WriteShape, WriteVar};
pub use skeleton::{
    create_skeleton, CoordSpec, CoordValues, DataVarSpec, SkeletonSpec, WriteDataType,
};
pub use sink::{write_batches, write_batches_partitioned, write_chunks};
