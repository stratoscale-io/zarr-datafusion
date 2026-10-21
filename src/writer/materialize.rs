//! Materialise a [`WriteShape`] into a concrete [`SkeletonSpec`].
//!
//! Phase 3 (docs/zarr-write-roundtrip-plan.md) derived the write's *structure*
//! without I/O. This is the seam that closes the loop: it loads the **source**
//! store's coordinate arrays, gathers them by any coordinate `WHERE` filter, and
//! assembles a `SkeletonSpec` for the **target** store — so a query alone produces
//! a store end to end.
//!
//! It *consumes* a `WriteShape` (from `derive_write_shape`); it never re-derives
//! the admission decision, so there is still one implementation of "what is a legal
//! write" (the Q1 concern in the plan discussion).
//!
//! **Scope of this cut:**
//! - **Local sources only.** Coordinate arrays are read from a `FilesystemStore`.
//! - **Coordinate values widen to Int64 / Float64.** A source `float32`/`int32`
//!   coordinate is written back at 64-bit width — lossless in value (every f32 is
//!   an exact f64), but the target coordinate's *dtype* differs from the source's.
//!   Data variables keep their exact width via the sink (gap 5); only coordinates
//!   widen here.
//! - **Axis-valued grids.** A coordinate `GROUP BY` reduces to the source axis's
//!   values. A *periodic* group key (month-of-time) would need its distinct-period
//!   values instead and is not materialised here.

use std::sync::Arc;

use zarrs::array::Array;
use zarrs::array::ArraySubset;
use zarrs::filesystem::FilesystemStore;

use datafusion::physical_plan::ExecutionPlan;

use crate::reader::filter::{CoordSelection, CoordValuesRef};
use crate::reader::filter::calculate_coord_ranges;

use super::plan::{derive_write_shape, find_scan, WriteShape};
use super::skeleton::{CoordSpec, CoordValues, DataVarSpec, SkeletonSpec};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Read a source coordinate array in full, widened to the writer's `CoordValues`
/// (`Int64` / `Float64`). Values are read raw — no CF-time conversion — so they
/// round-trip back to the stored representation.
fn load_coord(
    store: &Arc<FilesystemStore>,
    name: &str,
    dtype: &str,
) -> Result<CoordValues, BoxError> {
    let arr = Array::open(store.clone(), &format!("/{name}"))?;
    let full = ArraySubset::new_with_shape(arr.shape().to_vec());
    let read = |e: zarrs::array::ArrayError| -> BoxError { Box::new(e) };
    Ok(match dtype {
        "float32" => {
            let v = arr.retrieve_array_subset::<Vec<f32>>(&full).map_err(read)?;
            CoordValues::Float64(v.into_iter().map(|x| x as f64).collect())
        }
        "float64" => {
            CoordValues::Float64(arr.retrieve_array_subset::<Vec<f64>>(&full).map_err(read)?)
        }
        "int8" => int64_from(arr.retrieve_array_subset::<Vec<i8>>(&full).map_err(read)?),
        "int16" => int64_from(arr.retrieve_array_subset::<Vec<i16>>(&full).map_err(read)?),
        "int32" => int64_from(arr.retrieve_array_subset::<Vec<i32>>(&full).map_err(read)?),
        "uint8" => int64_from(arr.retrieve_array_subset::<Vec<u8>>(&full).map_err(read)?),
        "uint16" => int64_from(arr.retrieve_array_subset::<Vec<u16>>(&full).map_err(read)?),
        "uint32" => int64_from(arr.retrieve_array_subset::<Vec<u32>>(&full).map_err(read)?),
        // int64 and anything else fall back to i64.
        _ => CoordValues::Int64(arr.retrieve_array_subset::<Vec<i64>>(&full).map_err(read)?),
    })
}

/// Widen a signed/unsigned integer coordinate vector to an `Int64` coordinate.
fn int64_from<T>(raw: Vec<T>) -> CoordValues
where
    i64: TryFrom<T>,
{
    CoordValues::Int64(
        raw.into_iter()
            .map(|x| i64::try_from(x).ok().expect("integer coordinate fits in i64"))
            .collect(),
    )
}

/// A `CoordValuesRef` borrowing a writer `CoordValues`, for filter resolution.
fn as_ref(values: &CoordValues) -> CoordValuesRef<'_> {
    match values {
        CoordValues::Int64(v) => CoordValuesRef::Int64(v),
        CoordValues::Float64(v) => CoordValuesRef::Float64(v),
    }
}

/// Gather a coordinate's values by a resolved selection.
fn gather(values: &CoordValues, sel: &CoordSelection) -> CoordValues {
    match (values, sel) {
        (CoordValues::Int64(v), CoordSelection::Range(s, e)) => CoordValues::Int64(v[*s..*e].to_vec()),
        (CoordValues::Int64(v), CoordSelection::Indices(idx)) => {
            CoordValues::Int64(idx.iter().map(|&i| v[i]).collect())
        }
        (CoordValues::Float64(v), CoordSelection::Range(s, e)) => {
            CoordValues::Float64(v[*s..*e].to_vec())
        }
        (CoordValues::Float64(v), CoordSelection::Indices(idx)) => {
            CoordValues::Float64(idx.iter().map(|&i| v[i]).collect())
        }
    }
}

/// Materialise a derived [`WriteShape`] into a [`SkeletonSpec`] by loading the
/// source store's coordinate arrays (from `plan`'s scan) and gathering them by any
/// coordinate filter. `chunks` is the target chunk shape, one entry per grid axis.
pub fn materialize_spec(
    plan: &Arc<dyn ExecutionPlan>,
    shape: &WriteShape,
    chunks: Vec<u64>,
) -> Result<SkeletonSpec, BoxError> {
    let (_, zarr) = find_scan(plan).ok_or("plan has no single Zarr scan to read coordinates from")?;
    let meta = zarr.store_meta().ok_or("scan carries no store metadata")?;
    let store = Arc::new(FilesystemStore::new(zarr.path())?);

    // Load every source coordinate (needed to resolve filters, which reference the
    // full coordinate set), in cube-axis order.
    let coord_names: Vec<String> = meta.coords.iter().map(|c| c.name.clone()).collect();
    let coords: Vec<CoordValues> = meta
        .coords
        .iter()
        .map(|c| load_coord(&store, &c.name, &c.data_type))
        .collect::<Result<_, _>>()?;

    // Resolve any coordinate WHERE filters into per-axis selections.
    let selections: Option<Vec<CoordSelection>> = match zarr.coord_filters() {
        Some(filters) => {
            let refs: Vec<CoordValuesRef> = coords.iter().map(as_ref).collect();
            match calculate_coord_ranges(filters, &coord_names, &refs) {
                Some(sels) => Some(sels),
                // A filter that matches no value: the write would be empty.
                None => return Err("coordinate filter selects an empty grid".into()),
            }
        }
        None => None,
    };

    // Build the target grid axes, gathered by selection where present.
    let grid_coords: Vec<CoordSpec> = shape
        .grid_axis_source
        .iter()
        .zip(&shape.grid_axes)
        .map(|(&src, name)| {
            let values = match &selections {
                Some(sels) => gather(&coords[src], &sels[src]),
                None => coords[src].clone(),
            };
            CoordSpec::new(name.clone(), values)
        })
        .collect();

    let data_vars: Vec<DataVarSpec> = shape
        .data_vars
        .iter()
        .map(|v| DataVarSpec::new(v.name.clone(), v.data_type))
        .collect();

    Ok(SkeletonSpec::new(grid_coords, data_vars, chunks))
}

/// Convenience: derive the write's shape from `plan` and materialise it into a
/// `SkeletonSpec` in one call. Rejects (as an error) any plan that is not an
/// admissible write.
pub fn derive_skeleton_spec(
    plan: &Arc<dyn ExecutionPlan>,
    chunks: Vec<u64>,
) -> Result<SkeletonSpec, BoxError> {
    let shape: WriteShape = derive_write_shape(plan).map_err(|r| Box::new(r) as BoxError)?;
    materialize_spec(plan, &shape, chunks)
}
