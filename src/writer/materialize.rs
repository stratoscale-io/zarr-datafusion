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
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

use datafusion::physical_plan::ExecutionPlan;

use crate::optimizer::cardinality::pushdown::field_range;
use crate::reader::cf_time::apply_cf_time_conversion;
use crate::reader::coord::CoordValues as ReadCoordValues;
use crate::reader::filter::calculate_coord_ranges;
use crate::reader::filter::CoordSelection;
use crate::reader::schema_inference::ZarrArrayMeta;
use crate::reader::zarr_reader::coord_values_to_refs;

use super::plan::{derive_write_shape, find_scan, WriteShape};
use super::skeleton::{CoordSpec, CoordValues, DataVarSpec, SkeletonSpec};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Read a source coordinate array in full, widened to the writer's `CoordValues`
/// (`Int64` / `Float64`). Values are read raw — no CF-time conversion — so they
/// round-trip back to the stored representation.
async fn load_coord(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
    name: &str,
    dtype: &str,
) -> Result<CoordValues, BoxError> {
    let path = if prefix.as_ref().is_empty() {
        format!("/{name}")
    } else {
        format!("/{prefix}/{name}")
    };
    let arr = Array::async_open(store.clone(), &path).await?;
    let full = ArraySubset::new_with_shape(arr.shape().to_vec());
    macro_rules! get {
        ($t:ty) => {
            arr.async_retrieve_array_subset::<Vec<$t>>(&full)
                .await
                .map_err(|e| -> BoxError { Box::new(e) })?
        };
    }
    Ok(match dtype {
        "float32" => CoordValues::Float64(get!(f32).into_iter().map(f64::from).collect()),
        "float64" => CoordValues::Float64(get!(f64)),
        "int8" => int64_from(get!(i8)),
        "int16" => int64_from(get!(i16)),
        "int32" => int64_from(get!(i32)),
        "uint8" => int64_from(get!(u8)),
        "uint16" => int64_from(get!(u16)),
        "uint32" => int64_from(get!(u32)),
        // int64 and anything else fall back to i64.
        _ => CoordValues::Int64(get!(i64)),
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

/// A coordinate as the reader filters it: a CF time decoded to timestamps.
fn as_read(values: &CoordValues, coord: &ZarrArrayMeta) -> ReadCoordValues {
    let raw = match values {
        CoordValues::Int64(v) => ReadCoordValues::Int64(v.clone()),
        CoordValues::Float64(v) => ReadCoordValues::Float64(v.clone()),
    };
    apply_cf_time_conversion(raw, coord.cf_time_attrs.as_ref(), &coord.name)
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
pub async fn materialize_spec(
    plan: &Arc<dyn ExecutionPlan>,
    shape: &WriteShape,
    chunks: Vec<u64>,
) -> Result<SkeletonSpec, BoxError> {
    let (_, zarr) = find_scan(plan).ok_or("plan has no single Zarr scan to read coordinates from")?;
    let meta = zarr.store_meta().ok_or("scan carries no store metadata")?;
    let (store, prefix) = zarr.open_store().await?;

    // Load every source coordinate (needed to resolve filters, which reference the
    // full coordinate set), in cube-axis order.
    let coord_names: Vec<String> = meta.coords.iter().map(|c| c.name.clone()).collect();
    let mut coords: Vec<CoordValues> = Vec::with_capacity(meta.coords.len());
    for c in &meta.coords {
        coords.push(load_coord(&store, &prefix, &c.name, &c.data_type).await?);
    }

    // Resolve any coordinate WHERE filters into per-axis selections.
    let selections: Option<Vec<CoordSelection>> = match zarr.coord_filters() {
        Some(filters) => {
            let read: Vec<ReadCoordValues> =
                coords.iter().zip(&meta.coords).map(|(v, c)| as_read(v, c)).collect();
            let refs = coord_values_to_refs(&read);
            match calculate_coord_ranges(filters, &coord_names, &refs) {
                Some(sels) => Some(sels),
                // A filter that matches no value: the write would be empty.
                None => return Err("coordinate filter selects an empty grid".into()),
            }
        }
        None => None,
    };

    // Build the target grid axes, gathered by selection where present. A periodic
    // axis spans its field's whole range, whatever the selection.
    let grid_coords: Vec<CoordSpec> = shape
        .grid_axis_source
        .iter()
        .zip(&shape.grid_axes)
        .zip(&shape.grid_axis_field)
        .map(|((&src, name), field)| {
            let values = match (field, &selections) {
                (Some(field), _) => {
                    let (first, last) = field_range(field)
                        .ok_or_else(|| format!("'{field}' is not a periodic date_part"))?;
                    CoordValues::Int64((first..=last).collect())
                }
                (None, Some(sels)) => gather(&coords[src], &sels[src]),
                (None, None) => coords[src].clone(),
            };
            Ok(CoordSpec::new(name.clone(), values))
        })
        .collect::<Result<_, BoxError>>()?;

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
pub async fn derive_skeleton_spec(
    plan: &Arc<dyn ExecutionPlan>,
    chunks: Vec<u64>,
) -> Result<SkeletonSpec, BoxError> {
    let shape: WriteShape = derive_write_shape(plan).map_err(|r| Box::new(r) as BoxError)?;
    materialize_spec(plan, &shape, chunks).await
}
