//! Zarr array reader that flattens nD data into Arrow RecordBatches
//!
//! See [`super::schema_inference`] for assumptions about Zarr store structure
//! (1D coordinates, nD data variables as Cartesian product of coordinates).

use tracing::{debug, info, instrument, warn};

use arrow::{
    array::{ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, RecordBatchOptions},
    datatypes::{DataType, Schema, SchemaRef},
};
use datafusion::{
    common::DataFusionError, error::Result, execution::SendableRecordBatchStream,
    physical_plan::stream::RecordBatchStreamAdapter,
};
use futures::stream;
use std::sync::Arc;
use std::time::Instant;
use zarrs::{array::Array, array::ArraySubset, filesystem::FilesystemStore};

use super::cf_time::apply_cf_time_conversion;
use super::coord::{
    calculate_coord_limits, calculate_limited_subset, create_coord_dictionary_typed, CoordValues,
};
use super::filter::{
    calculate_coord_ranges, calculate_filtered_rows, determine_effective_coords,
    match_ranges_to_data_var, resolve_coord_selection, CoordFilters, CoordSelection,
    CoordValuesRef,
};
use super::scan_cache::ScanCache;
use super::schema_inference::discover_arrays;
use super::stats::SharedIoStats;
use super::tracked_store::TrackedStore;

fn zarr_err(e: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Extract the key type from a coordinate field's declared `Dictionary` type.
///
/// Falls back to `Int16` for non-dictionary fields (shouldn't happen for
/// coordinates, but keeps the reader robust to schema surprises).
fn dictionary_key_type(field_type: &DataType) -> &DataType {
    match field_type {
        DataType::Dictionary(key_type, _) => key_type,
        _ => &DataType::Int16,
    }
}

/// Get element size in bytes for a Zarr data type string
fn dtype_to_bytes(dtype: &str) -> u64 {
    match dtype {
        "float32" | "int32" | "uint32" => 4,
        "float64" | "int64" | "uint64" => 8,
        "int16" | "uint16" => 2,
        "int8" | "uint8" => 1,
        _ => 8, // Default assumption
    }
}

/// Get element size in bytes for an Arrow DataType
fn arrow_dtype_to_bytes(dtype: &DataType) -> u64 {
    match dtype {
        DataType::Float32 | DataType::Int32 | DataType::UInt32 => 4,
        DataType::Float64 | DataType::Int64 | DataType::UInt64 => 8,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int8 | DataType::UInt8 => 1,
        _ => 8, // Default assumption
    }
}

// =============================================================================
// Macros for type-dispatched array reading (reduces ~90 lines of duplication)
//
// We maintain both sync and async paths because Tokio uses thread pools for
// file I/O rather than io_uring, adding ~1-5μs overhead per operation. For
// Zarr's many-chunk workloads, sync is faster for local files.
// =============================================================================

/// Macro to read coordinate array values with type dispatch.
/// Handles both sync and async variants of zarrs array retrieval.
macro_rules! read_coord_values {
    // Sync version
    (sync, $arr:expr, $subset:expr, $dtype:expr) => {
        match $dtype {
            "float32" => {
                let vals = $arr
                    .retrieve_array_subset::<Vec<f32>>($subset)
                    .map_err(zarr_err)?;
                CoordValues::Float32(vals)
            }
            "float64" => {
                let vals = $arr
                    .retrieve_array_subset::<Vec<f64>>($subset)
                    .map_err(zarr_err)?;
                CoordValues::Float64(vals)
            }
            "int32" => {
                let vals = $arr
                    .retrieve_array_subset::<Vec<i32>>($subset)
                    .map_err(zarr_err)?;
                // Convert to i64 for uniform handling
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
            }
            "uint32" => {
                let vals = $arr
                    .retrieve_array_subset::<Vec<u32>>($subset)
                    .map_err(zarr_err)?;
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
            }
            "int16" => {
                let vals = $arr
                    .retrieve_array_subset::<Vec<i16>>($subset)
                    .map_err(zarr_err)?;
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
            }
            _ => {
                let vals = $arr
                    .retrieve_array_subset::<Vec<i64>>($subset)
                    .map_err(zarr_err)?;
                CoordValues::Int64(vals)
            }
        }
    };
    // Async version
    (async, $arr:expr, $subset:expr, $dtype:expr) => {
        match $dtype {
            "float32" => {
                let vals = $arr
                    .async_retrieve_array_subset::<Vec<f32>>($subset)
                    .await
                    .map_err(zarr_err)?;
                CoordValues::Float32(vals)
            }
            "float64" => {
                let vals = $arr
                    .async_retrieve_array_subset::<Vec<f64>>($subset)
                    .await
                    .map_err(zarr_err)?;
                CoordValues::Float64(vals)
            }
            "int32" => {
                let vals = $arr
                    .async_retrieve_array_subset::<Vec<i32>>($subset)
                    .await
                    .map_err(zarr_err)?;
                // Convert to i64 for uniform handling
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
            }
            "uint32" => {
                let vals = $arr
                    .async_retrieve_array_subset::<Vec<u32>>($subset)
                    .await
                    .map_err(zarr_err)?;
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
            }
            "int16" => {
                let vals = $arr
                    .async_retrieve_array_subset::<Vec<i16>>($subset)
                    .await
                    .map_err(zarr_err)?;
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
            }
            _ => {
                let vals = $arr
                    .async_retrieve_array_subset::<Vec<i64>>($subset)
                    .await
                    .map_err(zarr_err)?;
                CoordValues::Int64(vals)
            }
        }
    };
}

/// Macro to read data variable array with type dispatch.
/// Returns an ArrayRef based on the Arrow DataType.
macro_rules! read_data_array {
    // Sync version
    (sync, $arr:expr, $subset:expr, $data_type:expr) => {
        match $data_type {
            DataType::Float32 => {
                let vals = $arr
                    .retrieve_array_subset::<Vec<f32>>($subset)
                    .map_err(zarr_err)?;
                Arc::new(Float32Array::from(vals)) as ArrayRef
            }
            DataType::Float64 => {
                let vals = $arr
                    .retrieve_array_subset::<Vec<f64>>($subset)
                    .map_err(zarr_err)?;
                Arc::new(Float64Array::from(vals)) as ArrayRef
            }
            _ => {
                let vals = $arr
                    .retrieve_array_subset::<Vec<i64>>($subset)
                    .map_err(zarr_err)?;
                Arc::new(Int64Array::from(vals)) as ArrayRef
            }
        }
    };
    // Async version
    (async, $arr:expr, $subset:expr, $data_type:expr) => {
        match $data_type {
            DataType::Float32 => {
                let vals = $arr
                    .async_retrieve_array_subset::<Vec<f32>>($subset)
                    .await
                    .map_err(zarr_err)?;
                Arc::new(Float32Array::from(vals)) as ArrayRef
            }
            DataType::Float64 => {
                let vals = $arr
                    .async_retrieve_array_subset::<Vec<f64>>($subset)
                    .await
                    .map_err(zarr_err)?;
                Arc::new(Float64Array::from(vals)) as ArrayRef
            }
            _ => {
                let vals = $arr
                    .async_retrieve_array_subset::<Vec<i64>>($subset)
                    .await
                    .map_err(zarr_err)?;
                Arc::new(Int64Array::from(vals)) as ArrayRef
            }
        }
    };
}

// =============================================================================
// Helper functions to reduce sync/async duplication
// =============================================================================

/// Create an empty result stream when filter values are not found.
///
/// This consolidates the identical logic used in both sync and async paths when
/// a filter value doesn't match any coordinate values, requiring an empty result.
fn create_empty_result_stream(
    schema: &SchemaRef,
    projection: Option<&Vec<usize>>,
) -> Result<SendableRecordBatchStream> {
    let projected_schema = Arc::new(Schema::new(
        projection
            .map(|indices| {
                indices
                    .iter()
                    .map(|&i| schema.field(i).as_ref().clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| schema.fields().iter().map(|f| f.as_ref().clone()).collect()),
    ));
    let batch = RecordBatch::new_empty(projected_schema.clone());
    let stream = stream::iter(vec![Ok(batch)]);
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

/// Build projected schema, excluding coordinates that were skipped due to
/// mixed-dimensionality optimization.
///
/// This consolidates the identical schema building logic used in both sync and async paths.
/// Per-axis chunk extent that every projected data variable agrees on, in effective
/// axis order — the alignment the streaming window should snap its tiles to.
///
/// Only data variables constrain alignment; coordinate columns are 1-D and cheap.
/// Callers reach here under `all_full_cube`, which guarantees each projected var has
/// the same rank as the effective coordinate set, so `chunks[j]` corresponds to
/// effective axis `j`. The rank check below is belt-and-braces: a mismatch means the
/// positional mapping is not valid, and aligning on it would tile the wrong axis.
///
/// Returns all-ones (no constraint) when nothing usable is found, which reproduces
/// the previous unaligned behaviour rather than guessing.
fn projected_chunk_alignment(
    schema: &SchemaRef,
    projected_indices: &[usize],
    coord_names: &[String],
    data_vars: &[super::schema_inference::ZarrArrayMeta],
    sizes: &[u64],
) -> Vec<u64> {
    let per_var: Vec<&[u64]> = projected_indices
        .iter()
        .filter_map(|&i| {
            let name = schema.field(i).name();
            if coord_names.iter().any(|c| c == name) {
                return None;
            }
            data_vars
                .iter()
                .find(|v| &v.name == name)?
                .chunks
                .as_deref()
                .filter(|ch| ch.len() == sizes.len())
        })
        .collect();

    if per_var.is_empty() {
        return vec![1; sizes.len()];
    }
    crate::optimizer::cardinality::budget::chunk_alignment(&per_var, sizes)
}

/// Per-axis tile sizes for the streaming window, snapped to whole chunk rows.
///
/// Shared by the sync and async scans, which plan windows identically. Rounds the
/// tiled axis *down* to a whole number of chunk rows so the memory budget still caps
/// the batch: a tile thinner than its chunk makes every window re-fetch and
/// re-decompress the same chunks, but exceeding the budget risks an OOM, and a
/// re-read is the cheaper failure. When the target cannot hold even one chunk row,
/// the plan stays unaligned and the predicted cost is logged rather than paid
/// silently.
#[allow(clippy::too_many_arguments)]
fn plan_window_tiles(
    sizes: &[u64],
    ceiling_bytes: u64,
    row_width: usize,
    batch_size: usize,
    schema: &SchemaRef,
    projected_indices: &[usize],
    coord_names: &[String],
    data_vars: &[super::schema_inference::ZarrArrayMeta],
) -> Vec<usize> {
    use crate::optimizer::cardinality::budget;

    let align = projected_chunk_alignment(schema, projected_indices, coord_names, data_vars, sizes);

    // Plan-time inference: what this query's geometry actually needs, versus what it
    // is allowed. The window is the smaller — never sized from the machine, because
    // past one chunk row a bigger window reads exactly the same bytes.
    let need_bytes = budget::chunk_row_bytes(sizes, &align, row_width);
    let target = (ceiling_bytes / (row_width as u64).max(1)).max(batch_size as u64);

    let plan = budget::plan_streaming_chunked(sizes, target, &align);
    let amplification = plan.chunk_amplification(sizes, &align);

    if amplification > 1.0 {
        // Actionable rather than mysterious: say what it needs and what it has, so a
        // 37x re-read cliff reads as a setting to change instead of unexplained
        // slowness. This is the failure mode that stayed invisible for so long.
        warn!(
            amplification = format!("{amplification:.1}x"),
            needed = %crate::optimizer::cardinality::budget::human_bytes(need_bytes),
            ceiling = %crate::optimizer::cardinality::budget::human_bytes(ceiling_bytes as u128),
            tiles = ?plan.tiles,
            chunk_alignment = ?align,
            "read window is narrower than one chunk row, so chunks will be re-read; \
             raise the scan memory ceiling (ZARR_MEM_BUDGET_BYTES) to remove this"
        );
    } else {
        debug!(
            needed = %crate::optimizer::cardinality::budget::human_bytes(need_bytes),
            ceiling = %crate::optimizer::cardinality::budget::human_bytes(ceiling_bytes as u128),
            tiles = ?plan.tiles,
            "read window covers whole chunk rows; each chunk read once"
        );
    }

    plan.tiles.iter().map(|&t| t as usize).collect()
}

/// Split a batch into `batch_size`-row slices for emission.
///
/// The read window is sized from chunk geometry so each chunk is fetched once,
/// which makes it far wider than DataFusion's `batch_size`. Emitting it whole would
/// hand downstream operators a single enormous `RecordBatch` and break the batching
/// contract they pipeline against. `RecordBatch::slice` is zero-copy — the slices
/// share the window's buffers — so reading wide and emitting narrow costs nothing.
fn slice_to_batch_size(batch: RecordBatch, batch_size: usize) -> Vec<RecordBatch> {
    let rows = batch.num_rows();
    let step = batch_size.max(1);
    if rows <= step {
        return vec![batch];
    }
    let mut out = Vec::with_capacity(rows.div_ceil(step));
    let mut offset = 0;
    while offset < rows {
        let len = step.min(rows - offset);
        out.push(batch.slice(offset, len));
        offset += len;
    }
    out
}

fn build_projected_schema(
    schema: &SchemaRef,
    projected_indices: &[usize],
    coord_names: &[String],
    effective_coord_indices: &[usize],
) -> SchemaRef {
    let projected_fields: Vec<_> = projected_indices
        .iter()
        .filter_map(|&i| {
            let field = schema.field(i);
            let field_name = field.name();
            // Check if this is a coordinate that was skipped
            if let Some(coord_idx) = coord_names.iter().position(|n| n == field_name) {
                if !effective_coord_indices.contains(&coord_idx) {
                    return None; // Skip coordinates not in effective set
                }
            }
            Some(field.clone())
        })
        .collect();
    Arc::new(Schema::new(projected_fields))
}

/// Convert CoordValues to CoordValuesRef for filtering.
///
/// This consolidates the repeated conversion logic in both sync and async paths.
pub(crate) fn coord_values_to_refs(coord_values: &[CoordValues]) -> Vec<CoordValuesRef<'_>> {
    coord_values
        .iter()
        .map(|v| match v {
            CoordValues::Int64(vals) => CoordValuesRef::Int64(vals),
            CoordValues::Float32(vals) => CoordValuesRef::Float32(vals),
            CoordValues::Float64(vals) => CoordValuesRef::Float64(vals),
            CoordValues::TimestampMicros(vals) => CoordValuesRef::TimestampMicros(vals),
            CoordValues::Compact {
                encoding,
                is_timestamp,
            } => CoordValuesRef::Compact {
                encoding: *encoding,
                is_timestamp: *is_timestamp,
            },
        })
        .collect()
}

/// Calculate effective sizes and filtered rows based on coordinate selections.
///
/// Returns (effective_coord_sizes, effective_rows).
fn calculate_effective_sizes(
    coord_sizes: &[usize],
    selections: &Option<Vec<CoordSelection>>,
) -> (Vec<usize>, usize) {
    if let Some(ref sels) = selections {
        let sizes: Vec<usize> = sels.iter().map(|s| s.len()).collect();
        let rows = calculate_filtered_rows(sels);
        (sizes, rows)
    } else {
        (coord_sizes.to_vec(), coord_sizes.iter().product())
    }
}

/// Extract filtered coordinate values based on selections.
///
/// `Range` selections slice contiguously; `Indices` selections gather scattered positions.
fn extract_filtered_coords(
    coord_values: Vec<CoordValues>,
    selections: &Option<Vec<CoordSelection>>,
) -> Vec<CoordValues> {
    if let Some(ref sels) = selections {
        coord_values
            .iter()
            .zip(sels.iter())
            .map(|(values, sel)| match sel {
                CoordSelection::Range(start, end) => values.slice(*start, *end),
                CoordSelection::Indices(indices) => values.gather(indices),
            })
            .collect()
    } else {
        coord_values
    }
}

/// Expand coordinate selections into [`ReadPlan`]s for zarr reading.
///
/// All-Range selections produce a single plan (one read, `Keep::All`).
///
/// A selection containing scattered `Indices` (e.g. the surviving time positions
/// from `EXTRACT(month..)=12 AND EXTRACT(day..)=15 ...`) is handled one of two
/// ways:
///
/// - **Chunk-bucketed** (the heuristic): when the `Indices` coordinate maps to the
///   *outer* data-var dimension (dim 0) and that dimension's chunk extent is known
///   and > 1, the survivors are grouped per chunk via [`bucket_outer_indices`]. Each
///   occupied chunk becomes one read of a tight window plus a `Keep::Offsets` mask
///   that gathers the surviving rows. A chunk holding K survivors is read once
///   instead of K times, and empty chunks are skipped.
/// - **Per-index fallback** (previous behavior): otherwise, one single-step read
///   per surviving index with `Keep::All`. Used when chunk geometry is unknown, the
///   chunk length is 1, or the scattered dimension is not dim 0 (where the
///   `Keep::Offsets` row layout would not hold).
fn build_read_plans(
    selections: &[CoordSelection],
    coord_sizes: &[usize],
    data_var_shape: &[u64],
    data_var_chunks: Option<&[u64]>,
) -> Vec<ReadPlan> {
    // Build the ArraySubset for `selections` with coord `pos` replaced by `repl`.
    let subset_for = |pos: usize, repl: CoordSelection| -> ArraySubset {
        let expanded: Vec<CoordSelection> = selections
            .iter()
            .enumerate()
            .map(|(i, s)| if i == pos { repl.clone() } else { s.clone() })
            .collect();
        let array_ranges = match_ranges_to_data_var(coord_sizes, &expanded, data_var_shape)
            .unwrap_or_else(|| data_var_shape.iter().map(|&s| 0..s).collect());
        ArraySubset::new_with_ranges(&array_ranges)
    };

    let indices_pos = selections
        .iter()
        .position(|s| matches!(s, CoordSelection::Indices(_)));

    let pos = match indices_pos {
        None => {
            let array_ranges = match_ranges_to_data_var(coord_sizes, selections, data_var_shape)
                .unwrap_or_else(|| data_var_shape.iter().map(|&s| 0..s).collect());
            return vec![ReadPlan {
                subset: ArraySubset::new_with_ranges(&array_ranges),
                keep: Keep::All,
            }];
        }
        Some(pos) => pos,
    };

    let CoordSelection::Indices(ref indices) = selections[pos] else {
        unreachable!()
    };

    // Chunk-bucketing applies only when the scattered coord is the OUTER data-var
    // dimension (dim 0): the `Keep::Offsets` gather assumes each outer step is a
    // contiguous inner block, which holds only when that step is the slowest-varying
    // (row-major) axis. Map the coord to its data-var dim by size.
    let outer_dim = data_var_shape
        .iter()
        .position(|&d| d as usize == coord_sizes[pos]);
    let chunk_len = match (data_var_chunks, outer_dim) {
        (Some(chunks), Some(0)) => chunks.first().copied().map(|c| c as usize),
        _ => None,
    };

    if let Some(chunk_len) = chunk_len.filter(|&c| c > 1) {
        let buckets = bucket_outer_indices(indices, chunk_len);
        if !buckets.is_empty() {
            return buckets
                .into_iter()
                .map(|(start, end, offsets)| ReadPlan {
                    subset: subset_for(pos, CoordSelection::Range(start, end)),
                    keep: Keep::Offsets {
                        window_len: end - start,
                        offsets,
                    },
                })
                .collect();
        }
    }

    // Per-index fallback: one single-step read per surviving index.
    indices
        .iter()
        .map(|&idx| ReadPlan {
            subset: subset_for(pos, CoordSelection::Range(idx, idx + 1)),
            keep: Keep::All,
        })
        .collect()
}

/// Group sorted outer-axis indices into chunk-aligned read windows.
///
/// Given surviving indices on the outer (chunked) dimension — e.g. the scattered
/// time positions produced by `EXTRACT(month..)=12 AND EXTRACT(day..)=15 ...` —
/// and the chunk extent `chunk_len` along that dimension, returns one bucket per
/// chunk that contains at least one survivor.
///
/// Each bucket is `(window_start, window_end, offsets)`:
/// - `[window_start, window_end)` is a half-open window that never crosses a chunk
///   boundary (so reading it touches exactly one chunk), tightened to the span of
///   survivors it covers; and
/// - `offsets` are the survivors' positions *relative to `window_start`*, used to
///   gather the surviving rows out of the window after the read.
///
/// This is what turns "one read per surviving index" into "one read per occupied
/// chunk": a chunk holding K survivors is fetched/decompressed once instead of K
/// times, and a chunk holding none is never touched. `indices` must be sorted
/// ascending and de-duplicated (as produced by the filter resolver). Returns an
/// empty vec when `chunk_len == 0` (caller should fall back to the per-index path).
fn bucket_outer_indices(indices: &[usize], chunk_len: usize) -> Vec<(usize, usize, Vec<usize>)> {
    let mut buckets: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    if chunk_len == 0 {
        return buckets;
    }
    for &idx in indices {
        match buckets.last_mut() {
            // Same chunk as the open bucket: extend its window and record the offset.
            Some((start, end, offsets)) if idx / chunk_len == *start / chunk_len => {
                *end = idx + 1;
                offsets.push(idx - *start);
            }
            // First survivor of a new chunk: open a fresh bucket.
            _ => buckets.push((idx, idx + 1, vec![0])),
        }
    }
    buckets
}

/// What to keep from the array returned for a single [`ReadPlan`] subset read.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Keep {
    /// Keep the whole read (contiguous Range subsets, and the per-index fallback).
    All,
    /// The subset spans a chunk window of `window_len` outer steps; keep only the
    /// survivors at these `offsets` (relative to the window start). Each offset
    /// expands to a contiguous inner block of `len / window_len` rows.
    Offsets {
        window_len: usize,
        offsets: Vec<usize>,
    },
}

/// A single planned read: which `subset` to retrieve, and which rows to `keep`
/// from the result (see [`Keep`]).
struct ReadPlan {
    subset: ArraySubset,
    keep: Keep,
}

/// Gather the surviving rows from one subset's read result per its [`Keep`] rule.
///
/// For `Keep::All` the array is returned unchanged. For `Keep::Offsets` the array
/// is a flattened row-major block of `window_len` outer steps; each kept offset
/// selects the contiguous `inner = len / window_len` rows belonging to that outer
/// step, and the kept blocks are concatenated in offset order. This is what makes
/// a chunk-granular read produce exactly the per-index filter result.
fn apply_keep(array: ArrayRef, keep: &Keep) -> Result<ArrayRef> {
    match keep {
        Keep::All => Ok(array),
        Keep::Offsets {
            window_len,
            offsets,
        } => {
            let len = array.len();
            // window_len always divides len (len = window_len * inner). Guard a
            // zero window so a degenerate plan can't panic on divide-by-zero.
            let inner = if *window_len == 0 {
                0
            } else {
                len / *window_len
            };
            let parts: Vec<ArrayRef> = offsets
                .iter()
                .map(|&off| array.slice(off * inner, inner))
                .collect();
            let refs: Vec<&dyn arrow::array::Array> = parts.iter().map(|a| a.as_ref()).collect();
            arrow::compute::concat(&refs).map_err(DataFusionError::from)
        }
    }
}

/// Recalculate query coordinate sizes and rows for effective coordinates only.
///
/// This handles mixed-dimensionality optimization where only a subset of coordinates
/// are needed for the projected columns.
fn calculate_query_coord_sizes(
    coord_sizes: &[usize],
    effective_coord_sizes: &[usize],
    effective_coord_indices: &[usize],
    effective_rows: usize,
) -> (Vec<usize>, usize) {
    if effective_coord_indices.len() < coord_sizes.len() {
        let sizes: Vec<usize> = effective_coord_indices
            .iter()
            .map(|&i| effective_coord_sizes[i])
            .collect();
        let rows: usize = sizes.iter().product();
        info!(
            all_coords = coord_sizes.len(),
            effective_coords = effective_coord_indices.len(),
            query_rows = rows,
            original_rows = effective_rows,
            "Using reduced coordinate set for variable dimensionality"
        );
        (sizes, rows)
    } else {
        (effective_coord_sizes.to_vec(), effective_rows)
    }
}

/// Create the final result batch, handling empty projections (e.g., COUNT(*)).
fn create_result_batch(
    projected_schema: SchemaRef,
    result_arrays: Vec<ArrayRef>,
    final_rows: usize,
) -> Result<RecordBatch> {
    if result_arrays.is_empty() {
        info!(final_rows, "Empty projection - returning row count only");
        Ok(RecordBatch::try_new_with_options(
            projected_schema,
            result_arrays,
            &RecordBatchOptions::new().with_row_count(Some(final_rows)),
        )?)
    } else {
        Ok(RecordBatch::try_new(projected_schema, result_arrays)?)
    }
}

/// Apply limit to result arrays by slicing them.
fn apply_limit_to_arrays(
    result_arrays: Vec<ArrayRef>,
    limit: Option<usize>,
    max_rows: usize,
) -> Vec<ArrayRef> {
    if let Some(limit) = limit {
        let limit = limit.min(max_rows);
        result_arrays
            .into_iter()
            .map(|arr| arr.slice(0, limit))
            .collect()
    } else {
        result_arrays
    }
}

/// Intersect a partition's outer-dimension slice into the per-coordinate
/// selections that the rest of `read_zarr` consumes (effective sizes, filtered
/// coord values, and data-var subsets all derive from these).
///
/// - `coord_ranges = None`  => no filters yet; start from full-range selections
///   so we can still restrict the outer coordinate.
/// - `coord_ranges = Some`  => existing per-coordinate selections (one per coord).
/// - `outer_coord_idx`      => index (in coord order) of the coord to restrict.
/// - `partition_range`      => half-open `[start, end)` index slice on that coord.
/// - `coord_sizes`          => full length of each coordinate (to build defaults).
fn restrict_to_partition(
    coord_ranges: Option<Vec<CoordSelection>>,
    outer_coord_idx: usize,
    partition_sel: &CoordSelection,
    coord_sizes: &[usize],
) -> Option<Vec<CoordSelection>> {
    // FILL 5b: existing selections, or one full-range selection per coordinate.
    let mut sels: Vec<CoordSelection> = coord_ranges.unwrap_or_else(|| {
        coord_sizes
            .iter()
            .map(|&size| CoordSelection::Range(0, size))
            .collect()
    });

    // REPLACE the outer coordinate's selection with this partition's selection.
    // The partition selection is the head-resolved surviving sub-set for this
    // partition (a subset of the full filter result), so it is authoritative —
    // there is a single resolution site (the head), and the worker's own
    // re-resolution of the outer filter, if any, is discarded here. For the
    // unfiltered geometry case the partition selection is the partition's axis
    // slice, which is likewise exactly what this partition should read.
    sels[outer_coord_idx] = partition_sel.clone();

    Some(sels)
}

/// Narrow `coord_ranges` to a partition's outer-dimension slice, if any.
///
/// Maps the data var's axis-0 to its coordinate by size (coords are ordered to
/// match the axes, so the outer coordinate is the first whose size == data var
/// `shape[0]`) and applies the partition selection via [`restrict_to_partition`].
/// Shared by the sync ([`read_zarr`]) and async ([`read_zarr_async`]) readers.
fn apply_partition_selection(
    coord_ranges: Option<Vec<CoordSelection>>,
    partition_selection: Option<CoordSelection>,
    partition_extra: &[(usize, CoordSelection)],
    store_meta: &super::schema_inference::ZarrStoreMeta,
    coord_sizes: &[usize],
) -> Option<Vec<CoordSelection>> {
    // Step 1: apply the outer-axis selection (the axis-0 coordinate, identified by
    // size-match) — unchanged single-axis behavior, keeps the outer window read.
    let after_outer = match partition_selection {
        Some(sel) => {
            let outer_axis_len = store_meta
                .data_vars
                .first()
                .and_then(|dv| dv.shape.first())
                .copied();
            let outer_coord_idx =
                outer_axis_len.and_then(|len| coord_sizes.iter().position(|&s| s as u64 == len));
            match outer_coord_idx {
                Some(idx) => restrict_to_partition(coord_ranges, idx, &sel, coord_sizes),
                // Can't identify the outer coordinate => can't safely slice. scan()
                // only partitions when the dim is known, so this shouldn't happen;
                // stay defensive and fall back to no slicing.
                None => coord_ranges,
            }
        }
        None => coord_ranges,
    };

    // Step 2: apply any inner-axis (multi-axis box) restrictions. Each is a
    // chunk-aligned slice of the *surviving* set on that coordinate (the planner
    // splits the resolved selection), so — like the outer axis — it is the
    // authoritative subset for this partition and REPLACES that coord's selection.
    // Inner coords are read in full (no window optimization), so the replacement
    // is applied directly to the per-coord vector.
    if partition_extra.is_empty() {
        return after_outer;
    }
    let mut sels = after_outer.unwrap_or_else(|| {
        coord_sizes
            .iter()
            .map(|&size| CoordSelection::Range(0, size))
            .collect()
    });
    for (idx, sel) in partition_extra {
        if *idx < sels.len() {
            sels[*idx] = sel.clone();
        }
    }
    Some(sels)
}

/// Identify the OUTER coordinate (axis-0 of the first data var) by size-match.
/// Returns its index into `store_meta.coords`, or `None` if no coord matches.
fn identify_outer_coord(store_meta: &ZarrStoreMeta) -> Option<usize> {
    let outer_axis_len = store_meta
        .data_vars
        .first()
        .and_then(|dv| dv.shape.first())
        .copied()?;
    store_meta
        .coords
        .iter()
        .position(|c| c.shape.first().copied() == Some(outer_axis_len))
}

/// Plan a worker-side read of *only* this partition's slice of the outer coord.
///
/// The head ships the surviving outer-axis selection, so a worker never needs the
/// full (~10 MB) outer coordinate: it reads one contiguous window and gathers its
/// slice from that. Returns `(read_start, read_end, extract_sel)` where
/// `[read_start, read_end)` is the half-open window to fetch on the outer axis and
/// `extract_sel` is the selection *relative to that window* used to pull out the
/// surviving values.
///
/// - `Range(s, e)` => read exactly `[s, e)`; extract is the identity `Range(0, e-s)`.
/// - `Indices(v)`  => read the bounding window `[min, max+1)` in one request, then
///   gather with offsets shifted into the window. One bounding read is never worse
///   than the old full-axis read and is far cheaper when the indices cluster;
///   reading each scattered index separately would be thousands of tiny requests.
///
/// An empty selection reads nothing (`read_start == read_end`).
fn plan_outer_coord_read(sel: &CoordSelection) -> (u64, u64, CoordSelection) {
    match sel {
        CoordSelection::Range(s, e) => (
            *s as u64,
            *e as u64,
            CoordSelection::Range(0, e.saturating_sub(*s)),
        ),
        CoordSelection::Indices(v) => {
            let (Some(&lo), Some(&hi)) = (v.iter().min(), v.iter().max()) else {
                return (0, 0, CoordSelection::Indices(Vec::new()));
            };
            let rel: Vec<usize> = v.iter().map(|&i| i - lo).collect();
            (lo as u64, (hi + 1) as u64, CoordSelection::Indices(rel))
        }
    }
}

/// `[start, end)` widened to whole chunks of `chunk_len` within `len`: what
/// decoding the slice reads anyway, and the same for every partition in those
/// chunks. Unknown chunking leaves it as is.
fn widen_to_chunks(start: u64, end: u64, chunk_len: Option<u64>, len: u64) -> (u64, u64) {
    match chunk_len.filter(|&c| c > 0) {
        Some(c) if start < end => ((start / c) * c, end.div_ceil(c).saturating_mul(c).min(len)),
        _ => (start, end),
    }
}

fn shift_selection(sel: CoordSelection, by: usize) -> CoordSelection {
    match sel {
        CoordSelection::Range(s, e) => CoordSelection::Range(s + by, e + by),
        CoordSelection::Indices(v) => {
            CoordSelection::Indices(v.into_iter().map(|i| i + by).collect())
        }
    }
}

/// A view of `filters` with the outer coord's filter removed, when that coord's
/// selection is supplied by the partition (and so resolved on the head).
///
/// Re-resolving the outer filter on the worker would be both wasteful and unsound:
/// the worker only holds a partial slice of the outer coord, so `calculate_coord_ranges`
/// could mis-resolve (or wrongly return "no match") against it. Its selection is
/// REPLACED by [`apply_partition_selection`] regardless, so we drop it here.
/// Returns a borrow when there is nothing to strip (the common path).
fn filters_without_outer<'a>(
    filters: &'a CoordFilters,
    outer_idx: Option<usize>,
    coord_names: &[String],
) -> std::borrow::Cow<'a, CoordFilters> {
    match outer_idx {
        Some(i) if filters.filters.contains_key(&coord_names[i]) => {
            let mut f = filters.clone();
            f.filters.remove(&coord_names[i]);
            std::borrow::Cow::Owned(f)
        }
        _ => std::borrow::Cow::Borrowed(filters),
    }
}

/// Build the selections used to extract coordinate *values* from what was read.
///
/// Identical to `coord_ranges` (the absolute, data-var-read selections) except at
/// the outer coord, where we only read a window and so must extract with the
/// window-relative selection. Returns `coord_ranges` unchanged when the outer
/// coord was not sliced (no partitioning, or unidentifiable outer coord).
fn make_extract_selections(
    coord_ranges: &Option<Vec<CoordSelection>>,
    outer_idx: Option<usize>,
    outer_extract_sel: &Option<CoordSelection>,
) -> Option<Vec<CoordSelection>> {
    match (coord_ranges, outer_idx, outer_extract_sel) {
        (Some(sels), Some(oi), Some(esel)) => {
            let mut e = sels.clone();
            e[oi] = esel.clone();
            Some(e)
        }
        _ => coord_ranges.clone(),
    }
}

/// Outcome of resolving the outer-axis filter on the head, used to decide how to
/// partition the scan (see `split_selection`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OuterSelection {
    /// The outer filter resolved to this selection; partition by splitting it.
    Resolved(CoordSelection),
    /// A *present* outer filter matched no values — the scan yields nothing.
    Empty,
    /// No identifiable outer coordinate, or no filter on it. Fall back to
    /// whole-axis geometry partitioning; NO coordinate is read in this case.
    Unfiltered,
}

/// Read ONLY the outer coordinate from a LOCAL store and resolve its filter to a
/// selection on the outer axis. Run on the head at plan time so partitions can be
/// drawn around the *surviving* set instead of the full axis.
///
/// Reads nothing when there is no outer filter (returns `Unfiltered`), so
/// unfiltered scans keep paying zero plan-time I/O.
pub fn resolve_outer_selection(
    store_path: &str,
    store_meta: &ZarrStoreMeta,
    coord_filters: Option<&CoordFilters>,
) -> Result<OuterSelection> {
    let Some(idx) = identify_outer_coord(store_meta) else {
        return Ok(OuterSelection::Unfiltered);
    };
    let coord = &store_meta.coords[idx];
    let Some(filter) = coord_filters.and_then(|f| f.get(&coord.name)) else {
        return Ok(OuterSelection::Unfiltered);
    };

    // Read just this one coordinate's values (the whole axis).
    let store = Arc::new(FilesystemStore::new(store_path).map_err(zarr_err)?);
    let arr = Array::open(store, &format!("/{}", coord.name)).map_err(zarr_err)?;
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let raw = read_coord_values!(sync, arr, &subset, coord.data_type.as_str());
    let values = apply_cf_time_conversion(raw, coord.cf_time_attrs.as_ref(), &coord.name);

    let refs = coord_values_to_refs(std::slice::from_ref(&values));
    Ok(
        match resolve_coord_selection(&coord.name, Some(filter.as_slice()), &refs[0]) {
            Some(sel) => OuterSelection::Resolved(sel),
            None => OuterSelection::Empty,
        },
    )
}

/// Async (remote-store) variant of [`resolve_outer_selection`]. Reads only the
/// outer coordinate over the network when an outer filter is present.
pub async fn resolve_outer_selection_async(
    store: AsyncReadableListableStorage,
    prefix: &ObjectPath,
    store_meta: &ZarrStoreMeta,
    coord_filters: Option<&CoordFilters>,
) -> Result<OuterSelection> {
    let Some(idx) = identify_outer_coord(store_meta) else {
        return Ok(OuterSelection::Unfiltered);
    };
    let coord = &store_meta.coords[idx];
    let Some(filter) = coord_filters.and_then(|f| f.get(&coord.name)) else {
        return Ok(OuterSelection::Unfiltered);
    };

    let array_path = if prefix.as_ref().is_empty() {
        format!("/{}", coord.name)
    } else {
        format!("/{}/{}", prefix, coord.name)
    };
    let arr = Array::async_open(store, &array_path)
        .await
        .map_err(zarr_err)?;
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let raw = read_coord_values!(async, arr, &subset, coord.data_type.as_str());
    let values = apply_cf_time_conversion(raw, coord.cf_time_attrs.as_ref(), &coord.name);

    let refs = coord_values_to_refs(std::slice::from_ref(&values));
    Ok(
        match resolve_coord_selection(&coord.name, Some(filter.as_slice()), &refs[0]) {
            Some(sel) => OuterSelection::Resolved(sel),
            None => OuterSelection::Empty,
        },
    )
}

/// Constant context shared by every batch produced from one scan.
///
/// Block 0 Phase 1 extracted the per-batch column-building logic into
/// [`build_batch`] so each batch runs identical code; this struct carries the
/// inputs that do *not* change between batches. Phase 2 will call `build_batch`
/// once per outer-axis sub-selection (lazily), reusing one `BatchCtx`.
struct BatchCtx<'a> {
    store: &'a Arc<TrackedStore<FilesystemStore>>,
    store_meta: &'a ZarrStoreMeta,
    schema: &'a SchemaRef,
    projected_schema: &'a SchemaRef,
    projected_indices: &'a [usize],
    coord_names: &'a [String],
    effective_coord_indices: &'a [usize],
    coord_sizes: &'a [usize],
    stats: &'a Option<SharedIoStats>,
}

/// Build a single `RecordBatch` from a resolved selection state.
///
/// The varying inputs (`coord_ranges`, `query_coord_sizes`,
/// `filtered_coord_values`, `final_rows`) describe one slice of the scan; in
/// Phase 1 they cover the whole selection and this is called once. Phase 2 will
/// vary them per outer-axis sub-selection. Behavior is identical to the old
/// inlined block.
fn build_batch(
    ctx: &BatchCtx<'_>,
    coord_ranges: &Option<Vec<CoordSelection>>,
    query_coord_sizes: &[usize],
    filtered_coord_values: &[CoordValues],
    final_rows: usize,
    limit: Option<usize>,
    query_rows: usize,
) -> Result<RecordBatch> {
    let mut result_arrays: Vec<ArrayRef> = Vec::new();

    for idx in ctx.projected_indices {
        let field = ctx.schema.field(*idx);
        let field_name = field.name();

        // Check if this is a coordinate
        if let Some(coord_idx) = ctx.coord_names.iter().position(|n| n == field_name) {
            // Skip coordinates not relevant to the projected variables
            if !ctx.effective_coord_indices.contains(&coord_idx) {
                debug!(field = %field_name, "Skipping coordinate not used by projected variables");
                continue;
            }

            // Find position of this coordinate in the effective set
            let query_coord_idx = ctx
                .effective_coord_indices
                .iter()
                .position(|&i| i == coord_idx)
                .unwrap();

            // Create DictionaryArray for coordinate (memory efficient).
            // Key width comes from the schema field so the array matches the
            // declared dictionary key type (Int16/Int32/Int64).
            let dict_array = create_coord_dictionary_typed(
                &filtered_coord_values[coord_idx],
                query_coord_idx,
                query_coord_sizes,
                final_rows,
                dictionary_key_type(field.data_type()),
            );
            result_arrays.push(dict_array);
        } else {
            // Data variable - read filtered subset
            let read_start = Instant::now();
            let arr =
                Array::open(ctx.store.clone(), &format!("/{}", field_name)).map_err(zarr_err)?;
            let data_var_shape = arr.shape();

            // Chunk shape of this variable (from discovery metadata). Used by the
            // chunk-bucketing heuristic to coalesce scattered outer-axis (e.g. time)
            // reads onto chunk boundaries instead of one read per surviving index.
            let data_var_chunks: Option<Vec<u64>> = ctx
                .store_meta
                .data_vars
                .iter()
                .find(|v| v.name == *field_name)
                .and_then(|v| v.chunks.clone());
            debug!(field = %field_name, chunks = ?data_var_chunks, "Data variable chunk shape");

            // Build read plans: one for all-Range filters; for scattered outer-axis
            // (e.g. time) indices, one per occupied chunk with a keep-mask.
            let plans = if let Some(sels) = coord_ranges {
                build_read_plans(
                    sels,
                    ctx.coord_sizes,
                    data_var_shape,
                    data_var_chunks.as_deref(),
                )
            } else {
                vec![ReadPlan {
                    subset: ArraySubset::new_with_shape(arr.shape().to_vec()),
                    keep: Keep::All,
                }]
            };
            // Elements actually fetched (chunk windows may include non-survivors).
            let num_elements: u64 = plans.iter().map(|p| p.subset.num_elements()).sum();

            let mut parts: Vec<ArrayRef> = Vec::with_capacity(plans.len());
            for plan in &plans {
                let raw = read_data_array!(sync, arr, &plan.subset, field.data_type());
                parts.push(apply_keep(raw, &plan.keep)?);
            }
            let array: ArrayRef = if parts.len() == 1 {
                parts.remove(0)
            } else {
                let refs: Vec<&dyn arrow::array::Array> =
                    parts.iter().map(|a| a.as_ref()).collect();
                arrow::compute::concat(&refs).map_err(DataFusionError::from)?
            };

            if let Some(s) = ctx.stats {
                let bytes = num_elements * arrow_dtype_to_bytes(field.data_type());
                s.record_data(bytes, read_start.elapsed());
            }
            result_arrays.push(array);
        }
    }

    // Apply limit if specified (slice the already-filtered arrays)
    let result_arrays = apply_limit_to_arrays(result_arrays, limit, query_rows);

    create_result_batch(ctx.projected_schema.clone(), result_arrays, final_rows)
}

/// Chop an outer-axis selection into contiguous sub-selections of at most
/// `max_steps` surviving indices each (the last may be shorter). Order and
/// membership are preserved, so concatenating the per-window batches reproduces
/// the un-windowed result. An empty selection yields one empty window so the scan
/// still emits a single empty batch with the projected schema.
fn window_outer_selection(sel: &CoordSelection, max_steps: usize) -> Vec<CoordSelection> {
    let max_steps = max_steps.max(1);
    match sel {
        CoordSelection::Range(s, e) => {
            if e <= s {
                return vec![CoordSelection::Range(*s, *s)];
            }
            let mut out = Vec::with_capacity((e - s).div_ceil(max_steps));
            let mut a = *s;
            while a < *e {
                let b = (a + max_steps).min(*e);
                out.push(CoordSelection::Range(a, b));
                a = b;
            }
            out
        }
        CoordSelection::Indices(v) => {
            if v.is_empty() {
                return vec![CoordSelection::Indices(Vec::new())];
            }
            v.chunks(max_steps)
                .map(|c| CoordSelection::Indices(c.to_vec()))
                .collect()
        }
    }
}

/// One axis's window within a tiled batch: the sliced coordinate selection plus
/// the survivor offset/len into that axis's filtered coordinate values (the same
/// offset the single-axis path advances by today).
#[derive(Debug, Clone, PartialEq, Eq)]
struct AxisWindow {
    sel: CoordSelection,
    offset: usize,
    len: usize,
}

/// One axis's window list, tiled at `tile` steps, with cumulative survivor offsets.
/// Reuses [`window_outer_selection`], so a `tile` covering the whole axis yields
/// exactly one window.
fn axis_windows(sel: &CoordSelection, tile: usize) -> Vec<AxisWindow> {
    let mut offset = 0;
    window_outer_selection(sel, tile)
        .into_iter()
        .map(|w| {
            let len = w.len();
            let aw = AxisWindow {
                sel: w,
                offset,
                len,
            };
            offset += len;
            aw
        })
        .collect()
}

/// A lazy, row-major (axis 0 slowest) odometer over the tile grid: the Cartesian
/// product of each axis's windows. Yields on demand so a scan never materializes
/// `n_batches` windows — the whole point on cubes where that count is huge. When
/// every inner axis is whole (one window each), the grid collapses to the current
/// outer-only line, reproducing today's behavior exactly.
struct TileGrid {
    per_axis: Vec<Vec<AxisWindow>>,
    cursor: Vec<usize>,
    done: bool,
}

impl TileGrid {
    fn new(sels: &[CoordSelection], tiles: &[usize]) -> Self {
        let per_axis: Vec<Vec<AxisWindow>> = sels
            .iter()
            .zip(tiles)
            .map(|(s, &t)| axis_windows(s, t))
            .collect();
        let cursor = vec![0; per_axis.len()];
        TileGrid {
            per_axis,
            cursor,
            done: false,
        }
    }

    /// Total batches = product of per-axis window counts.
    fn n_batches(&self) -> usize {
        self.per_axis.iter().map(|w| w.len()).product()
    }

    /// The current batch's per-axis windows, then advance the odometer (innermost
    /// axis fastest). `None` once the grid is exhausted.
    fn next(&mut self) -> Option<Vec<AxisWindow>> {
        if self.done {
            return None;
        }
        let batch: Vec<AxisWindow> = self
            .cursor
            .iter()
            .enumerate()
            .map(|(ax, &i)| self.per_axis[ax][i].clone())
            .collect();

        // Odometer carry, from the innermost axis outward.
        let mut ax = self.cursor.len();
        loop {
            if ax == 0 {
                self.done = true;
                break;
            }
            ax -= 1;
            self.cursor[ax] += 1;
            if self.cursor[ax] < self.per_axis[ax].len() {
                break;
            }
            self.cursor[ax] = 0;
        }
        Some(batch)
    }
}

#[cfg(test)]
mod tilegrid_tests {
    use super::*;
    use std::collections::HashSet;

    fn sel_members(s: &CoordSelection) -> Vec<usize> {
        match s {
            CoordSelection::Range(a, b) => (*a..*b).collect(),
            CoordSelection::Indices(v) => v.clone(),
        }
    }

    /// The Cartesian product of per-axis members = the selection's index tuples.
    fn tuple_set(sels: &[CoordSelection]) -> HashSet<Vec<usize>> {
        let members: Vec<Vec<usize>> = sels.iter().map(sel_members).collect();
        let mut out = HashSet::new();
        if members.iter().any(|m| m.is_empty()) {
            return out; // an empty axis -> no tuples
        }
        let mut cur = vec![0usize; members.len()];
        fn rec(m: &[Vec<usize>], ax: usize, cur: &mut Vec<usize>, out: &mut HashSet<Vec<usize>>) {
            if ax == m.len() {
                out.insert(cur.clone());
                return;
            }
            for &v in &m[ax] {
                cur[ax] = v;
                rec(m, ax + 1, cur, out);
            }
        }
        rec(&members, 0, &mut cur, &mut out);
        out
    }

    #[test]
    fn tilegrid_reproduces_full_row_set() {
        let sels = vec![
            CoordSelection::Range(0, 5),
            CoordSelection::Indices(vec![1, 3, 4, 7]),
            CoordSelection::Range(2, 9),
        ];
        let tiles = [2usize, 3, 4]; // every axis tiled
        let expected_batches: usize = sels
            .iter()
            .zip(&tiles)
            .map(|(s, &t)| axis_windows(s, t).len())
            .product();

        let mut grid = TileGrid::new(&sels, &tiles);
        assert_eq!(grid.n_batches(), expected_batches);

        let full = tuple_set(&sels);
        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        let mut n = 0;
        while let Some(batch) = grid.next() {
            n += 1;
            // each window's len <= its tile
            for (aw, &t) in batch.iter().zip(&tiles) {
                assert!(aw.len <= t, "window len {} exceeds tile {}", aw.len, t);
            }
            let batch_sels: Vec<CoordSelection> = batch.iter().map(|w| w.sel.clone()).collect();
            for tup in tuple_set(&batch_sels) {
                assert!(seen.insert(tup), "batches overlap");
            }
        }
        assert_eq!(n, expected_batches);
        assert_eq!(seen, full); // disjoint exact cover
    }

    #[test]
    fn tilegrid_degenerates_to_outer_only() {
        let sels = vec![CoordSelection::Range(0, 10), CoordSelection::Range(0, 4)];
        let tiles = [3usize, 4]; // inner axis whole (extent 4)
        let mut grid = TileGrid::new(&sels, &tiles);
        let outer = window_outer_selection(&sels[0], 3);
        assert_eq!(grid.n_batches(), outer.len());
        for w in &outer {
            let b = grid.next().unwrap();
            assert_eq!(b[0].sel, *w);
            assert_eq!(b[1].sel, CoordSelection::Range(0, 4)); // inner untouched
        }
        assert!(grid.next().is_none());
    }

    #[test]
    fn axis_windows_offsets_and_bounds() {
        let ws = axis_windows(&CoordSelection::Range(0, 7), 3);
        assert_eq!(
            ws.iter().map(|w| (w.offset, w.len)).collect::<Vec<_>>(),
            vec![(0, 3), (3, 3), (6, 1)]
        );
        // Indices tile: chunks with cumulative offsets.
        let wi = axis_windows(&CoordSelection::Indices(vec![2, 5, 8, 11, 14]), 2);
        assert_eq!(
            wi.iter().map(|w| (w.offset, w.len)).collect::<Vec<_>>(),
            vec![(0, 2), (2, 2), (4, 1)]
        );
    }

    #[test]
    fn tilegrid_empty_selection_yields_one_empty_batch() {
        let sels = vec![CoordSelection::Range(3, 3), CoordSelection::Range(0, 4)];
        let mut grid = TileGrid::new(&sels, &[4usize, 4]);
        let b = grid.next().unwrap();
        assert_eq!(b[0].len, 0);
        assert!(grid.next().is_none());
    }
}

/// Lazy, memory-bounded scan state: produces one `RecordBatch` per outer-axis
/// window on demand (Block 0 Phase 2). Owns everything `build_batch` needs so the
/// stream can outlive `read_zarr`'s stack frame; a fresh `BatchCtx` borrowing this
/// state is built per window.
struct WindowedScan {
    store: Arc<TrackedStore<FilesystemStore>>,
    store_meta: ZarrStoreMeta,
    schema: SchemaRef,
    projected_schema: SchemaRef,
    projected_indices: Vec<usize>,
    coord_names: Vec<String>,
    effective_coord_indices: Vec<usize>,
    coord_sizes: Vec<usize>,
    stats: Option<SharedIoStats>,
    query_coord_sizes: Vec<usize>,
    filtered_coord_values: Vec<CoordValues>,
    coord_ranges: Option<Vec<CoordSelection>>,
    /// Lazy odometer over the per-axis tile grid (effective-axis order). Replaces
    /// the old single outer-axis cursor.
    grid: TileGrid,
    limit: Option<usize>,
    /// Rows emitted so far (drives the LIMIT stop).
    emitted: usize,
    /// DataFusion's output batch target. The read window is chosen from chunk
    /// geometry and is usually much larger, so a window is sliced into batches of
    /// this size before being emitted — reading wide and emitting narrow.
    batch_size: usize,
    /// Slices of the current window still to be emitted.
    pending: std::collections::VecDeque<RecordBatch>,
}

impl WindowedScan {
    /// Emit the next output batch, honouring DataFusion's `batch_size`.
    ///
    /// The read window is sized from chunk geometry so each chunk is fetched once,
    /// which makes it far wider than `batch_size`. Emitting it whole would hand
    /// downstream operators a single enormous `RecordBatch`; instead each window is
    /// sliced here. `RecordBatch::slice` is zero-copy — the slices share the
    /// window's buffers, so this costs no extra memory.
    fn next_batch(&mut self) -> Option<Result<RecordBatch>> {
        loop {
            if let Some(batch) = self.pending.pop_front() {
                return Some(Ok(batch));
            }
            match self.next_window()? {
                Ok(window) => self.push_slices(window),
                Err(e) => return Some(Err(e)),
            }
        }
    }

    /// Split one window into `batch_size`-row slices, queued for emission.
    fn push_slices(&mut self, window: RecordBatch) {
        if window.num_rows() == 0 {
            return;
        }
        self.pending
            .extend(slice_to_batch_size(window, self.batch_size));
    }

    /// Build the next tile's batch, or `None` when the grid is exhausted or the
    /// LIMIT has been reached (later tiles are never read — laziness).
    fn next_window(&mut self) -> Option<Result<RecordBatch>> {
        if let Some(limit) = self.limit {
            if self.emitted >= limit {
                return None;
            }
        }

        // The next tile: one window per effective axis (effective-axis order).
        let aw = self.grid.next()?;
        let window_rows: usize = aw.iter().map(|w| w.len).product();

        // Narrow the resolved selection to this tile. `query_coord_sizes` and `aw`
        // are effective-ordered; `filtered_coord_values` / `coord_ranges` /
        // `coord_sizes` are coord-index-ordered, bridged by effective_coord_indices.
        let mut query_coord_sizes = self.query_coord_sizes.clone();
        for (j, w) in aw.iter().enumerate() {
            query_coord_sizes[j] = w.len;
        }
        let filtered_coord_values: Vec<CoordValues> = self
            .filtered_coord_values
            .iter()
            .enumerate()
            .map(
                |(c, v)| match self.effective_coord_indices.iter().position(|&ec| ec == c) {
                    Some(j) => v.slice(aw[j].offset, aw[j].offset + aw[j].len),
                    None => v.clone(),
                },
            )
            .collect();
        let mut sels = self.coord_ranges.clone().unwrap_or_else(|| {
            self.coord_sizes
                .iter()
                .map(|&s| CoordSelection::Range(0, s))
                .collect()
        });
        for (j, w) in aw.iter().enumerate() {
            sels[self.effective_coord_indices[j]] = w.sel.clone();
        }
        let coord_ranges = Some(sels);

        // LIMIT as a running cap: this tile contributes at most the rows still owed.
        let (final_rows, limit_w) = match self.limit {
            Some(limit) => {
                let remaining = limit - self.emitted;
                (remaining.min(window_rows), Some(remaining))
            }
            None => (window_rows, None),
        };

        let ctx = BatchCtx {
            store: &self.store,
            store_meta: &self.store_meta,
            schema: &self.schema,
            projected_schema: &self.projected_schema,
            projected_indices: &self.projected_indices,
            coord_names: &self.coord_names,
            effective_coord_indices: &self.effective_coord_indices,
            coord_sizes: &self.coord_sizes,
            stats: &self.stats,
        };
        let batch = build_batch(
            &ctx,
            &coord_ranges,
            &query_coord_sizes,
            &filtered_coord_values,
            final_rows,
            limit_w,
            window_rows,
        );

        if let Ok(ref b) = batch {
            self.emitted += b.num_rows();
        }
        Some(batch)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn read_zarr(
    store_path: &str,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: Option<SharedIoStats>,
    coord_filters: Option<CoordFilters>,
    // Outer-axis selection for this partition (intersected with the filter
    // selection). `None` => read the whole (filtered) store, the legacy
    // single-partition path. (Plain `//`: doc comments aren't allowed on params.)
    partition_selection: Option<CoordSelection>,
    // Inner-axis `(coord index, selection)` restrictions for a multi-axis (N-D
    // box) partition; empty for an outer-only partition. See PartitionSpec::extra.
    partition_extra: Vec<(usize, CoordSelection)>,
    // Target rows per emitted RecordBatch (DataFusion's batch_size). Drives the
    // outer-axis windowing of the streaming scan.
    batch_size: usize,
    // Rule-set memory budget in bytes (Phase 5); `None` => ZARR_MEM_BUDGET_BYTES,
    // then batch_size. Converted to a row target by the reader's own row_width.
    stream_budget_bytes: Option<u64>,
) -> Result<SendableRecordBatchStream> {
    let fs_store = Arc::new(FilesystemStore::new(store_path).map_err(zarr_err)?);

    // Wrap with TrackedStore if stats are provided
    let store: Arc<TrackedStore<FilesystemStore>> = Arc::new(TrackedStore::new(
        fs_store,
        stats.clone().unwrap_or_default(),
    ));

    // Discover store structure (with timing)
    let meta_start = Instant::now();
    let store_meta = discover_arrays(store_path).map_err(DataFusionError::External)?;
    if let Some(ref s) = stats {
        // TODO: Track actual metadata bytes read in discover_arrays() instead of estimating
        let meta_bytes = (store_meta.coords.len() + store_meta.data_vars.len()) as u64 * 500;
        s.record_metadata(meta_bytes, meta_start.elapsed());
    }

    let coord_names: Vec<_> = store_meta.coords.iter().map(|c| c.name.clone()).collect();
    let coord_types: Vec<_> = store_meta
        .coords
        .iter()
        .map(|c| c.data_type.clone())
        .collect();

    // When this partition supplies the outer-axis selection, we read ONLY that
    // slice of the outer coord (not the whole axis). `None` => no partitioning,
    // or the outer coord can't be identified => read everything as before.
    let outer_slice_idx = partition_selection
        .as_ref()
        .and_then(|_| identify_outer_coord(&store_meta));
    // The selection (relative to the slice we read) used to extract the outer
    // coord's surviving values; set while reading the outer coord below.
    let mut outer_extract_sel: Option<CoordSelection> = None;

    // Load coordinate arrays and get their sizes
    let mut coord_sizes: Vec<usize> = Vec::new();
    let mut coord_values: Vec<CoordValues> = Vec::new();

    for (i, (coord, dtype)) in store_meta.coords.iter().zip(coord_types.iter()).enumerate() {
        let read_start = Instant::now();
        let arr = Array::open(store.clone(), &format!("/{}", coord.name)).map_err(zarr_err)?;
        let size = arr.shape()[0] as usize;
        coord_sizes.push(size);

        let element_bytes = dtype_to_bytes(dtype);
        let (raw_values, read_len) = if Some(i) == outer_slice_idx {
            // Read only this partition's window of the outer coord.
            let sel = partition_selection
                .as_ref()
                .expect("outer_slice_idx implies Some");
            let (rs, re, extract_sel) = plan_outer_coord_read(sel);
            outer_extract_sel = Some(extract_sel);
            let subset = ArraySubset::new_with_ranges(std::slice::from_ref(&(rs..re)));
            (
                read_coord_values!(sync, arr, &subset, dtype.as_str()),
                (re - rs) as usize,
            )
        } else {
            let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
            (read_coord_values!(sync, arr, &subset, dtype.as_str()), size)
        };

        // Apply CF time conversion using helper function
        let values =
            apply_cf_time_conversion(raw_values, coord.cf_time_attrs.as_ref(), &coord.name);

        if let Some(ref s) = stats {
            let bytes = read_len as u64 * element_bytes;
            s.record_coord(bytes, read_start.elapsed());
        }
        coord_values.push(values);
    }

    // Total rows = product of all coordinate sizes (before filtering)
    let total_rows: usize = coord_sizes.iter().product();

    // Calculate coordinate ranges based on filters
    let coord_ranges = if let Some(ref filters) = coord_filters {
        let coord_refs = coord_values_to_refs(&coord_values);
        // Skip re-resolving the outer filter when this partition supplied its
        // (head-resolved) selection — we only hold a slice of that coord here.
        let filters = filters_without_outer(filters, outer_slice_idx, &coord_names);

        match calculate_coord_ranges(&filters, &coord_names, &coord_refs) {
            Some(ranges) => {
                let filtered_rows = calculate_filtered_rows(&ranges);
                let reduction_pct = 100.0 * (1.0 - (filtered_rows as f64 / total_rows as f64));
                info!(
                    total_rows,
                    filtered_rows,
                    reduction_pct = format!("{:.2}%", reduction_pct),
                    filters = ?filters.filters.keys().collect::<Vec<_>>(),
                    "Filter pushdown optimization"
                );
                Some(ranges)
            }
            None => {
                // Filter value not found - return empty result
                warn!("Filter value not found in coordinates - returning empty result");
                return create_empty_result_stream(&schema, projection.as_ref());
            }
        }
    } else {
        None
    };

    // Narrow the read to this partition's outer-dimension slice, if any. These
    // selections carry ABSOLUTE outer-axis indices (they drive the data-var
    // reads), so keep them distinct from the relative `extract_selections` below.
    let coord_ranges =
        apply_partition_selection(
            coord_ranges,
            partition_selection,
            &partition_extra,
            &store_meta,
            &coord_sizes,
        );

    // Calculate effective sizes based on filters
    let (effective_coord_sizes, effective_rows) =
        calculate_effective_sizes(&coord_sizes, &coord_ranges);

    // Extract filtered coordinate values. The outer coord was read as a partial
    // window, so it is extracted with a window-relative selection rather than the
    // absolute one used for the data-var reads.
    let extract_selections =
        make_extract_selections(&coord_ranges, outer_slice_idx, &outer_extract_sel);
    let filtered_coord_values = extract_filtered_coords(coord_values, &extract_selections);

    let total_columns = schema.fields().len();
    let projected_indices = projection.unwrap_or_else(|| (0..total_columns).collect());

    // Log projection optimization effect
    let skipped_columns = total_columns - projected_indices.len();
    if skipped_columns > 0 {
        let projected_names: Vec<_> = projected_indices
            .iter()
            .map(|&i| schema.field(i).name().as_str())
            .collect();
        info!(
            reading = projected_indices.len(),
            skipping = skipped_columns,
            columns = ?projected_names,
            "Projection optimization"
        );
    } else {
        info!(
            columns = total_columns,
            "No projection optimization (all columns)"
        );
    }

    // ==========================================================================
    // Mixed-dimensionality handling: determine which coordinates are actually
    // needed for the projected columns (data variables or coordinates only)
    // ==========================================================================
    // Separate projected columns into coordinates and data variables
    let (projected_coord_names, projected_var_names): (Vec<&str>, Vec<&str>) = projected_indices
        .iter()
        .map(|&i| schema.field(i).name().as_str())
        .partition(|name| coord_names.contains(&name.to_string()));

    // Determine effective coordinates for the projected columns
    // Pass limit to enable coordinate-only optimization when LIMIT is present
    let effective_coord_indices = determine_effective_coords(
        &projected_var_names,
        &projected_coord_names,
        &store_meta.data_vars,
        &coord_names,
        &coord_sizes,
        limit,
    )
    .map_err(DataFusionError::Plan)?;

    // Recalculate effective sizes and rows for the relevant coordinates only
    let (query_coord_sizes, query_rows) = calculate_query_coord_sizes(
        &coord_sizes,
        &effective_coord_sizes,
        &effective_coord_indices,
        effective_rows,
    );

    // Apply limit (after filter and dimensionality reduction)
    let final_rows = limit.map(|l| l.min(query_rows)).unwrap_or(query_rows);
    if let Some(limit) = limit {
        if limit < query_rows {
            let reduction_pct = 100.0 * (1.0 - (final_rows as f64 / query_rows as f64));
            info!(
                query_rows,
                final_rows,
                reduction_pct = format!("{:.2}%", reduction_pct),
                "Limit optimization"
            );
        }
    }

    // Build projected schema (constant across batches), excluding skipped coords.
    let projected_schema = build_projected_schema(
        &schema,
        &projected_indices,
        &coord_names,
        &effective_coord_indices,
    );

    // Observe-only (Phase 2/3 exact-cardinality): at debug level log exact
    // cardinality/inner-rows/touched-chunks/fallback; and, if a memory budget is
    // configured, warn when the predicted peak exceeds it. Drives no decision.
    crate::optimizer::cardinality::predicate::observe_scan(
        &store_meta.coords,
        &effective_coord_indices,
        coord_ranges.as_deref(),
        &coord_sizes,
        &schema,
        &projected_indices,
        &coord_names,
        &store_meta.data_vars,
        &projected_schema,
        batch_size,
    );

    // Plan outer-axis windows. Windowing slices the OUTER (most-significant)
    // effective coordinate, scaling every read by the window — which is only
    // transparent when each projected data var spans the full *effective*
    // coordinate set (the coords this projection uses). Comparing against the
    // effective set, not every store coordinate, is what lets a var that simply
    // doesn't use some coord (e.g. SST in a store that also has `level`) still
    // stream. A var missing an *effective* coord (broadcast across the outer axis,
    // e.g. a static field selected next to a time-varying one) can't be windowed,
    // so it falls back to a single batch (still correct, just not streamed).
    // Coordinate columns are always safe. See docs/block-0-impl.md "Known
    // limitations" for the worked example.
    let all_full_cube = projected_indices.iter().all(|&i| {
        let name = schema.field(i).name();
        coord_names.iter().any(|c| c == name)
            || store_meta
                .data_vars
                .iter()
                .find(|v| &v.name == name)
                .is_some_and(|v| v.shape.len() == effective_coord_indices.len())
    });
    // Build the tile grid from the budget-driven streaming plan. Windowing is only
    // safe when every projected var spans the full effective coord set
    // (`all_full_cube`); otherwise fall through to the single-batch path.
    let grid =
        if all_full_cube && !effective_coord_indices.is_empty() && !query_coord_sizes.is_empty() {
            // Phase 4.3: a byte budget (ZARR_MEM_BUDGET_BYTES) drives multi-axis tiling;
            // absent one, the target is batch_size and this reproduces the previous
            // outer-only `max_steps = batch_size / inner_rows`.
            let row_width =
                crate::optimizer::cardinality::cost::row_width(&projected_schema).max(1);
            // Ceiling precedence: rule-set bytes (Phase 5, already divided by the
            // partition count) -> env bytes -> the 256 MiB default. The window itself
            // is then inferred from chunk geometry inside `plan_window_tiles`, capped
            // by this ceiling: `batch_size` is an output-batching preference and is
            // no longer allowed to masquerade as a memory budget.
            let ceiling_bytes = crate::optimizer::cardinality::budget::resolve_scan_ceiling(
                stream_budget_bytes.or_else(|| {
                    crate::optimizer::cardinality::budget::MemoryBudget::from_env().map(|b| b.bytes)
                }),
                1,
            );
            let sizes: Vec<u64> = query_coord_sizes.iter().map(|&s| s as u64).collect();
            let tiles = plan_window_tiles(
                &sizes,
                ceiling_bytes,
                row_width,
                batch_size,
                &schema,
                &projected_indices,
                &coord_names,
                &store_meta.data_vars,
            );
            let effective_sels: Vec<CoordSelection> = (0..effective_coord_indices.len())
                .map(|j| match &coord_ranges {
                    Some(sels) => sels[effective_coord_indices[j]].clone(),
                    None => CoordSelection::Range(0, query_coord_sizes[j]),
                })
                .collect();
            Some(TileGrid::new(&effective_sels, &tiles))
        } else {
            None
        };

    // Single batch (small result, mixed dimensionality, or no windowable axis):
    // one batch, byte-identical to the un-windowed path.
    if grid.as_ref().map(|g| g.n_batches()).unwrap_or(1) <= 1 {
        // Phase 4.4: this path cannot tile (small result, or the mixed-dim fallback
        // that streaming can't window). If enforcement is on and a budget is set,
        // reject an over-budget batch (the Gap-2 case) instead of OOMing. No-op
        // unless ZARR_MEM_ENFORCE is set.
        crate::optimizer::cardinality::budget::admit_single_batch(
            final_rows,
            crate::optimizer::cardinality::cost::row_width(&projected_schema),
        )
        .map_err(|inf| DataFusionError::ResourcesExhausted(inf.to_string()))?;
        let ctx = BatchCtx {
            store: &store,
            store_meta: &store_meta,
            schema: &schema,
            projected_schema: &projected_schema,
            projected_indices: &projected_indices,
            coord_names: &coord_names,
            effective_coord_indices: &effective_coord_indices,
            coord_sizes: &coord_sizes,
            stats: &stats,
        };
        let batch = build_batch(
            &ctx,
            &coord_ranges,
            &query_coord_sizes,
            &filtered_coord_values,
            final_rows,
            limit,
            query_rows,
        )?;
        let stream = stream::iter(slice_to_batch_size(batch, batch_size).into_iter().map(Ok));
        return Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )));
    }

    // Multiple tiles: emit one batch per tile, lazily, bounding peak memory to
    // ~one tile instead of the whole selection.
    let grid = grid.expect("grid is Some when n_batches > 1");
    info!(
        num_batches = grid.n_batches(),
        batch_size, "Streaming scan: tiling selection"
    );
    let scan = WindowedScan {
        batch_size,
        pending: std::collections::VecDeque::new(),
        store,
        store_meta,
        schema,
        projected_schema: projected_schema.clone(),
        projected_indices,
        coord_names,
        effective_coord_indices,
        coord_sizes,
        stats,
        query_coord_sizes,
        filtered_coord_values,
        coord_ranges,
        grid,
        limit,
        emitted: 0,
    };
    let stream = stream::unfold(scan, |mut scan| async move {
        scan.next_batch().map(|batch| (batch, scan))
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

// =============================================================================
// Async version for remote object stores
// =============================================================================

use super::schema_inference::{discover_arrays_async, ZarrStoreMeta};
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

/// Async twin of [`BatchCtx`]: constant context for [`build_batch_async`].
struct AsyncBatchCtx<'a> {
    store: &'a AsyncReadableListableStorage,
    cache: &'a ScanCache,
    prefix: &'a ObjectPath,
    store_meta: &'a ZarrStoreMeta,
    schema: &'a SchemaRef,
    projected_schema: &'a SchemaRef,
    projected_indices: &'a [usize],
    coord_names: &'a [String],
    effective_coord_indices: &'a [usize],
    coord_sizes: &'a [usize],
    stats: &'a Option<SharedIoStats>,
}

/// Async twin of [`build_batch`]: builds one `RecordBatch` from a resolved
/// selection state using async object-store reads. `final_rows` is the async
/// path's `effective_rows` (limit already folded into it); `total_rows` feeds the
/// no-filter limit-subset read optimization, which only fires when `coord_ranges`
/// is `None` (i.e. the single, un-windowed batch).
#[allow(clippy::too_many_arguments)]
async fn build_batch_async(
    ctx: &AsyncBatchCtx<'_>,
    coord_ranges: &Option<Vec<CoordSelection>>,
    query_coord_sizes: &[usize],
    filtered_coord_values: &[CoordValues],
    final_rows: usize,
    limit: Option<usize>,
    query_rows: usize,
    total_rows: usize,
) -> Result<RecordBatch> {
    let mut result_arrays: Vec<ArrayRef> = Vec::new();

    for idx in ctx.projected_indices {
        let field = ctx.schema.field(*idx);
        let field_name = field.name();

        if let Some(coord_idx) = ctx.coord_names.iter().position(|n| n == field_name) {
            if !ctx.effective_coord_indices.contains(&coord_idx) {
                debug!(field = %field_name, "Skipping coordinate not used by projected variables");
                continue;
            }
            let query_coord_idx = ctx
                .effective_coord_indices
                .iter()
                .position(|&i| i == coord_idx)
                .unwrap();
            let dict_array = create_coord_dictionary_typed(
                &filtered_coord_values[coord_idx],
                query_coord_idx,
                query_coord_sizes,
                final_rows,
                dictionary_key_type(field.data_type()),
            );
            result_arrays.push(dict_array);
        } else {
            let read_start = Instant::now();
            let array_path = if ctx.prefix.as_ref().is_empty() {
                format!("/{}", field_name)
            } else {
                format!("/{}/{}", ctx.prefix, field_name)
            };
            let arr = ctx.cache.array(ctx.store, &array_path).await?;
            let data_var_shape = arr.shape();
            let data_var_chunks: Option<Vec<u64>> = ctx
                .store_meta
                .data_vars
                .iter()
                .find(|v| v.name == *field_name)
                .and_then(|v| v.chunks.clone());

            let plans = if let Some(sels) = coord_ranges {
                build_read_plans(
                    sels,
                    ctx.coord_sizes,
                    data_var_shape,
                    data_var_chunks.as_deref(),
                )
            } else if final_rows < total_rows {
                // No filter, but a LIMIT: read only the leading rows.
                let ranges = calculate_limited_subset(arr.shape(), final_rows);
                vec![ReadPlan {
                    subset: ArraySubset::new_with_ranges(&ranges),
                    keep: Keep::All,
                }]
            } else {
                vec![ReadPlan {
                    subset: ArraySubset::new_with_shape(arr.shape().to_vec()),
                    keep: Keep::All,
                }]
            };
            let num_elements: u64 = plans.iter().map(|p| p.subset.num_elements()).sum();

            let mut parts: Vec<ArrayRef> = Vec::with_capacity(plans.len());
            for plan in &plans {
                let raw = read_data_array!(async, arr, &plan.subset, field.data_type());
                parts.push(apply_keep(raw, &plan.keep)?);
            }
            let array: ArrayRef = if parts.len() == 1 {
                parts.remove(0)
            } else {
                let refs: Vec<&dyn arrow::array::Array> =
                    parts.iter().map(|a| a.as_ref()).collect();
                arrow::compute::concat(&refs).map_err(DataFusionError::from)?
            };

            if let Some(s) = ctx.stats {
                let bytes = num_elements * arrow_dtype_to_bytes(field.data_type());
                s.record_data(bytes, read_start.elapsed());
            }
            result_arrays.push(array);
        }
    }

    let result_arrays = apply_limit_to_arrays(result_arrays, limit, query_rows);
    create_result_batch(ctx.projected_schema.clone(), result_arrays, final_rows)
}

/// Async twin of [`WindowedScan`]: produces one `RecordBatch` per outer-axis
/// window via async reads, on demand.
struct AsyncWindowedScan {
    store: AsyncReadableListableStorage,
    cache: Arc<ScanCache>,
    prefix: ObjectPath,
    store_meta: ZarrStoreMeta,
    schema: SchemaRef,
    projected_schema: SchemaRef,
    projected_indices: Vec<usize>,
    coord_names: Vec<String>,
    effective_coord_indices: Vec<usize>,
    coord_sizes: Vec<usize>,
    stats: Option<SharedIoStats>,
    query_coord_sizes: Vec<usize>,
    filtered_coord_values: Vec<CoordValues>,
    coord_ranges: Option<Vec<CoordSelection>>,
    /// Lazy odometer over the per-axis tile grid (effective-axis order).
    grid: TileGrid,
    limit: Option<usize>,
    total_rows: usize,
    /// Rows emitted so far (drives the LIMIT stop).
    emitted: usize,
    /// DataFusion's output batch target; see `WindowedScan::batch_size`.
    batch_size: usize,
    /// Slices of the current window still to be emitted.
    pending: std::collections::VecDeque<RecordBatch>,
}

impl AsyncWindowedScan {
    /// Emit the next output batch, honouring `batch_size`; see
    /// [`WindowedScan::next_batch`] for why reads and emissions are decoupled.
    async fn next_batch(&mut self) -> Option<Result<RecordBatch>> {
        loop {
            if let Some(batch) = self.pending.pop_front() {
                return Some(Ok(batch));
            }
            match self.next_window().await? {
                Ok(window) => self.push_slices(window),
                Err(e) => return Some(Err(e)),
            }
        }
    }

    /// Split one window into `batch_size`-row slices, queued for emission.
    fn push_slices(&mut self, window: RecordBatch) {
        if window.num_rows() == 0 {
            return;
        }
        self.pending
            .extend(slice_to_batch_size(window, self.batch_size));
    }

    async fn next_window(&mut self) -> Option<Result<RecordBatch>> {
        if let Some(limit) = self.limit {
            if self.emitted >= limit {
                return None;
            }
        }

        // The next tile: one window per effective axis (effective-axis order).
        let aw = self.grid.next()?;
        let window_rows: usize = aw.iter().map(|w| w.len).product();

        // sizes/aw are effective-ordered; coord values / read-ranges are
        // coord-index-ordered, bridged by effective_coord_indices.
        let mut query_coord_sizes = self.query_coord_sizes.clone();
        for (j, w) in aw.iter().enumerate() {
            query_coord_sizes[j] = w.len;
        }
        let filtered_coord_values: Vec<CoordValues> = self
            .filtered_coord_values
            .iter()
            .enumerate()
            .map(
                |(c, v)| match self.effective_coord_indices.iter().position(|&ec| ec == c) {
                    Some(j) => v.slice(aw[j].offset, aw[j].offset + aw[j].len),
                    None => v.clone(),
                },
            )
            .collect();
        let mut sels = self.coord_ranges.clone().unwrap_or_else(|| {
            self.coord_sizes
                .iter()
                .map(|&s| CoordSelection::Range(0, s))
                .collect()
        });
        for (j, w) in aw.iter().enumerate() {
            sels[self.effective_coord_indices[j]] = w.sel.clone();
        }
        let coord_ranges = Some(sels);

        let (final_rows, limit_w) = match self.limit {
            Some(limit) => {
                let remaining = limit - self.emitted;
                (remaining.min(window_rows), Some(remaining))
            }
            None => (window_rows, None),
        };

        let ctx = AsyncBatchCtx {
            store: &self.store,
            cache: &self.cache,
            prefix: &self.prefix,
            store_meta: &self.store_meta,
            schema: &self.schema,
            projected_schema: &self.projected_schema,
            projected_indices: &self.projected_indices,
            coord_names: &self.coord_names,
            effective_coord_indices: &self.effective_coord_indices,
            coord_sizes: &self.coord_sizes,
            stats: &self.stats,
        };
        let batch = build_batch_async(
            &ctx,
            &coord_ranges,
            &query_coord_sizes,
            &filtered_coord_values,
            final_rows,
            limit_w,
            window_rows,
            self.total_rows,
        )
        .await;

        if let Ok(ref b) = batch {
            self.emitted += b.num_rows();
        }
        Some(batch)
    }
}

/// Async version of read_zarr for remote object stores
#[allow(clippy::too_many_arguments)]
#[instrument(level = "info", skip_all)]
pub async fn read_zarr_async(
    store: AsyncReadableListableStorage,
    prefix: &ObjectPath,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: Option<SharedIoStats>,
    cached_meta: Option<ZarrStoreMeta>,
    coord_filters: Option<CoordFilters>,
    // Outer-axis selection for this partition; `None` => whole store.
    partition_selection: Option<CoordSelection>,
    // Inner-axis `(coord index, selection)` restrictions for a multi-axis (N-D
    // box) partition; empty for an outer-only partition. See PartitionSpec::extra.
    partition_extra: Vec<(usize, CoordSelection)>,
    // Target rows per emitted RecordBatch (DataFusion's batch_size). Drives the
    // outer-axis windowing of the streaming scan.
    batch_size: usize,
    // Rule-set memory budget in bytes (Phase 5); `None` => ZARR_MEM_BUDGET_BYTES,
    // then batch_size. Converted to a row target by the reader's own row_width.
    stream_budget_bytes: Option<u64>,
    // Shared by the partitions of one scan, which read the same arrays.
    cache: Arc<ScanCache>,
) -> Result<SendableRecordBatchStream> {
    info!("Starting async Zarr read");

    // Use cached metadata if available, otherwise discover
    let store_meta = if let Some(meta) = cached_meta {
        info!("Using cached metadata");
        meta
    } else {
        debug!("Discovering store metadata");
        let meta_start = Instant::now();
        let meta = discover_arrays_async(&store, prefix)
            .await
            .map_err(DataFusionError::External)?;
        debug!(elapsed = ?meta_start.elapsed(), "Metadata discovery complete");

        if let Some(ref s) = stats {
            // TODO: Track actual metadata bytes read
            let meta_bytes = (meta.coords.len() + meta.data_vars.len()) as u64 * 500;
            s.record_metadata(meta_bytes, meta_start.elapsed());
        }
        meta
    };

    let coord_names: Vec<_> = store_meta.coords.iter().map(|c| c.name.clone()).collect();
    let coord_types: Vec<_> = store_meta
        .coords
        .iter()
        .map(|c| c.data_type.clone())
        .collect();

    // Get coordinate sizes from metadata (already discovered)
    let coord_sizes: Vec<usize> = store_meta
        .coords
        .iter()
        .map(|c| c.shape[0] as usize)
        .collect();
    debug!(?coord_names, ?coord_sizes, "Coordinate info");

    // Total rows = product of all coordinate sizes (before filtering)
    let total_rows: usize = coord_sizes.iter().product();

    // When this partition supplies the outer-axis selection, read ONLY that slice
    // of the outer coord over the network instead of the whole axis (see the sync
    // path for the rationale). `None` => no partitioning / unidentifiable outer.
    let outer_slice_idx = partition_selection
        .as_ref()
        .and_then(|_| identify_outer_coord(&store_meta));

    // First, load all coordinate values (needed for filter matching)
    debug!("Loading coordinate values for filter matching");
    let outer_read = outer_slice_idx.map(|i| {
        let sel = partition_selection
            .as_ref()
            .expect("outer_slice_idx implies Some");
        let (rs, re, extract_sel) = plan_outer_coord_read(sel);
        let coord = &store_meta.coords[i];
        let chunk_len = coord.chunks.as_ref().and_then(|c| c.first().copied());
        let (cs, ce) = widen_to_chunks(rs, re, chunk_len, coord.shape[0]);
        (
            i,
            (cs, ce, shift_selection(extract_sel, (rs - cs) as usize)),
        )
    });
    let all_coord_values: Vec<CoordValues> = futures::future::try_join_all(
        store_meta
            .coords
            .iter()
            .zip(coord_types.iter())
            .enumerate()
            .map(|(i, (coord, dtype))| {
                let (store, cache, stats) = (&store, &cache, &stats);
                let outer_read = outer_read.as_ref().filter(|(o, _)| *o == i);
                async move {
                    let array_path = if prefix.as_ref().is_empty() {
                        format!("/{}", coord.name)
                    } else {
                        format!("/{}/{}", prefix, coord.name)
                    };
                    let arr = &cache.array(store, &array_path).await?;
                    let read = |subset: ArraySubset, read_len: usize| async move {
                        let read_start = Instant::now();
                        let raw_values = read_coord_values!(async, arr, &subset, dtype.as_str());
                        if let Some(s) = stats {
                            let bytes = read_len as u64 * dtype_to_bytes(dtype);
                            s.record_coord(bytes, read_start.elapsed());
                        }
                        Ok::<_, DataFusionError>(apply_cf_time_conversion(
                            raw_values,
                            coord.cf_time_attrs.as_ref(),
                            &coord.name,
                        ))
                    };
                    match outer_read {
                        Some((_, (cs, ce, _))) => {
                            let subset =
                                ArraySubset::new_with_ranges(std::slice::from_ref(&(*cs..*ce)));
                            cache
                                .coord(&format!("{array_path}[{cs}..{ce}]"), || {
                                    read(subset, (ce - cs) as usize)
                                })
                                .await
                        }
                        None => {
                            let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
                            cache
                                .coord(&array_path, || read(subset, coord.shape[0] as usize))
                                .await
                        }
                    }
                }
            }),
    )
    .await?;
    let outer_extract_sel: Option<CoordSelection> =
        outer_read.map(|(_, (_, _, extract_sel))| extract_sel);

    // Calculate coordinate ranges based on filters
    let coord_ranges = if let Some(ref filters) = coord_filters {
        let coord_refs = coord_values_to_refs(&all_coord_values);
        // Skip re-resolving the outer filter when this partition supplied its
        // (head-resolved) selection — we only hold a slice of that coord here.
        let filters = filters_without_outer(filters, outer_slice_idx, &coord_names);

        match calculate_coord_ranges(&filters, &coord_names, &coord_refs) {
            Some(ranges) => {
                let filtered_rows = calculate_filtered_rows(&ranges);
                let reduction_pct = 100.0 * (1.0 - (filtered_rows as f64 / total_rows as f64));
                info!(
                    total_rows,
                    filtered_rows,
                    reduction_pct = format!("{:.2}%", reduction_pct),
                    filters = ?filters.filters.keys().collect::<Vec<_>>(),
                    "Filter pushdown optimization"
                );
                Some(ranges)
            }
            None => {
                // Filter value not found - return empty result
                warn!("Filter value not found in coordinates - returning empty result");
                return create_empty_result_stream(&schema, projection.as_ref());
            }
        }
    } else {
        None
    };
    debug!(?coord_ranges, "Coordinate ranges calculated");

    // Narrow the read to this partition's outer-dimension slice, if any. These
    // selections carry ABSOLUTE outer-axis indices (they drive the data-var
    // reads), so keep them distinct from the relative `extract_selections` below.
    let coord_ranges =
        apply_partition_selection(
            coord_ranges,
            partition_selection,
            &partition_extra,
            &store_meta,
            &coord_sizes,
        );

    // Calculate effective sizes based on filters
    let (effective_coord_sizes, rows_after_filter) =
        calculate_effective_sizes(&coord_sizes, &coord_ranges);

    // Extract filtered coordinate values. The outer coord was read as a partial
    // window, so it is extracted with a window-relative selection rather than the
    // absolute one used for the data-var reads.
    let extract_selections =
        make_extract_selections(&coord_ranges, outer_slice_idx, &outer_extract_sel);
    let filtered_coord_values = extract_filtered_coords(all_coord_values, &extract_selections);

    // Apply limit (after filter reduction)
    let effective_rows = limit
        .map(|l| l.min(rows_after_filter))
        .unwrap_or(rows_after_filter);

    // Log limit optimization effect
    if effective_rows < rows_after_filter {
        let reduction_pct = 100.0 * (1.0 - (effective_rows as f64 / rows_after_filter as f64));
        info!(
            rows_after_filter,
            effective_rows,
            reduction_pct = format!("{:.2}%", reduction_pct),
            "Limit optimization applied"
        );
    }

    // Calculate how many values we need from each coordinate (for limit optimization on top of filter)
    // TODO: Use this for optimized coordinate loading in the future
    let _coord_value_limits = if effective_rows < rows_after_filter {
        calculate_coord_limits(&effective_coord_sizes, effective_rows)
    } else {
        effective_coord_sizes.clone()
    };

    info!("Coordinates loaded and filtered");

    let total_columns = schema.fields().len();
    let projected_indices = projection.unwrap_or_else(|| (0..total_columns).collect());

    // Log projection optimization effect
    let skipped_columns = total_columns - projected_indices.len();
    if skipped_columns > 0 {
        let projected_names: Vec<_> = projected_indices
            .iter()
            .map(|&i| schema.field(i).name().as_str())
            .collect();
        info!(
            reading = projected_indices.len(),
            skipping = skipped_columns,
            columns = ?projected_names,
            "Projection optimization"
        );
    } else {
        info!(
            columns = total_columns,
            "No projection optimization (all columns)"
        )
    }

    // ==========================================================================
    // Mixed-dimensionality handling: determine which coordinates are actually
    // needed for the projected columns (data variables or coordinates only)
    // ==========================================================================
    // Separate projected columns into coordinates and data variables
    let (projected_coord_names, projected_var_names): (Vec<&str>, Vec<&str>) = projected_indices
        .iter()
        .map(|&i| schema.field(i).name().as_str())
        .partition(|name| coord_names.contains(&name.to_string()));

    // Determine effective coordinates for the projected columns
    // Pass limit to enable coordinate-only optimization when LIMIT is present
    let effective_coord_indices = determine_effective_coords(
        &projected_var_names,
        &projected_coord_names,
        &store_meta.data_vars,
        &coord_names,
        &coord_sizes,
        limit,
    )
    .map_err(DataFusionError::Plan)?;

    // Recalculate effective sizes and rows for the relevant coordinates only
    let (query_coord_sizes, query_rows) = calculate_query_coord_sizes(
        &coord_sizes,
        &effective_coord_sizes,
        &effective_coord_indices,
        rows_after_filter,
    );

    // Recalculate effective_rows with limit applied to query_rows
    let effective_rows = limit.map(|l| l.min(query_rows)).unwrap_or(query_rows);

    // `effective_rows` already folds LIMIT into query_rows; it is the async twin
    // of the sync path's `final_rows`.
    let final_rows = effective_rows;

    // Build projected schema (constant across batches), excluding skipped coords.
    let projected_schema = build_projected_schema(
        &schema,
        &projected_indices,
        &coord_names,
        &effective_coord_indices,
    );

    // Observe-only (Phase 2/3 exact-cardinality): see the sync path for rationale.
    crate::optimizer::cardinality::predicate::observe_scan(
        &store_meta.coords,
        &effective_coord_indices,
        coord_ranges.as_deref(),
        &coord_sizes,
        &schema,
        &projected_indices,
        &coord_names,
        &store_meta.data_vars,
        &projected_schema,
        batch_size,
    );

    // Plan outer-axis windows — same logic and safety gate as the sync path
    // (`read_zarr`): window only when every projected data var spans the full
    // *effective* coordinate set, so a var missing an effective coord (broadcast
    // across the outer axis) falls back to a single batch.
    let all_full_cube = projected_indices.iter().all(|&i| {
        let name = schema.field(i).name();
        coord_names.iter().any(|c| c == name)
            || store_meta
                .data_vars
                .iter()
                .find(|v| &v.name == name)
                .is_some_and(|v| v.shape.len() == effective_coord_indices.len())
    });
    // Build the tile grid from the budget-driven streaming plan (see the sync path
    // in `read_zarr` for the rationale); only safe when `all_full_cube`.
    let grid =
        if all_full_cube && !effective_coord_indices.is_empty() && !query_coord_sizes.is_empty() {
            let row_width =
                crate::optimizer::cardinality::cost::row_width(&projected_schema).max(1);
            // Ceiling precedence: rule-set bytes (Phase 5, already divided by the
            // partition count) -> env bytes -> the 256 MiB default. The window itself
            // is then inferred from chunk geometry inside `plan_window_tiles`, capped
            // by this ceiling: `batch_size` is an output-batching preference and is
            // no longer allowed to masquerade as a memory budget.
            let ceiling_bytes = crate::optimizer::cardinality::budget::resolve_scan_ceiling(
                stream_budget_bytes.or_else(|| {
                    crate::optimizer::cardinality::budget::MemoryBudget::from_env().map(|b| b.bytes)
                }),
                1,
            );
            let sizes: Vec<u64> = query_coord_sizes.iter().map(|&s| s as u64).collect();
            let tiles = plan_window_tiles(
                &sizes,
                ceiling_bytes,
                row_width,
                batch_size,
                &schema,
                &projected_indices,
                &coord_names,
                &store_meta.data_vars,
            );
            let effective_sels: Vec<CoordSelection> = (0..effective_coord_indices.len())
                .map(|j| match &coord_ranges {
                    Some(sels) => sels[effective_coord_indices[j]].clone(),
                    None => CoordSelection::Range(0, query_coord_sizes[j]),
                })
                .collect();
            Some(TileGrid::new(&effective_sels, &tiles))
        } else {
            None
        };

    // Single batch (small result, mixed dimensionality, or no windowable axis):
    // one batch, byte-identical to the un-windowed path (incl. the no-filter
    // limit-subset read optimization inside build_batch_async).
    if grid.as_ref().map(|g| g.n_batches()).unwrap_or(1) <= 1 {
        // Phase 4.4: same enforcement as the sync path (see `read_zarr`).
        crate::optimizer::cardinality::budget::admit_single_batch(
            final_rows,
            crate::optimizer::cardinality::cost::row_width(&projected_schema),
        )
        .map_err(|inf| DataFusionError::ResourcesExhausted(inf.to_string()))?;
        let ctx = AsyncBatchCtx {
            store: &store,
            cache: &cache,
            prefix,
            store_meta: &store_meta,
            schema: &schema,
            projected_schema: &projected_schema,
            projected_indices: &projected_indices,
            coord_names: &coord_names,
            effective_coord_indices: &effective_coord_indices,
            coord_sizes: &coord_sizes,
            stats: &stats,
        };
        let batch = build_batch_async(
            &ctx,
            &coord_ranges,
            &query_coord_sizes,
            &filtered_coord_values,
            final_rows,
            limit,
            query_rows,
            total_rows,
        )
        .await?;
        let stream = stream::iter(slice_to_batch_size(batch, batch_size).into_iter().map(Ok));
        return Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )));
    }

    // Multiple tiles: emit one batch per tile, lazily, bounding peak memory to
    // ~one tile instead of the whole selection.
    let grid = grid.expect("grid is Some when n_batches > 1");
    info!(
        num_batches = grid.n_batches(),
        batch_size, "Streaming async scan: tiling selection"
    );
    let scan = AsyncWindowedScan {
        batch_size,
        pending: std::collections::VecDeque::new(),
        store,
        cache,
        prefix: prefix.clone(),
        store_meta,
        schema,
        projected_schema: projected_schema.clone(),
        projected_indices,
        coord_names,
        effective_coord_indices,
        coord_sizes,
        stats,
        query_coord_sizes,
        filtered_coord_values,
        coord_ranges,
        grid,
        limit,
        total_rows,
        emitted: 0,
    };
    let stream = stream::unfold(scan, |mut scan| async move {
        scan.next_batch().await.map(|batch| (batch, scan))
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

#[cfg(test)]
mod bucket_tests {
    use super::bucket_outer_indices;

    #[test]
    fn empty_indices_yield_no_buckets() {
        assert!(bucket_outer_indices(&[], 10).is_empty());
    }

    #[test]
    fn chunk_len_zero_falls_back_to_empty() {
        // Caller treats empty as "no chunk geometry" and uses the per-index path.
        assert!(bucket_outer_indices(&[1, 2, 3], 0).is_empty());
    }

    #[test]
    fn chunk_len_one_is_one_bucket_per_index() {
        // Degenerate chunking: per-chunk == per-index.
        let got = bucket_outer_indices(&[3, 5, 8], 1);
        assert_eq!(got, vec![(3, 4, vec![0]), (5, 6, vec![0]), (8, 9, vec![0])]);
    }

    #[test]
    fn survivors_in_one_chunk_coalesce_into_a_tight_window() {
        // 3 and 5 share chunk 0 (len 10): one read of [3,6), offsets 0 and 2.
        let got = bucket_outer_indices(&[3, 5], 10);
        assert_eq!(got, vec![(3, 6, vec![0, 2])]);
    }

    #[test]
    fn survivors_across_chunks_split_per_chunk() {
        // 3 -> chunk 0, 12 -> chunk 1: two separate single-element windows.
        let got = bucket_outer_indices(&[3, 12], 10);
        assert_eq!(got, vec![(3, 4, vec![0]), (12, 13, vec![0])]);
    }

    #[test]
    fn survivors_straddling_a_boundary_are_split() {
        // 8,9 in chunk 0; 10,11 in chunk 1. Windows never cross the boundary.
        let got = bucket_outer_indices(&[8, 9, 10, 11], 10);
        assert_eq!(got, vec![(8, 10, vec![0, 1]), (10, 12, vec![0, 1])]);
    }
}

#[cfg(test)]
mod plan_tests {
    use super::{build_read_plans, Keep};
    use crate::reader::filter::CoordSelection;

    // (start, end_exclusive) ranges of a plan's subset, per dimension.
    fn ranges(p: &super::ReadPlan) -> Vec<(u64, u64)> {
        p.subset
            .start()
            .iter()
            .zip(p.subset.end_exc())
            .map(|(&s, e)| (s, e))
            .collect()
    }

    #[test]
    fn all_range_is_a_single_keep_all_plan() {
        // No Indices anywhere -> one read, keep everything.
        let sels = vec![
            CoordSelection::Range(2, 5),
            CoordSelection::Range(0, 2),
            CoordSelection::Range(0, 3),
        ];
        let plans = build_read_plans(&sels, &[20, 2, 3], &[20, 2, 3], Some(&[10, 2, 3]));
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].keep, Keep::All);
        assert_eq!(ranges(&plans[0]), vec![(2, 5), (0, 2), (0, 3)]);
    }

    #[test]
    fn outer_indices_bucketed_by_chunk_when_chunk_len_gt_1() {
        // time=Indices([3,5,12]) on dim 0, chunk_len 10:
        //   chunk 0 -> survivors 3,5 -> window [3,6), offsets [0,2]
        //   chunk 1 -> survivor 12   -> window [12,13), offsets [0]
        let sels = vec![
            CoordSelection::Indices(vec![3, 5, 12]),
            CoordSelection::Range(0, 2),
            CoordSelection::Range(0, 3),
        ];
        let plans = build_read_plans(&sels, &[20, 2, 3], &[20, 2, 3], Some(&[10, 2, 3]));
        assert_eq!(plans.len(), 2);

        assert_eq!(ranges(&plans[0]), vec![(3, 6), (0, 2), (0, 3)]);
        assert_eq!(
            plans[0].keep,
            Keep::Offsets {
                window_len: 3,
                offsets: vec![0, 2]
            }
        );

        assert_eq!(ranges(&plans[1]), vec![(12, 13), (0, 2), (0, 3)]);
        assert_eq!(
            plans[1].keep,
            Keep::Offsets {
                window_len: 1,
                offsets: vec![0]
            }
        );
    }

    #[test]
    fn chunk_len_1_falls_back_to_one_plan_per_index() {
        // Each time step is its own chunk -> per-index reads, keep all.
        let sels = vec![
            CoordSelection::Indices(vec![3, 5, 12]),
            CoordSelection::Range(0, 2),
            CoordSelection::Range(0, 3),
        ];
        let plans = build_read_plans(&sels, &[20, 2, 3], &[20, 2, 3], Some(&[1, 2, 3]));
        assert_eq!(plans.len(), 3);
        assert!(plans.iter().all(|p| p.keep == Keep::All));
        assert_eq!(ranges(&plans[0]), vec![(3, 4), (0, 2), (0, 3)]);
        assert_eq!(ranges(&plans[2]), vec![(12, 13), (0, 2), (0, 3)]);
    }

    #[test]
    fn unknown_chunk_shape_falls_back_to_per_index() {
        let sels = vec![
            CoordSelection::Indices(vec![3, 5]),
            CoordSelection::Range(0, 2),
            CoordSelection::Range(0, 3),
        ];
        let plans = build_read_plans(&sels, &[20, 2, 3], &[20, 2, 3], None);
        assert_eq!(plans.len(), 2);
        assert!(plans.iter().all(|p| p.keep == Keep::All));
    }

    #[test]
    fn indices_not_on_outer_dim_falls_back_to_per_index() {
        // Indices on the middle coord (dim 1) -> Keep::Offsets layout would be
        // wrong, so we keep the safe per-index path.
        let sels = vec![
            CoordSelection::Range(0, 20),
            CoordSelection::Indices(vec![0, 1]),
            CoordSelection::Range(0, 3),
        ];
        let plans = build_read_plans(&sels, &[20, 2, 3], &[20, 2, 3], Some(&[10, 1, 3]));
        assert_eq!(plans.len(), 2);
        assert!(plans.iter().all(|p| p.keep == Keep::All));
    }
}

#[cfg(test)]
mod keep_tests {
    use super::{apply_keep, Keep};
    use arrow::array::{Array, ArrayRef, Int32Array};
    use std::sync::Arc;

    fn arr(v: &[i32]) -> ArrayRef {
        Arc::new(Int32Array::from(v.to_vec()))
    }
    fn vals(a: &ArrayRef) -> Vec<i32> {
        let a = a.as_any().downcast_ref::<Int32Array>().unwrap();
        a.iter().map(|x| x.unwrap()).collect()
    }

    #[test]
    fn keep_all_returns_unchanged() {
        let out = apply_keep(arr(&[1, 2, 3, 4]), &Keep::All).unwrap();
        assert_eq!(vals(&out), vec![1, 2, 3, 4]);
    }

    #[test]
    fn keep_offsets_gathers_inner_blocks() {
        // window_len=3 outer steps, inner=2: [s0,s0, s1,s1, s2,s2].
        // Keep offsets 0 and 2 -> first and third 2-row blocks.
        let a = arr(&[10, 11, 20, 21, 30, 31]);
        let out = apply_keep(
            a,
            &Keep::Offsets {
                window_len: 3,
                offsets: vec![0, 2],
            },
        )
        .unwrap();
        assert_eq!(vals(&out), vec![10, 11, 30, 31]);
    }

    #[test]
    fn keep_offsets_scalar_inner() {
        // inner == 1 (window_len == len): pick individual rows.
        let a = arr(&[5, 6, 7, 8]);
        let out = apply_keep(
            a,
            &Keep::Offsets {
                window_len: 4,
                offsets: vec![1, 3],
            },
        )
        .unwrap();
        assert_eq!(vals(&out), vec![6, 8]);
    }
}

#[cfg(test)]
mod resolve_outer_tests {
    use super::*;
    use crate::reader::filter::{CoordFilterKind, CoordFilters};
    use datafusion::common::ScalarValue;

    // Synthetic v3 store: time(7), lat(10), lon(10); temperature/humidity shape
    // [7,10,10]. The outer axis (7) size-matches `time`, so it is the outer coord.
    const STORE: &str = "data/synthetic_v3.zarr";

    fn one_filter(coord: &str, value: i64) -> CoordFilters {
        let mut f = CoordFilters::new();
        f.push(coord, CoordFilterKind::Eq(ScalarValue::Int64(Some(value))));
        f
    }

    #[test]
    fn unfiltered_when_filter_is_only_on_inner_coord() {
        let meta = discover_arrays(STORE).unwrap();
        // A filter on lat (inner) leaves the outer (time) unconstrained -> no read.
        let filters = one_filter("lat", 3);
        let out = resolve_outer_selection(STORE, &meta, Some(&filters)).unwrap();
        assert_eq!(out, OuterSelection::Unfiltered);
    }

    #[test]
    fn unfiltered_when_no_filters_at_all() {
        let meta = discover_arrays(STORE).unwrap();
        let out = resolve_outer_selection(STORE, &meta, None).unwrap();
        assert_eq!(out, OuterSelection::Unfiltered);
    }

    #[test]
    fn resolves_outer_time_equality_to_range() {
        let meta = discover_arrays(STORE).unwrap();
        // time values are 0..=6; time = 3 -> the single index range [3, 4).
        let filters = one_filter("time", 3);
        let out = resolve_outer_selection(STORE, &meta, Some(&filters)).unwrap();
        assert_eq!(out, OuterSelection::Resolved(CoordSelection::Range(3, 4)));
    }

    #[test]
    fn empty_when_outer_filter_matches_nothing() {
        let meta = discover_arrays(STORE).unwrap();
        let filters = one_filter("time", 999);
        let out = resolve_outer_selection(STORE, &meta, Some(&filters)).unwrap();
        assert_eq!(out, OuterSelection::Empty);
    }
}

#[cfg(test)]
mod outer_read_tests {
    use super::*;

    // ── plan_outer_coord_read (pure window/offset math) ──────────────────────

    #[test]
    fn range_reads_exact_window_identity_extract() {
        // A contiguous slice is read verbatim; extraction is the identity range.
        let (rs, re, ext) = plan_outer_coord_read(&CoordSelection::Range(8, 16));
        assert_eq!((rs, re), (8, 16));
        assert_eq!(ext, CoordSelection::Range(0, 8));
    }

    #[test]
    fn empty_range_reads_nothing() {
        let (rs, re, ext) = plan_outer_coord_read(&CoordSelection::Range(5, 5));
        assert_eq!((rs, re), (5, 5));
        assert_eq!(ext, CoordSelection::Range(0, 0));
    }

    #[test]
    fn indices_read_bounding_window_with_shifted_offsets() {
        // Scattered indices => one bounding read [min, max+1) and offsets shifted
        // into that window, so a later gather reconstructs the exact positions.
        let (rs, re, ext) = plan_outer_coord_read(&CoordSelection::Indices(vec![3, 5, 8]));
        assert_eq!((rs, re), (3, 9));
        assert_eq!(ext, CoordSelection::Indices(vec![0, 2, 5]));
    }

    #[test]
    fn empty_indices_read_nothing() {
        let (rs, re, ext) = plan_outer_coord_read(&CoordSelection::Indices(vec![]));
        assert_eq!((rs, re), (0, 0));
        assert_eq!(ext, CoordSelection::Indices(vec![]));
    }

    // ── end-to-end: partitioned read == full read (against real data) ────────
    //
    // Synthetic v3: time(7) × lat(10) × lon(10) = 700 rows, time is the outer
    // coord. The flattening is Cartesian with time most-significant, so time `t`
    // occupies rows [t*100, (t+1)*100).

    use crate::reader::schema_inference::infer_schema;
    use arrow::compute::concat_batches;
    use arrow::record_batch::RecordBatch;
    use futures::executor::block_on;
    use futures::TryStreamExt;

    const STORE: &str = "data/synthetic_v3.zarr";

    fn read(partition: Option<CoordSelection>) -> RecordBatch {
        let schema = Arc::new(infer_schema(STORE).unwrap());
        let stream = read_zarr(
            STORE,
            schema.clone(),
            None,
            None,
            None,
            None,
            partition,
            Vec::new(),
            8192,
            None,
        )
        .unwrap();
        let batches: Vec<RecordBatch> = block_on(stream.try_collect()).unwrap();
        concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    #[test]
    fn full_axis_partition_matches_unpartitioned() {
        let full = read(None);
        // A single partition spanning the whole outer axis must be identical.
        let whole = read(Some(CoordSelection::Range(0, 7)));
        assert_eq!(full, whole);
    }

    #[test]
    fn complementary_range_partitions_reconstruct_full() {
        let full = read(None);
        let p0 = read(Some(CoordSelection::Range(0, 3))); // time 0..3 -> 300 rows
        let p1 = read(Some(CoordSelection::Range(3, 7))); // time 3..7 -> 400 rows
        assert_eq!(p0.num_rows(), 300);
        assert_eq!(p1.num_rows(), 400);
        let rebuilt = concat_batches(&full.schema(), &[p0, p1]).unwrap();
        assert_eq!(full, rebuilt);
    }

    #[test]
    fn scattered_indices_partition_matches_those_time_blocks() {
        let full = read(None);
        // Read only time {1, 3, 5} via the bounding-window + gather path.
        let got = read(Some(CoordSelection::Indices(vec![1, 3, 5])));
        assert_eq!(got.num_rows(), 300);
        // Expected: full rows for time 1, then 3, then 5 (100 rows each).
        let expected = concat_batches(
            &full.schema(),
            &[
                full.slice(100, 100),
                full.slice(300, 100),
                full.slice(500, 100),
            ],
        )
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn empty_partition_yields_no_rows() {
        assert_eq!(read(Some(CoordSelection::Range(2, 2))).num_rows(), 0);
    }

    // ── multi-axis (N-D box) partition transparency ──────────────────────────
    //
    // `partition_extra` restricts INNER axes so a partition reads an N-D box of
    // chunks. Coords follow the data var's native dimension order [time(0),
    // lat(1), lon(2)]; time is the outer axis, so lat/lon are the inner axes.
    // These prove a box read returns exactly the right rows.
    const LAT: usize = 1;

    /// Read one box: `outer` on the outer axis, `extra` on inner axes.
    fn read_box(outer: Option<CoordSelection>, extra: Vec<(usize, CoordSelection)>) -> RecordBatch {
        let schema = Arc::new(infer_schema(STORE).unwrap());
        let stream = read_zarr(
            STORE,
            schema.clone(),
            None,
            None,
            None,
            None,
            outer,
            extra,
            8192,
            None,
        )
        .unwrap();
        let batches: Vec<RecordBatch> = block_on(stream.try_collect()).unwrap();
        concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    /// Order-independent fingerprint of the `temperature` column (i64/f32/f64).
    fn temp_sum(rb: &RecordBatch) -> f64 {
        use arrow::array::{Float32Array, Float64Array, Int64Array};
        let idx = rb.schema().index_of("temperature").unwrap();
        let col = rb.column(idx);
        if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
            (0..a.len()).map(|i| a.value(i) as f64).sum()
        } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
            (0..a.len()).map(|i| a.value(i) as f64).sum()
        } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
            (0..a.len()).map(|i| a.value(i)).sum()
        } else {
            panic!("unexpected temperature type: {:?}", col.data_type());
        }
    }

    #[test]
    fn full_range_extra_is_noop() {
        // An inner restriction spanning the whole lat axis must be identical to no
        // restriction at all (same rows, same order).
        let full = read(None);
        let same = read_box(
            Some(CoordSelection::Range(0, 7)),
            vec![(LAT, CoordSelection::Range(0, 10))],
        );
        assert_eq!(full, same);
    }

    #[test]
    fn inner_axis_box_partitions_reconstruct_full() {
        // Split the INNER lat axis into [0,5) and [5,10), each spanning all time &
        // lon. Together they must reconstruct the full read — verified by row count
        // and an order-independent column sum (lat-split rows interleave, so a plain
        // concat wouldn't match order).
        let full = read(None);
        let a = read_box(None, vec![(LAT, CoordSelection::Range(0, 5))]);
        let b = read_box(None, vec![(LAT, CoordSelection::Range(5, 10))]);
        assert_eq!(a.num_rows(), 350, "7 time × 5 lat × 10 lon");
        assert_eq!(b.num_rows(), 350);
        assert_eq!(a.num_rows() + b.num_rows(), full.num_rows());
        let union = temp_sum(&a) + temp_sum(&b);
        assert!(
            (union - temp_sum(&full)).abs() < 1e-2,
            "box union sum {union} != full {}",
            temp_sum(&full)
        );
    }

    #[test]
    fn outer_and_inner_box_grid_reconstructs_full() {
        // A 2×2 grid over (time, lat) — the real multi-axis case — reconstructs the
        // full cube exactly (row count + order-independent sum).
        let full = read(None);
        let grid = [
            (CoordSelection::Range(0, 3), CoordSelection::Range(0, 5)),
            (CoordSelection::Range(0, 3), CoordSelection::Range(5, 10)),
            (CoordSelection::Range(3, 7), CoordSelection::Range(0, 5)),
            (CoordSelection::Range(3, 7), CoordSelection::Range(5, 10)),
        ];
        let mut rows = 0usize;
        let mut sum = 0.0f64;
        for (t, lat) in grid {
            let rb = read_box(Some(t), vec![(LAT, lat)]);
            rows += rb.num_rows();
            sum += temp_sum(&rb);
        }
        assert_eq!(rows, full.num_rows());
        assert!((sum - temp_sum(&full)).abs() < 1e-2);
    }
}

#[cfg(test)]
mod window_tests {
    use super::window_outer_selection;
    use crate::reader::filter::CoordSelection;

    fn r(s: usize, e: usize) -> CoordSelection {
        CoordSelection::Range(s, e)
    }
    fn ix(v: &[usize]) -> CoordSelection {
        CoordSelection::Indices(v.to_vec())
    }

    #[test]
    fn range_splits_evenly() {
        assert_eq!(
            window_outer_selection(&r(0, 10), 5),
            vec![r(0, 5), r(5, 10)]
        );
    }

    #[test]
    fn range_last_window_is_shorter() {
        assert_eq!(
            window_outer_selection(&r(0, 7), 3),
            vec![r(0, 3), r(3, 6), r(6, 7)]
        );
    }

    #[test]
    fn range_larger_than_len_is_one_window() {
        assert_eq!(window_outer_selection(&r(2, 5), 100), vec![r(2, 5)]);
    }

    #[test]
    fn range_preserves_offset() {
        // Windows carry absolute indices, not zero-based.
        assert_eq!(
            window_outer_selection(&r(4, 9), 2),
            vec![r(4, 6), r(6, 8), r(8, 9)]
        );
    }

    #[test]
    fn indices_chunk_in_order() {
        assert_eq!(
            window_outer_selection(&ix(&[1, 3, 5, 7]), 2),
            vec![ix(&[1, 3]), ix(&[5, 7])]
        );
    }

    #[test]
    fn indices_last_chunk_is_shorter() {
        assert_eq!(
            window_outer_selection(&ix(&[1, 3, 5, 7, 9]), 2),
            vec![ix(&[1, 3]), ix(&[5, 7]), ix(&[9])]
        );
    }

    #[test]
    fn empty_selection_yields_one_empty_window() {
        assert_eq!(window_outer_selection(&r(5, 5), 4), vec![r(5, 5)]);
        assert_eq!(window_outer_selection(&ix(&[]), 4), vec![ix(&[])]);
    }

    #[test]
    fn zero_max_steps_is_clamped_to_one() {
        assert_eq!(window_outer_selection(&r(0, 2), 0), vec![r(0, 1), r(1, 2)]);
    }

    #[test]
    fn windows_cover_the_whole_selection() {
        // The union of windows equals the input, in order (the concat invariant).
        for &(s, e) in &[(0usize, 13usize), (3, 3), (7, 20)] {
            for w in 1..=6 {
                let pieces = window_outer_selection(&r(s, e), w);
                let rebuilt: Vec<usize> = pieces
                    .iter()
                    .flat_map(|p| match p {
                        CoordSelection::Range(a, b) => (*a..*b).collect::<Vec<_>>(),
                        CoordSelection::Indices(v) => v.clone(),
                    })
                    .collect();
                let expected: Vec<usize> = (s..e).collect();
                assert_eq!(rebuilt, expected, "range=({s},{e}) w={w}");
            }
        }
    }
}

// End-to-end tests for the async streaming path (`read_zarr_async`), driven over
// a local object store so they run in CI without network. Mirrors the sync-path
// streaming coverage: streamed == reference, and LIMIT stops early (Phase 3).
#[cfg(test)]
mod async_streaming_tests {
    use super::*;
    use crate::reader::stats::ZarrIoStats;
    use arrow::record_batch::RecordBatch;
    use futures::TryStreamExt;
    use zarrs_object_store::object_store::local::LocalFileSystem;
    use zarrs_object_store::AsyncObjectStore;

    const STORE: &str = "data/synthetic_v3.zarr"; // time(7) × lat(10) × lon(10)

    fn local_async_store() -> (AsyncReadableListableStorage, ObjectPath) {
        let abs = std::fs::canonicalize(STORE).unwrap();
        let fs = LocalFileSystem::new_with_prefix(abs).unwrap();
        let store: AsyncReadableListableStorage = Arc::new(AsyncObjectStore::new(fs));
        (store, ObjectPath::from(""))
    }

    async fn read_async(
        limit: Option<usize>,
        batch_size: usize,
        stats: Option<SharedIoStats>,
    ) -> Vec<RecordBatch> {
        let (schema, meta) =
            crate::reader::schema_inference::infer_schema_with_meta(STORE).unwrap();
        let (store, prefix) = local_async_store();
        let stream = read_zarr_async(
            store,
            &prefix,
            Arc::new(schema),
            None,
            limit,
            stats,
            Some(meta),
            None,
            None,
            Vec::new(),
            batch_size,
            None,
            Arc::default(),
        )
        .await
        .unwrap();
        stream.try_collect().await.unwrap()
    }

    /// Each one-step time partition, read concurrently through `cache`.
    async fn read_partitions(
        cache: impl Fn() -> Arc<ScanCache>,
        stats: SharedIoStats,
    ) -> Vec<RecordBatch> {
        let (schema, meta) =
            crate::reader::schema_inference::infer_schema_with_meta(STORE).unwrap();
        let schema = Arc::new(schema);
        let (store, prefix) = local_async_store();
        let partitions = (0..7).map(|t| {
            let (store, prefix, schema, meta) =
                (store.clone(), prefix.clone(), schema.clone(), meta.clone());
            let (stats, cache) = (stats.clone(), cache());
            async move {
                let stream = read_zarr_async(
                    store,
                    &prefix,
                    schema,
                    None,
                    None,
                    Some(stats),
                    Some(meta),
                    None,
                    Some(CoordSelection::Range(t, t + 1)),
                    Vec::new(),
                    10_000_000,
                    None,
                    cache,
                )
                .await?;
                stream.try_collect::<Vec<_>>().await
            }
        });
        futures::future::try_join_all(partitions)
            .await
            .unwrap()
            .concat()
    }

    #[tokio::test]
    async fn partitions_sharing_a_cache_read_each_coordinate_chunk_once() {
        let coord_reads = |stats: &SharedIoStats| {
            stats
                .coord_arrays
                .load(std::sync::atomic::Ordering::Relaxed)
        };
        let alone = Arc::new(ZarrIoStats::default());
        let reference = read_partitions(Arc::default, alone.clone()).await;
        let shared = Arc::new(ZarrIoStats::default());
        let cache = Arc::new(ScanCache::default());
        let batches = read_partitions(|| cache.clone(), shared.clone()).await;

        // time, lat and lon for each of the 7 partitions, or each once: time is
        // one chunk, which every partition's step falls in.
        assert_eq!(coord_reads(&alone), 21);
        assert_eq!(coord_reads(&shared), 3);
        assert_eq!(rows(&batches), 700);
        assert_eq!(rendered(&batches), rendered(&reference));
    }

    #[test]
    fn a_slice_widens_to_the_chunks_it_falls_in() {
        assert_eq!(widen_to_chunks(3, 5, Some(4), 10), (0, 8));
        assert_eq!(widen_to_chunks(8, 9, Some(4), 10), (8, 10));
        assert_eq!(widen_to_chunks(4, 8, Some(4), 10), (4, 8));
        assert_eq!(widen_to_chunks(3, 5, None, 10), (3, 5));
        assert_eq!(widen_to_chunks(0, 0, Some(4), 10), (0, 0));
    }

    fn rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    fn rendered(batches: &[RecordBatch]) -> String {
        arrow::util::pretty::pretty_format_batches(batches)
            .unwrap()
            .to_string()
    }

    #[tokio::test]
    async fn async_streaming_is_transparent() {
        // batch_size 150, inner_rows=100 => 1 time step/window => 7 windows of 100.
        let streamed = read_async(None, 150, None).await;
        let reference = read_async(None, 10_000_000, None).await;
        assert!(
            streamed.len() > 1,
            "expected multiple batches, got {}",
            streamed.len()
        );
        assert_eq!(reference.len(), 1, "reference reads in one batch");
        assert_eq!(rows(&streamed), 700);
        assert_eq!(rendered(&streamed), rendered(&reference));
    }

    #[tokio::test]
    async fn async_streaming_limit_reads_less() {
        // LIMIT 100 (one time plane) with 100-row windows: the lazy stream should
        // stop after the first window and read far fewer data bytes than a full
        // scan — proof the remote path streams instead of materializing everything.
        let full_stats = Arc::new(ZarrIoStats::default());
        let full = read_async(None, 100, Some(full_stats.clone())).await;
        assert_eq!(rows(&full), 700);

        let limit_stats = Arc::new(ZarrIoStats::default());
        let limited = read_async(Some(100), 100, Some(limit_stats.clone())).await;
        assert_eq!(rows(&limited), 100, "LIMIT caps total rows");

        let full_bytes = full_stats
            .data_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        let limit_bytes = limit_stats
            .data_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            limit_bytes < full_bytes,
            "LIMIT should read fewer data bytes ({limit_bytes}) than a full scan ({full_bytes})"
        );

        // And the rows match the first 100 of a full read.
        let reference = read_async(None, 10_000_000, None).await;
        assert_eq!(rendered(&limited), rendered(&[reference[0].slice(0, 100)]));
    }
}
