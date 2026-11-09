//! Chunk-writing sink: scatter Arrow rows into a skeleton's data-variable chunks.
//!
//! Phase 2 of docs/zarr-write-roundtrip-plan.md. The central constraint:
//!
//! > A Zarr chunk must be written as a complete n-D tile, but rows arrive in scan
//! > order.
//!
//! The resolution is to **scatter each row into a dense buffer by its grid index**
//! — no ordering requirement, so no `SortExec` and no `ORDER BY`, and a
//! row-oriented source in arbitrary order costs nothing (§5.6). Row -> index is a
//! binary search of each coordinate value into its axis; a value that is not on the
//! axis is a **loud error**, never a silent drop, because it means the source and
//! target grids disagree.
//!
//! Two entry points:
//! - [`write_batches`] — a single writer holding the whole array in memory.
//! - [`write_batches_partitioned`] — `N` concurrent writers, each owning a
//!   chunk-aligned outer slab. Because [`plan_partitions`] boundaries are
//!   chunk-aligned, no two writers ever touch the same chunk, so the §5.4
//!   last-write-wins hazard cannot arise (Phase 5). Both share the [`write_range`]
//!   core and give identical results.
//!
//! **Known limits of this cut:**
//!
//! - **Whole array (or slab) materialised in memory.** Each writer builds a dense
//!   `Vec` for its region and hands it to zarrs in one `store_array_subset` call,
//!   which owns all chunk decomposition (ragged edge chunks included).
//!   Partitioning bounds a *writer's* buffer to its slab; bounding *accumulation*
//!   too (a streaming input shuffle, plan gap 3) waits on the `DataSink` driver.
//! - **Plain columns only.** Coordinate columns are read as plain typed arrays, not
//!   the `DictionaryArray` fast path (§5.6, Phase 6). Build and test where nothing
//!   clever can hide a bug first.
//! - **Default fills only.** A variable with a custom `fill_value` is refused: this
//!   sink writes the *whole* region including the holes, so the hole value it writes
//!   must equal the array's declared fill, and it only knows the defaults (0 / NaN).

use std::collections::BTreeMap;

use arrow::array::{Array as _, Float32Array, Float64Array, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::DataType as ArrowType;
use arrow::record_batch::RecordBatch;
use tracing::{debug, info};
use zarrs::array::Array;
use zarrs::array::ArraySubset;
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

use crate::physical_plan::partition::plan_partitions;
use super::skeleton::{CoordValues, SkeletonSpec, WriteDataType};
use super::store::open_target;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A dense, row-major target buffer for one data variable, initialised to the
/// variable's fill value.
enum Buffer {
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Buffer {
    fn filled(dtype: WriteDataType, total: usize) -> Self {
        match dtype {
            WriteDataType::Int64 => Buffer::I64(vec![0i64; total]),
            WriteDataType::Float32 => Buffer::F32(vec![f32::NAN; total]),
            WriteDataType::Float64 => Buffer::F64(vec![f64::NAN; total]),
        }
    }
}

/// One coordinate column of a batch, cast to its axis's element type, paired with
/// the axis values it indexes into.
enum AxisCol<'a> {
    I64 { col: Int64Array, axis: &'a [i64] },
    F64 { col: Float64Array, axis: &'a [f64] },
}

impl AxisCol<'_> {
    /// The global axis index of the value in row `r`. `Err` if the value is null or
    /// not present on the axis — both mean the row does not belong on this grid.
    fn index_at(&self, name: &str, r: usize) -> Result<usize, BoxError> {
        match self {
            AxisCol::I64 { col, axis } => {
                if col.is_null(r) {
                    return Err(format!("null value in coordinate column '{name}'").into());
                }
                let v = col.value(r);
                let found = if descending(axis) {
                    axis.binary_search_by(|a| v.cmp(a))
                } else {
                    axis.binary_search(&v)
                };
                found.map_err(|_| {
                    format!("value {v} in column '{name}' is not on axis '{name}'").into()
                })
            }
            AxisCol::F64 { col, axis } => {
                if col.is_null(r) {
                    return Err(format!("null value in coordinate column '{name}'").into());
                }
                let v = col.value(r);
                let reverse = descending(axis);
                axis.binary_search_by(|a| {
                    let order = a
                        .partial_cmp(&v)
                        .expect("axis and coordinate values are non-NaN");
                    if reverse {
                        order.reverse()
                    } else {
                        order
                    }
                })
                .map_err(|_| format!("value {v} in column '{name}' is not on axis '{name}'").into())
            }
        }
    }
}

/// Whether an axis's values run from high to low, as latitude often does.
fn descending<T: PartialOrd>(axis: &[T]) -> bool {
    axis.first() > axis.last()
}

/// One data-variable column of a batch, cast to the variable's element type.
enum VarCol {
    I64(Int64Array),
    F32(Float32Array),
    F64(Float64Array),
}

impl VarCol {
    /// Write the value in row `r` into `buf` at `pos`. A null leaves the fill value
    /// in place (null == missing == hole).
    fn scatter(&self, r: usize, buf: &mut Buffer, pos: usize) {
        match (self, buf) {
            (VarCol::I64(c), Buffer::I64(b)) => {
                if !c.is_null(r) {
                    b[pos] = c.value(r);
                }
            }
            (VarCol::F32(c), Buffer::F32(b)) => {
                if !c.is_null(r) {
                    b[pos] = c.value(r);
                }
            }
            (VarCol::F64(c), Buffer::F64(b)) => {
                if !c.is_null(r) {
                    b[pos] = c.value(r);
                }
            }
            // Buffer and column are built from the same WriteDataType, so the
            // variants always agree.
            _ => unreachable!("buffer and column element types diverged"),
        }
    }
}

/// Cast `col` to `target` and downcast to a concrete Arrow array.
fn cast_to(
    batch: &RecordBatch,
    name: &str,
    target: &ArrowType,
) -> Result<arrow::array::ArrayRef, BoxError> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| format!("column '{name}' not found in input"))?;
    cast(batch.column(idx), target).map_err(|e| format!("cast '{name}' to {target:?}: {e}").into())
}

/// Row-major strides for `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// A custom fill would disagree with the holes this sink writes (see module docs).
fn reject_custom_fills(spec: &SkeletonSpec) -> Result<(), BoxError> {
    if let Some(v) = spec.data_vars.iter().find(|v| v.fill_value.is_some()) {
        return Err(format!(
            "data variable '{}' has a custom fill_value; the sink writes the whole \
             target region and only knows the default fills (0 / NaN)",
            v.name
        )
        .into());
    }
    Ok(())
}

/// A batch's coordinate columns, one per grid axis, cast to their axes' types.
fn axis_columns<'a>(
    batch: &RecordBatch,
    spec: &'a SkeletonSpec,
) -> Result<Vec<AxisCol<'a>>, BoxError> {
    spec.coords
        .iter()
        .map(|c| match &c.values {
            CoordValues::Int64(axis) => {
                let arr = cast_to(batch, &c.name, &ArrowType::Int64)?;
                let col = arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or("cast to Int64 did not yield Int64Array")?
                    .clone();
                Ok(AxisCol::I64 { col, axis })
            }
            CoordValues::Float64(axis) => {
                let arr = cast_to(batch, &c.name, &ArrowType::Float64)?;
                let col = arr
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or("cast to Float64 did not yield Float64Array")?
                    .clone();
                Ok(AxisCol::F64 { col, axis })
            }
        })
        .collect()
}

/// A batch's data-variable columns, cast to their variables' types.
fn var_columns(batch: &RecordBatch, spec: &SkeletonSpec) -> Result<Vec<VarCol>, BoxError> {
    spec.data_vars
        .iter()
        .map(|v| match v.data_type {
            WriteDataType::Int64 => {
                let arr = cast_to(batch, &v.name, &ArrowType::Int64)?;
                Ok(VarCol::I64(
                    arr.as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or("cast to Int64 did not yield Int64Array")?
                        .clone(),
                ))
            }
            WriteDataType::Float32 => {
                let arr = cast_to(batch, &v.name, &ArrowType::Float32)?;
                Ok(VarCol::F32(
                    arr.as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or("cast to Float32 did not yield Float32Array")?
                        .clone(),
                ))
            }
            WriteDataType::Float64 => {
                let arr = cast_to(batch, &v.name, &ArrowType::Float64)?;
                Ok(VarCol::F64(
                    arr.as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or("cast to Float64 did not yield Float64Array")?
                        .clone(),
                ))
            }
        })
        .collect()
}

/// Write the rows whose outer-axis (axis 0) index falls in `[outer_start,
/// outer_end)` into that chunk-aligned slab of the target.
///
/// The slab is full on every inner axis and clipped only on the outer axis, so it
/// touches a *disjoint, whole* set of chunks: two calls with non-overlapping outer
/// ranges never write the same chunk, which is what makes them safe to run
/// concurrently (§5.4). Rows outside the range are ignored, so every partition can
/// see the same input and take only its share.
fn write_range(
    store: &ReadableWritableListableStorage,
    spec: &SkeletonSpec,
    shape: &[usize],
    strides: &[usize],
    batches: &[RecordBatch],
    outer_start: usize,
    outer_end: usize,
) -> Result<u64, BoxError> {
    let outer_stride = strides[0]; // = product of inner-axis sizes
    let slab_rows = (outer_end - outer_start) * outer_stride;
    let base = outer_start * outer_stride;

    let mut buffers: Vec<Buffer> = spec
        .data_vars
        .iter()
        .map(|v| Buffer::filled(v.data_type, slab_rows))
        .collect();
    let arrays: Vec<Array<dyn ReadableWritableListableStorageTraits>> = spec
        .data_vars
        .iter()
        .map(|v| Array::open(store.clone(), &format!("/{}", v.name)))
        .collect::<Result<_, _>>()?;

    let mut rows_written: u64 = 0;

    for batch in batches {
        let n = batch.num_rows();

        let axis_cols = axis_columns(batch, spec)?;
        let var_cols = var_columns(batch, spec)?;

        for r in 0..n {
            // Resolve the outer index first, so rows outside this slab are cheaply
            // skipped (and off-grid values still error, in every partition).
            let outer_idx = axis_cols[0].index_at(&spec.coords[0].name, r)?;
            if outer_idx < outer_start || outer_idx >= outer_end {
                continue;
            }
            let mut pos = outer_idx * strides[0];
            for (axis_idx, ac) in axis_cols.iter().enumerate().skip(1) {
                pos += ac.index_at(&spec.coords[axis_idx].name, r)? * strides[axis_idx];
            }
            let pos_slab = pos - base;
            for (v, vc) in var_cols.iter().enumerate() {
                vc.scatter(r, &mut buffers[v], pos_slab);
            }
            rows_written += 1;
        }
    }

    // One subset write over the slab: outer axis clipped to the range, inner axes
    // full. zarrs decomposes it into chunks (inner edge chunks included).
    let mut ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(shape.len());
    ranges.push(outer_start as u64..outer_end as u64);
    for &s in &shape[1..] {
        ranges.push(0..s as u64);
    }
    let subset = ArraySubset::new_with_ranges(&ranges);
    for (v, (array, buf)) in arrays.iter().zip(buffers).enumerate() {
        match buf {
            Buffer::I64(b) => array.store_array_subset(&subset, b.as_slice()),
            Buffer::F32(b) => array.store_array_subset(&subset, b.as_slice()),
            Buffer::F64(b) => array.store_array_subset(&subset, b.as_slice()),
        }
        .map_err(|e| format!("store '{}': {e}", spec.data_vars[v].name))?;
    }
    debug!(outer_start, outer_end, rows_written, "slab written");
    Ok(rows_written)
}

/// Write data-variable chunks into an existing skeleton by scattering the rows of
/// `batches` into the target grid. Single writer, whole array in memory.
///
/// `spec` must be the same spec the skeleton was created from — it supplies the
/// coordinate axis values used to map each row to its grid index. Returns the
/// number of rows written.
pub fn write_batches(
    store_path: &str,
    spec: &SkeletonSpec,
    batches: impl IntoIterator<Item = RecordBatch>,
) -> Result<u64, BoxError> {
    reject_custom_fills(spec)?;
    let shape: Vec<usize> = spec.coords.iter().map(|c| c.values.len()).collect();
    let strides = row_major_strides(&shape);
    let batches: Vec<RecordBatch> = batches.into_iter().collect();
    let store = open_target(store_path)?;
    let n = write_range(&store, spec, &shape, &strides, &batches, 0, shape[0])?;
    info!(store_path, rows = n, "sink wrote data variables");
    Ok(n)
}

/// Write in parallel across `target_partitions` writers, each owning a
/// chunk-aligned slab of the outer axis.
///
/// The partitioning comes from [`plan_partitions`], whose boundaries are
/// **chunk-aligned** — so each target chunk lies wholly within one partition and no
/// two writers ever touch the same chunk (§5.4). That disjointness is what makes
/// the concurrent writes correct: last-write-wins cannot arise when there is only
/// ever one writer per chunk. Result is identical to [`write_batches`]; more
/// partitions bound each writer's buffer to its slab rather than the whole array.
pub fn write_batches_partitioned(
    store_path: &str,
    spec: &SkeletonSpec,
    batches: impl IntoIterator<Item = RecordBatch>,
    target_partitions: usize,
) -> Result<u64, BoxError> {
    reject_custom_fills(spec)?;
    let shape: Vec<usize> = spec.coords.iter().map(|c| c.values.len()).collect();
    let strides = row_major_strides(&shape);
    let batches: Vec<RecordBatch> = batches.into_iter().collect();

    let store = open_target(store_path)?;
    let ranges: Vec<(usize, usize)> = plan_partitions(shape[0] as u64, spec.chunks[0], target_partitions)
        .iter()
        .filter_map(|p| p.as_range())
        .collect();

    // Each closure borrows shared, immutable state and writes a disjoint slab.
    let results: Vec<Result<u64, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = ranges
            .iter()
            .map(|&(a, b)| {
                let (store, spec, shape, strides, batches) =
                    (&store, spec, &shape, &strides, &batches);
                scope.spawn(move || {
                    write_range(store, spec, shape, strides, batches, a, b)
                        .map_err(|e| e.to_string())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("writer thread panicked".into())))
            .collect()
    });

    let mut total = 0u64;
    for r in results {
        total += r.map_err(BoxError::from)?;
    }
    info!(
        store_path,
        rows = total,
        partitions = ranges.len(),
        "sink wrote data variables (partitioned)"
    );
    Ok(total)
}

/// Write the chunks `batches`' rows fall in into an existing skeleton, each chunk
/// whole: its cells without a row take the fill value. Chunks no row falls in are
/// left as they are, so writers whose rows fall in different chunks can fill one
/// store between them; two writing the same chunk would each erase the other's
/// cells. Returns the number of rows written.
pub fn write_chunks(
    store_path: &str,
    spec: &SkeletonSpec,
    batches: impl IntoIterator<Item = RecordBatch>,
) -> Result<u64, BoxError> {
    reject_custom_fills(spec)?;
    let shape: Vec<usize> = spec.coords.iter().map(|c| c.values.len()).collect();
    let chunk: Vec<usize> = spec.chunks.iter().map(|&c| c as usize).collect();
    // A chunk at the end of an axis is cut short by it.
    let extent = |index: &[usize]| -> Vec<usize> {
        index
            .iter()
            .zip(&chunk)
            .zip(&shape)
            .map(|((&i, &c), &s)| c.min(s - i * c))
            .collect()
    };

    let mut touched: BTreeMap<Vec<usize>, Vec<Buffer>> = BTreeMap::new();
    let mut rows_written = 0u64;
    for batch in batches {
        let axis_cols = axis_columns(&batch, spec)?;
        let var_cols = var_columns(&batch, spec)?;
        for r in 0..batch.num_rows() {
            let at = axis_cols
                .iter()
                .zip(&spec.coords)
                .map(|(ac, c)| ac.index_at(&c.name, r))
                .collect::<Result<Vec<_>, _>>()?;
            let index: Vec<usize> = at.iter().zip(&chunk).map(|(&i, &c)| i / c).collect();
            let size = extent(&index);
            let mut pos = 0;
            for ((&i, &c), &n) in at.iter().zip(&chunk).zip(&size) {
                pos = pos * n + i % c;
            }
            let buffers = touched.entry(index).or_insert_with(|| {
                let cells = size.iter().product();
                spec.data_vars
                    .iter()
                    .map(|v| Buffer::filled(v.data_type, cells))
                    .collect()
            });
            for (vc, buf) in var_cols.iter().zip(buffers.iter_mut()) {
                vc.scatter(r, buf, pos);
            }
            rows_written += 1;
        }
    }

    let store = open_target(store_path)?;
    let arrays: Vec<Array<dyn ReadableWritableListableStorageTraits>> = spec
        .data_vars
        .iter()
        .map(|v| Array::open(store.clone(), &format!("/{}", v.name)))
        .collect::<Result<_, _>>()?;
    let chunks_written = touched.len();
    for (index, buffers) in touched {
        let ranges: Vec<std::ops::Range<u64>> = index
            .iter()
            .zip(&chunk)
            .zip(extent(&index))
            .map(|((&i, &c), n)| (i * c) as u64..(i * c + n) as u64)
            .collect();
        let subset = ArraySubset::new_with_ranges(&ranges);
        for (v, (array, buf)) in arrays.iter().zip(buffers).enumerate() {
            match buf {
                Buffer::I64(b) => array.store_array_subset(&subset, b.as_slice()),
                Buffer::F32(b) => array.store_array_subset(&subset, b.as_slice()),
                Buffer::F64(b) => array.store_array_subset(&subset, b.as_slice()),
            }
            .map_err(|e| format!("store '{}': {e}", spec.data_vars[v].name))?;
        }
    }
    info!(
        store_path,
        rows = rows_written,
        chunks = chunks_written,
        "sink wrote chunks"
    );
    Ok(rows_written)
}
