use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::stats::{ColumnStatistics, Precision, Statistics};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::{datasource::TableProvider, error::Result, physical_plan::ExecutionPlan};
use std::sync::Arc;
use tracing::{debug, info};
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

use crate::physical_plan::partition::{plan_partitions, split_selection, PartitionSpec};
use crate::physical_plan::zarr_exec::{StoreKind, ZarrExec};
use crate::reader::filter::{is_date_part_filter, parse_coord_filters, CoordFilters, CoordSelection};
use crate::reader::schema_inference::{ZarrArrayMeta, ZarrStoreMeta};
use crate::reader::storage::is_remote_url;
use crate::reader::virtual_store::{is_virtualizarr_store, VirtualStoreAdapter};
use crate::reader::zarr_reader::{
    resolve_outer_selection, resolve_outer_selection_async, OuterSelection,
};

/// Cached remote store info (store, prefix, metadata)
pub type CachedRemoteStore = Option<(AsyncReadableListableStorage, ObjectPath, ZarrStoreMeta)>;

/// Cached VirtualiZarr adapter (pre-loaded refs and metadata)
pub type CachedVirtualiZarrAdapter = Option<Arc<VirtualStoreAdapter>>;

pub struct ZarrTable {
    schema: SchemaRef,
    path: String,
    /// Cached async store and metadata for remote URLs (avoids recreating on each query)
    cached_remote: CachedRemoteStore,
    /// Store metadata for statistics (used for count optimization)
    store_meta: Option<ZarrStoreMeta>,
    /// Cached VirtualiZarr adapter for remote VirtualiZarr stores
    cached_virtualizarr: CachedVirtualiZarrAdapter,
    /// How a scan reopens the store when it has no cached one, as on a worker.
    store_kind: StoreKind,
}

impl std::fmt::Debug for ZarrTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrTable")
            .field("schema", &self.schema)
            .field("path", &self.path)
            .field(
                "cached_remote",
                &self.cached_remote.as_ref().map(|(_, p, _)| p),
            )
            .field(
                "total_rows",
                &self.store_meta.as_ref().map(|m| m.total_rows),
            )
            .field(
                "has_virtualizarr_adapter",
                &self.cached_virtualizarr.is_some(),
            )
            .finish()
    }
}

impl ZarrTable {
    pub fn new(schema: SchemaRef, path: impl Into<String>) -> Self {
        Self {
            schema,
            path: path.into(),
            cached_remote: None,
            store_meta: None,
            cached_virtualizarr: None,
            store_kind: StoreKind::Zarr,
        }
    }

    /// Create a ZarrTable with store metadata (for local paths)
    pub fn with_metadata(
        schema: SchemaRef,
        path: impl Into<String>,
        metadata: ZarrStoreMeta,
    ) -> Self {
        Self {
            schema,
            path: path.into(),
            cached_remote: None,
            store_meta: Some(metadata),
            cached_virtualizarr: None,
            store_kind: StoreKind::Zarr,
        }
    }

    /// Create a ZarrTable with a cached async store and metadata (for remote URLs)
    pub fn with_cached_remote(
        schema: SchemaRef,
        path: impl Into<String>,
        store: AsyncReadableListableStorage,
        prefix: ObjectPath,
        metadata: ZarrStoreMeta,
    ) -> Self {
        Self {
            schema,
            path: path.into(),
            cached_remote: Some((store, prefix, metadata.clone())),
            store_meta: Some(metadata),
            cached_virtualizarr: None,
            store_kind: StoreKind::Zarr,
        }
    }

    /// Create a ZarrTable for a remote VirtualiZarr store with cached adapter
    pub fn with_remote_virtualizarr(
        schema: SchemaRef,
        path: impl Into<String>,
        adapter: Arc<VirtualStoreAdapter>,
        metadata: ZarrStoreMeta,
    ) -> Self {
        Self {
            schema,
            path: path.into(),
            cached_remote: None,
            store_meta: Some(metadata),
            cached_virtualizarr: Some(adapter),
            store_kind: StoreKind::Zarr,
        }
    }

    /// Create a ZarrTable backed by an icechunk store (local or remote).
    ///
    /// The icechunk store is parked in the `cached_remote` slot (keyed at
    /// `prefix`, which is the group path within the repo, or root) so schema
    /// inference, partition planning and the async read path treat it like any
    /// other remote store.
    /// Its scans carry how to reopen the repo.
    pub fn with_cached_icechunk(
        schema: SchemaRef,
        path: impl Into<String>,
        store: AsyncReadableListableStorage,
        prefix: ObjectPath,
        metadata: ZarrStoreMeta,
        anonymous: bool,
    ) -> Self {
        let group = (!prefix.as_ref().is_empty()).then(|| prefix.to_string());
        Self {
            schema,
            path: path.into(),
            cached_remote: Some((store, prefix, metadata.clone())),
            store_meta: Some(metadata),
            cached_virtualizarr: None,
            store_kind: StoreKind::Icechunk { group, anonymous },
        }
    }

    /// Get the store metadata (for dimension info display)
    pub fn store_meta(&self) -> Option<&ZarrStoreMeta> {
        self.store_meta.as_ref()
    }

    /// Get the store path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Decide this scan's output partitions.
    ///
    /// When an outer-axis filter is present, this resolves it to the *surviving*
    /// index set (reading only the outer coordinate, on the head, at plan time)
    /// and splits THAT across partitions — so a narrow filter still fans out
    /// across the cluster. With no outer filter it falls back to splitting the
    /// full axis geometrically (no coordinate read).
    ///
    /// Returns an EMPTY vec to mean "single unpartitioned read" — used for
    /// VirtualiZarr stores (their read path doesn't take a range yet) and when we
    /// lack the metadata to slice safely.
    async fn plan_scan_partitions(
        &self,
        state: &dyn Session,
        limit: Option<usize>,
        coord_filters: Option<&CoordFilters>,
    ) -> Vec<PartitionSpec> {
        // Guard 0: a LIMIT is a GLOBAL row cap, but it gets pushed into each
        // partition's read independently — which is unsound across multiple
        // partitions (each would return its own first-N rows). Until per-
        // partition fetch is handled by a global limit above the scan, keep
        // limited queries single-partition (where pushdown stays correct).
        if limit.is_some() {
            return Vec::new();
        }

        // Guard 1: local and remote stores partition; VirtualiZarr does not yet
        // (its async read path ignores the partition selection).
        if is_virtualizarr_store(&self.path) {
            return Vec::new();
        }

        // Guard 2: we need metadata and a data variable to size the outer axis.
        let Some(meta) = &self.store_meta else {
            return Vec::new();
        };
        let Some(data_var) = meta.data_vars.first() else {
            return Vec::new();
        };
        // The reader maps the partition selection to the coordinate whose size
        // equals the data var's outer-axis length (`shape[0]`). If no such
        // coordinate exists we can't slice safely, so don't partition.
        let outer_len: u64 = data_var.shape.first().copied().unwrap_or(0);
        let outer_coord_exists = meta
            .coords
            .iter()
            .any(|c| c.shape.first().copied() == Some(outer_len));
        if !outer_coord_exists {
            return Vec::new();
        }

        let target_partitions: usize = state.config().target_partitions();
        // Unknown chunking => treat the whole axis as one chunk (=> one partition).
        let chunk_len: u64 = data_var
            .chunks
            .as_ref()
            .and_then(|c| c.first().copied())
            .unwrap_or(outer_len);

        // Resolve the outer filter on the head (reads only the outer coord, and
        // only when an outer filter exists). Remote and icechunk stores resolve
        // via the cached async store the head set up; workers never reach here.
        // A local icechunk path must not reach the sync resolver, which would
        // open it as a plain Zarr store.
        let outer = match &self.cached_remote {
            Some((store, prefix, _)) => {
                resolve_outer_selection_async(store.clone(), prefix, meta, coord_filters).await
            }
            // No cached async store on the head (unexpected): fall back to
            // geometry partitioning rather than building a store here.
            None if is_remote_url(&self.path) => Ok(OuterSelection::Unfiltered),
            None => resolve_outer_selection(&self.path, meta, coord_filters),
        };

        let outer_specs = match outer {
            // Partition the SURVIVING set so a narrow filter still fans out.
            Ok(OuterSelection::Resolved(sel)) => split_selection(&sel, chunk_len, target_partitions)
                .into_iter()
                .map(PartitionSpec::from_outer)
                .collect(),
            // A present outer filter matched nothing: one empty partition.
            Ok(OuterSelection::Empty) => vec![PartitionSpec::range(0, 0)],
            // No outer filter: split the full axis geometrically (no coord read).
            Ok(OuterSelection::Unfiltered) => {
                plan_partitions(outer_len, chunk_len, target_partitions)
            }
            // A resolution read error shouldn't abort the query — fall back to
            // geometry partitioning (still correct, just not load-balanced).
            Err(e) => {
                debug!(error = %e, "outer-selection resolution failed; using geometry partitioning");
                plan_partitions(outer_len, chunk_len, target_partitions)
            }
        };

        // Multi-axis fan-out: if the outer axis alone under-parallelizes (fewer
        // partitions than `target`), split the best *inner* axis too so a scan
        // whose outer axis is coarsely chunked still uses the whole machine.
        fan_out_inner(outer_specs, meta, data_var, coord_filters, target_partitions)
    }
}

/// Map data-variable axis `axis` to its coordinate index in `meta.coords`, using
/// the variable's `dimension_names`. Returns `None` when names are unavailable —
/// inner fan-out then declines (a size-match fallback would be ambiguous when two
/// axes share a length).
fn coord_index_for_axis(
    data_var: &ZarrArrayMeta,
    axis: usize,
    meta: &ZarrStoreMeta,
) -> Option<usize> {
    let name = data_var.dimensions.as_ref()?.get(axis)?;
    meta.coords.iter().position(|c| &c.name == name)
}

/// Fan a scan out across an inner axis when the outer axis alone yields fewer
/// partitions than `target`.
///
/// `outer_specs` are the outer-only partitions already planned. If they under-fill
/// the parallelism target, this splits one or more *unfiltered*, multiply-chunked
/// inner axes geometrically and returns the Cartesian product
/// (`outer × inner₁ × inner₂ × …`) as N-D box partitions — never exceeding
/// `target`, never splitting a chunk across partitions. Inner axes are split most-
/// chunked first, greedily, until the inner budget (`target / p_outer`) is used up,
/// so a cube whose axes are individually too coarse can still fill the machine.
/// Metadata-only: no coordinate reads. Skips filtered axes (their surviving set
/// isn't resolved here, and `extra` would overwrite the filter selection — see H2).
/// Declines (returns `outer_specs` unchanged) for a limited/empty plan, when already
/// at target, when dimension names are missing, or when no inner axis offers extra
/// parallelism.
fn fan_out_inner(
    outer_specs: Vec<PartitionSpec>,
    meta: &ZarrStoreMeta,
    data_var: &ZarrArrayMeta,
    coord_filters: Option<&CoordFilters>,
    target: usize,
) -> Vec<PartitionSpec> {
    let p_outer = outer_specs.len();
    // Already at/over target, or a single empty partition (outer filter matched
    // nothing) — nothing to gain from inner splitting.
    if p_outer >= target || (p_outer == 1 && outer_specs[0].is_empty()) {
        return outer_specs;
    }
    let inner_budget = target / p_outer;
    if inner_budget <= 1 {
        return outer_specs;
    }

    // Gather candidate inner axes (any axis but 0) that are unfiltered and offer
    // chunk-level parallelism (>1 chunk). Skip filtered axes: their surviving set
    // isn't resolved here, and `extra` would overwrite the filter selection.
    let mut candidates: Vec<(usize, usize, u64, u64)> = Vec::new(); // (coord_idx, n_chunks, extent, chunk_len)
    for axis in 1..data_var.shape.len() {
        let Some(coord_idx) = coord_index_for_axis(data_var, axis, meta) else {
            continue;
        };
        let coord_name = &meta.coords[coord_idx].name;
        if coord_filters.is_some_and(|f| f.get(coord_name).is_some()) {
            continue;
        }
        let extent = data_var.shape[axis];
        let chunk_len = data_var
            .chunks
            .as_ref()
            .and_then(|c| c.get(axis).copied())
            .unwrap_or(extent)
            .max(1);
        let n_chunks = extent.div_ceil(chunk_len) as usize;
        if n_chunks > 1 {
            candidates.push((coord_idx, n_chunks, extent, chunk_len));
        }
    }
    // Most-chunked first (deterministic tie-break by coord index).
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    // Greedily split axes until the inner budget is spent. `acc` is the running
    // product of the splits chosen so far; `allowed` is how many more factors we
    // may still multiply in without exceeding the budget.
    let mut split_axes: Vec<(usize, Vec<CoordSelection>)> = Vec::new();
    let mut acc = 1usize;
    for (coord_idx, n_chunks, extent, chunk_len) in candidates {
        let allowed = inner_budget / acc;
        if allowed <= 1 {
            break;
        }
        let k = n_chunks.min(allowed);
        let pieces: Vec<CoordSelection> = plan_partitions(extent, chunk_len, k)
            .into_iter()
            .map(|p| p.outer)
            .collect();
        if pieces.len() <= 1 {
            continue;
        }
        acc *= pieces.len();
        split_axes.push((coord_idx, pieces));
    }
    if split_axes.is_empty() {
        return outer_specs;
    }

    // Cartesian product of the per-axis piece lists -> one `extra` list per inner
    // box (each entry a distinct coordinate's chunk-aligned slice).
    let mut inner_combos: Vec<Vec<(usize, CoordSelection)>> = vec![Vec::new()];
    for (coord_idx, pieces) in &split_axes {
        let mut next = Vec::with_capacity(inner_combos.len() * pieces.len());
        for combo in &inner_combos {
            for piece in pieces {
                let mut c = combo.clone();
                c.push((*coord_idx, piece.clone()));
                next.push(c);
            }
        }
        inner_combos = next;
    }

    // Each outer slice × each inner box.
    let mut boxes = Vec::with_capacity(p_outer * inner_combos.len());
    for outer in &outer_specs {
        for combo in &inner_combos {
            boxes.push(PartitionSpec::from_outer(outer.outer.clone()).with_extra(combo.clone()));
        }
    }
    boxes
}

#[async_trait]
impl TableProvider for ZarrTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> datafusion::datasource::TableType {
        datafusion::datasource::TableType::Base
    }

    /// Indicate which filters can be pushed down to the scan
    ///
    /// Returns `Inexact` for all filters - we'll handle coordinate equality
    /// filters during scan, but DataFusion should still apply filters post-scan
    /// for correctness (in case we miss any).
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|f| {
                // DatePart filters are handled exactly by ZarrExec (via index gather),
                // so tell DataFusion to drop the FilterExec — otherwise it re-evaluates
                // date_part() on Dictionary-encoded columns and hits a type error.
                if is_date_part_filter(f) {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Inexact
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion::logical_expr::Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Log projection pushdown
        let total_columns = self.schema.fields().len();
        if let Some(indices) = projection {
            let projected_names: Vec<_> = indices
                .iter()
                .map(|&i| self.schema.field(i).name().as_str())
                .collect();
            info!(
                projected = indices.len(),
                total = total_columns,
                columns = ?projected_names,
                "Projection pushdown"
            );
        } else {
            info!(
                projected = total_columns,
                total = total_columns,
                "No projection pushdown (all columns)"
            );
        }

        // Log limit pushdown
        if let Some(limit) = limit {
            info!(limit, "Limit pushdown");
        }

        // Parse coordinate filters for filter pushdown
        debug!(
            num_filters = filters.len(),
            filters = ?filters,
            "Filters passed to scan()"
        );
        let coord_filters = if let Some(meta) = &self.store_meta {
            let coord_names: Vec<String> = meta.coords.iter().map(|c| c.name.clone()).collect();
            debug!(?coord_names, "Coordinate names from metadata");
            let parsed = parse_coord_filters(filters, &coord_names);
            if !parsed.is_empty() {
                info!(
                    num_filters = parsed.len(),
                    coords = ?parsed.filters.keys().collect::<Vec<_>>(),
                    "Filter pushdown"
                );
                Some(parsed)
            } else {
                None
            }
        } else {
            // No metadata available - can't do filter pushdown
            None
        };

        // ── Plan output partitions ────────────────────────────────────────
        // Resolve any outer-axis filter to the surviving set and split THAT
        // across partitions (local + remote); VirtualiZarr stays single-partition.
        let partitions = self
            .plan_scan_partitions(state, limit, coord_filters.as_ref())
            .await;

        let exec = ZarrExec::new(
            self.schema.clone(),
            self.path.clone(),
            projection.cloned(),
            limit,
            self.cached_remote.clone(),
            coord_filters,
            self.cached_virtualizarr.clone(),
        );

        // FILL 6d: attach the planned slices (builder consumes & returns exec).
        let exec = exec
            .with_partitions(partitions)
            .with_store_meta(self.store_meta.clone())
            .with_store_kind(self.store_kind.clone());

        Ok(Arc::new(exec))
    }

    /// Return statistics for this table
    ///
    /// This enables DataFusion's optimizer to convert count(*) and count(column)
    /// queries into constant values without scanning the data.
    ///
    /// For coordinate columns, we also provide:
    /// - min_value/max_value: Enables MIN(coord)/MAX(coord) optimization
    /// - distinct_count: Number of unique coordinate values
    fn statistics(&self) -> Option<Statistics> {
        let meta = self.store_meta.as_ref()?;

        // Build column statistics
        let column_statistics: Vec<ColumnStatistics> = self
            .schema
            .fields()
            .iter()
            .map(|field| {
                let field_name = field.name();

                // Check if this is a coordinate column with min/max
                if let Some(coord) = meta.coords.iter().find(|c| &c.name == field_name) {
                    if let Some((min, max)) = coord.coord_min_max {
                        // Coordinates have distinct_count = shape[0] (number of unique values)
                        let distinct_count = coord.shape[0] as usize;

                        // Convert min/max to ScalarValue based on the underlying type
                        // Dictionary types have a value type inside
                        let (min_value, max_value) = match field.data_type() {
                            arrow::datatypes::DataType::Dictionary(_, value_type) => {
                                scalar_values_from_f64(min, max, value_type.as_ref())
                            }
                            dt => scalar_values_from_f64(min, max, dt),
                        };

                        info!(
                            coord = %field_name,
                            min = %min_value,
                            max = %max_value,
                            distinct = distinct_count,
                            "Coordinate statistics"
                        );

                        return ColumnStatistics {
                            null_count: Precision::Exact(0),
                            min_value: Precision::Exact(min_value),
                            max_value: Precision::Exact(max_value),
                            distinct_count: Precision::Exact(distinct_count),
                            ..Default::default()
                        };
                    }
                }

                // Default: only null_count for data variables
                ColumnStatistics {
                    null_count: Precision::Exact(0),
                    ..Default::default()
                }
            })
            .collect();

        info!(
            total_rows = meta.total_rows,
            num_columns = column_statistics.len(),
            "Providing statistics for query optimization"
        );

        Some(Statistics {
            num_rows: Precision::Exact(meta.total_rows),
            total_byte_size: Precision::Absent,
            column_statistics,
        })
    }
}

/// Convert f64 min/max values to appropriate ScalarValue based on Arrow data type
fn scalar_values_from_f64(
    min: f64,
    max: f64,
    data_type: &arrow::datatypes::DataType,
) -> (
    datafusion::common::ScalarValue,
    datafusion::common::ScalarValue,
) {
    use arrow::datatypes::DataType;
    use datafusion::common::ScalarValue;

    match data_type {
        DataType::Float64 => (
            ScalarValue::Float64(Some(min)),
            ScalarValue::Float64(Some(max)),
        ),
        DataType::Float32 => (
            ScalarValue::Float32(Some(min as f32)),
            ScalarValue::Float32(Some(max as f32)),
        ),
        DataType::Int64 => (
            ScalarValue::Int64(Some(min as i64)),
            ScalarValue::Int64(Some(max as i64)),
        ),
        DataType::Int32 => (
            ScalarValue::Int32(Some(min as i32)),
            ScalarValue::Int32(Some(max as i32)),
        ),
        DataType::Int16 => (
            ScalarValue::Int16(Some(min as i16)),
            ScalarValue::Int16(Some(max as i16)),
        ),
        DataType::UInt64 => (
            ScalarValue::UInt64(Some(min as u64)),
            ScalarValue::UInt64(Some(max as u64)),
        ),
        DataType::UInt32 => (
            ScalarValue::UInt32(Some(min as u32)),
            ScalarValue::UInt32(Some(max as u32)),
        ),
        // Fallback to Float64
        _ => (
            ScalarValue::Float64(Some(min)),
            ScalarValue::Float64(Some(max)),
        ),
    }
}

#[cfg(test)]
mod fanout_tests {
    use super::*;
    use crate::reader::filter::CoordSelection;

    fn coord(name: &str, size: u64) -> ZarrArrayMeta {
        ZarrArrayMeta {
            name: name.into(),
            data_type: "int64".into(),
            shape: vec![size],
            chunks: Some(vec![size]),
            coord_min_max: None,
            cf_time_attrs: None,
            dimensions: Some(vec![name.into()]),
        }
    }

    fn data_var(dims: &[&str], shape: &[u64], chunks: &[u64]) -> ZarrArrayMeta {
        ZarrArrayMeta {
            name: "t".into(),
            data_type: "float32".into(),
            shape: shape.to_vec(),
            chunks: Some(chunks.to_vec()),
            coord_min_max: None,
            cf_time_attrs: None,
            dimensions: Some(dims.iter().map(|s| s.to_string()).collect()),
        }
    }

    fn meta(coords: Vec<ZarrArrayMeta>, dv: &ZarrArrayMeta) -> ZarrStoreMeta {
        ZarrStoreMeta {
            total_rows: 0,
            coords,
            data_vars: vec![dv.clone()],
        }
    }

    /// Three outer (time) partitions, as the outer split would produce.
    fn outer3() -> Vec<PartitionSpec> {
        vec![
            PartitionSpec::range(0, 1),
            PartitionSpec::range(1, 2),
            PartitionSpec::range(2, 3),
        ]
    }

    #[test]
    fn fans_out_across_best_inner_axis() {
        // temperature[time=3, lat=721, lon=1440], lat has 4 chunks, lon 2.
        let dv = data_var(&["time", "lat", "lon"], &[3, 721, 1440], &[1, 181, 720]);
        let m = meta(vec![coord("time", 3), coord("lat", 721), coord("lon", 1440)], &dv);
        // target 8, p_outer 3 => inner_budget 2 => split lat (4 chunks) into 2.
        let out = fan_out_inner(outer3(), &m, &dv, None, 8);
        assert_eq!(out.len(), 6, "3 outer × 2 inner");
        // Every box restricts lat (coord index 1) and preserves its outer slice.
        for b in &out {
            assert_eq!(b.extra.len(), 1);
            assert_eq!(b.extra[0].0, 1, "inner axis is lat (coord idx 1)");
        }
        // Union of inner pieces covers [0, 721).
        let lat_ranges: Vec<_> = out[..2]
            .iter()
            .map(|b| b.extra[0].1.as_range().unwrap())
            .collect();
        assert_eq!(lat_ranges[0].0, 0);
        assert_eq!(lat_ranges.last().unwrap().1, 721);
    }

    #[test]
    fn never_exceeds_target() {
        let dv = data_var(&["time", "lat", "lon"], &[3, 721, 1440], &[1, 181, 720]);
        let m = meta(vec![coord("time", 3), coord("lat", 721), coord("lon", 1440)], &dv);
        for target in 4..=32 {
            let out = fan_out_inner(outer3(), &m, &dv, None, target);
            assert!(out.len() <= target, "target {target} => {} boxes", out.len());
        }
    }

    #[test]
    fn declines_when_outer_already_at_target() {
        let dv = data_var(&["time", "lat", "lon"], &[3, 721, 1440], &[1, 181, 720]);
        let m = meta(vec![coord("time", 3), coord("lat", 721), coord("lon", 1440)], &dv);
        // target 3 == p_outer 3 => no fan-out.
        let out = fan_out_inner(outer3(), &m, &dv, None, 3);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|b| b.extra.is_empty()));
    }

    #[test]
    fn declines_when_inner_axes_single_chunked() {
        // Inner lat/lon each one chunk (like synthetic) => nothing to split.
        let dv = data_var(&["time", "lat", "lon"], &[3, 10, 10], &[1, 10, 10]);
        let m = meta(vec![coord("time", 3), coord("lat", 10), coord("lon", 10)], &dv);
        let out = fan_out_inner(outer3(), &m, &dv, None, 16);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|b| b.extra.is_empty()));
    }

    #[test]
    fn single_outer_chunk_parallelizes_on_inner() {
        // The headline case: outer axis is one chunk -> without fan-out this is a
        // single partition; with it, we fan out across the inner axes. Budget 8,
        // lat has 4 chunks and lon has 2, so N-D fan-out uses both: 4 × 2 = 8.
        let dv = data_var(&["time", "lat", "lon"], &[1, 721, 1440], &[1, 181, 720]);
        let m = meta(vec![coord("time", 1), coord("lat", 721), coord("lon", 1440)], &dv);
        let outer1 = vec![PartitionSpec::range(0, 1)];
        let out = fan_out_inner(outer1, &m, &dv, None, 8);
        assert_eq!(out.len(), 8, "1 outer × (4 lat × 2 lon)");
        assert!(out.iter().all(|b| b.extra.len() == 2), "boxes span 2 inner axes");
    }

    #[test]
    fn nd_fan_out_splits_multiple_inner_axes_most_chunked_first() {
        // Two splittable inner axes; budget lets us use both. lat (4 chunks) is
        // split before lon (2) — most-chunked first — and the product caps at budget.
        let dv = data_var(&["time", "lat", "lon"], &[2, 721, 1440], &[1, 181, 720]);
        let m = meta(vec![coord("time", 2), coord("lat", 721), coord("lon", 1440)], &dv);
        // 2 outer × budget 8 (target 16 / p_outer 2 = 8) => lat 4 × lon 2 = 8 inner.
        let outer = vec![PartitionSpec::range(0, 1), PartitionSpec::range(1, 2)];
        let out = fan_out_inner(outer, &m, &dv, None, 16);
        assert_eq!(out.len(), 16, "2 outer × 4 lat × 2 lon");
        // Every box restricts both inner axes; coord indices are lat(1) and lon(2).
        for b in &out {
            assert_eq!(b.extra.len(), 2);
            let idxs: Vec<usize> = b.extra.iter().map(|(i, _)| *i).collect();
            assert_eq!(idxs, vec![1, 2], "lat then lon (most-chunked first)");
        }
    }

    #[test]
    fn nd_fan_out_stops_at_budget() {
        // Budget only affords one extra factor beyond the first axis: lat (4) alone
        // fills budget 4, so lon is NOT split.
        let dv = data_var(&["time", "lat", "lon"], &[1, 721, 1440], &[1, 181, 720]);
        let m = meta(vec![coord("time", 1), coord("lat", 721), coord("lon", 1440)], &dv);
        let out = fan_out_inner(vec![PartitionSpec::range(0, 1)], &m, &dv, None, 4);
        assert_eq!(out.len(), 4, "1 outer × 4 lat; budget 4 spent on lat alone");
        assert!(out.iter().all(|b| b.extra.len() == 1));
    }

    #[test]
    fn declines_for_empty_partition() {
        let dv = data_var(&["time", "lat", "lon"], &[3, 721, 1440], &[1, 181, 720]);
        let m = meta(vec![coord("time", 3), coord("lat", 721), coord("lon", 1440)], &dv);
        let empty = vec![PartitionSpec::range(0, 0)];
        let out = fan_out_inner(empty, &m, &dv, None, 8);
        assert_eq!(out.len(), 1);
        assert!(out[0].is_empty());
    }

    #[test]
    fn declines_without_dimension_names() {
        // No dimension_names => can't map axes unambiguously => no fan-out.
        let mut dv = data_var(&["time", "lat", "lon"], &[3, 721, 1440], &[1, 181, 720]);
        dv.dimensions = None;
        let m = meta(vec![coord("time", 3), coord("lat", 721), coord("lon", 1440)], &dv);
        let out = fan_out_inner(outer3(), &m, &dv, None, 8);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|b| b.extra.is_empty()));
    }

    #[test]
    fn skips_filtered_inner_axis() {
        // lat (best, 4 chunks) is filtered => fall back to lon (2 chunks).
        let dv = data_var(&["time", "lat", "lon"], &[3, 721, 1440], &[1, 181, 720]);
        let m = meta(vec![coord("time", 3), coord("lat", 721), coord("lon", 1440)], &dv);
        let mut filters = CoordFilters::default();
        filters
            .filters
            .insert("lat".into(), vec![crate::reader::filter::CoordFilterKind::Eq(
                datafusion::scalar::ScalarValue::Float64(Some(1.0)),
            )]);
        let out = fan_out_inner(outer3(), &m, &dv, Some(&filters), 8);
        // lon has 2 chunks, budget 2 => split lon into 2 => 3×2 = 6 boxes on lon (idx 2).
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|b| b.extra[0].0 == 2), "fell back to lon (idx 2)");
        let _ = CoordSelection::Range(0, 0); // keep import used
    }
}
