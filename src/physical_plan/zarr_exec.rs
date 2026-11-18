use crate::physical_plan::partition::PartitionSpec;
use crate::reader::filter::{CoordFilters, CoordSelection};
use crate::reader::scan_cache::ScanCache;
use crate::reader::schema_inference::ZarrStoreMeta;
use crate::reader::stats::{SharedIoStats, ZarrIoStats};
use crate::reader::storage::is_remote_url;
use crate::reader::virtual_store::{is_virtualizarr_store, VirtualStoreAdapter};
use crate::reader::zarr_reader::{read_zarr, read_zarr_async};
use arrow::datatypes::{Schema, SchemaRef};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::{
    common::DataFusionError,
    physical_expr::EquivalenceProperties,
    physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

/// Cached remote store info (store, prefix, metadata)
pub type CachedRemoteStore = Option<(AsyncReadableListableStorage, ObjectPath, ZarrStoreMeta)>;

/// Cached VirtualiZarr adapter (pre-loaded refs and metadata)
pub type CachedVirtualiZarrAdapter = Option<Arc<VirtualStoreAdapter>>;

/// How to reopen a scan's store from its path when no cached store is attached,
/// as on a worker that decoded the scan.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StoreKind {
    /// Plain Zarr, or a VirtualiZarr store, told apart by its path.
    #[default]
    Zarr,
    /// An icechunk repository, read at `group` (the root when `None`).
    Icechunk {
        group: Option<String>,
        anonymous: bool,
    },
}

#[derive(Clone)]
pub struct ZarrExec {
    schema: SchemaRef,
    path: String,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    properties: Arc<PlanProperties>,
    io_stats: SharedIoStats,
    /// Cached remote store and metadata (avoids recreating on each query)
    cached_remote: CachedRemoteStore,
    /// Coordinate filters for filter pushdown (e.g., time = X, hybrid = Y)
    coord_filters: Option<CoordFilters>,
    /// Cached VirtualiZarr adapter for remote VirtualiZarr stores
    cached_virtualizarr: CachedVirtualiZarrAdapter,
    /// Outer-dimension slices, one per output partition.
    /// EMPTY means "single unpartitioned read" (legacy behavior) — that keeps
    /// every existing `ZarrExec::new` caller (codec, tests) working unchanged.
    /// Populated by `with_partitions` from the planner in `ZarrTable::scan`.
    partitions: Vec<PartitionSpec>,
    /// Memory budget in bytes, set by the cardinality optimizer rule (Phase 5).
    /// `None` => the reader falls back to `ZARR_MEM_BUDGET_BYTES`, then `batch_size`
    /// — i.e. unchanged from Phase 4. The reader converts bytes -> row target via
    /// its own projected `row_width`.
    stream_budget_bytes: Option<u64>,
    /// Store metadata (coordinates in cube-axis order + data variables), attached
    /// so plan-level rules can reason about the cube: classify columns as
    /// coordinate vs. data variable, map a coordinate to its axis, and read axis
    /// extents. Used by the aggregate-pushdown recognizer (Phase 7). `None` for
    /// callers that don't supply it (codec/tests) — recognition then declines.
    store_meta: Option<ZarrStoreMeta>,
    store_kind: StoreKind,
    /// Shared by this scan's partitions; a new scan starts with an empty one.
    scan_cache: Arc<ScanCache>,
}

impl std::fmt::Debug for ZarrExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrExec")
            .field("schema", &self.schema)
            .field("path", &self.path)
            .field("projection", &self.projection)
            .field("limit", &self.limit)
            .field("has_cached_remote", &self.cached_remote.is_some())
            .field(
                "has_cached_virtualizarr",
                &self.cached_virtualizarr.is_some(),
            )
            .field(
                "coord_filters",
                &self
                    .coord_filters
                    .as_ref()
                    .map(|f| f.filters.keys().collect::<Vec<_>>()),
            )
            .finish()
    }
}

impl DisplayAs for ZarrExec {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let mut parts = vec![format!("path={}", self.path)];
        if let Some(limit) = self.limit {
            parts.push(format!("limit={}", limit));
        }
        if let Some(ref filters) = self.coord_filters {
            if !filters.is_empty() {
                let filter_strs: Vec<_> = filters
                    .filters
                    .iter()
                    .map(|(k, v)| {
                        let joined: Vec<String> = v.iter().map(|f| f.to_string()).collect();
                        format!("{}{}", k, joined.join(" AND "))
                    })
                    .collect();
                parts.push(format!("filters=[{}]", filter_strs.join(", ")));
            }
        }
        write!(f, "ZarrExec: {}", parts.join(", "))
    }
}

impl ZarrExec {
    pub fn new(
        schema: SchemaRef,
        path: String,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        cached_remote: CachedRemoteStore,
        coord_filters: Option<CoordFilters>,
        cached_virtualizarr: CachedVirtualiZarrAdapter,
    ) -> Self {
        // Start with a single, unpartitioned read (empty `partitions`).
        let properties = Self::build_properties(&schema, projection.as_ref(), 1);
        Self {
            schema,
            path,
            projection,
            limit,
            properties,
            io_stats: Arc::new(ZarrIoStats::new()),
            cached_remote,
            coord_filters,
            cached_virtualizarr,
            partitions: Vec::new(),
            stream_budget_bytes: None,
            store_meta: None,
            store_kind: StoreKind::Zarr,
            scan_cache: Arc::default(),
        }
    }

    /// Build the `PlanProperties` for a given output partition count.
    ///
    /// Shared by `new` (count = 1) and `with_partitions` (count = number of
    /// slices). `&SchemaRef` / `Option<&Vec<usize>>` are *borrows* — this only
    /// reads them to derive the projected schema, it doesn't take ownership.
    fn build_properties(
        schema: &SchemaRef,
        projection: Option<&Vec<usize>>,
        num_partitions: usize,
    ) -> Arc<PlanProperties> {
        // Compute projected schema for plan properties
        let projected_schema = if let Some(indices) = projection {
            Arc::new(Schema::new(
                indices
                    .iter()
                    .map(|&i| schema.field(i).clone())
                    .collect::<Vec<_>>(),
            ))
        } else {
            schema.clone()
        };

        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(projected_schema),
            Partitioning::UnknownPartitioning(num_partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }

    /// Attach the planner's partition slices and re-derive `PlanProperties` so
    /// the plan advertises `partitions.len()` output partitions.
    ///
    /// Takes `self` by value (`mut self`) and returns it — the builder idiom:
    /// `ZarrExec::new(..).with_partitions(specs)`.
    pub fn with_partitions(mut self, partitions: Vec<PartitionSpec>) -> Self {
        // ▢ FILL 2: how many output partitions should we advertise?
        // An empty `partitions` must still mean ONE partition (legacy single
        // read), so floor the count at 1. Hint: `.len().max(1)`.
        let num_partitions = partitions.len().max(1);
        self.properties =
            Self::build_properties(&self.schema, self.projection.as_ref(), num_partitions);
        self.partitions = partitions;
        self
    }

    /// Get I/O statistics collected during execution
    pub fn io_stats(&self) -> SharedIoStats {
        self.io_stats.clone()
    }

    /// Get the current limit
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Get the full (unprojected) schema
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Get the store path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get the projection indices, if any
    pub fn projection(&self) -> Option<&Vec<usize>> {
        self.projection.as_ref()
    }

    /// Get the coordinate filters
    pub fn coord_filters(&self) -> Option<&CoordFilters> {
        self.coord_filters.as_ref()
    }

    /// Set the streaming memory budget in bytes (from the cardinality optimizer
    /// rule). Builder idiom: `ZarrExec::new(..).with_stream_budget_bytes(Some(b))`.
    pub fn with_stream_budget_bytes(mut self, bytes: Option<u64>) -> Self {
        self.stream_budget_bytes = bytes;
        self
    }

    /// The rule-set streaming memory budget in bytes, if any.
    pub fn stream_budget_bytes(&self) -> Option<u64> {
        self.stream_budget_bytes
    }

    /// Attach store metadata (coords in cube-axis order + data vars). Builder idiom.
    pub fn with_store_meta(mut self, meta: Option<ZarrStoreMeta>) -> Self {
        self.store_meta = meta;
        self
    }

    /// Store metadata, if attached — coordinates (cube-axis order) and data variables.
    pub fn with_store_kind(mut self, kind: StoreKind) -> Self {
        self.store_kind = kind;
        self
    }

    pub fn store_kind(&self) -> &StoreKind {
        &self.store_kind
    }

    pub fn store_meta(&self) -> Option<&ZarrStoreMeta> {
        self.store_meta.as_ref()
    }

    /// The scan's store, opened for async reads, and the prefix its arrays are under.
    pub async fn open_store(
        &self,
    ) -> datafusion::error::Result<(AsyncReadableListableStorage, ObjectPath)> {
        if let Some((store, prefix, _)) = &self.cached_remote {
            return Ok((store.clone(), prefix.clone()));
        }
        if self.store_kind == StoreKind::Zarr && !is_remote_url(&self.path) {
            use zarrs_object_store::object_store::local::LocalFileSystem;
            use zarrs_object_store::AsyncObjectStore;
            let fs = LocalFileSystem::new_with_prefix(&self.path)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            return Ok((Arc::new(AsyncObjectStore::new(fs)), ObjectPath::default()));
        }
        open_uncached(&self.path, self.store_kind.clone()).await
    }

    /// Get this scan's partition slices (empty == single unpartitioned read).
    /// Used by the distributed codec (to ship them) and the `TaskEstimator`
    /// (to split them across worker tasks).
    pub fn partitions(&self) -> &[PartitionSpec] {
        &self.partitions
    }

    /// Create a new ZarrExec with the given limit
    pub fn with_limit(&self, limit: Option<usize>) -> Self {
        Self::new(
            self.schema.clone(),
            self.path.clone(),
            self.projection.clone(),
            limit,
            self.cached_remote.clone(),
            self.coord_filters.clone(),
            self.cached_virtualizarr.clone(),
        )
        // `new` resets partitions to empty; re-attach ours so a limited copy
        // keeps the same output partitioning. `.clone()` because `self` is
        // borrowed (`&self`) and we can't move the Vec out of it.
        .with_partitions(self.partitions.clone())
        .with_stream_budget_bytes(self.stream_budget_bytes)
        .with_store_meta(self.store_meta.clone())
        .with_store_kind(self.store_kind.clone())
    }
}
impl ExecutionPlan for ZarrExec {
    fn name(&self) -> &str {
        "ZarrExec"
    }

    fn children(&self) -> Vec<&std::sync::Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn properties(&self) -> &Arc<datafusion::physical_plan::PlanProperties> {
        &self.properties
    }

    fn with_new_children(
        self: std::sync::Arc<Self>,
        _children: Vec<std::sync::Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<std::sync::Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: std::sync::Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
        // DataFusion decides the target rows per emitted RecordBatch via session
        // config; we honor it as the *output* batch target. It is deliberately not
        // the read window — see `stream_budget_bytes` below.
        let batch_size = context.session_config().batch_size();

        // Resolve this partition's share of the scan memory ceiling. Every partition
        // streams its own window concurrently, so the scan-wide ceiling has to be
        // divided by the partition count or an N-core box holds N times the intended
        // memory. Done here rather than in the reader because the partition count is
        // a session/planner fact, not something the reader can see.
        let n_partitions = self.partitions.len().max(1);
        let stream_budget_bytes =
            Some(crate::optimizer::cardinality::budget::resolve_scan_ceiling(
                self.stream_budget_bytes,
                n_partitions,
            ));

        // Look up this partition's outer-axis selection. Empty `partitions` =>
        // legacy whole-store read (`None`). The selection is intersected with the
        // filter-derived ranges inside the reader.
        let partition_selection: Option<CoordSelection> = if self.partitions.is_empty() {
            None
        } else {
            Some(self.partitions[partition].outer.clone())
        };
        // Inner-axis restrictions for a multi-axis (N-D box) partition; empty for
        // an outer-only partition (single-axis, the common case).
        let partition_extra: Vec<(usize, CoordSelection)> = if self.partitions.is_empty() {
            Vec::new()
        } else {
            self.partitions[partition].extra.clone()
        };

        info!(
            path = %self.path,
            partition,
            partition_selection = ?partition_selection,
            limit = ?self.limit,
            projection = ?self.projection,
            has_cached_remote = self.cached_remote.is_some(),
            has_cached_virtualizarr = self.cached_virtualizarr.is_some(),
            has_coord_filters = self.coord_filters.is_some(),
            "ZarrExec::execute called"
        );

        // Check for cached VirtualiZarr adapter first (remote VirtualiZarr)
        if let Some(adapter) = &self.cached_virtualizarr {
            info!("Using cached VirtualiZarr adapter (remote) execution path");
            return execute_virtualizarr_with_adapter(
                adapter.clone(),
                self.schema.clone(),
                self.projection.clone(),
                self.limit,
                self.io_stats.clone(),
                self.coord_filters.clone(),
                batch_size,
                stream_budget_bytes,
                self.scan_cache.clone(),
            );
        }

        if is_virtualizarr_store(&self.path) {
            info!("Using VirtualiZarr (local) execution path");
            execute_virtualizarr(
                self.path.clone(),
                self.schema.clone(),
                self.projection.clone(),
                self.limit,
                self.io_stats.clone(),
                self.coord_filters.clone(),
                batch_size,
                stream_budget_bytes,
                self.scan_cache.clone(),
            )
        } else if is_remote_url(&self.path)
            || self.cached_remote.is_some()
            || self.store_kind != StoreKind::Zarr
        {
            // A local icechunk store is read through the same async path.
            info!("Using remote (async) execution path");
            execute_remote(
                self.path.clone(),
                self.store_kind.clone(),
                self.schema.clone(),
                self.projection.clone(),
                self.limit,
                self.io_stats.clone(),
                self.cached_remote.clone(),
                self.coord_filters.clone(),
                partition_selection,
                partition_extra,
                batch_size,
                stream_budget_bytes,
                self.scan_cache.clone(),
            )
        } else {
            info!("Using local (sync) execution path");
            read_zarr(
                &self.path,
                self.schema.clone(),
                self.projection.clone(),
                self.limit,
                Some(self.io_stats.clone()),
                self.coord_filters.clone(),
                partition_selection,
                partition_extra,
                batch_size,
                stream_budget_bytes,
            )
        }
    }
}

// =============================================================================
// Helper functions to reduce duplication across execute_* functions
// =============================================================================

/// Build projected schema from full schema and projection indices.
///
/// This consolidates the identical schema projection logic used across all execute functions.
fn build_exec_projected_schema(schema: &SchemaRef, projection: Option<&Vec<usize>>) -> SchemaRef {
    if let Some(indices) = projection {
        Arc::new(Schema::new(
            indices
                .iter()
                .map(|&i| schema.field(i).clone())
                .collect::<Vec<_>>(),
        ))
    } else {
        schema.clone()
    }
}

/// Parameters for async read execution.
///
/// Encapsulates the common parameters passed to all async execute functions.
struct AsyncReadParams {
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: SharedIoStats,
    coord_filters: Option<CoordFilters>,
    /// Outer-axis selection for this partition; `None` => whole store.
    partition_selection: Option<CoordSelection>,
    /// Inner-axis `(coord index, selection)` restrictions for a multi-axis (N-D
    /// box) partition; empty for an outer-only partition.
    partition_extra: Vec<(usize, CoordSelection)>,
    /// Target rows per emitted RecordBatch (DataFusion's batch_size).
    batch_size: usize,
    /// Rule-set streaming row target (Phase 5); `None` => env/batch_size fallback.
    stream_budget_bytes: Option<u64>,
    cache: Arc<ScanCache>,
}

/// Execute an async read with the given store setup function.
///
/// This consolidates the common stream creation pattern used across all async execute functions:
/// 1. Create projected schema
/// 2. Create async stream that sets up store and calls read_zarr_async
/// 3. Collect batches and wrap in RecordBatchStreamAdapter
fn execute_async_read<F, Fut>(
    params: AsyncReadParams,
    store_setup: F,
    debug_name: &'static str,
) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<
            Output = datafusion::error::Result<(
                AsyncReadableListableStorage,
                ObjectPath,
                Option<ZarrStoreMeta>,
            )>,
        > + Send,
{
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream::{self, TryStreamExt};

    let projected_schema = build_exec_projected_schema(&params.schema, params.projection.as_ref());

    let stream = stream::once(async move {
        debug!("{} stream polled - starting async execution", debug_name);

        // Set up the store using the provided function
        let (store, prefix, cached_meta) = store_setup().await?;

        // Read the data
        debug!("Starting read_zarr_async for {}", debug_name);
        let result_stream = read_zarr_async(
            store,
            &prefix,
            params.schema,
            params.projection,
            params.limit,
            Some(params.stats),
            cached_meta,
            params.coord_filters,
            params.partition_selection,
            params.partition_extra,
            params.batch_size,
            params.stream_budget_bytes,
            params.cache,
        )
        .await?;

        // Return the reader's stream directly so batches flow lazily — collecting
        // here would materialize the whole result and defeat streaming.
        debug!("{} reader stream ready", debug_name);
        Ok::<_, DataFusionError>(result_stream)
    })
    .try_flatten();

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

/// Execute read from remote object store
#[allow(clippy::too_many_arguments)]
fn execute_remote(
    path: String,
    store_kind: StoreKind,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: SharedIoStats,
    cached_remote: CachedRemoteStore,
    coord_filters: Option<CoordFilters>,
    partition_selection: Option<CoordSelection>,
    partition_extra: Vec<(usize, CoordSelection)>,
    batch_size: usize,
    stream_budget_bytes: Option<u64>,
    cache: Arc<ScanCache>,
) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
    use crate::reader::async_tracked_store::AsyncTrackedStore;

    debug!(path = %path, has_cached_remote = cached_remote.is_some(), "Setting up remote execution stream");

    // Byte counting is attached here, per execution, rather than where the store is
    // built. The remote store is created once at CREATE EXTERNAL TABLE time and cached
    // for the life of the table, but `stats` is per-query — baking a handle into the
    // cached store would make every query report the running total of the session.
    let io_stats = stats.clone();

    let params = AsyncReadParams {
        schema,
        projection,
        limit,
        stats,
        coord_filters,
        partition_selection,
        partition_extra,
        batch_size,
        stream_budget_bytes,
        cache,
    };

    execute_async_read(
        params,
        move || async move {
            let (store, prefix, meta) = if let Some((store, prefix, meta)) = cached_remote {
                info!("Using cached async store and metadata");
                (store, prefix, Some(meta))
            } else {
                debug!("Creating async store (no cache)");
                let (store, prefix) = open_uncached(&path, store_kind).await?;
                debug!(prefix = %prefix, "Async store created");
                (store, prefix, None)
            };

            // Wrap whichever store we ended up with, so cached and freshly-created
            // stores are both counted. Partitions each build their own wrapper but
            // share one `Arc<ZarrIoStats>`, so the atomic counters aggregate.
            let tracked: AsyncReadableListableStorage =
                Arc::new(AsyncTrackedStore::new(store, Some(io_stats)));
            Ok((tracked, prefix, meta))
        },
        "Remote",
    )
}

async fn open_uncached(
    path: &str,
    store_kind: StoreKind,
) -> datafusion::error::Result<(AsyncReadableListableStorage, ObjectPath)> {
    use crate::reader::storage::create_async_store;
    match store_kind {
        StoreKind::Zarr => create_async_store(path)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e))),
        StoreKind::Icechunk { group, anonymous } => Ok((
            open_icechunk(path, anonymous).await?,
            ObjectPath::from(group.unwrap_or_default().trim_matches('/')),
        )),
    }
}

#[cfg(feature = "icechunk")]
async fn open_icechunk(
    path: &str,
    anonymous: bool,
) -> datafusion::error::Result<AsyncReadableListableStorage> {
    use crate::reader::icechunk_store::{open_icechunk_async, open_icechunk_remote};
    if is_remote_url(path) {
        open_icechunk_remote(path, anonymous).await
    } else {
        open_icechunk_async(path).await
    }
    .map_err(DataFusionError::External)
}

#[cfg(not(feature = "icechunk"))]
async fn open_icechunk(
    path: &str,
    _anonymous: bool,
) -> datafusion::error::Result<AsyncReadableListableStorage> {
    Err(DataFusionError::Plan(format!(
        "{path} is an icechunk store, which needs the `icechunk` feature"
    )))
}

/// Execute read from VirtualiZarr Parquet reference store
#[allow(clippy::too_many_arguments)]
fn execute_virtualizarr(
    path: String,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: SharedIoStats,
    coord_filters: Option<CoordFilters>,
    batch_size: usize,
    stream_budget_bytes: Option<u64>,
    cache: Arc<ScanCache>,
) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
    use crate::reader::schema_inference::discover_arrays;

    debug!(path = %path, "Setting up VirtualiZarr execution stream");

    // Pre-discover metadata from local .zmetadata (avoids async discovery issues)
    let cached_meta = discover_arrays(&path).map_err(DataFusionError::External)?;

    let params = AsyncReadParams {
        schema,
        projection,
        limit,
        stats,
        coord_filters,
        // VirtualiZarr stays single-partition for now (scan() doesn't partition it).
        partition_selection: None,
        partition_extra: Vec::new(),
        batch_size,
        stream_budget_bytes,
        cache,
    };

    execute_async_read(
        params,
        move || async move {
            // Create VirtualStoreAdapter
            let adapter = VirtualStoreAdapter::new(&path).map_err(DataFusionError::External)?;
            let store: AsyncReadableListableStorage = Arc::new(adapter);
            let prefix = ObjectPath::from("");
            Ok((store, prefix, Some(cached_meta)))
        },
        "VirtualiZarr",
    )
}

/// Execute read from a cached VirtualiZarr adapter (for remote VirtualiZarr stores)
#[allow(clippy::too_many_arguments)]
fn execute_virtualizarr_with_adapter(
    adapter: Arc<VirtualStoreAdapter>,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: SharedIoStats,
    coord_filters: Option<CoordFilters>,
    batch_size: usize,
    stream_budget_bytes: Option<u64>,
    cache: Arc<ScanCache>,
) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
    debug!("Setting up remote VirtualiZarr execution with cached adapter");

    let params = AsyncReadParams {
        schema,
        projection,
        limit,
        stats,
        coord_filters,
        partition_selection: None,
        partition_extra: Vec::new(),
        batch_size,
        stream_budget_bytes,
        cache,
    };

    execute_async_read(
        params,
        move || async move {
            let store: AsyncReadableListableStorage = adapter;
            let prefix = ObjectPath::from("");
            Ok((store, prefix, None))
        },
        "Remote VirtualiZarr",
    )
}
