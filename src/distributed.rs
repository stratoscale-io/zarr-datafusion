//! datafusion-distributed integration: shared wiring for the `worker` and
//! `head` binaries.
//!
//! Both sides must agree on two things or queries fail at execution time:
//!   * the [`ZarrPhysicalCodec`], so a `ZarrExec` serialized on the head can be
//!     decoded on the worker, and
//!   * the metric UDFs, so an aggregate stage running on a worker can resolve
//!     `rmse`, `mae`, etc.
//!
//! [`configure_distributed_builder`] registers the codec on a
//! `SessionStateBuilder`. The metric UDFs are registered separately, on the
//! built `SessionContext`, via [`crate::udfs::register_metric_udfs`]: they must
//! be *appended* to the context's function registry, not set on the builder
//! with `with_aggregate_functions`/`with_scalar_functions`, which *replace* the
//! list and so drop DataFusion's built-in `avg`/`sum`/`count`.

use std::sync::Arc;

use datafusion::common::DataFusionError;
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_distributed::{
    DistributedExt, DistributedLeafExec, TaskEstimation, TaskEstimator, WorkerResolver,
};
use url::Url;

use crate::physical_plan::codec::ZarrPhysicalCodec;
use crate::physical_plan::partition::distribute_specs_across_tasks;
use crate::physical_plan::zarr_exec::ZarrExec;

/// A fixed list of worker URLs (e.g. parsed from `WORKER_URLS`).
///
/// Each entry is one worker: the planner caps tasks-per-stage at the number of
/// URLs, so never collapse the cluster behind a single load-balancer VIP — list
/// the individual worker addresses.
#[derive(Clone, Debug)]
pub struct StaticWorkerResolver {
    urls: Vec<Url>,
}

impl StaticWorkerResolver {
    pub fn new(urls: Vec<Url>) -> Self {
        Self { urls }
    }

    /// Parse a comma-separated list of worker URLs, e.g.
    /// `http://worker1:8080,http://worker2:8080`.
    pub fn from_csv(s: &str) -> Result<Self, url::ParseError> {
        s.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(Url::parse)
            .collect::<Result<Vec<_>, _>>()
            .map(Self::new)
    }

    pub fn urls(&self) -> &[Url] {
        &self.urls
    }
}

impl WorkerResolver for StaticWorkerResolver {
    fn get_urls(&self) -> Result<Vec<Url>, DataFusionError> {
        Ok(self.urls.clone())
    }
}

/// [`TaskEstimator`] that lets `datafusion-distributed` spread a partitioned
/// [`ZarrExec`] across the worker pool.
///
/// Without it, the planner assigns every leaf `Maximum(1)` task, so all of a
/// scan's partitions run on one worker. We advertise one task per partition
/// slice (the planner caps it at the worker count) and, in `scale_up_leaf_node`,
/// hand each task a disjoint subset of the slices.
#[derive(Debug, Default)]
pub struct ZarrTaskEstimator;

impl TaskEstimator for ZarrTaskEstimator {
    fn task_estimation(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        _cfg: &ConfigOptions,
    ) -> Option<TaskEstimation> {
        let exec = plan.downcast_ref::<ZarrExec>()?;
        // Distribute only when there are >= 2 slices to spread; otherwise let the
        // default `Maximum(1)` apply (no gain from a network boundary). `desired`
        // is a soft hint the planner caps at the available worker count.
        let n = exec.partitions().len();
        (n >= 2).then(|| TaskEstimation::desired(n))
    }

    fn scale_up_leaf_node(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        task_count: usize,
        _cfg: &ConfigOptions,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(exec) = plan.downcast_ref::<ZarrExec>() else {
            return Ok(None);
        };
        let specs = exec.partitions();
        if specs.len() < 2 || task_count <= 1 {
            return Ok(None); // nothing to distribute
        }

        // One per-task ZarrExec variant, each holding a disjoint (padded) subset
        // of the slices. Caches are dropped (`None`): the codec drops them on the
        // wire anyway and each worker rebuilds the store from `path`.
        let variants = distribute_specs_across_tasks(specs, task_count)
            .into_iter()
            .map(|group| {
                Arc::new(
                    ZarrExec::new(
                        exec.schema().clone(),
                        exec.path().to_string(),
                        exec.projection().cloned(),
                        exec.limit(),
                        None,
                        exec.coord_filters().cloned(),
                        None,
                    )
                    .with_partitions(group)
                    .with_store_kind(exec.store_kind().clone()),
                ) as Arc<dyn ExecutionPlan>
            });

        let leaf = DistributedLeafExec::try_new(Arc::clone(plan), variants)?;
        Ok(Some(Arc::new(leaf)))
    }
}

/// Register the Zarr physical codec and task estimator on a session builder.
///
/// Call this on both the head (coordinator) and every worker so the two sides
/// agree on the wire format and on how scans fan out. Pair it with
/// [`crate::udfs::register_metric_udfs`] on the resulting `SessionContext` so
/// both sides also resolve the same metric functions without clobbering
/// DataFusion's built-in aggregates.
pub fn configure_distributed_builder(builder: SessionStateBuilder) -> SessionStateBuilder {
    builder
        .with_distributed_user_codec(ZarrPhysicalCodec)
        .with_distributed_task_estimator(ZarrTaskEstimator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_plan::partition::PartitionSpec;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::empty::EmptyExec;

    /// A `ZarrExec` carrying `n` single-chunk partition slices `[i, i+1)`.
    fn zarr_with_partitions(n: u64) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("lat", DataType::Int64, false),
            Field::new("temperature", DataType::Int64, true),
        ]));
        let specs: Vec<PartitionSpec> = (0..n).map(|i| PartitionSpec::range(i, i + 1)).collect();
        Arc::new(
            ZarrExec::new(
                schema,
                "/tmp/x.zarr".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .with_partitions(specs),
        )
    }

    #[test]
    fn estimates_one_task_per_partition() {
        let plan = zarr_with_partitions(7);
        let est = ZarrTaskEstimator
            .task_estimation(&plan, &ConfigOptions::default())
            .expect("ZarrExec with >=2 partitions should estimate");
        assert_eq!(est.task_count.as_usize(), 7);
    }

    #[test]
    fn no_estimation_for_single_partition() {
        // <2 slices: nothing to distribute, let the default Maximum(1) apply.
        let plan = zarr_with_partitions(1);
        assert!(ZarrTaskEstimator
            .task_estimation(&plan, &ConfigOptions::default())
            .is_none());
    }

    #[test]
    fn no_estimation_for_non_zarr_leaf() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(schema));
        assert!(ZarrTaskEstimator
            .task_estimation(&plan, &ConfigOptions::default())
            .is_none());
    }

    #[test]
    fn scale_up_splits_into_uniform_disjoint_variants() {
        let plan = zarr_with_partitions(7);
        let scaled = ZarrTaskEstimator
            .scale_up_leaf_node(&plan, 3, &ConfigOptions::default())
            .unwrap()
            .expect("should produce a DistributedLeafExec");
        let leaf = scaled
            .downcast_ref::<DistributedLeafExec>()
            .expect("scaled node should be a DistributedLeafExec");
        let variants = leaf.variants();

        // One variant per task.
        assert_eq!(variants.len(), 3);

        // All variants must report the SAME partition count (DistributedLeafExec
        // requires it) — here ceil(7/3) = 3.
        let counts: Vec<usize> = variants
            .iter()
            .map(|v| v.properties().partitioning.partition_count())
            .collect();
        assert!(counts.iter().all(|&c| c == 3), "non-uniform: {counts:?}");

        // The non-empty (real) specs across all variants must equal the original
        // 7 slices, in order — disjoint and complete, no dup/drop.
        let real: Vec<PartitionSpec> = variants
            .iter()
            .flat_map(|v| {
                v.downcast_ref::<ZarrExec>()
                    .expect("variant is a ZarrExec")
                    .partitions()
                    .iter()
                    .filter(|s| !s.is_empty())
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect();
        let expected: Vec<PartitionSpec> = (0..7).map(|i| PartitionSpec::range(i, i + 1)).collect();
        assert_eq!(real, expected);
    }

    #[test]
    fn scale_up_single_task_is_noop() {
        // task_count == 1: nothing to distribute, leave the plan unchanged.
        let plan = zarr_with_partitions(7);
        assert!(ZarrTaskEstimator
            .scale_up_leaf_node(&plan, 1, &ConfigOptions::default())
            .unwrap()
            .is_none());
    }
}
