//! `DataSink` driver: expose the Zarr write path as a DataFusion `ExecutionPlan`.
//!
//! §6 of docs/zarr-write-roundtrip-plan.md. The write pipeline so far is a set of
//! functions; this wraps it as a native DataFusion sink so a query plan can *write*
//! a Zarr store as part of execution. That is the prerequisite for both the
//! `COPY TO` verb and a streaming input shuffle.
//!
//! The load-bearing design point (§3.1 / Q1): the target grid comes from
//! *derivation at plan time*, never from the data stream. So the sink is
//! **constructed with the `SkeletonSpec` already derived** (via
//! `derive_skeleton_spec`); `write_all` only consumes the stream and writes. It does
//! not re-derive anything, so there is still one implementation of "what is a legal
//! write".
//!
//! `DataSinkExec` requires a single-partition input (it executes only partition 0),
//! so [`zarr_write_exec`] coalesces first. The *write-side* parallelism is our own:
//! `write_batches_partitioned` fans the collected batches across chunk-aligned
//! slabs (Phase 5), independent of the DataFusion input partitioning.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::datasource::sink::{DataSink, DataSinkExec};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, SendableRecordBatchStream,
};
use futures::StreamExt;

use super::skeleton::{create_skeleton, SkeletonSpec};
use super::sink::{write_batches, write_batches_partitioned, write_chunks};

/// A DataFusion sink that writes its input stream into a Zarr store.
///
/// Holds the fully-derived target spec; execution creates the skeleton and writes
/// the data-variable chunks, or with [`writing_chunks`](Self::writing_chunks),
/// writes only the chunks its rows fall in to a skeleton created beforehand.
#[derive(Debug, Clone)]
pub struct ZarrDataSink {
    store_path: String,
    spec: SkeletonSpec,
    input_schema: SchemaRef,
    /// Chunk-aligned write partitions (Phase 5). `1` is the single-writer path.
    target_partitions: usize,
    writes_chunks: bool,
}

impl ZarrDataSink {
    pub fn new(
        store_path: impl Into<String>,
        spec: SkeletonSpec,
        input_schema: SchemaRef,
        target_partitions: usize,
    ) -> Self {
        Self {
            store_path: store_path.into(),
            spec,
            input_schema,
            target_partitions: target_partitions.max(1),
            writes_chunks: false,
        }
    }

    /// This sink, writing only the chunks its rows fall in, to a store already
    /// created from its spec, as several sinks whose rows fall in different
    /// chunks can between them.
    pub fn writing_chunks(mut self) -> Self {
        self.writes_chunks = true;
        self
    }

    pub fn writes_chunks(&self) -> bool {
        self.writes_chunks
    }

    pub fn store_path(&self) -> &str {
        &self.store_path
    }

    pub fn spec(&self) -> &SkeletonSpec {
        &self.spec
    }

    pub fn target_partitions(&self) -> usize {
        self.target_partitions
    }
}

impl DisplayAs for ZarrDataSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ZarrDataSink(path={}, vars={}, partitions={}{})",
            self.store_path,
            self.spec.data_vars.len(),
            self.target_partitions,
            if self.writes_chunks { ", chunks" } else { "" },
        )
    }
}

#[async_trait]
impl DataSink for ZarrDataSink {
    fn schema(&self) -> &SchemaRef {
        &self.input_schema
    }

    async fn write_all(
        &self,
        mut data: SendableRecordBatchStream,
        _context: &Arc<TaskContext>,
    ) -> Result<u64> {
        // The input is coalesced to one partition (see zarr_write_exec), so this is
        // the whole result. Drain it, then hand off to the (blocking) writer.
        let mut batches = Vec::new();
        while let Some(batch) = data.next().await {
            batches.push(batch?);
        }

        let store_path = self.store_path.clone();
        let spec = self.spec.clone();
        let parts = self.target_partitions;
        let writes_chunks = self.writes_chunks;

        tokio::task::spawn_blocking(move || {
            if writes_chunks {
                return write_chunks(&store_path, &spec, batches);
            }
            create_skeleton(&store_path, &spec)?;
            if parts > 1 {
                write_batches_partitioned(&store_path, &spec, batches, parts)
            } else {
                write_batches(&store_path, &spec, batches)
            }
        })
        .await
        .map_err(|e| DataFusionError::Execution(format!("zarr write task join error: {e}")))?
        .map_err(DataFusionError::External)
    }
}

/// Build an `ExecutionPlan` that writes `input`'s rows into a Zarr store at
/// `store_path`, using an already-derived `spec`. Executing the returned plan
/// creates the skeleton and writes the data; it yields a single `count` row.
///
/// The input is coalesced to one partition because `DataSinkExec` reads only
/// partition 0; write-side parallelism comes from `target_partitions` (Phase 5).
pub fn zarr_write_exec(
    input: Arc<dyn ExecutionPlan>,
    store_path: impl Into<String>,
    spec: SkeletonSpec,
    target_partitions: usize,
) -> Arc<dyn ExecutionPlan> {
    let input = if input.output_partitioning().partition_count() > 1 {
        Arc::new(CoalescePartitionsExec::new(input)) as Arc<dyn ExecutionPlan>
    } else {
        input
    };
    let schema = input.schema();
    let sink = Arc::new(ZarrDataSink::new(store_path, spec, schema, target_partitions));
    Arc::new(DataSinkExec::new(input, sink, None))
}
