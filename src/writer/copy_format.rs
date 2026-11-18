//! `COPY TO ... STORED AS ZARR` — the SQL verb for the write path.
//!
//! §6 of docs/zarr-write-roundtrip-plan.md. DataFusion routes `COPY (query) TO
//! 'path' STORED AS <fmt> OPTIONS(...)` to a registered [`FileFormatFactory`]. This
//! implements a write-only Zarr format: `create_writer_physical_plan` derives the
//! target grid from the query's plan (never the stream — §3.1/Q1) and returns the
//! [`zarr_write_exec`] sink node.
//!
//! Reads are unaffected: `CREATE EXTERNAL TABLE ... STORED AS ZARR` goes through the
//! separate `ZarrTableFactory` (a `TableProviderFactory`), a different registry. So
//! the read methods here are unreachable and stubbed.
//!
//! ```sql
//! COPY (SELECT x, y, (b08 - b04) / (b08 + b04) AS ndvi FROM scene)
//!   TO 'data/ndvi.zarr' STORED AS ZARR OPTIONS ('chunks' '512,512');
//! ```
//!
//! `OPTIONS`: `chunks` (comma-separated target chunk shape; defaults to the source's
//! chunking when the source is Zarr) and `partitions` (concurrent writers, Phase 5;
//! default 1).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;

use datafusion::catalog::Session;
use datafusion::common::{GetExt, Statistics};
use datafusion::datasource::file_format::file_compression_type::FileCompressionType;
use datafusion::datasource::file_format::{FileFormat, FileFormatFactory};
use datafusion::datasource::physical_plan::{FileScanConfig, FileSinkConfig, FileSource};
use datafusion::datasource::table_schema::TableSchema;
use datafusion::error::{DataFusionError, Result};
use datafusion::object_store::{ObjectMeta, ObjectStore};
use datafusion::physical_expr::LexRequirement;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

use super::data_sink::zarr_write_exec;
use super::materialize::derive_skeleton_spec;
use super::plan::find_scan;

const ZARR_EXT: &str = "zarr";

/// Factory registered under `STORED AS ZARR` for `COPY TO`.
#[derive(Debug, Default)]
pub struct ZarrFormatFactory;

impl GetExt for ZarrFormatFactory {
    fn get_ext(&self) -> String {
        ZARR_EXT.to_string()
    }
}

impl FileFormatFactory for ZarrFormatFactory {
    fn create(
        &self,
        _state: &dyn Session,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        // DataFusion may namespace custom OPTIONS under `format.`; accept both.
        let get = |k: &str| {
            options
                .get(k)
                .or_else(|| options.get(&format!("format.{k}")))
        };

        let chunks = match get("chunks") {
            Some(s) => Some(
                s.split(',')
                    .map(|p| p.trim().parse::<u64>())
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| DataFusionError::Plan(format!("invalid 'chunks' option: {e}")))?,
            ),
            None => None,
        };
        let partitions = match get("partitions") {
            Some(s) => s
                .trim()
                .parse::<usize>()
                .map_err(|e| DataFusionError::Plan(format!("invalid 'partitions' option: {e}")))?,
            None => 1,
        };

        Ok(Arc::new(ZarrFileFormat { chunks, partitions }))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(ZarrFileFormat {
            chunks: None,
            partitions: 1,
        })
    }
}

/// The write-only Zarr file format.
#[derive(Debug)]
pub struct ZarrFileFormat {
    /// Target chunk shape; `None` copies the source's chunking.
    chunks: Option<Vec<u64>>,
    /// Concurrent write partitions (Phase 5).
    partitions: usize,
}

#[async_trait]
impl FileFormat for ZarrFileFormat {
    fn get_ext(&self) -> String {
        ZARR_EXT.to_string()
    }

    fn get_ext_with_compression(&self, _c: &FileCompressionType) -> Result<String> {
        Ok(ZARR_EXT.to_string())
    }

    fn compression_type(&self) -> Option<FileCompressionType> {
        None
    }

    async fn infer_schema(
        &self,
        _state: &dyn Session,
        _store: &Arc<dyn ObjectStore>,
        _objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        Err(DataFusionError::NotImplemented(
            "Zarr reads use CREATE EXTERNAL TABLE STORED AS ZARR, not this write format".into(),
        ))
    }

    async fn infer_stats(
        &self,
        _state: &dyn Session,
        _store: &Arc<dyn ObjectStore>,
        _table_schema: SchemaRef,
        _object: &ObjectMeta,
    ) -> Result<Statistics> {
        Err(DataFusionError::NotImplemented(
            "the Zarr write format does not read".into(),
        ))
    }

    async fn create_physical_plan(
        &self,
        _state: &dyn Session,
        _conf: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::NotImplemented(
            "the Zarr write format does not read".into(),
        ))
    }

    async fn create_writer_physical_plan(
        &self,
        input: Arc<dyn ExecutionPlan>,
        _state: &dyn Session,
        conf: FileSinkConfig,
        _order_requirements: Option<LexRequirement>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let target = conf
            .original_url
            .strip_prefix("file://")
            .unwrap_or(&conf.original_url)
            .to_string();

        let chunks = match &self.chunks {
            Some(c) => c.clone(),
            None => source_chunks(&input).ok_or_else(|| {
                DataFusionError::Plan(
                    "COPY TO ... STORED AS ZARR needs OPTIONS('chunks' '...') \
                     (or a Zarr source whose chunking can be copied)"
                        .into(),
                )
            })?,
        };

        let spec = derive_skeleton_spec(&input, chunks).await.map_err(DataFusionError::External)?;
        Ok(zarr_write_exec(input, target, spec, self.partitions))
    }

    fn file_source(&self, _table_schema: TableSchema) -> Arc<dyn FileSource> {
        unimplemented!("the Zarr write format has no read FileSource; reads use ZarrTableFactory")
    }
}

/// The source store's data-variable chunk shape, if the plan reads one Zarr store.
fn source_chunks(input: &Arc<dyn ExecutionPlan>) -> Option<Vec<u64>> {
    let (_, zarr) = find_scan(input)?;
    zarr.store_meta()?.data_vars.first()?.chunks.clone()
}

/// Register `STORED AS ZARR` for `COPY TO` on a context. Idempotent-ish: overwrites
/// any existing registration under the `zarr` extension.
pub fn register_zarr_write_format(ctx: &SessionContext) -> Result<()> {
    ctx.state_ref()
        .write()
        .register_file_format(Arc::new(ZarrFormatFactory), true)
}
