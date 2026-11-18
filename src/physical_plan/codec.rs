//! Serialization support for [`ZarrExec`] so it can be shipped across a
//! datafusion-distributed cluster.
//!
//! datafusion-distributed ships physical stages from the head to workers as
//! bytes. Built-in operators are handled by DataFusion's protobuf codec, but
//! custom `ExecutionPlan`s like [`ZarrExec`] need a [`PhysicalExtensionCodec`]
//! that knows how to encode and decode them. [`ZarrPhysicalCodec`] is registered
//! on both sides via `configure_distributed_builder` (see `crate::distributed`).
//!
//! Only the *logical* inputs of `ZarrExec` are serialized: `schema`, `path`,
//! `projection`, `limit`, `coord_filters`, `partitions` and `store_kind`. The
//! live store caches (`cached_remote`, `cached_virtualizarr`), the derived
//! `PlanProperties`, and the runtime `io_stats` are intentionally dropped —
//! `ZarrExec::execute()` re-creates the store from `path` and `store_kind` when
//! the caches are absent, which is exactly the right behavior on a remote worker.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::{DataFusionError, Result, ScalarValue, TableReference};
use datafusion::datasource::sink::DataSinkExec;
use datafusion::datasource::TableProvider;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::logical_plan::LogicalExtensionCodec;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use datafusion_proto::protobuf;
use prost::Message;
use serde::{Deserialize, Serialize};

use crate::datasource::zarr::ZarrTable;
use crate::physical_plan::partition::PartitionSpec;
use crate::physical_plan::zarr_exec::{StoreKind, ZarrExec};
use crate::reader::filter::{CoordFilterKind, CoordFilters};
use crate::reader::schema_inference::ZarrStoreMeta;
use crate::writer::{SkeletonSpec, ZarrDataSink};

// =============================================================================
// ScalarValue / Schema <-> bytes, via datafusion-proto's protobuf conversions
// =============================================================================

fn scalar_to_bytes(value: &ScalarValue) -> Result<Vec<u8>> {
    let proto = protobuf::ScalarValue::try_from(value)
        .map_err(|e| DataFusionError::Internal(format!("encode ScalarValue: {e}")))?;
    Ok(proto.encode_to_vec())
}

fn scalar_from_bytes(bytes: &[u8]) -> Result<ScalarValue> {
    let proto = protobuf::ScalarValue::decode(bytes)
        .map_err(|e| DataFusionError::Internal(format!("decode ScalarValue proto: {e}")))?;
    ScalarValue::try_from(&proto)
        .map_err(|e| DataFusionError::Internal(format!("decode ScalarValue: {e}")))
}

fn schema_to_bytes(schema: &Schema) -> Result<Vec<u8>> {
    let proto = protobuf::Schema::try_from(schema)
        .map_err(|e| DataFusionError::Internal(format!("encode Schema: {e}")))?;
    Ok(proto.encode_to_vec())
}

fn schema_from_bytes(bytes: &[u8]) -> Result<Schema> {
    let proto = protobuf::Schema::decode(bytes)
        .map_err(|e| DataFusionError::Internal(format!("decode Schema proto: {e}")))?;
    Schema::try_from(&proto).map_err(|e| DataFusionError::Internal(format!("decode Schema: {e}")))
}

// =============================================================================
// Serde DTOs mirroring the serializable subset of ZarrExec
// =============================================================================

/// Serializable mirror of [`CoordFilterKind`]. `ScalarValue` payloads are stored
/// as protobuf-encoded bytes since `ScalarValue` itself is not serde-friendly.
#[derive(Serialize, Deserialize)]
enum CoordFilterKindDto {
    Eq(Vec<u8>),
    Range {
        low: Option<Vec<u8>>,
        high: Option<Vec<u8>>,
        low_inclusive: bool,
        high_inclusive: bool,
    },
    DatePart {
        field: String,
        value: i32,
    },
    InList(Vec<Vec<u8>>),
    DatePartSet {
        field: String,
        values: Vec<i32>,
    },
}

impl CoordFilterKindDto {
    fn from_kind(kind: &CoordFilterKind) -> Result<Self> {
        Ok(match kind {
            CoordFilterKind::Eq(v) => CoordFilterKindDto::Eq(scalar_to_bytes(v)?),
            CoordFilterKind::Range {
                low,
                high,
                low_inclusive,
                high_inclusive,
            } => CoordFilterKindDto::Range {
                low: low.as_ref().map(scalar_to_bytes).transpose()?,
                high: high.as_ref().map(scalar_to_bytes).transpose()?,
                low_inclusive: *low_inclusive,
                high_inclusive: *high_inclusive,
            },
            CoordFilterKind::DatePart { field, value } => CoordFilterKindDto::DatePart {
                field: field.clone(),
                value: *value,
            },
            CoordFilterKind::InList(values) => CoordFilterKindDto::InList(
                values
                    .iter()
                    .map(scalar_to_bytes)
                    .collect::<Result<Vec<_>>>()?,
            ),
            CoordFilterKind::DatePartSet { field, values } => CoordFilterKindDto::DatePartSet {
                field: field.clone(),
                values: values.clone(),
            },
        })
    }

    fn into_kind(self) -> Result<CoordFilterKind> {
        Ok(match self {
            CoordFilterKindDto::Eq(bytes) => CoordFilterKind::Eq(scalar_from_bytes(&bytes)?),
            CoordFilterKindDto::Range {
                low,
                high,
                low_inclusive,
                high_inclusive,
            } => CoordFilterKind::Range {
                low: low.as_deref().map(scalar_from_bytes).transpose()?,
                high: high.as_deref().map(scalar_from_bytes).transpose()?,
                low_inclusive,
                high_inclusive,
            },
            CoordFilterKindDto::DatePart { field, value } => {
                CoordFilterKind::DatePart { field, value }
            }
            CoordFilterKindDto::InList(byte_vals) => CoordFilterKind::InList(
                byte_vals
                    .iter()
                    .map(|b| scalar_from_bytes(b))
                    .collect::<Result<Vec<_>>>()?,
            ),
            CoordFilterKindDto::DatePartSet { field, values } => {
                CoordFilterKind::DatePartSet { field, values }
            }
        })
    }
}

/// Serializable mirror of the serializable subset of [`ZarrExec`].
#[derive(Serialize, Deserialize)]
struct ZarrExecDto {
    /// Protobuf-encoded full (unprojected) Arrow schema.
    schema: Vec<u8>,
    path: String,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    coord_filters: Option<HashMap<String, Vec<CoordFilterKindDto>>>,
    /// Per-task partition slices. MUST be carried: each worker decodes a
    /// distinct subset and reads only those chunks. If dropped, every worker
    /// re-scans the whole store. `serde(default)` keeps old payloads decodable
    /// as an empty (single-partition) list.
    #[serde(default)]
    partitions: Vec<PartitionSpec>,
    #[serde(default)]
    store_kind: StoreKind,
}

impl ZarrExecDto {
    fn from_exec(exec: &ZarrExec) -> Result<Self> {
        let coord_filters = exec
            .coord_filters()
            .map(|cf| {
                cf.filters
                    .iter()
                    .map(|(k, kinds)| {
                        let dtos = kinds
                            .iter()
                            .map(CoordFilterKindDto::from_kind)
                            .collect::<Result<Vec<_>>>()?;
                        Ok((k.clone(), dtos))
                    })
                    .collect::<Result<HashMap<_, _>>>()
            })
            .transpose()?;

        Ok(Self {
            schema: schema_to_bytes(exec.schema())?,
            path: exec.path().to_string(),
            projection: exec.projection().cloned(),
            limit: exec.limit(),
            coord_filters,
            partitions: exec.partitions().to_vec(),
            store_kind: exec.store_kind().clone(),
        })
    }

    fn into_exec(self) -> Result<ZarrExec> {
        let schema = Arc::new(schema_from_bytes(&self.schema)?);

        let coord_filters = self
            .coord_filters
            .map(|map| {
                let filters = map
                    .into_iter()
                    .map(|(k, dtos)| {
                        let kinds = dtos
                            .into_iter()
                            .map(|d| d.into_kind())
                            .collect::<Result<Vec<_>>>()?;
                        Ok((k, kinds))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;
                Ok::<_, DataFusionError>(CoordFilters { filters })
            })
            .transpose()?;

        // Caches are dropped on the wire; execute() re-creates the store from
        // `path` and `store_kind`.
        // Reattach the partition slices so the decoded exec reports the right
        // partition count and execute() reads only this task's chunks.
        Ok(ZarrExec::new(
            schema,
            self.path,
            self.projection,
            self.limit,
            None,
            coord_filters,
            None,
        )
        .with_partitions(self.partitions)
        .with_store_kind(self.store_kind))
    }
}

// =============================================================================
// PhysicalExtensionCodec
// =============================================================================

/// A [`DataSinkExec`] writing through a [`ZarrDataSink`]; its input travels as
/// the node's child, and supplies the sink's schema.
#[derive(Serialize, Deserialize)]
struct ZarrSinkDto {
    zarr_sink: ZarrSinkFields,
}

#[derive(Serialize, Deserialize)]
struct ZarrSinkFields {
    store_path: String,
    spec: SkeletonSpec,
    target_partitions: usize,
    writes_chunks: bool,
}

/// Encodes/decodes [`ZarrExec`], and a [`DataSinkExec`] over a [`ZarrDataSink`],
/// for datafusion-distributed. Register this on the head and every worker (via
/// `with_distributed_user_codec`) so both sides agree on the wire format.
#[derive(Debug, Default)]
pub struct ZarrPhysicalCodec;

impl PhysicalExtensionCodec for ZarrPhysicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        _ctx: &TaskContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Ok(ZarrSinkDto { zarr_sink: f }) = serde_json::from_slice(buf) {
            let [input] = inputs else {
                return Err(DataFusionError::Internal(
                    "a Zarr sink takes one input".to_string(),
                ));
            };
            let sink = ZarrDataSink::new(f.store_path, f.spec, input.schema(), f.target_partitions);
            let sink = if f.writes_chunks {
                sink.writing_chunks()
            } else {
                sink
            };
            return Ok(Arc::new(DataSinkExec::new(
                input.clone(),
                Arc::new(sink),
                None,
            )));
        }
        let dto: ZarrExecDto = serde_json::from_slice(buf)
            .map_err(|e| DataFusionError::Internal(format!("decode ZarrExecDto: {e}")))?;
        Ok(Arc::new(dto.into_exec()?))
    }

    fn try_encode(&self, node: Arc<dyn ExecutionPlan>, buf: &mut Vec<u8>) -> Result<()> {
        let sink = node
            .downcast_ref::<DataSinkExec>()
            .and_then(|exec| exec.sink().downcast_ref::<ZarrDataSink>());
        if let Some(sink) = sink {
            let dto = ZarrSinkDto {
                zarr_sink: ZarrSinkFields {
                    store_path: sink.store_path().to_string(),
                    spec: sink.spec().clone(),
                    target_partitions: sink.target_partitions(),
                    writes_chunks: sink.writes_chunks(),
                },
            };
            let bytes = serde_json::to_vec(&dto)
                .map_err(|e| DataFusionError::Internal(format!("encode ZarrSinkDto: {e}")))?;
            buf.extend_from_slice(&bytes);
            return Ok(());
        }
        let exec = node
            .downcast_ref::<ZarrExec>()
            .ok_or_else(|| DataFusionError::Internal("not a ZarrExec".to_string()))?;
        let dto = ZarrExecDto::from_exec(exec)?;
        let bytes = serde_json::to_vec(&dto)
            .map_err(|e| DataFusionError::Internal(format!("encode ZarrExecDto: {e}")))?;
        buf.extend_from_slice(&bytes);
        Ok(())
    }
}

// =============================================================================
// LogicalExtensionCodec
// =============================================================================

/// Serializable mirror of the reconstructable subset of [`ZarrTable`].
///
/// The Arrow schema is *not* serialized here — the framework passes it
/// separately to `try_decode_table_provider`. The live store caches
/// (`cached_remote`, `cached_virtualizarr`) are dropped; they are re-created
/// from `path` on use. `store_meta` is carried so the decoder can do
/// statistics-based optimization and filter-pushdown parsing without re-opening
/// the store.
#[derive(Serialize, Deserialize)]
struct ZarrTableDto {
    path: String,
    store_meta: Option<ZarrStoreMeta>,
}

/// Encodes/decodes [`ZarrTable`] so a `TableScan` over a Zarr store can travel
/// from a client to a planner. Companion to [`ZarrPhysicalCodec`].
///
/// NOTE: this logical codec is currently unused by the live datafusion-distributed
/// path — the head keeps the `TableProvider` local and ships only the physical
/// `ZarrExec`, so only [`ZarrPhysicalCodec`] is registered. It is retained (and
/// unit-tested) for round-tripping a full `TableScan`, e.g. if logical-plan
/// distribution is needed later.
#[derive(Debug, Default)]
pub struct ZarrLogicalCodec;

impl LogicalExtensionCodec for ZarrLogicalCodec {
    // We have no custom LogicalPlan Extension nodes, only a custom TableProvider.
    fn try_decode(
        &self,
        _buf: &[u8],
        _inputs: &[LogicalPlan],
        _ctx: &TaskContext,
    ) -> Result<Extension> {
        Err(DataFusionError::NotImplemented(
            "ZarrLogicalCodec has no logical Extension nodes".to_string(),
        ))
    }

    fn try_encode(&self, _node: &Extension, _buf: &mut Vec<u8>) -> Result<()> {
        Err(DataFusionError::NotImplemented(
            "ZarrLogicalCodec has no logical Extension nodes".to_string(),
        ))
    }

    fn try_decode_table_provider(
        &self,
        buf: &[u8],
        _table_ref: &TableReference,
        schema: SchemaRef,
        _ctx: &TaskContext,
    ) -> Result<Arc<dyn TableProvider>> {
        let dto: ZarrTableDto = serde_json::from_slice(buf)
            .map_err(|e| DataFusionError::Internal(format!("decode ZarrTableDto: {e}")))?;
        // Caches are dropped on the wire; the scan rebuilds them from `path`.
        let table = match dto.store_meta {
            Some(meta) => ZarrTable::with_metadata(schema, dto.path, meta),
            None => ZarrTable::new(schema, dto.path),
        };
        Ok(Arc::new(table))
    }

    fn try_encode_table_provider(
        &self,
        _table_ref: &TableReference,
        node: Arc<dyn TableProvider>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        let table = node
            .downcast_ref::<ZarrTable>()
            .ok_or_else(|| DataFusionError::Internal("not a ZarrTable".to_string()))?;
        let dto = ZarrTableDto {
            path: table.path().to_string(),
            store_meta: table.store_meta().cloned(),
        };
        let bytes = serde_json::to_vec(&dto)
            .map_err(|e| DataFusionError::Internal(format!("encode ZarrTableDto: {e}")))?;
        buf.extend_from_slice(&bytes);
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field};

    fn sample_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("time", DataType::Int64, false),
            Field::new("lat", DataType::Float64, false),
            Field::new("temperature", DataType::Float64, true),
        ]))
    }

    fn sample_filters() -> CoordFilters {
        let mut filters = CoordFilters::new();
        // Two AND-composed filters on one coordinate exercises the Vec path.
        filters.push("time", CoordFilterKind::Eq(ScalarValue::Int64(Some(42))));
        filters.push(
            "time",
            CoordFilterKind::DatePartSet {
                field: "HOUR".to_string(),
                values: vec![0, 12],
            },
        );
        filters.push(
            "lat",
            CoordFilterKind::Range {
                low: Some(ScalarValue::Float64(Some(10.0))),
                high: Some(ScalarValue::Float64(Some(20.0))),
                low_inclusive: true,
                high_inclusive: false,
            },
        );
        filters.push(
            "month",
            CoordFilterKind::DatePart {
                field: "MONTH".to_string(),
                value: 12,
            },
        );
        filters.push(
            "level",
            CoordFilterKind::InList(vec![
                ScalarValue::Int64(Some(500)),
                ScalarValue::Int64(Some(850)),
            ]),
        );
        filters
    }

    #[test]
    fn roundtrip_preserves_a_zarr_sink() {
        use crate::writer::{CoordSpec, CoordValues, DataVarSpec, WriteDataType};
        use datafusion::physical_plan::empty::EmptyExec;
        use datafusion::prelude::SessionContext;
        use datafusion_proto::bytes::{
            physical_plan_from_bytes_with_extension_codec,
            physical_plan_to_bytes_with_extension_codec,
        };
        use zarrs::array::FillValue;

        let mut var = DataVarSpec::new("clim", WriteDataType::Float32);
        var.fill_value = Some(FillValue::from(-1.0f32));
        let spec = SkeletonSpec::new(
            vec![
                CoordSpec::new("hour", CoordValues::Int64((0..24).collect())),
                CoordSpec::new("lat", CoordValues::Float64(vec![90.0, 89.75])),
            ],
            vec![var],
            vec![24, 12],
        );
        let input = Arc::new(EmptyExec::new(sample_schema()));
        let sink = ZarrDataSink::new("/out.zarr", spec, input.schema(), 3).writing_chunks();
        let exec = Arc::new(DataSinkExec::new(input, Arc::new(sink), None));

        let bytes = physical_plan_to_bytes_with_extension_codec(exec, &ZarrPhysicalCodec).unwrap();
        let ctx = SessionContext::new();
        let decoded = physical_plan_from_bytes_with_extension_codec(
            &bytes,
            &ctx.task_ctx(),
            &ZarrPhysicalCodec,
        )
        .unwrap();

        let exec = decoded.downcast_ref::<DataSinkExec>().expect("a sink");
        assert_eq!(exec.input().schema(), sample_schema());
        let sink = exec
            .sink()
            .downcast_ref::<ZarrDataSink>()
            .expect("a Zarr sink");
        assert_eq!(sink.store_path(), "/out.zarr");
        assert!(sink.writes_chunks());
        assert_eq!(sink.target_partitions(), 3);
        let spec = sink.spec();
        assert_eq!(spec.chunks, vec![24, 12]);
        assert_eq!(spec.coords[0].name, "hour");
        assert_eq!(spec.coords[1].values.len(), 2);
        assert_eq!(spec.data_vars[0].fill_value, Some(FillValue::from(-1.0f32)));
    }

    #[test]
    fn roundtrip_preserves_all_fields() {
        let schema = sample_schema();
        let exec = ZarrExec::new(
            schema.clone(),
            "gs://bucket/data.zarr".to_string(),
            Some(vec![0, 2]),
            Some(100),
            None,
            Some(sample_filters()),
            None,
        );

        let codec = ZarrPhysicalCodec;
        let ctx = TaskContext::default();

        let mut buf = Vec::new();
        codec
            .try_encode(Arc::new(exec) as Arc<dyn ExecutionPlan>, &mut buf)
            .unwrap();

        let decoded = codec.try_decode(&buf, &[], &ctx).unwrap();
        let decoded = decoded.downcast_ref::<ZarrExec>().unwrap();

        assert_eq!(decoded.path(), "gs://bucket/data.zarr");
        assert_eq!(decoded.projection(), Some(&vec![0, 2]));
        assert_eq!(decoded.limit(), Some(100));
        assert_eq!(decoded.schema().as_ref(), schema.as_ref());

        let filters = decoded.coord_filters().expect("filters preserved");
        assert_eq!(filters.len(), 4); // time, lat, month, level

        // `time` carries two AND-composed filters (Eq + DatePartSet).
        let time = filters.get("time").unwrap();
        assert_eq!(time.len(), 2);
        match &time[0] {
            CoordFilterKind::Eq(ScalarValue::Int64(Some(v))) => assert_eq!(*v, 42),
            other => panic!("unexpected: {other:?}"),
        }
        match &time[1] {
            CoordFilterKind::DatePartSet { field, values } => {
                assert_eq!(field, "HOUR");
                assert_eq!(values, &vec![0, 12]);
            }
            other => panic!("unexpected: {other:?}"),
        }
        match &filters.get("lat").unwrap()[0] {
            CoordFilterKind::Range {
                low,
                high,
                low_inclusive,
                high_inclusive,
            } => {
                assert_eq!(*low, Some(ScalarValue::Float64(Some(10.0))));
                assert_eq!(*high, Some(ScalarValue::Float64(Some(20.0))));
                assert!(*low_inclusive);
                assert!(!*high_inclusive);
            }
            other => panic!("unexpected: {other:?}"),
        }
        match &filters.get("month").unwrap()[0] {
            CoordFilterKind::DatePart { field, value } => {
                assert_eq!(field, "MONTH");
                assert_eq!(*value, 12);
            }
            other => panic!("unexpected: {other:?}"),
        }
        match &filters.get("level").unwrap()[0] {
            CoordFilterKind::InList(values) => {
                assert_eq!(
                    values,
                    &vec![ScalarValue::Int64(Some(500)), ScalarValue::Int64(Some(850))]
                );
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn roundtrip_preserves_store_kind() {
        let kinds = [
            StoreKind::Zarr,
            StoreKind::Icechunk {
                group: Some("single/spatial".to_string()),
                anonymous: true,
            },
            StoreKind::Icechunk {
                group: None,
                anonymous: false,
            },
        ];
        for kind in kinds {
            let exec = ZarrExec::new(
                sample_schema(),
                "s3://bucket/repo".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .with_store_kind(kind.clone());
            let codec = ZarrPhysicalCodec;
            let mut buf = Vec::new();
            codec
                .try_encode(Arc::new(exec) as Arc<dyn ExecutionPlan>, &mut buf)
                .unwrap();
            let decoded = codec
                .try_decode(&buf, &[], &TaskContext::default())
                .unwrap();
            assert_eq!(
                decoded.downcast_ref::<ZarrExec>().unwrap().store_kind(),
                &kind
            );
        }
    }

    #[test]
    fn payload_without_store_kind_is_plain_zarr() {
        let dto = ZarrExecDto::from_exec(&ZarrExec::new(
            sample_schema(),
            "/tmp/local.zarr".to_string(),
            None,
            None,
            None,
            None,
            None,
        ))
        .unwrap();
        let mut json = serde_json::to_value(&dto).unwrap();
        json.as_object_mut().unwrap().remove("store_kind");
        let decoded = ZarrPhysicalCodec
            .try_decode(
                &serde_json::to_vec(&json).unwrap(),
                &[],
                &TaskContext::default(),
            )
            .unwrap();
        assert_eq!(
            decoded.downcast_ref::<ZarrExec>().unwrap().store_kind(),
            &StoreKind::Zarr
        );
    }

    #[test]
    fn roundtrip_preserves_partitions() {
        use crate::physical_plan::partition::PartitionSpec;

        let schema = sample_schema();
        let specs = vec![PartitionSpec::range(0, 3), PartitionSpec::range(3, 7)];
        let exec = ZarrExec::new(
            schema.clone(),
            "/tmp/local.zarr".to_string(),
            None,
            None,
            None,
            None,
            None,
        )
        .with_partitions(specs.clone());

        // Sanity: the exec advertises 2 output partitions before the round trip.
        assert_eq!(exec.partitions(), specs.as_slice());
        assert_eq!(exec.properties().partitioning.partition_count(), 2);

        let codec = ZarrPhysicalCodec;
        let ctx = TaskContext::default();
        let mut buf = Vec::new();
        codec
            .try_encode(Arc::new(exec) as Arc<dyn ExecutionPlan>, &mut buf)
            .unwrap();
        let decoded = codec.try_decode(&buf, &[], &ctx).unwrap();
        let decoded = decoded.downcast_ref::<ZarrExec>().unwrap();

        // The slices AND the derived partition count must survive the wire,
        // else workers would each re-scan the whole store.
        assert_eq!(decoded.partitions(), specs.as_slice());
        assert_eq!(decoded.properties().partitioning.partition_count(), 2);
    }

    #[test]
    fn roundtrip_preserves_indices_partitions() {
        // A partition can carry a scattered Indices selection (produced once a
        // resolved date-part filter is split). It must survive the wire intact —
        // a worker that decoded it as a Range or dropped it would read the wrong
        // chunks. This covers the variant the geometry-only path never exercises.
        use crate::physical_plan::partition::PartitionSpec;
        use crate::reader::filter::CoordSelection;

        let schema = sample_schema();
        let specs = vec![
            PartitionSpec::from_outer(CoordSelection::Indices(vec![3, 17, 41])),
            PartitionSpec::from_outer(CoordSelection::Indices(vec![58, 90])),
        ];
        let exec = ZarrExec::new(
            schema,
            "/tmp/local.zarr".to_string(),
            None,
            None,
            None,
            None,
            None,
        )
        .with_partitions(specs.clone());

        let codec = ZarrPhysicalCodec;
        let ctx = TaskContext::default();
        let mut buf = Vec::new();
        codec
            .try_encode(Arc::new(exec) as Arc<dyn ExecutionPlan>, &mut buf)
            .unwrap();
        let decoded = codec.try_decode(&buf, &[], &ctx).unwrap();
        let decoded = decoded.downcast_ref::<ZarrExec>().unwrap();

        assert_eq!(decoded.partitions(), specs.as_slice());
        assert_eq!(decoded.properties().partitioning.partition_count(), 2);
    }

    #[test]
    fn roundtrip_empty_partitions_stays_single() {
        // A non-partitioned exec must decode as single-partition (legacy path).
        let schema = sample_schema();
        let exec = ZarrExec::new(
            schema,
            "/tmp/local.zarr".to_string(),
            None,
            None,
            None,
            None,
            None,
        );
        assert!(exec.partitions().is_empty());

        let codec = ZarrPhysicalCodec;
        let ctx = TaskContext::default();
        let mut buf = Vec::new();
        codec
            .try_encode(Arc::new(exec) as Arc<dyn ExecutionPlan>, &mut buf)
            .unwrap();
        let decoded = codec.try_decode(&buf, &[], &ctx).unwrap();
        let decoded = decoded.downcast_ref::<ZarrExec>().unwrap();

        assert!(decoded.partitions().is_empty());
        assert_eq!(decoded.properties().partitioning.partition_count(), 1);
    }

    #[test]
    fn roundtrip_with_no_filters_or_projection() {
        let schema = sample_schema();
        let exec = ZarrExec::new(
            schema.clone(),
            "/tmp/local.zarr".to_string(),
            None,
            None,
            None,
            None,
            None,
        );

        let codec = ZarrPhysicalCodec;
        let ctx = TaskContext::default();

        let mut buf = Vec::new();
        codec
            .try_encode(Arc::new(exec) as Arc<dyn ExecutionPlan>, &mut buf)
            .unwrap();
        let decoded = codec.try_decode(&buf, &[], &ctx).unwrap();
        let decoded = decoded.downcast_ref::<ZarrExec>().unwrap();

        assert_eq!(decoded.path(), "/tmp/local.zarr");
        assert!(decoded.projection().is_none());
        assert!(decoded.limit().is_none());
        assert!(decoded.coord_filters().is_none());
    }

    fn sample_store_meta() -> ZarrStoreMeta {
        use crate::reader::schema_inference::ZarrArrayMeta;
        let time = ZarrArrayMeta {
            name: "time".to_string(),
            data_type: "int64".to_string(),
            shape: vec![7],
            chunks: Some(vec![7]),
            coord_min_max: Some((0.0, 6.0)),
            cf_time_attrs: None,
            dimensions: Some(vec!["time".to_string()]),
        };
        let lat = ZarrArrayMeta {
            name: "lat".to_string(),
            data_type: "float64".to_string(),
            shape: vec![10],
            chunks: Some(vec![10]),
            coord_min_max: Some((-90.0, 90.0)),
            cf_time_attrs: None,
            dimensions: Some(vec!["lat".to_string()]),
        };
        let temperature = ZarrArrayMeta {
            name: "temperature".to_string(),
            data_type: "float64".to_string(),
            shape: vec![7, 10],
            chunks: Some(vec![7, 10]),
            coord_min_max: None,
            cf_time_attrs: None,
            dimensions: Some(vec!["time".to_string(), "lat".to_string()]),
        };
        ZarrStoreMeta {
            coords: vec![lat, time],
            data_vars: vec![temperature],
            total_rows: 70,
        }
    }

    #[test]
    fn table_provider_roundtrip_preserves_path_and_meta() {
        let schema = sample_schema();
        let table =
            ZarrTable::with_metadata(schema.clone(), "gs://bucket/data.zarr", sample_store_meta());

        let codec = ZarrLogicalCodec;
        let table_ref = TableReference::bare("weather");
        let ctx = TaskContext::default();

        let mut buf = Vec::new();
        codec
            .try_encode_table_provider(
                &table_ref,
                Arc::new(table) as Arc<dyn TableProvider>,
                &mut buf,
            )
            .unwrap();

        let decoded = codec
            .try_decode_table_provider(&buf, &table_ref, schema.clone(), &ctx)
            .unwrap();
        let decoded = decoded.downcast_ref::<ZarrTable>().unwrap();

        assert_eq!(decoded.path(), "gs://bucket/data.zarr");
        assert_eq!(decoded.schema().as_ref(), schema.as_ref());

        let meta = decoded.store_meta().expect("store_meta preserved");
        assert_eq!(meta.total_rows, 70);
        assert_eq!(meta.coords.len(), 2);
        assert_eq!(meta.data_vars.len(), 1);
        assert_eq!(meta.coords[1].name, "time");
        assert_eq!(meta.coords[1].coord_min_max, Some((0.0, 6.0)));
        assert_eq!(meta.data_vars[0].name, "temperature");
        assert_eq!(meta.data_vars[0].shape, vec![7, 10]);
    }

    #[test]
    fn table_provider_roundtrip_without_meta() {
        let schema = sample_schema();
        let table = ZarrTable::new(schema.clone(), "/tmp/local.zarr");

        let codec = ZarrLogicalCodec;
        let table_ref = TableReference::bare("weather");
        let ctx = TaskContext::default();

        let mut buf = Vec::new();
        codec
            .try_encode_table_provider(
                &table_ref,
                Arc::new(table) as Arc<dyn TableProvider>,
                &mut buf,
            )
            .unwrap();
        let decoded = codec
            .try_decode_table_provider(&buf, &table_ref, schema.clone(), &ctx)
            .unwrap();
        let decoded = decoded.downcast_ref::<ZarrTable>().unwrap();

        assert_eq!(decoded.path(), "/tmp/local.zarr");
        assert!(decoded.store_meta().is_none());
    }
}
