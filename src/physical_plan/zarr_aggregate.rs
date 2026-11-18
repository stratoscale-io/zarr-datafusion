//! `ZarrAggregateExec` — aggregate pushdown (Phases 7.3 global, 7.4 coordinate
//! `GROUP BY`, 7.5 periodic `GROUP BY`).
//!
//! A recognized `SUM`/`COUNT`/`AVG`/`MIN`/`MAX` over a `ZarrExec` — with no `GROUP BY`
//! (7.3), grouped on coordinate columns (7.4), or on a periodic function of one such
//! as `date_part('month', time)` (7.5) — is rewritten (by
//! [`crate::optimizer::CardinalityRule`]) into this operator, which **replaces the
//! whole `AggregateExec ← ZarrExec` subtree**. It drives the (projected, streaming)
//! `ZarrExec` child and folds its batches into per-group accumulators — keying each
//! row on the evaluated `GROUP BY` expressions — instead of letting DataFusion
//! hash-aggregate flattened rows. The group count is computed exactly
//! (`group_cardinality`) and admitted against a budget, so the group table is known
//! to fit. Emits one row per group in the aggregate's output schema
//! (`[group cols…, agg cols…]`).
//!
//! Note (7.5): grouping by `date_part` over a dictionary-encoded coordinate *fails*
//! in stock DataFusion (a dictionary return-type assertion), so pushdown here is an
//! *enabler* — it decodes the coordinate before evaluating the key, running a query
//! the engine otherwise cannot.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::scalar::ScalarValue;
use futures::TryStreamExt;

use crate::optimizer::cardinality::pushdown::{AggKind, AggSpec};

/// One accumulator, enough to finalize any supported aggregate.
#[derive(Clone, Default)]
struct Acc {
    count: u128,
    // TODO(phase7): SUM accumulates in f64, which loses precision for large int64
    // sums (> 2^53). Accumulate integer inputs in i128 to keep SUM/COUNT bit-exact.
    sum: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl Acc {
    fn push(&mut self, v: f64) {
        self.count += 1;
        self.sum += v;
        self.min = Some(self.min.map_or(v, |m| m.min(v)));
        self.max = Some(self.max.map_or(v, |m| m.max(v)));
    }

    /// The finalized value as the aggregate's *natural* array (one element), before
    /// casting to DataFusion's declared output type.
    fn finalize_one(&self, kind: AggKind) -> ArrayRef {
        match kind {
            AggKind::Count => Arc::new(Int64Array::from(vec![self.count as i64])),
            AggKind::Sum => Arc::new(Float64Array::from(vec![
                (self.count > 0).then_some(self.sum)
            ])),
            AggKind::Avg => Arc::new(Float64Array::from(vec![
                (self.count > 0).then(|| self.sum / self.count as f64)
            ])),
            AggKind::Min => Arc::new(Float64Array::from(vec![self.min])),
            AggKind::Max => Arc::new(Float64Array::from(vec![self.max])),
        }
    }
}

/// The accumulator set for one group (parallel to the aggregate list).
type Group = Vec<Acc>;

/// Physical operator: fold a `ZarrExec` scan into grouped aggregates.
pub struct ZarrAggregateExec {
    /// The scan to read (already projected to the group + aggregate columns).
    input: Arc<dyn ExecutionPlan>,
    /// `GROUP BY` key expressions over the *scan* schema, in output-schema order.
    /// Empty => global. Evaluating them per batch handles a bare coordinate column
    /// (7.4) and a periodic function like `date_part('month', time)` (7.5) uniformly
    /// — and, because we read the result value directly (not through DataFusion's
    /// aggregate), a periodic group works even though DataFusion's own path asserts
    /// on `date_part` over a dictionary-encoded coordinate.
    group_exprs: Vec<Arc<dyn PhysicalExpr>>,
    /// Aggregates to compute, in output-schema order (after the group columns).
    aggs: Vec<AggSpec>,
    /// The aggregate's output schema: `[group cols…, agg cols…]`.
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl ZarrAggregateExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        group_exprs: Vec<Arc<dyn PhysicalExpr>>,
        aggs: Vec<AggSpec>,
        schema: SchemaRef,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            input,
            group_exprs,
            aggs,
            schema,
            properties,
        }
    }

    fn new_group(&self) -> Group {
        vec![Acc::default(); self.aggs.len()]
    }

    /// Fold one aggregate at `row` of its (pre-cast) column into `acc`.
    fn fold_cell(acc: &mut Acc, kind: AggKind, col: Option<&Float64Array>, row: usize) {
        match (kind, col) {
            (AggKind::Count, None) => acc.count += 1, // COUNT(*)
            (AggKind::Count, Some(a)) => {
                if a.is_valid(row) {
                    acc.count += 1; // COUNT(col): non-null only
                }
            }
            (_, Some(a)) => {
                if a.is_valid(row) {
                    acc.push(a.value(row));
                }
            }
            (_, None) => {} // non-count without a column — recognizer never emits this
        }
    }

    /// Fold one batch into `groups`.
    // TODO(phase7): folds the streamed data batches, so the reader still materializes
    // each window's data column, and (grouped) builds the coordinate column. A bespoke
    // chunk-fold reader keyed on the axis index would avoid both.
    fn fold_batch(
        &self,
        groups: &mut HashMap<Vec<ScalarValue>, Group>,
        batch: &RecordBatch,
    ) -> Result<()> {
        // Cast each aggregate's argument column to f64 once per batch (None = COUNT(*)).
        let agg_cols: Vec<Option<Float64Array>> = self
            .aggs
            .iter()
            .map(|spec| match spec.column.as_deref() {
                None => Ok(None),
                Some(name) => {
                    let array = batch.column_by_name(name).ok_or_else(|| {
                        DataFusionError::Internal(format!("aggregate column {name} not in scan"))
                    })?;
                    let f64arr = arrow::compute::cast(array, &DataType::Float64)?;
                    Ok(Some(
                        f64arr
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .expect("cast to Float64")
                            .clone(),
                    ))
                }
            })
            .collect::<Result<_>>()?;

        // Global (no GROUP BY): one accumulator set, no per-row keying.
        if self.group_exprs.is_empty() {
            let group = groups
                .get_mut(&Vec::new())
                .expect("global group pre-inserted");
            for (i, spec) in self.aggs.iter().enumerate() {
                for row in 0..batch.num_rows() {
                    Self::fold_cell(&mut group[i], spec.kind, agg_cols[i].as_ref(), row);
                }
            }
            return Ok(());
        }

        // Grouped: evaluate each group-key expression on the batch, then key each row
        // by the resulting values. Evaluating the expression (rather than reading a
        // named column) handles coordinates and periodic functions uniformly.
        //
        // First decode dictionary-encoded coordinate columns to their value type:
        // `date_part` over a dictionary returns a dictionary, tripping
        // `ScalarFunctionExpr::evaluate`'s return-type assertion (the very reason the
        // un-pushed query fails). Decoded input makes it return the plain Int32.
        let decoded = decode_dictionaries(batch)?;
        let group_arrays: Vec<ArrayRef> = self
            .group_exprs
            .iter()
            .map(|e| e.evaluate(&decoded)?.into_array(decoded.num_rows()))
            .collect::<Result<_>>()?;

        for row in 0..batch.num_rows() {
            let key: Vec<ScalarValue> = group_arrays
                .iter()
                .map(|arr| ScalarValue::try_from_array(arr, row))
                .collect::<Result<_>>()?;
            let group = groups.entry(key).or_insert_with(|| self.new_group());
            for (i, spec) in self.aggs.iter().enumerate() {
                Self::fold_cell(&mut group[i], spec.kind, agg_cols[i].as_ref(), row);
            }
        }
        Ok(())
    }

    /// Build the output batch: one row per group, `[group cols…, agg cols…]`.
    fn finalize(&self, groups: &HashMap<Vec<ScalarValue>, Group>) -> Result<RecordBatch> {
        let n_group = self.group_exprs.len();
        // Fix a stable group order (map iteration order, materialized once).
        let keys: Vec<&Vec<ScalarValue>> = groups.keys().collect();

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(n_group + self.aggs.len());

        // Group columns from the keys (empty when no group survived the selection).
        for j in 0..n_group {
            let field_ty = self.schema.field(j).data_type();
            let col = if keys.is_empty() {
                arrow::array::new_empty_array(field_ty)
            } else {
                let arr = ScalarValue::iter_to_array(keys.iter().map(|k| k[j].clone()))?;
                cast_to(&arr, field_ty)?
            };
            columns.push(col);
        }

        // Aggregate columns: finalize each group, concatenate.
        for (i, spec) in self.aggs.iter().enumerate() {
            let per_group: Vec<ArrayRef> = keys
                .iter()
                .map(|k| groups[*k][i].finalize_one(spec.kind))
                .collect();
            let refs: Vec<&dyn Array> = per_group.iter().map(|a| a.as_ref()).collect();
            let natural = if refs.is_empty() {
                // No groups (e.g. empty grouped selection): an empty column.
                spec.kind.empty_array()
            } else {
                arrow::compute::concat(&refs)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
            };
            columns.push(cast_to(
                &natural,
                self.schema.field(n_group + i).data_type(),
            )?);
        }

        RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl AggKind {
    /// A zero-row array of this aggregate's natural type (for the no-groups case).
    fn empty_array(self) -> ArrayRef {
        match self {
            AggKind::Count => Arc::new(Int64Array::from(Vec::<i64>::new())),
            _ => Arc::new(Float64Array::from(Vec::<Option<f64>>::new())),
        }
    }
}

/// Cast `arr` to `ty` only when needed (identity types skip the copy).
fn cast_to(arr: &ArrayRef, ty: &DataType) -> Result<ArrayRef> {
    if arr.data_type() == ty {
        Ok(arr.clone())
    } else {
        Ok(arrow::compute::cast(arr, ty)?)
    }
}

/// Decode any dictionary-encoded columns of `batch` to their value type, leaving
/// other columns untouched. Coordinate columns are dictionary-encoded; decoding
/// them lets scalar functions (notably `date_part`) evaluate without the
/// dictionary-return-type assertion. A no-op batch is returned as a cheap clone.
fn decode_dictionaries(batch: &RecordBatch) -> Result<RecordBatch> {
    use arrow::datatypes::{Field, Schema};
    if !batch
        .schema()
        .fields()
        .iter()
        .any(|f| matches!(f.data_type(), DataType::Dictionary(_, _)))
    {
        return Ok(batch.clone());
    }
    let columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| match c.data_type() {
            DataType::Dictionary(_, v) => Ok(arrow::compute::cast(c, v)?),
            _ => Ok(c.clone()),
        })
        .collect::<Result<_>>()?;
    let fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| match f.data_type() {
            DataType::Dictionary(_, v) => Field::new(f.name(), v.as_ref().clone(), f.is_nullable()),
            _ => f.as_ref().clone(),
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

impl std::fmt::Debug for ZarrAggregateExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ZarrAggregateExec(n_group={}, aggs={:?})",
            self.group_exprs.len(),
            self.aggs
        )
    }
}

impl DisplayAs for ZarrAggregateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let names: Vec<String> = self
            .aggs
            .iter()
            .map(|a| format!("{:?}({})", a.kind, a.column.as_deref().unwrap_or("*")))
            .collect();
        if self.group_exprs.is_empty() {
            write!(f, "ZarrAggregateExec: {}", names.join(", "))
        } else {
            let keys: Vec<String> = self.group_exprs.iter().map(|e| e.to_string()).collect();
            write!(
                f,
                "ZarrAggregateExec: {} GROUP BY [{}]",
                names.join(", "),
                keys.join(", ")
            )
        }
    }
}

impl ExecutionPlan for ZarrAggregateExec {
    fn name(&self) -> &str {
        "ZarrAggregateExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(ZarrAggregateExec::new(
            children[0].clone(),
            self.group_exprs.clone(),
            self.aggs.clone(),
            self.schema.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "ZarrAggregateExec has one output partition, got {partition}"
            )));
        }
        // Clone the small config into the fold future.
        let this = ZarrAggregateExec::new(
            self.input.clone(),
            self.group_exprs.clone(),
            self.aggs.clone(),
            self.schema.clone(),
        );
        let input = self.input.clone();
        let n_in = input.properties().output_partitioning().partition_count();

        let fut = async move {
            let mut groups: HashMap<Vec<ScalarValue>, Group> = HashMap::new();
            // A global aggregate always emits exactly one row, even over no data.
            if this.group_exprs.is_empty() {
                groups.insert(Vec::new(), this.new_group());
            }
            // TODO(phase7): partitions are folded sequentially, serializing the scan.
            // Fold each partition concurrently into its own map and merge to keep the
            // multi-partition (fan-out) scan parallelism.
            for p in 0..n_in {
                let mut stream = input.execute(p, context.clone())?;
                while let Some(batch) = stream.try_next().await? {
                    this.fold_batch(&mut groups, &batch)?;
                }
            }
            this.finalize(&groups)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}
