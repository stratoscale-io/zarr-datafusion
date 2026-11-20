//! Admission and grid derivation: from the physical plan feeding the sink, decide
//! whether it is a well-defined write and derive the target grid's *structure*.
//!
//! Phase 3 of docs/zarr-write-roundtrip-plan.md. The governing question (§5.8) is
//! *what is the admissible shape of the query feeding the sink?* — and the answer
//! must be **expressed in terms of `optimizer::cardinality`, not reinvented**.
//! Admission and derivation are the same act: a query is admissible *because* we
//! could derive its target grid; a query that cannot be lowered is exactly the one
//! that is rejected. So this is a fallible constructor, not a boolean gate.
//!
//! This module derives the grid *structure* — which source axes form the target
//! grid, and the data variables to write — with **no store I/O**. Materialising the
//! grid's coordinate *values* (loading the source coord arrays and gathering them
//! by any subset filter) is a downstream step that *consumes* a [`WriteShape`]; it
//! does not re-derive it, so there is still one implementation of "what is a legal
//! write" (the Q1 concern in the plan discussion).
//!
//! ## The rule that closes the §5.4 corruption hole
//!
//! A Zarr chunk is written whole, and the sink scatters rows by grid index. If a
//! source coordinate axis is neither **projected** (kept as an output coordinate)
//! nor **reduced** (grouped away by an aggregate), then many source rows collapse
//! onto the same target cell and the last write silently wins. So:
//!
//! - **Pure projection** (no `GROUP BY`): *every* source axis must appear as an
//!   output coordinate column. A missing axis is rejected as an ambiguous collapse.
//! - **Reduce** (`GROUP BY` on coordinate axes): the target grid is the group-key
//!   axes; the remaining axes are legitimately reduced. Admissible exactly when the
//!   aggregate is of pushable shape — which is `recognize()`'s existing decision,
//!   reused rather than duplicated. A periodic key, `date_part('hour', time)`, is an
//!   axis of its own, spanning the field's range (0 to 23 for `hour`), and each key
//!   is named by the output column it reaches, so `AS hour` names the axis.
//! - **Projection over a join with another Zarr store** (no `GROUP BY`): the
//!   probe side's store is the source, and the join must equate every coordinate
//!   of the other, so each source row meets at most one of its rows.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::JoinType;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::expressions::CastExpr;
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::PhysicalExpr;

use crate::optimizer::cardinality::pushdown::{coordinate_key, date_part_field, recognize};
use crate::optimizer::cardinality::GroupKey;
use crate::physical_plan::zarr_exec::ZarrExec;

use super::skeleton::WriteDataType;

/// A data variable the write will produce: its output name and element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteVar {
    pub name: String,
    pub data_type: WriteDataType,
}

/// The structural shape of a write derived from the plan feeding the sink.
///
/// Carries the target grid's axes (in dimension order — a subset/reorder of the
/// source axes) and the data variables to write. Coordinate *values* are not here:
/// materialising them is a downstream step that consumes this shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteShape {
    /// Target grid axis names, in source dimension order.
    pub grid_axes: Vec<String>,
    /// Source-axis index (into the store's coordinate list) for each grid axis,
    /// in the same order as `grid_axes`, or `None` for an axis of a joined
    /// table's column.
    pub grid_axis_source: Vec<Option<usize>>,
    /// For a grid axis of a joined table's column, the column, in the same order
    /// as `grid_axes`: the axis is its distinct values.
    pub grid_axis_table: Vec<Option<String>>,
    /// For a grid axis that is a periodic function of its source axis, the
    /// `date_part` field (`hour`, `month`, …), in the same order as `grid_axes`.
    pub grid_axis_field: Vec<Option<String>>,
    /// Data variables to write.
    pub data_vars: Vec<WriteVar>,
    /// True when the write reduces the source cube (the plan has a `GROUP BY`);
    /// false for a pure projection.
    pub is_reduce: bool,
}

/// Why a plan is not an admissible write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// No single-source `ZarrExec` reachable — e.g. a join, or a non-Zarr source.
    NotSingleZarrScan,
    /// The scan carries no store metadata, so axes cannot be resolved.
    NoStoreMeta,
    /// A pure projection dropped a source axis (neither projected nor reduced),
    /// which would collapse rows onto shared cells.
    DroppedAxis(String),
    /// The plan has an aggregate, but not of a pushable shape (`recognize` declined
    /// — e.g. a computed aggregate argument, `DISTINCT`, or a non-coordinate key).
    UnpushableAggregate,
    /// A global aggregate (no `GROUP BY`) yields a scalar, not a grid to write.
    ScalarAggregate,
    /// An output data column has a type with no writable Zarr mapping.
    UnwritableColumn { name: String, why: String },
    /// The output has no data variables to write.
    NoDataVariables,
    /// A group key is not in the output, so its rows have no place on the grid.
    KeyNotInOutput(String),
    /// A write over a join has no `GROUP BY`, so a cell may get several rows.
    JoinWithoutGroupBy,
    /// A group key names a column of both the scan and a joined table.
    AmbiguousKey(String),
    /// A group key of a write over a join is neither over the scan's coordinates
    /// nor a column of a joined table.
    UnplaceableKey(String),
}

impl fmt::Display for RejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSingleZarrScan => write!(
                f,
                "not a single-source Zarr scan (a write must read one Zarr store, \
                 joined only by inner joins to tables that read no Zarr)"
            ),
            Self::NoStoreMeta => write!(f, "the Zarr scan carries no store metadata"),
            Self::DroppedAxis(a) => write!(
                f,
                "coordinate axis '{a}' is neither selected nor grouped: without it \
                 in the output, multiple source rows collapse onto the same target \
                 cell (last write wins). Select it, or reduce it with GROUP BY."
            ),
            Self::UnpushableAggregate => write!(
                f,
                "the aggregate is not of a writable shape (a computed argument, \
                 DISTINCT, or a non-coordinate GROUP BY key)"
            ),
            Self::ScalarAggregate => write!(
                f,
                "a global aggregate produces a single scalar, not a grid to write; \
                 add a GROUP BY on coordinate axes"
            ),
            Self::UnwritableColumn { name, why } => {
                write!(f, "output column '{name}' cannot be written: {why}")
            }
            Self::NoDataVariables => write!(
                f,
                "the query outputs no data variables to write (only coordinates)"
            ),
            Self::KeyNotInOutput(key) => write!(
                f,
                "group key '{key}' is not in the output, so its rows have no place \
                 on the grid; select it"
            ),
            Self::JoinWithoutGroupBy => write!(
                f,
                "a write over a join needs a GROUP BY, unless it joins a Zarr store \
                 on every one of its coordinates: otherwise a join can give a cell \
                 several rows"
            ),
            Self::AmbiguousKey(key) => write!(
                f,
                "group key '{key}' is a column of both the Zarr scan and a joined \
                 table; rename one"
            ),
            Self::UnplaceableKey(key) => write!(
                f,
                "group key '{key}' of a write over a join is neither a coordinate (or \
                 a date_part of one) nor a column of a joined table"
            ),
        }
    }
}

impl std::error::Error for RejectReason {}

/// What a write reads: its one Zarr scan, the aggregate above it if any, and
/// the tables inner joins put beside it, which read no Zarr.
pub(crate) struct Source<'a> {
    pub agg: Option<&'a AggregateExec>,
    pub zarr: &'a ZarrExec,
    pub tables: Vec<&'a Arc<dyn ExecutionPlan>>,
    /// The join each of `tables` is joined by.
    pub joins: Vec<&'a Arc<dyn ExecutionPlan>>,
}

/// Walk down to the `ZarrExec`, capturing an `AggregateExec` on the way, through
/// single-child nodes and inner joins whose other side reads no Zarr. Returns
/// `None` (not admissible) at any other node with a number of children other
/// than one. Mirrors `cardinality::rule::descend_to_zarr`, but the aggregate is
/// optional so a pure projection is reachable too.
pub(crate) fn find_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<Source<'_>> {
    fn go<'a>(
        node: &'a Arc<dyn ExecutionPlan>,
        agg: Option<&'a AggregateExec>,
        mut tables: Vec<&'a Arc<dyn ExecutionPlan>>,
        mut joins: Vec<&'a Arc<dyn ExecutionPlan>>,
    ) -> Option<Source<'a>> {
        if let Some(zarr) = node.downcast_ref::<ZarrExec>() {
            return Some(Source {
                agg,
                zarr,
                tables,
                joins,
            });
        }
        let agg = node.downcast_ref::<AggregateExec>().or(agg);
        match node.children()[..] {
            [child] => go(child, agg, tables, joins),
            [left, right] if inner_join(node) => {
                let (scan, table) = match (reads_zarr(left), reads_zarr(right)) {
                    (true, false) => (left, right),
                    (false, true) => (right, left),
                    // the build side, as a table, if the probe's store is the grid
                    (true, true) if agg.is_none() => (right, left),
                    _ => return None,
                };
                tables.push(table);
                joins.push(node);
                go(scan, agg, tables, joins)
            }
            _ => None,
        }
    }
    go(plan, None, Vec::new(), Vec::new())
}

/// Whether `node` is a join that keeps only matching rows.
fn inner_join(node: &Arc<dyn ExecutionPlan>) -> bool {
    if let Some(join) = node.downcast_ref::<HashJoinExec>() {
        return *join.join_type() == JoinType::Inner;
    }
    if let Some(join) = node.downcast_ref::<NestedLoopJoinExec>() {
        return *join.join_type() == JoinType::Inner;
    }
    node.is::<CrossJoinExec>()
}

/// Whether `table` reads a Zarr store row by row, and `join` equates every one
/// of its coordinates, so a row of the other side meets at most one of its rows.
fn keyed_by_its_coordinates(join: &Arc<dyn ExecutionPlan>, table: &Arc<dyn ExecutionPlan>) -> bool {
    let Some(join) = join.downcast_ref::<HashJoinExec>() else {
        return false;
    };
    let Some(zarr) = row_scan(table) else {
        return false;
    };
    let Some(meta) = zarr.store_meta() else {
        return false;
    };
    let left = Arc::ptr_eq(join.left(), table);
    let keys: HashSet<String> = join
        .on()
        .iter()
        .filter_map(|(l, r)| scan_column(table, if left { l } else { r }))
        .collect();
    meta.coords.iter().all(|c| keys.contains(&c.name))
}

/// The Zarr scan `plan` reads, if each of its rows is one of the scan's.
fn row_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<&ZarrExec> {
    let mut node = plan;
    loop {
        if let Some(zarr) = node.downcast_ref::<ZarrExec>() {
            return Some(zarr);
        }
        let rowwise = node.is::<ProjectionExec>()
            || node.is::<FilterExec>()
            || node.is::<RepartitionExec>()
            || node.is::<CoalescePartitionsExec>()
            || node.is::<CooperativeExec>();
        match node.children()[..] {
            [child] if rowwise => node = child,
            _ => return None,
        }
    }
}

/// The scan column under `plan` whose values `expr`, over `plan`'s output, tells
/// apart: the column itself, or a cast of it that loses nothing.
fn scan_column(plan: &Arc<dyn ExecutionPlan>, expr: &Arc<dyn PhysicalExpr>) -> Option<String> {
    let mut node = plan;
    let mut expr = expr.clone();
    loop {
        let index = loop {
            if let Some(column) = expr.downcast_ref::<Column>() {
                break column.index();
            }
            let cast = expr.downcast_ref::<CastExpr>()?;
            let from = match cast.expr().data_type(&node.schema()).ok()? {
                // a coordinate is read as a dictionary of its values
                DataType::Dictionary(_, values) => *values,
                from => from,
            };
            let to = cast.cast_type();
            let lossless = from == *to
                || (from.is_integer() && matches!(to, DataType::Int64 | DataType::Float64))
                || (from.is_floating() && *to == DataType::Float64);
            if !lossless {
                return None;
            }
            expr = cast.expr().clone();
        };
        if node.is::<ZarrExec>() {
            return Some(node.schema().field(index).name().clone());
        }
        if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
            expr = projection.expr().get(index)?.expr.clone();
        } else if let Some(filter) = node.downcast_ref::<FilterExec>() {
            let index = filter.projection().as_ref().map_or(index, |p| p[index]);
            expr = Arc::new(Column::new(
                filter.input().schema().field(index).name(),
                index,
            ));
        } else if node.is::<RepartitionExec>()
            || node.is::<CoalescePartitionsExec>()
            || node.is::<CooperativeExec>()
        {
            expr = Arc::new(Column::new(node.schema().field(index).name(), index));
        } else {
            return None;
        }
        let [child] = node.children()[..] else {
            return None;
        };
        node = child;
    }
}

fn reads_zarr(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let mut found = false;
    let _ = plan.apply(|node| {
        found = node.is::<ZarrExec>();
        Ok(if found {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    });
    found
}

/// For each output column of `plan`, the column of the topmost aggregate's output
/// it carries unchanged, if any: through projections of a bare column, and nodes
/// that keep their input's columns.
fn aggregate_columns(plan: &Arc<dyn ExecutionPlan>) -> Vec<Option<usize>> {
    fn go(node: &Arc<dyn ExecutionPlan>, columns: Vec<Option<usize>>) -> Vec<Option<usize>> {
        if node.is::<AggregateExec>() {
            return columns;
        }
        let [child] = node.children()[..] else {
            return vec![None; columns.len()];
        };
        let columns = if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
            let exprs = projection.expr();
            columns
                .into_iter()
                .map(|c| Some(exprs[c?].expr.downcast_ref::<Column>()?.index()))
                .collect()
        } else if child.schema().fields() == node.schema().fields() {
            columns
        } else {
            vec![None; columns.len()]
        };
        go(child, columns)
    }
    let width = plan.schema().fields().len();
    go(plan, (0..width).map(Some).collect())
}

/// Derive the write's structural shape from the physical plan feeding the sink, or
/// reject it. See the module docs for the admission rule.
pub fn derive_write_shape(plan: &Arc<dyn ExecutionPlan>) -> Result<WriteShape, RejectReason> {
    let source = find_scan(plan).ok_or(RejectReason::NotSingleZarrScan)?;
    let keyed = source.agg.is_none()
        && source
            .tables
            .iter()
            .zip(&source.joins)
            .all(|(table, join)| keyed_by_its_coordinates(join, table));
    if !source.tables.is_empty() && !keyed {
        return joined_write_shape(plan, &source);
    }
    let (maybe_agg, zarr) = (source.agg, source.zarr);
    let meta = zarr.store_meta().ok_or(RejectReason::NoStoreMeta)?;

    // Source axes, in dimension order, with a name -> index lookup.
    let axis_names: Vec<&str> = meta.coords.iter().map(|c| c.name.as_str()).collect();
    let axis_index = |name: &str| axis_names.iter().position(|a| *a == name);

    let out = plan.schema();

    // A reduce's grid axes, one per group key, each named by the output column it
    // reaches, in source dimension order.
    let mut keys: Vec<(usize, String, Option<String>)> = Vec::new();
    if let Some(agg) = maybe_agg {
        let cand = recognize(agg, zarr).ok_or(RejectReason::UnpushableAggregate)?;
        if cand.group_keys.is_empty() {
            return Err(RejectReason::ScalarAggregate);
        }
        let reached = aggregate_columns(plan);
        let exprs = agg.group_expr().expr();
        for (k, key) in cand.group_keys.iter().enumerate() {
            let name = reached
                .iter()
                .position(|c| *c == Some(k))
                .map(|j| out.field(j).name().clone())
                .ok_or_else(|| RejectReason::KeyNotInOutput(cand.group_names[k].clone()))?;
            let field = match key {
                GroupKey::Axis(_) => None,
                GroupKey::Periodic { .. } => date_part_field(&exprs[k].0),
            };
            keys.push((key.axis(), name, field));
        }
        // The coordinate list follows the store's listing, not the data's dimensions.
        let dims = meta.data_vars.first().and_then(|v| v.dimensions.as_ref());
        keys.sort_by_key(|(axis, _, _)| {
            dims.and_then(|d| d.iter().position(|n| *n == axis_names[*axis]))
                .unwrap_or(*axis)
        });
    }
    let is_key = |name: &str| keys.iter().any(|(_, key, _)| key == name);

    let data_vars = data_vars(&out, |name| {
        if maybe_agg.is_some() {
            is_key(name)
        } else {
            axis_index(name).is_some()
        }
    })?;

    if maybe_agg.is_some() {
        let (grid_axis_source, grid_axes, grid_axis_field) = keys.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut sources, mut names, mut fields), (source, name, field)| {
                sources.push(Some(source));
                names.push(name);
                fields.push(field);
                (sources, names, fields)
            },
        );
        return Ok(WriteShape {
            grid_axis_table: vec![None; grid_axes.len()],
            grid_axes,
            grid_axis_source,
            grid_axis_field,
            data_vars,
            is_reduce: true,
        });
    }

    // Pure projection: every source axis must be present as an output coordinate,
    // or rows collapse (§5.4).
    let projected: std::collections::HashSet<&str> = out
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .filter(|n| axis_index(n).is_some())
        .collect();
    for name in &axis_names {
        if !projected.contains(name) {
            return Err(RejectReason::DroppedAxis(name.to_string()));
        }
    }

    Ok(WriteShape {
        grid_axes: axis_names.iter().map(|a| a.to_string()).collect(),
        grid_axis_source: (0..axis_names.len()).map(Some).collect(),
        grid_axis_table: vec![None; axis_names.len()],
        grid_axis_field: vec![None; axis_names.len()],
        data_vars,
        is_reduce: false,
    })
}

/// Every output column that isn't a coordinate is a data variable to write. Its
/// element type is the widened Arrow type (gap 5).
fn data_vars(
    out: &Schema,
    coordinate: impl Fn(&str) -> bool,
) -> Result<Vec<WriteVar>, RejectReason> {
    let mut vars = Vec::new();
    for field in out.fields() {
        if coordinate(field.name()) {
            continue;
        }
        let data_type = WriteDataType::from_arrow(field.data_type()).map_err(|e| {
            RejectReason::UnwritableColumn {
                name: field.name().clone(),
                why: e.to_string(),
            }
        })?;
        vars.push(WriteVar {
            name: field.name().clone(),
            data_type,
        });
    }
    if vars.is_empty() {
        return Err(RejectReason::NoDataVariables);
    }
    Ok(vars)
}

/// The shape of a write over inner joins of the scan with tables that read no
/// Zarr. It must reduce, and each group key is an axis: over the scan's
/// coordinates as for a single scan, in source dimension order, then each a
/// joined table's column, in `GROUP BY` order, whose values are the column's.
/// Any aggregate is admissible: a `GROUP BY` gives each key tuple, so each
/// cell, one row.
fn joined_write_shape(
    plan: &Arc<dyn ExecutionPlan>,
    source: &Source<'_>,
) -> Result<WriteShape, RejectReason> {
    let agg = source.agg.ok_or(RejectReason::JoinWithoutGroupBy)?;
    let meta = source.zarr.store_meta().ok_or(RejectReason::NoStoreMeta)?;
    let coord_axis: HashMap<&str, usize> = meta
        .coords
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.as_str(), i))
        .collect();
    let scan_columns: HashSet<String> = source
        .zarr
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let table_columns: HashSet<String> = source
        .tables
        .iter()
        .flat_map(|table| {
            table
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>()
        })
        .collect();

    let exprs = agg.group_expr().expr();
    if exprs.is_empty() {
        return Err(RejectReason::ScalarAggregate);
    }
    let out = plan.schema();
    let reached = aggregate_columns(plan);
    let mut scan_keys: Vec<(usize, String, Option<String>)> = Vec::new();
    let mut table_keys: Vec<(String, String)> = Vec::new();
    for (k, (expr, group_name)) in exprs.iter().enumerate() {
        let name = reached
            .iter()
            .position(|c| *c == Some(k))
            .map(|j| out.field(j).name().clone())
            .ok_or_else(|| RejectReason::KeyNotInOutput(group_name.clone()))?;
        let column = expr.downcast_ref::<Column>().map(|c| c.name().to_string());
        match column {
            Some(column) if table_columns.contains(&column) => {
                if scan_columns.contains(&column) {
                    return Err(RejectReason::AmbiguousKey(column));
                }
                table_keys.push((name, column));
            }
            _ => {
                let key = coordinate_key(expr, &coord_axis)
                    .ok_or_else(|| RejectReason::UnplaceableKey(group_name.clone()))?;
                let field = match key {
                    GroupKey::Axis(_) => None,
                    GroupKey::Periodic { .. } => date_part_field(expr),
                };
                scan_keys.push((key.axis(), name, field));
            }
        }
    }
    let dims = meta.data_vars.first().and_then(|v| v.dimensions.as_ref());
    scan_keys.sort_by_key(|(axis, _, _)| {
        dims.and_then(|d| d.iter().position(|n| *n == meta.coords[*axis].name))
            .unwrap_or(*axis)
    });

    let is_key = |name: &str| {
        scan_keys.iter().any(|(_, key, _)| key == name)
            || table_keys.iter().any(|(key, _)| key == name)
    };
    let data_vars = data_vars(&out, is_key)?;
    let tables = table_keys.len();
    Ok(WriteShape {
        grid_axes: scan_keys
            .iter()
            .map(|(_, name, _)| name.clone())
            .chain(table_keys.iter().map(|(name, _)| name.clone()))
            .collect(),
        grid_axis_source: scan_keys
            .iter()
            .map(|(axis, _, _)| Some(*axis))
            .chain(std::iter::repeat_n(None, tables))
            .collect(),
        grid_axis_field: scan_keys
            .iter()
            .map(|(_, _, field)| field.clone())
            .chain(std::iter::repeat_n(None, tables))
            .collect(),
        grid_axis_table: std::iter::repeat_n(None, scan_keys.len())
            .chain(table_keys.into_iter().map(|(_, column)| Some(column)))
            .collect(),
        data_vars,
        is_reduce: true,
    })
}
