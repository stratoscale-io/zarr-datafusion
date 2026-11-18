//! Recognizing aggregate-pushdown opportunities (Phase 7.2, observe-only).
//!
//! An `AGG(data_var) [GROUP BY coord]` directly over a `ZarrExec` can, in principle,
//! be folded chunk-by-chunk during the scan instead of materializing the flattened
//! selection — *if* the number of output groups is known to fit in memory. This
//! module implements the **recognizer**: it inspects a physical `AggregateExec` over
//! a `ZarrExec`, decides whether the aggregates and grouping are of a pushable shape,
//! and (in Phase 7.2) only *reports* the opportunity. The rewrite lands in 7.3.
//!
//! Pushable shape (this phase):
//! - aggregates are `SUM`/`COUNT`/`AVG`/`MIN`/`MAX`, none `DISTINCT`;
//! - each aggregate argument is a **bare data-variable column**, modulo casts (or
//!   `COUNT(*)`/`COUNT(coord)`). An argument that *computes* over an array, such as
//!   `AVG(sst - 273.15)`, is declined: `AggSpec` can only name an array, so pushing
//!   it would drop the arithmetic and silently return `AVG(sst)`;
//! - `GROUP BY` is empty, or on **coordinate columns** (each mapped to a cube axis).
//!   Grouping by a periodic function of a coordinate (e.g. month-of-time) is deferred
//!   to Phase 7.5.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::expressions::{CastExpr, Column, Literal, TryCastExpr};
use datafusion::scalar::ScalarValue;

use super::{group_cardinality, AxisSet, GroupKey, ProductSet};
use crate::physical_plan::zarr_exec::ZarrExec;
use crate::reader::schema_inference::ZarrStoreMeta;

/// A pushable aggregate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggKind {
    Sum,
    Count,
    Avg,
    Min,
    Max,
}

/// One aggregate to push: its function and the data-variable column it reads
/// (`None` for `COUNT(*)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggSpec {
    pub kind: AggKind,
    pub column: Option<String>,
}

/// A recognized aggregate-pushdown opportunity over a `ZarrExec` scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushdownCandidate {
    pub aggs: Vec<AggSpec>,
    pub group_keys: Vec<GroupKey>,
    /// Coordinate names behind `group_keys`, in order — for logging/EXPLAIN.
    pub group_names: Vec<String>,
}

/// Decide whether `agg` directly over `zarr` is a pushable-shape aggregate.
///
/// Returns `None` (decline) when the store metadata is absent, any aggregate is
/// `DISTINCT` or an unsupported function, an aggregate reads a coordinate (other
/// than `COUNT`), or a `GROUP BY` key isn't a bare coordinate column. Pure and
/// side-effect-free — the caller decides what to do with a recognized candidate.
pub fn recognize(agg: &AggregateExec, zarr: &ZarrExec) -> Option<PushdownCandidate> {
    let meta = zarr.store_meta()?;
    let coord_axis: HashMap<&str, usize> = meta
        .coords
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.as_str(), i))
        .collect();
    let data_vars: HashSet<&str> = meta.data_vars.iter().map(|d| d.name.as_str()).collect();

    // --- aggregate functions ---
    let mut aggs = Vec::new();
    for a in agg.aggr_expr() {
        if a.is_distinct() {
            return None;
        }
        let kind = match a.fun().name() {
            "sum" => AggKind::Sum,
            "count" => AggKind::Count,
            "avg" => AggKind::Avg,
            "min" => AggKind::Min,
            "max" => AggKind::Max,
            _ => return None,
        };
        // An argument that is a computation over an array (e.g. `sst - 273.15`) has no
        // representation in `AggSpec`; decline so DataFusion evaluates it correctly.
        let column = match aggregate_arg(&a.expressions()) {
            AggArg::Star => None,
            AggArg::Column(name) => Some(name),
            AggArg::Unsupported => return None,
        };
        let ok = match (&kind, &column) {
            // COUNT(*) / COUNT(coord) / COUNT(data_var) all just count rows.
            (AggKind::Count, _) => true,
            // Other aggregates must read a data variable, not a coordinate.
            (_, Some(name)) => data_vars.contains(name.as_str()),
            (_, None) => false,
        };
        if !ok {
            return None;
        }
        aggs.push(AggSpec { kind, column });
    }
    if aggs.is_empty() {
        return None;
    }

    // --- grouping ---
    // Each group key must be a bare coordinate column (→ Axis) or a periodic function
    // of one, e.g. `date_part('month', time)` (→ Periodic). Anything else declines.
    let mut group_keys = Vec::new();
    let mut group_names = Vec::new();
    for (expr, name) in agg.group_expr().expr() {
        let key = column_axis(expr, &coord_axis)
            .map(GroupKey::Axis)
            .or_else(|| {
                date_part_periodic(expr, &coord_axis)
                    .map(|(axis, period)| GroupKey::Periodic { axis, period })
            })?;
        group_keys.push(key);
        group_names.push(name.clone());
    }

    Some(PushdownCandidate {
        aggs,
        group_keys,
        group_names,
    })
}

/// The cube axis of a group key that is a bare coordinate column.
fn column_axis(expr: &Arc<dyn PhysicalExpr>, coord_axis: &HashMap<&str, usize>) -> Option<usize> {
    let col = expr.downcast_ref::<Column>()?;
    coord_axis.get(col.name()).copied()
}

/// A group key of the form `date_part('<field>', <coord>)`, `<coord>` perhaps cast,
/// → `(axis, period)`, where
/// `period` bounds the distinct values (month→12, hour→24, …). `year` and unknown
/// fields are *not* periodic and decline.
fn date_part_periodic(
    expr: &Arc<dyn PhysicalExpr>,
    coord_axis: &HashMap<&str, usize>,
) -> Option<(usize, u64)> {
    let sf = expr.downcast_ref::<ScalarFunctionExpr>()?;
    if sf.fun().name() != "date_part" {
        return None;
    }
    let args = sf.args();
    if args.len() != 2 {
        return None;
    }
    let period = period_of(&literal_str(&args[0])?)?;
    // Through a cast: `date_part` refuses a dictionary-encoded time unless cast.
    let column = bare_column(&args[1])?;
    let axis = coord_axis.get(column.as_str()).copied()?;
    Some((axis, period))
}

/// The field of a periodic `date_part` group key, such as `hour`, lowercased.
pub(crate) fn date_part_field(expr: &Arc<dyn PhysicalExpr>) -> Option<String> {
    let sf = expr.downcast_ref::<ScalarFunctionExpr>()?;
    if sf.fun().name() != "date_part" {
        return None;
    }
    let field = literal_str(sf.args().first()?)?.to_ascii_lowercase();
    period_of(&field).map(|_| field)
}

/// The string value of a UTF-8 literal expression, if it is one.
fn literal_str(expr: &Arc<dyn PhysicalExpr>) -> Option<String> {
    match expr.downcast_ref::<Literal>()?.value() {
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => Some(s.clone()),
        _ => None,
    }
}

/// The period (max distinct values) of a `date_part` field, or `None` if the field
/// is unbounded (`year`) or unrecognized.
fn period_of(field: &str) -> Option<u64> {
    let (first, last) = field_range(field)?;
    Some((last - first + 1) as u64)
}

/// The first and last values of a periodic `date_part` field, as DataFusion
/// numbers them.
pub(crate) fn field_range(field: &str) -> Option<(i64, i64)> {
    Some(match field.to_ascii_lowercase().as_str() {
        "month" => (1, 12),
        "day" => (1, 31),
        "hour" => (0, 23),
        "minute" => (0, 59),
        "second" => (0, 59),
        "quarter" => (1, 4),
        "doy" | "dayofyear" => (1, 366),
        "dow" | "dayofweek" => (0, 6),
        "week" => (1, 53),
        _ => return None,
    })
}

/// The whole cube as an unfiltered selection — its group count is the *maximum*
/// possible, so `group_cardinality(universe, keys) <= budget` guarantees any
/// filtered subset fits too. Metadata-only (no coordinate reads).
pub fn universe(meta: &ZarrStoreMeta) -> ProductSet {
    let axes = meta
        .coords
        .iter()
        .map(|c| AxisSet::interval(0, c.shape.first().copied().unwrap_or(0)))
        .collect();
    ProductSet::single(axes)
}

/// Upper bound on the output group count for `cand` over `zarr`'s cube — the group
/// count of the full (unfiltered) cube under the candidate's keys.
pub fn max_group_count(cand: &PushdownCandidate, meta: &ZarrStoreMeta) -> u128 {
    group_cardinality(&universe(meta), &cand.group_keys)
}

/// What an aggregate reads, as far as `AggSpec` is able to represent it.
///
/// `AggSpec` can only say "apply this function to this array", so an argument that
/// is a *computation* over an array has no representation here and must decline the
/// pushdown — see [`aggregate_arg`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum AggArg {
    /// No value argument: `COUNT(*)`, which lowers to `count(Int64(1))`.
    Star,
    /// A bare column, possibly wrapped in casts.
    Column(String),
    /// An expression we cannot fold chunk-wise (e.g. `sst - 273.15`).
    Unsupported,
}

/// The column an aggregate reads, searching *through* the casts that DataFusion
/// inserts (`AVG(temperature)` lowers to `avg(CAST(temperature AS Float64))`).
///
/// Casts are the *only* wrapper we see through. Descending through arbitrary
/// expressions would let `AVG(sst - 273.15)` be recognized as `Avg(sst)` and
/// silently drop the arithmetic, since the rewritten node inherits the original
/// aggregate's schema and the projection above it merely relabels column 0.
fn aggregate_arg(exprs: &[Arc<dyn PhysicalExpr>]) -> AggArg {
    match exprs {
        // No argument at all — a genuine `COUNT(*)`.
        [] => AggArg::Star,
        [e] => match bare_column(e) {
            Some(name) => AggArg::Column(name),
            // `COUNT(*)` lowers to a literal argument; any other expression is
            // a computation we cannot represent.
            None if e.downcast_ref::<Literal>().is_some() => AggArg::Star,
            None => AggArg::Unsupported,
        },
        // Multi-argument aggregates are not pushable.
        _ => AggArg::Unsupported,
    }
}

/// The name of a bare column, seeing through cast wrappers only.
fn bare_column(e: &Arc<dyn PhysicalExpr>) -> Option<String> {
    if let Some(c) = e.downcast_ref::<Column>() {
        return Some(c.name().to_string());
    }
    if let Some(c) = e.downcast_ref::<CastExpr>() {
        return bare_column(c.expr());
    }
    if let Some(c) = e.downcast_ref::<TryCastExpr>() {
        return bare_column(c.expr());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::schema_inference::{ZarrArrayMeta, ZarrStoreMeta};

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

    fn meta() -> ZarrStoreMeta {
        // Cube axes: [time, lat, lon]; one data variable `temperature`.
        ZarrStoreMeta {
            total_rows: 0,
            coords: vec![coord("time", 12), coord("lat", 5), coord("lon", 7)],
            data_vars: vec![ZarrArrayMeta {
                name: "temperature".into(),
                data_type: "float32".into(),
                shape: vec![12, 5, 7],
                chunks: Some(vec![1, 5, 7]),
                coord_min_max: None,
                cf_time_attrs: None,
                dimensions: Some(vec!["time".into(), "lat".into(), "lon".into()]),
            }],
        }
    }

    #[test]
    fn universe_group_count_matches_axis_extents() {
        let m = meta();
        // GROUP BY lat -> 5 groups; GROUP BY lat, lon -> 35; global -> 1.
        assert_eq!(group_cardinality(&universe(&m), &[GroupKey::Axis(1)]), 5);
        assert_eq!(
            group_cardinality(&universe(&m), &[GroupKey::Axis(1), GroupKey::Axis(2)]),
            35
        );
        assert_eq!(group_cardinality(&universe(&m), &[]), 1);
    }
}
