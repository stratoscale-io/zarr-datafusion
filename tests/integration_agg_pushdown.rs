//! Phase 7.2 — aggregate-pushdown *recognition* (observe-only).
//!
//! These drive real SQL through DataFusion to a physical plan, locate the
//! `AggregateExec` sitting directly over the `ZarrExec`, and assert what the
//! recognizer makes of it. No rewrite happens yet (7.3); this pins the matcher's
//! accept/decline behavior.

mod common;

use common::*;

use std::sync::Arc;

use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

use zarr_datafusion::optimizer::cardinality::pushdown::{recognize, AggKind};
use zarr_datafusion::optimizer::cardinality::GroupKey;
use zarr_datafusion::optimizer::CardinalityRule;
use zarr_datafusion::physical_plan::zarr_aggregate::ZarrAggregateExec;
use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;

/// Find the first `AggregateExec` whose input is a `ZarrExec`.
fn find_agg_over_zarr(plan: &Arc<dyn ExecutionPlan>) -> Option<(&AggregateExec, &ZarrExec)> {
    if let Some(agg) = plan.downcast_ref::<AggregateExec>() {
        if let Some(z) = agg.input().downcast_ref::<ZarrExec>() {
            return Some((agg, z));
        }
    }
    for c in plan.children() {
        if let Some(found) = find_agg_over_zarr(c) {
            return Some(found);
        }
    }
    None
}

async fn plan_for(sql: &str) -> Arc<dyn ExecutionPlan> {
    // Plain context (no stats rules) so MIN/MAX/COUNT aren't answered logically and
    // a real AggregateExec survives into the physical plan.
    let ctx = SessionContext::new();
    register_zarr_table(&ctx, "t", SYNTHETIC_V3);
    get_physical_plan(&ctx, sql).await
}

#[tokio::test]
async fn recognizes_global_sum_over_data_var() {
    let plan = plan_for("SELECT SUM(temperature) AS s FROM t").await;
    let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
    let cand = recognize(agg, z).expect("recognized as pushable");
    assert_eq!(cand.aggs.len(), 1);
    assert_eq!(cand.aggs[0].kind, AggKind::Sum);
    assert_eq!(cand.aggs[0].column.as_deref(), Some("temperature"));
    assert!(cand.group_keys.is_empty(), "no GROUP BY => global aggregate");
}

#[tokio::test]
async fn recognizes_group_by_coordinate() {
    // Synthetic cube axes are [time, lat, lon]; GROUP BY lat => axis 1.
    let plan = plan_for("SELECT lat, AVG(temperature) AS a FROM t GROUP BY lat").await;
    let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
    let cand = recognize(agg, z).expect("recognized as pushable");
    assert_eq!(cand.aggs[0].kind, AggKind::Avg);
    assert_eq!(cand.group_keys, vec![GroupKey::Axis(1)]);
    assert_eq!(cand.group_names, vec!["lat".to_string()]);
}

#[tokio::test]
async fn declines_distinct_aggregate() {
    let plan = plan_for("SELECT COUNT(DISTINCT temperature) AS c FROM t").await;
    let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
    assert!(recognize(agg, z).is_none(), "DISTINCT is not pushable");
}

#[tokio::test]
async fn declines_non_count_aggregate_over_coordinate() {
    // MAX over a coordinate (not a data variable) isn't a pushable data aggregate.
    let plan = plan_for("SELECT MAX(lat) AS m FROM t").await;
    let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
    assert!(recognize(agg, z).is_none());
}

#[tokio::test]
async fn declines_group_by_data_variable() {
    // Grouping by a data variable has no cube-axis mapping.
    let plan = plan_for("SELECT temperature, COUNT(*) AS c FROM t GROUP BY temperature").await;
    let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
    assert!(recognize(agg, z).is_none());
}

// ---- Phase 7.3: global aggregate pushdown (rewrite + equivalence) ----

/// A context with the cardinality rule registered — global aggregates get pushed.
fn ctx_pushdown() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(CardinalityRule::new()))
        .build();
    SessionContext::new_with_state(state)
}

fn contains_node<T: ExecutionPlan + 'static>(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan.downcast_ref::<T>().is_some() {
        return true;
    }
    plan.children().iter().any(|c| contains_node::<T>(c))
}

async fn run(ctx: &SessionContext, sql: &str) -> String {
    run_on(ctx, SYNTHETIC_V3, sql).await
}

async fn run_on(ctx: &SessionContext, path: &str, sql: &str) -> String {
    register_zarr_table(ctx, "t", path);
    let batches = execute_query(ctx, sql).await;
    // Compare on the rendered table: `pretty_format_batches` decodes dictionary
    // columns to their values (so different dict encodings compare equal) and is
    // independent of how rows are split across batches. Zero rows (empty grouped
    // selection: 0 batches vs. one 0-row batch) canonicalizes to a single token.
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    if total == 0 {
        return "EMPTY".to_string();
    }
    arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn global_aggregate_rewritten_to_zarr_aggregate() {
    let ctx = ctx_pushdown();
    register_zarr_table(&ctx, "t", SYNTHETIC_V3);
    let plan = get_physical_plan(&ctx, "SELECT SUM(temperature) AS s FROM t").await;
    assert!(
        contains_node::<ZarrAggregateExec>(&plan),
        "pushdown should introduce ZarrAggregateExec"
    );
    // The scan is now wrapped by ZarrAggregateExec — no bare AggregateExec remains.
    assert!(
        !contains_node::<AggregateExec>(&plan),
        "the AggregateExec should have been replaced"
    );
}

#[tokio::test]
async fn global_aggregates_are_value_equivalent() {
    let queries = [
        "SELECT SUM(temperature) AS v FROM t",
        "SELECT COUNT(*) AS v FROM t",
        "SELECT COUNT(temperature) AS v FROM t",
        "SELECT AVG(temperature) AS v FROM t",
        "SELECT MIN(temperature) AS v FROM t",
        "SELECT MAX(temperature) AS v FROM t",
        "SELECT SUM(temperature) AS a, COUNT(*) AS b, AVG(temperature) AS c FROM t",
        // With a coordinate filter (narrows the read).
        "SELECT SUM(temperature) AS v FROM t WHERE lat = -10.0",
        // Empty selection — aggregates of nothing.
        "SELECT SUM(temperature) AS s, COUNT(*) AS c, MIN(temperature) AS m FROM t WHERE time = 99999",
    ];
    for sql in queries {
        let baseline = run(&SessionContext::new(), sql).await;
        let pushed = run(&ctx_pushdown(), sql).await;
        assert_eq!(baseline, pushed, "value mismatch for `{sql}`");
    }
}

// ---- Regression: aggregate arguments that COMPUTE over an array ----
//
// `AggSpec` can only name an array, so an argument like `temperature - 273.15` has no
// representation in a pushed aggregate. The recognizer used to search *through* any
// wrapper expression for the first `Column`, so it accepted these as `Avg(temperature)`
// and silently dropped the arithmetic — the rewritten node inherits the original
// aggregate's schema, and the projection above it only relabels column 0, so the wrong
// value flowed out under the right name with no error. Unit conversions (K->degC,
// m->mm) are the common form, which made this a routine query returning a plausible
// number in the wrong unit.

/// Queries whose aggregate argument is a computation, not a bare column.
const EXPRESSION_ARG_QUERIES: [&str; 5] = [
    "SELECT AVG(temperature - 273.15) AS v FROM t",
    "SELECT SUM(temperature * 2.0) AS v FROM t",
    "SELECT MIN(temperature + 1.0) AS v FROM t",
    "SELECT MAX(temperature / 2.0) AS v FROM t",
    "SELECT COUNT(temperature - 273.15) AS v FROM t",
];

#[tokio::test]
async fn declines_aggregate_over_expression_argument() {
    for sql in EXPRESSION_ARG_QUERIES {
        let plan = plan_for(sql).await;
        let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
        assert!(
            recognize(agg, z).is_none(),
            "must decline a computed argument: `{sql}`"
        );
    }
}

#[tokio::test]
async fn expression_argument_is_not_pushed() {
    for sql in EXPRESSION_ARG_QUERIES {
        let ctx = ctx_pushdown();
        register_zarr_table(&ctx, "t", SYNTHETIC_V3);
        let plan = get_physical_plan(&ctx, sql).await;
        assert!(
            !contains_node::<ZarrAggregateExec>(&plan),
            "a computed argument must not be pushed: `{sql}`"
        );
        assert!(
            contains_node::<AggregateExec>(&plan),
            "it must fall back to DataFusion's aggregate: `{sql}`"
        );
    }
}

#[tokio::test]
async fn expression_argument_aggregates_are_value_equivalent() {
    for sql in EXPRESSION_ARG_QUERIES {
        let baseline = run(&SessionContext::new(), sql).await;
        let pushed = run(&ctx_pushdown(), sql).await;
        assert_eq!(baseline, pushed, "value mismatch for `{sql}`");
    }
}

#[tokio::test]
async fn unit_conversion_is_not_silently_dropped() {
    // The exact shape that failed against ARCO-ERA5: `AVG(sst - 273.15)` returned the
    // Kelvin mean. Guard the *semantics*, not just baseline agreement — if the offset
    // were dropped again, these two would coincide.
    let converted = run(
        &ctx_pushdown(),
        "SELECT AVG(temperature - 273.15) AS v FROM t",
    )
    .await;
    let raw = run(&ctx_pushdown(), "SELECT AVG(temperature) AS v FROM t").await;
    assert_ne!(
        converted, raw,
        "`- 273.15` was dropped: the aggregate ignored its argument expression"
    );
}

#[tokio::test]
async fn bare_and_cast_wrapped_columns_still_push() {
    // Tightening the recognizer must not cost the optimization it exists for:
    // `AVG(temperature)` lowers to `avg(CAST(temperature AS Float64))`, and seeing
    // through that cast is exactly why `bare_column` recurses at all.
    for sql in [
        "SELECT AVG(temperature) AS v FROM t",
        "SELECT SUM(temperature) AS v FROM t",
        "SELECT COUNT(*) AS v FROM t",
        "SELECT COUNT(temperature) AS v FROM t",
    ] {
        let ctx = ctx_pushdown();
        register_zarr_table(&ctx, "t", SYNTHETIC_V3);
        let plan = get_physical_plan(&ctx, sql).await;
        assert!(
            contains_node::<ZarrAggregateExec>(&plan),
            "a bare/cast column must still push: `{sql}`"
        );
    }
}

// ---- Phase 7.4: GROUP BY coordinate axis ----

#[tokio::test]
async fn group_by_coordinate_rewritten_to_zarr_aggregate() {
    let ctx = ctx_pushdown();
    register_zarr_table(&ctx, "t", SYNTHETIC_V3);
    let plan = get_physical_plan(&ctx, "SELECT lat, AVG(temperature) AS a FROM t GROUP BY lat").await;
    assert!(contains_node::<ZarrAggregateExec>(&plan));
    assert!(!contains_node::<AggregateExec>(&plan));
}

#[tokio::test]
async fn group_by_coordinate_is_value_equivalent() {
    // ORDER BY makes the (otherwise unordered) grouped output deterministic so the
    // pushed and baseline results compare directly.
    let queries = [
        "SELECT lat, COUNT(*) AS c FROM t GROUP BY lat ORDER BY lat",
        "SELECT lat, AVG(temperature) AS a FROM t GROUP BY lat ORDER BY lat",
        "SELECT lat, SUM(temperature) AS s, MIN(temperature) AS mn, MAX(temperature) AS mx \
         FROM t GROUP BY lat ORDER BY lat",
        // Two grouping coordinates (100 groups).
        "SELECT lat, lon, AVG(temperature) AS a FROM t GROUP BY lat, lon ORDER BY lat, lon",
        // Group by the outer coordinate under a filter on another.
        "SELECT time, SUM(temperature) AS s FROM t WHERE lat = 5 GROUP BY time ORDER BY time",
        // Empty grouped selection => zero groups on both sides.
        "SELECT lat, COUNT(*) AS c FROM t WHERE time = 99999 GROUP BY lat ORDER BY lat",
    ];
    for sql in queries {
        let baseline = run(&SessionContext::new(), sql).await;
        let pushed = run(&ctx_pushdown(), sql).await;
        assert_eq!(baseline, pushed, "value mismatch for `{sql}`");
    }
}

// ---- Phase 7.5: periodic GROUP BY (date_part) — pushdown as an enabler ----

const MONTHLY: &str = "data/monthly_v3.zarr"; // 6 monthly timestamps, temp = 0,10..50

#[tokio::test]
async fn periodic_group_by_rewritten_to_zarr_aggregate() {
    let ctx = ctx_pushdown();
    register_zarr_table(&ctx, "t", MONTHLY);
    let plan = get_physical_plan(
        &ctx,
        "SELECT EXTRACT(month FROM time) AS m, SUM(temperature) AS s FROM t GROUP BY EXTRACT(month FROM time)",
    )
    .await;
    assert!(contains_node::<ZarrAggregateExec>(&plan));
    assert!(!contains_node::<AggregateExec>(&plan));
}

#[tokio::test]
async fn periodic_group_by_enables_a_query_datafusion_cannot_run() {
    // (1) Enabler: baseline DataFusion cannot GROUP BY date_part over a dictionary
    //     coordinate — it asserts on the dictionary return type.
    let base = SessionContext::new();
    register_zarr_table(&base, "t", MONTHLY);
    let baseline = base
        .sql("SELECT EXTRACT(month FROM time) m, SUM(temperature) s FROM t GROUP BY EXTRACT(month FROM time)")
        .await
        .unwrap()
        .collect()
        .await;
    assert!(
        baseline.is_err(),
        "baseline is expected to fail on date_part over a dict coordinate"
    );

    // (2) Anchor: GROUP BY the raw time coordinate IS baseline-valid, and pushdown
    //     matches it — so time-grouped aggregates are a trusted reference.
    let aggs_by_time = "SELECT SUM(temperature) AS s, COUNT(*) AS c, AVG(temperature) AS a, \
                        MIN(temperature) AS mn, MAX(temperature) AS mx \
                        FROM t GROUP BY time ORDER BY time";
    let by_time_base = run_on(&SessionContext::new(), MONTHLY, aggs_by_time).await;
    let by_time_push = run_on(&ctx_pushdown(), MONTHLY, aggs_by_time).await;
    assert_eq!(by_time_base, by_time_push, "GROUP BY time must be equivalent");

    // (3) Correctness: each month has exactly one timestamp in this fixture, so the
    //     month-grouped aggregates must equal the time-grouped ones (same order).
    let aggs_by_month = "SELECT SUM(temperature) AS s, COUNT(*) AS c, AVG(temperature) AS a, \
                         MIN(temperature) AS mn, MAX(temperature) AS mx \
                         FROM t GROUP BY EXTRACT(month FROM time) ORDER BY EXTRACT(month FROM time)";
    let by_month_push = run_on(&ctx_pushdown(), MONTHLY, aggs_by_month).await;
    assert_eq!(
        by_month_push, by_time_push,
        "periodic-grouped aggregates must match the time-grouped reference"
    );
}

#[tokio::test]
async fn periodic_group_by_of_a_cast_time_is_pushed_with_the_same_answer() {
    let sql = "SELECT date_part('month', CAST(time AS TIMESTAMP)) AS m, SUM(temperature) AS s \
               FROM t GROUP BY date_part('month', CAST(time AS TIMESTAMP)) ORDER BY m";
    let ctx = ctx_pushdown();
    register_zarr_table(&ctx, "t", MONTHLY);
    let plan = get_physical_plan(&ctx, sql).await;
    assert!(contains_node::<ZarrAggregateExec>(&plan));

    let baseline = run_on(&SessionContext::new(), MONTHLY, sql).await;
    let pushed = run_on(&ctx_pushdown(), MONTHLY, sql).await;
    assert_eq!(baseline, pushed);
}

#[tokio::test]
async fn periodic_group_single_month_matches_global() {
    // era5_v3 is entirely January, so GROUP BY month yields one group whose aggregates
    // equal the (baseline-valid) global aggregate — a cross-check on real data.
    let global = "SELECT SUM(temperature) AS s, COUNT(*) AS c FROM t";
    let by_month = "SELECT SUM(temperature) AS s, COUNT(*) AS c FROM t GROUP BY EXTRACT(month FROM time)";
    let global_base = run_on(&SessionContext::new(), ERA5_V3, global).await;
    let month_push = run_on(&ctx_pushdown(), ERA5_V3, by_month).await;
    assert_eq!(global_base, month_push);
}

// ---- Phase 7.6: viability fallback ----

/// Pushdown context with an explicit group budget.
fn ctx_with_group_budget(n: u128) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(CardinalityRule::new().with_group_budget(n)))
        .build();
    SessionContext::new_with_state(state)
}

#[tokio::test]
async fn over_budget_group_by_falls_back_to_datafusion() {
    // GROUP BY lat, lon => 100 groups over the synthetic universe. With a budget of
    // 50 the aggregate is not pushed; DataFusion handles it, still correctly.
    let ctx = ctx_with_group_budget(50);
    register_zarr_table(&ctx, "t", SYNTHETIC_V3);
    let plan = get_physical_plan(&ctx, "SELECT lat, lon, COUNT(*) AS c FROM t GROUP BY lat, lon").await;
    assert!(
        !contains_node::<ZarrAggregateExec>(&plan),
        "an over-budget group-by must not be pushed"
    );
    assert!(
        contains_node::<AggregateExec>(&plan),
        "it falls back to DataFusion's aggregate"
    );

    let sql = "SELECT lat, lon, COUNT(*) AS c FROM t GROUP BY lat, lon ORDER BY lat, lon";
    let baseline = run(&SessionContext::new(), sql).await;
    let fallback = run(&ctx_with_group_budget(50), sql).await;
    assert_eq!(baseline, fallback, "fallback result must be correct");
}

#[tokio::test]
async fn within_budget_group_by_is_pushed() {
    // The same shape, but a budget of 200 admits the 100 groups -> pushed.
    let ctx = ctx_with_group_budget(200);
    register_zarr_table(&ctx, "t", SYNTHETIC_V3);
    let plan = get_physical_plan(&ctx, "SELECT lat, lon, COUNT(*) AS c FROM t GROUP BY lat, lon").await;
    assert!(contains_node::<ZarrAggregateExec>(&plan));
    assert!(!contains_node::<AggregateExec>(&plan));
}
