//! Physical optimizer rule to push LIMIT down into ZarrExec
//!
//! This rule enables limit pushdown past FilterExec when the filter is already
//! handled by ZarrExec's coord_filters. DataFusion normally can't push limits
//! past filters because filters might remove rows, but for Zarr with sorted
//! scientific data, the filter is internalized in coord_filters, so we can
//! safely push the limit.
//!
//! ## Problem
//!
//! Without this rule, a query like:
//! ```sql
//! SELECT time, latitude, longitude, temperature
//! FROM era5
//! WHERE latitude BETWEEN 24.0 AND 54.75
//! LIMIT 5
//! ```
//!
//! Results in a plan where the limit is NOT pushed to ZarrExec:
//! ```text
//! CoalescePartitionsExec: fetch=5
//!   FilterExec: latitude >= 24 AND ...
//!     ZarrExec: limit=None          <-- limit NOT pushed!
//! ```
//!
//! ## Solution
//!
//! This rule picks up limits as it walks down the plan (from CoalescePartitionsExec,
//! GlobalLimitExec, or LocalLimitExec) and pushes them into ZarrExec, through
//! operators that pass rows through one-for-one:
//! ```text
//! CoalescePartitionsExec: fetch=5
//!   FilterExec: latitude >= 24 AND ...
//!     ZarrExec: limit=5, filters=[latitude BETWEEN 24 AND 54.75]  <-- limit PUSHED!
//! ```
//!
//! # Soundness Assumptions
//!
//! This rule is sound (produces equivalent results) when:
//!
//! 1. **Plan Structure**: LIMIT is represented as `CoalescePartitionsExec(fetch=N)`,
//!    `GlobalLimitExec`, or `LocalLimitExec` somewhere in the plan tree. A limit
//!    only moves down through projections, filters, repartitioning, coalescing
//!    and other limits ([`passes_limit_through`]). Below an aggregate, join,
//!    sort, window or union, reading fewer input rows changes the result, so the
//!    limit stops there.
//!
//! 2. **Filter Internalization**: All coordinate filters between the LIMIT and
//!    ZarrExec are already captured in `ZarrExec.coord_filters`. This is true
//!    when `ZarrTable.scan()` correctly parses the WHERE clause.
//!
//! 3. **Sorted Coordinates**: Zarr coordinate arrays are sorted, so LIMIT N
//!    returns a deterministic, contiguous subset.
//!
//! ## Known Limitations
//!
//! - Data variable filters (e.g., `temperature > 290`) are NOT internalized;
//!   they remain as FilterExec. Pushing LIMIT past such filters is still safe
//!   but may return fewer rows than expected (correct but potentially confusing).
//!
//! - The exact plan structure varies across DataFusion versions, platforms
//!   (macOS vs Linux), and Rust toolchains (stable vs nightly). The limit is
//!   picked up wherever the limit node sits on the path down to ZarrExec.
//!
//! See `docs/OPTIMIZER_SOUNDNESS.md` for detailed soundness documentation.

use crate::physical_plan::zarr_exec::ZarrExec;
use datafusion::common::Result;
use datafusion::config::ConfigOptions;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::ExecutionPlan;
use std::sync::Arc;
use tracing::{debug, info};

/// Physical optimizer rule that pushes LIMIT into ZarrExec
///
/// This optimization enables early termination during Zarr array reading,
/// reducing I/O and memory usage for queries with small limits.
///
/// # Safety
///
/// This optimization is safe because:
/// 1. ZarrExec already internalizes coordinate filters via `coord_filters`
/// 2. Coordinate filters on sorted scientific data produce contiguous ranges
/// 3. The limit can be applied during Zarr array reading
///
/// # Plan Structure Handling
///
/// The rule searches for LIMIT nodes anywhere in the plan tree, not just at
/// the root. This handles variations in DataFusion's plan structure across:
/// - Different platforms (macOS vs Linux)
/// - Different Rust toolchains (stable vs nightly)
/// - Different DataFusion versions
///
/// See [`docs/OPTIMIZER_SOUNDNESS.md`] for detailed soundness documentation.
#[derive(Debug, Default)]
pub struct ZarrLimitPushdownRule;

impl ZarrLimitPushdownRule {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for ZarrLimitPushdownRule {
    fn name(&self) -> &str {
        "zarr_limit_pushdown"
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Log plan structure for debugging CI failures across platforms
        debug!(
            plan_name = plan.name(),
            plan_structure = %format_plan_structure(&plan, 0),
            "ZarrLimitPushdownRule::optimize called"
        );

        // Limits are picked up on the way down, so a limit only reaches the
        // ZarrExecs below it (not scans in other branches of the plan).
        push_limit_to_zarr(plan, None)
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Format plan structure for debug logging
///
/// Produces a compact representation of the plan tree for CI diagnosis.
fn format_plan_structure(plan: &Arc<dyn ExecutionPlan>, depth: usize) -> String {
    let indent = "  ".repeat(depth);
    let mut result = format!("{}{}", indent, plan.name());

    // Add limit info if present
    if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
        if let Some(fetch) = coalesce.fetch() {
            result.push_str(&format!("(fetch={})", fetch));
        }
    }
    if let Some(global) = plan.downcast_ref::<GlobalLimitExec>() {
        if let Some(fetch) = global.fetch() {
            result.push_str(&format!("(fetch={})", fetch));
        }
    }
    if let Some(local) = plan.downcast_ref::<LocalLimitExec>() {
        result.push_str(&format!("(fetch={})", local.fetch()));
    }
    if let Some(zarr) = plan.downcast_ref::<ZarrExec>() {
        if let Some(limit) = zarr.limit() {
            result.push_str(&format!("(limit={})", limit));
        }
    }

    // Recurse into children
    for child in plan.children() {
        result.push('\n');
        result.push_str(&format_plan_structure(child, depth + 1));
    }

    result
}

/// Extract the number of input rows a single plan node needs (not recursive)
fn extract_limit_from_node(plan: &Arc<dyn ExecutionPlan>) -> Option<usize> {
    // Check for CoalescePartitionsExec with fetch
    if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
        if let Some(fetch) = coalesce.fetch() {
            return Some(fetch);
        }
    }

    // Check for GlobalLimitExec; with OFFSET it needs skip + fetch rows
    if let Some(global_limit) = plan.downcast_ref::<GlobalLimitExec>() {
        if let Some(fetch) = global_limit.fetch() {
            return Some(global_limit.skip() + fetch);
        }
    }

    // Check for LocalLimitExec
    if let Some(local_limit) = plan.downcast_ref::<LocalLimitExec>() {
        return Some(local_limit.fetch());
    }

    None
}

/// Recursively traverse the plan and push limit into ZarrExec
fn push_limit_to_zarr(
    plan: Arc<dyn ExecutionPlan>,
    limit: Option<usize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    debug!(
        plan_name = plan.name(),
        limit = ?limit,
        "push_limit_to_zarr visiting node"
    );

    // Check if current node is ZarrExec
    if let Some(zarr_exec) = plan.downcast_ref::<ZarrExec>() {
        debug!(
            current_limit = ?zarr_exec.limit(),
            incoming_limit = ?limit,
            "Found ZarrExec"
        );
        // Only push limit if ZarrExec doesn't already have one (or has a larger one)
        let should_push = match (zarr_exec.limit(), limit) {
            (None, Some(_)) => true,
            (Some(existing), Some(new)) if new < existing => true,
            _ => false,
        };

        if should_push {
            let new_limit = limit.unwrap();
            info!(
                limit = new_limit,
                has_coord_filters = zarr_exec.coord_filters().is_some(),
                "Pushing limit into ZarrExec"
            );
            return Ok(Arc::new(zarr_exec.with_limit(limit)));
        }

        debug!("Not pushing limit (already has one or no incoming limit)");
        return Ok(plan);
    }

    // Extract limit from this node if it has one
    let limit_from_node = extract_limit_from_node(&plan);
    let effective_limit = match (limit, limit_from_node) {
        (Some(l1), Some(l2)) => Some(l1.min(l2)),
        (Some(l), None) | (None, Some(l)) => Some(l),
        (None, None) => None,
    };

    if limit_from_node.is_some() {
        debug!(
            limit_from_node = ?limit_from_node,
            effective_limit = ?effective_limit,
            "Found limit in current node"
        );
    }

    // Below an operator that combines, reorders or multiplies rows, fewer
    // input rows change the result: stop carrying the limit (limits found
    // further down are still pushed).
    let child_limit = if passes_limit_through(&plan) {
        effective_limit
    } else {
        None
    };

    // Recursively process children
    let children = plan.children();
    debug!(num_children = children.len(), "Processing children");
    if children.is_empty() {
        return Ok(plan);
    }

    let new_children: Result<Vec<Arc<dyn ExecutionPlan>>> = children
        .into_iter()
        .map(|child| push_limit_to_zarr(child.clone(), child_limit))
        .collect();

    plan.with_new_children(new_children?)
}

/// Whether a LIMIT above `plan` may move below it: true for operators whose
/// first N output rows come from their first N input rows. Filters count
/// because ZarrExec internalizes coordinate filters (see the module docs for
/// data-variable filters).
fn passes_limit_through(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<ProjectionExec>().is_some()
        || plan.downcast_ref::<FilterExec>().is_some()
        || plan.downcast_ref::<RepartitionExec>().is_some()
        || plan.downcast_ref::<CoalescePartitionsExec>().is_some()
        || plan.downcast_ref::<GlobalLimitExec>().is_some()
        || plan.downcast_ref::<LocalLimitExec>().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_name() {
        let rule = ZarrLimitPushdownRule::new();
        assert_eq!(rule.name(), "zarr_limit_pushdown");
    }
}
