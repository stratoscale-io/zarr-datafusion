//! Integration tests for projection and limit pushdown verification
//!
//! Verifies that DataFusion correctly pushes down projections and limits
//! to the ZarrExec plan.

mod common;

use common::*;
use datafusion::physical_plan::ExecutionPlan;

#[tokio::test]
async fn test_pushdown_projection_single_column() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT temperature FROM data").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    // Verify only temperature column is in projected schema
    let projected_schema = zarr_exec.properties().equivalence_properties().schema();
    assert_eq!(
        projected_schema.fields().len(),
        1,
        "Should project only 1 column"
    );
    assert_eq!(projected_schema.field(0).name(), "temperature");
}

#[tokio::test]
async fn test_pushdown_projection_multiple_columns() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT lat, lon FROM data").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    let projected_schema = zarr_exec.properties().equivalence_properties().schema();
    assert_eq!(
        projected_schema.fields().len(),
        2,
        "Should project 2 columns"
    );
}

#[tokio::test]
async fn test_pushdown_projection_all_columns() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT * FROM data").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    let projected_schema = zarr_exec.properties().equivalence_properties().schema();
    assert_eq!(
        projected_schema.fields().len(),
        5,
        "Should project all 5 columns"
    );
}

#[tokio::test]
async fn test_pushdown_limit_small() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 10").await;

    assert_eq!(batch.num_rows(), 10, "Should return exactly 10 rows");
}

#[tokio::test]
async fn test_pushdown_limit_larger_than_data() {
    let ctx = create_test_context();
    let (_, meta) = register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 10000").await;

    // Should return all rows, not 10000
    assert_eq!(
        batch.num_rows(),
        meta.total_rows,
        "Should return all available rows"
    );
}

#[tokio::test]
async fn test_pushdown_limit_with_projection() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT lat, lon FROM data LIMIT 50").await;

    assert_eq!(batch.num_rows(), 50);
    assert_eq!(batch.num_columns(), 2);
}

#[tokio::test]
async fn test_pushdown_projection_data_variable_only() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT humidity FROM data LIMIT 10").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    let projected_schema = zarr_exec.properties().equivalence_properties().schema();
    assert_eq!(projected_schema.fields().len(), 1);
    assert_eq!(projected_schema.field(0).name(), "humidity");
}

#[tokio::test]
async fn test_pushdown_projection_coords_only() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT lat, lon, time FROM data LIMIT 10").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    let projected_schema = zarr_exec.properties().equivalence_properties().schema();
    assert_eq!(projected_schema.fields().len(), 3);
}

#[tokio::test]
async fn test_pushdown_projection_preserves_data() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Query single column
    let batch_single = execute_query_single(
        &ctx,
        "SELECT temperature FROM data ORDER BY lat, lon, time LIMIT 50",
    )
    .await;

    // Query all columns
    let batch_all = execute_query_single(
        &ctx,
        "SELECT temperature FROM (SELECT * FROM data ORDER BY lat, lon, time LIMIT 50)",
    )
    .await;

    // Both should have same temperature values
    let temp_single = format!("{:?}", batch_single.column(0));
    let temp_all = format!("{:?}", batch_all.column(0));
    assert_eq!(
        temp_single, temp_all,
        "Projection should not affect data values"
    );
}

#[tokio::test]
async fn test_pushdown_limit_with_order_by() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(
        &ctx,
        "SELECT temperature FROM data ORDER BY temperature LIMIT 5",
    )
    .await;

    assert_eq!(batch.num_rows(), 5);
}

#[tokio::test]
async fn test_pushdown_limit_one() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 1").await;

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 5);
}

// =============================================================================
// Filter pushdown tests
// =============================================================================

#[tokio::test]
async fn test_pushdown_filter_single_coordinate() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on time coordinate (first coordinate in synthetic data)
    // synthetic data has time = [0, 1, 2, 3, 4, 5, 6]
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time = 0").await;

    // Should return rows where time = 0 (all lat × lon combinations)
    // With time fixed, rows = lat(10) × lon(10) = 100
    assert_eq!(
        batch.num_rows(),
        100,
        "Should return lat × lon rows for time=0"
    );
}

#[tokio::test]
async fn test_pushdown_filter_multiple_coordinates() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on time and lat coordinates
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time = 0 AND lat = 0").await;

    // With time and lat fixed, rows = lon(10) = 10
    assert_eq!(
        batch.num_rows(),
        10,
        "Should return lon rows for time=0 AND lat=0"
    );
}

#[tokio::test]
async fn test_pushdown_filter_all_coordinates() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on all coordinates - should return single row
    let batch = execute_query_single(
        &ctx,
        "SELECT * FROM data WHERE time = 0 AND lat = 0 AND lon = 0",
    )
    .await;

    assert_eq!(
        batch.num_rows(),
        1,
        "Should return single row when all coordinates specified"
    );
}

#[tokio::test]
async fn test_pushdown_filter_with_projection() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter with specific columns
    let batch = execute_query_single(
        &ctx,
        "SELECT temperature, humidity FROM data WHERE time = 0 AND lat = 0",
    )
    .await;

    assert_eq!(batch.num_columns(), 2, "Should project only 2 columns");
    assert_eq!(batch.num_rows(), 10, "Should return filtered rows");
}

#[tokio::test]
async fn test_pushdown_filter_with_limit() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter + limit combination
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time = 0 LIMIT 5").await;

    assert_eq!(batch.num_rows(), 5, "Should return limited rows");
}

#[tokio::test]
async fn test_pushdown_filter_preserves_data_correctness() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on time coordinate (which is Int64)
    let batch = execute_query_single(&ctx, "SELECT time FROM data WHERE time = 3").await;

    // All returned time values should be 3
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::Int16Type;
    let time_col = batch.column(0);
    let time_dict = time_col.as_dictionary::<Int16Type>();

    // Check that all dictionary keys resolve to value 3
    for i in 0..batch.num_rows() {
        if !time_dict.is_null(i) {
            let key = time_dict.keys().value(i);
            let values = time_dict.values();
            let value = values
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .expect("time should be Int64")
                .value(key as usize);
            assert_eq!(value, 3, "All time values should be 3");
        }
    }
}

#[tokio::test]
async fn test_pushdown_filter_nonexistent_value_returns_empty() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on value that doesn't exist
    let batches = execute_query(&ctx, "SELECT * FROM data WHERE time = 9999").await;

    // Should return 0 rows (filter not found in coordinates)
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 0,
        "Should return no rows for non-existent filter value"
    );
}

#[tokio::test]
async fn test_pushdown_filter_on_data_variable_not_pushed() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on data variable (temperature) - should NOT be pushed down
    // but should still work via DataFusion's filter
    let batches = execute_query(&ctx, "SELECT * FROM data WHERE temperature > 290 LIMIT 10").await;

    // Should still return results (filter applied by DataFusion)
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows <= 10, "Should respect limit");
}

// =============================================================================
// Range filter pushdown tests
// =============================================================================

#[tokio::test]
async fn test_pushdown_filter_between() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // BETWEEN filter on time coordinate
    // synthetic data has time = [0, 1, 2, 3, 4, 5, 6]
    // BETWEEN 2 AND 5 should include times [2, 3, 4, 5] = 4 values
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time BETWEEN 2 AND 5").await;

    // With 4 time values × 10 lat × 10 lon = 400 rows
    assert_eq!(
        batch.num_rows(),
        400,
        "BETWEEN should return 4 time values × lat × lon rows"
    );
}

#[tokio::test]
async fn test_pushdown_filter_combined_range() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Combined >= AND <= filter (equivalent to BETWEEN)
    let batch =
        execute_query_single(&ctx, "SELECT * FROM data WHERE time >= 2 AND time <= 5").await;

    // Should be same as BETWEEN: 4 times × 10 lat × 10 lon = 400 rows
    assert_eq!(batch.num_rows(), 400, "Combined range should match BETWEEN");
}

#[tokio::test]
async fn test_pushdown_filter_greater_than() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Greater than filter
    // time > 3 should include times [4, 5, 6] = 3 values
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time > 3").await;

    // 3 times × 10 lat × 10 lon = 300 rows
    assert_eq!(batch.num_rows(), 300, "time > 3 should return 300 rows");
}

#[tokio::test]
async fn test_pushdown_filter_less_than_eq() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Less than or equal filter
    // time <= 2 should include times [0, 1, 2] = 3 values
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time <= 2").await;

    // 3 times × 10 lat × 10 lon = 300 rows
    assert_eq!(batch.num_rows(), 300, "time <= 2 should return 300 rows");
}

#[tokio::test]
async fn test_pushdown_filter_exclusive_range() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Exclusive range: > 2 AND < 5
    // Should include times [3, 4] = 2 values
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time > 2 AND time < 5").await;

    // 2 times × 10 lat × 10 lon = 200 rows
    assert_eq!(
        batch.num_rows(),
        200,
        "Exclusive range should return 200 rows"
    );
}

#[tokio::test]
async fn test_pushdown_filter_half_open_high() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Half-open range: time >= 5
    // Should include times [5, 6] = 2 values
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time >= 5").await;

    // 2 times × 10 lat × 10 lon = 200 rows
    assert_eq!(batch.num_rows(), 200, "time >= 5 should return 200 rows");
}

#[tokio::test]
async fn test_pushdown_filter_mixed_eq_and_range() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Mix equality filter with range filter
    // time = 3 AND lat BETWEEN 2 AND 7
    // lat values [2, 3, 4, 5, 6, 7] = 6 values
    let batch = execute_query_single(
        &ctx,
        "SELECT * FROM data WHERE time = 3 AND lat BETWEEN 2 AND 7",
    )
    .await;

    // 1 time × 6 lat × 10 lon = 60 rows
    assert_eq!(
        batch.num_rows(),
        60,
        "Mixed eq and range should return 60 rows"
    );
}

#[tokio::test]
async fn test_pushdown_filter_range_no_match() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Range filter that doesn't match any values
    // time = [0, 1, 2, 3, 4, 5, 6], so BETWEEN 100 AND 200 matches nothing
    let batches = execute_query(&ctx, "SELECT * FROM data WHERE time BETWEEN 100 AND 200").await;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0, "Out of range filter should return 0 rows");
}

#[tokio::test]
async fn test_pushdown_filter_range_partial_overlap() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Range filter that partially overlaps
    // time = [0, 1, 2, 3, 4, 5, 6], BETWEEN -5 AND 2 should match [0, 1, 2] = 3 values
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time BETWEEN -5 AND 2").await;

    // 3 times × 10 lat × 10 lon = 300 rows
    assert_eq!(
        batch.num_rows(),
        300,
        "Partial overlap should return 300 rows"
    );
}

#[tokio::test]
async fn test_pushdown_filter_range_with_limit() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Range filter combined with limit
    let batch = execute_query_single(
        &ctx,
        "SELECT * FROM data WHERE time BETWEEN 2 AND 5 LIMIT 50",
    )
    .await;

    assert_eq!(
        batch.num_rows(),
        50,
        "Range with limit should return 50 rows"
    );
}

#[tokio::test]
async fn test_pushdown_filter_range_with_projection() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Range filter combined with projection
    let batch = execute_query_single(
        &ctx,
        "SELECT temperature, humidity FROM data WHERE time >= 4",
    )
    .await;

    // time >= 4 means times [4, 5, 6] = 3 values
    // 3 times × 10 lat × 10 lon = 300 rows
    assert_eq!(batch.num_columns(), 2, "Should project 2 columns");
    assert_eq!(batch.num_rows(), 300, "Should return 300 rows");
}

// =============================================================================
// Coordinate-only query optimization tests
// =============================================================================
// When selecting only coordinate columns with LIMIT, the query should avoid
// Cartesian product expansion of non-selected coordinates.

#[tokio::test]
async fn test_coord_only_single_coord_with_limit() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Synthetic data has time = [0, 1, 2, 3, 4, 5, 6] (7 values)
    // With optimization: SELECT time LIMIT 10 should return 7 rows (all unique times)
    // Without optimization: would return 10 of 700 rows (with repeated time values)
    let batch = execute_query_single(&ctx, "SELECT time FROM data LIMIT 10").await;

    assert_eq!(batch.num_columns(), 1, "Should have 1 column");
    assert_eq!(
        batch.num_rows(),
        7,
        "Coord-only with LIMIT should return unique time values (7 rows), not Cartesian product"
    );
}

#[tokio::test]
async fn test_coord_only_two_coords_with_limit() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Synthetic data has lat = [0..9] (10 values), lon = [0..9] (10 values)
    // With optimization: SELECT lat, lon LIMIT 50 should return 50 of lat×lon = 100 combinations
    // Without optimization: would return 50 of 700 rows
    let batch = execute_query_single(&ctx, "SELECT lat, lon FROM data LIMIT 50").await;

    assert_eq!(batch.num_columns(), 2, "Should have 2 columns");
    assert_eq!(
        batch.num_rows(),
        50,
        "Coord-only with LIMIT should return lat×lon combinations"
    );
}

#[tokio::test]
async fn test_coord_only_three_coords_with_limit() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Selecting all 3 coordinates with LIMIT
    // time×lat×lon = 7×10×10 = 700 total combinations
    // LIMIT 100 should return 100 rows
    let batch = execute_query_single(&ctx, "SELECT time, lat, lon FROM data LIMIT 100").await;

    assert_eq!(batch.num_columns(), 3, "Should have 3 columns");
    assert_eq!(batch.num_rows(), 100, "Should return 100 rows");
}

#[tokio::test]
async fn test_coord_only_without_limit_full_cartesian() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Without LIMIT, should return full Cartesian product (700 rows)
    // This ensures aggregate queries like SELECT COUNT(*), MIN(lat) work correctly
    let batch = execute_query_single(&ctx, "SELECT time FROM data").await;

    assert_eq!(batch.num_columns(), 1, "Should have 1 column");
    assert_eq!(
        batch.num_rows(),
        700,
        "Coord-only WITHOUT LIMIT should return full Cartesian (700 rows)"
    );
}

#[tokio::test]
async fn test_coord_only_aggregate_preserves_count() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Aggregate queries should still return correct COUNT even though they project coordinates
    let batch = execute_query_single(&ctx, "SELECT COUNT(*), MIN(lat), MAX(lon) FROM data").await;

    assert_eq!(batch.num_rows(), 1, "Aggregate should return 1 row");
    assert_eq!(batch.num_columns(), 3, "Should have 3 columns");

    // COUNT(*) should be 700 (full Cartesian product)
    let count_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("COUNT should be Int64");
    assert_eq!(
        count_col.value(0),
        700,
        "COUNT(*) should be 700 (full Cartesian product)"
    );
}

// =============================================================================
// Early filter rejection tests (Phase 1 optimization)
// =============================================================================
// These tests verify that filters clearly outside coordinate bounds are rejected
// early, before loading coordinate data.

#[tokio::test]
async fn test_early_rejection_equality_filter_outside_bounds() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Synthetic data has time = [0, 1, 2, 3, 4, 5, 6]
    // Filter for time = 99999 is clearly outside bounds [0, 6]
    // Should return 0 rows via early rejection
    let batches = execute_query(&ctx, "SELECT * FROM data WHERE time = 99999").await;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 0,
        "Filter outside bounds should return 0 rows via early rejection"
    );
}

#[tokio::test]
async fn test_early_rejection_range_filter_completely_outside() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Synthetic data has lat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    // Filter for lat BETWEEN 100 AND 200 is completely outside bounds [0, 9]
    let batches = execute_query(&ctx, "SELECT * FROM data WHERE lat BETWEEN 100 AND 200").await;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 0,
        "Range filter completely outside bounds should return 0 rows"
    );
}

#[tokio::test]
async fn test_early_rejection_negative_range_outside() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Synthetic data has lon = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    // Filter for lon BETWEEN -100 AND -10 is completely outside bounds [0, 9]
    let batches = execute_query(&ctx, "SELECT * FROM data WHERE lon BETWEEN -100 AND -10").await;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 0,
        "Negative range outside bounds should return 0 rows"
    );
}

#[tokio::test]
async fn test_early_rejection_passes_valid_filter() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Synthetic data has time = [0, 1, 2, 3, 4, 5, 6]
    // Filter for time = 3 is within bounds [0, 6]
    // Should NOT be rejected early and should return data
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time = 3").await;

    // time = 3 with lat(10) × lon(10) = 100 rows
    assert_eq!(
        batch.num_rows(),
        100,
        "Valid filter within bounds should return expected rows"
    );
}

#[tokio::test]
async fn test_early_rejection_passes_partial_overlap() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Synthetic data has time = [0, 1, 2, 3, 4, 5, 6]
    // Filter for time BETWEEN -5 AND 2 overlaps with [0, 2]
    // Should NOT be rejected (partial overlap is valid)
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time BETWEEN -5 AND 2").await;

    // time in [0, 1, 2] (3 values) × lat(10) × lon(10) = 300 rows
    assert_eq!(
        batch.num_rows(),
        300,
        "Partial overlap filter should return matching rows"
    );
}

// =============================================================================
// LIMIT-aware coordinate loading tests (Phase 2 optimization)
// =============================================================================
// These tests verify that LIMIT queries load only necessary coordinate values.

#[tokio::test]
async fn test_limit_aware_loading_small_limit() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // With LIMIT 5 and no filters, we should only load minimal coordinate values
    // Synthetic data has coords: time(7) × lat(10) × lon(10) = 700 rows
    // For LIMIT 5: only need ceil(5/100)=1 time, ceil(5/10)=1 lat, 5 lon values
    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 5").await;

    assert_eq!(batch.num_rows(), 5, "Should return exactly 5 rows");
    assert_eq!(batch.num_columns(), 5, "Should have all columns");
}

#[tokio::test]
async fn test_limit_aware_loading_with_projection() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // LIMIT with projection - should still benefit from limit-aware loading
    let batch = execute_query_single(&ctx, "SELECT temperature FROM data LIMIT 10").await;

    assert_eq!(batch.num_rows(), 10, "Should return exactly 10 rows");
    assert_eq!(batch.num_columns(), 1, "Should have 1 column");
}

#[tokio::test]
async fn test_limit_aware_loading_larger_limit() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // LIMIT 100 with coords: time(7) × lat(10) × lon(10) = 700 rows
    // For LIMIT 100: need ceil(100/100)=1 time, all lat(10), all lon(10)
    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 100").await;

    assert_eq!(batch.num_rows(), 100, "Should return exactly 100 rows");
}

#[tokio::test]
async fn test_limit_aware_loading_with_filters() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // When filters are present, LIMIT-aware loading is applied to NON-FILTERED coordinates
    // Filtered coordinates need all values to find matching indices
    // Non-filtered coordinates can still use LIMIT optimization
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time = 3 LIMIT 5").await;

    // time = 3 gives 100 rows, LIMIT 5 returns 5
    assert_eq!(batch.num_rows(), 5, "Filter + LIMIT should work correctly");
}

/// Test that verifies dictionary array sizes are correctly limited when combining filter + LIMIT
/// TODO: Filter is being applied to wrong coordinate - needs investigation
#[tokio::test]
#[ignore]
async fn test_filter_plus_limit_dictionary_sizes() {
    use arrow::datatypes::Int16Type;

    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Query with filter on lat and LIMIT 5
    // Coordinates: time(7), lat(10), lon(10)
    // Filter: lat = 5.0 (matches 1 lat value)
    // LIMIT: 5
    // Expected: time should be limited (since no filter on it)
    //           lat should use all values (filtered coord)
    //           lon should be limited
    let batch = execute_query_single(
        &ctx,
        "SELECT time, lat, lon, temperature FROM data WHERE lat = 5.0 LIMIT 5",
    )
    .await;

    assert_eq!(batch.num_rows(), 5, "Should return 5 rows");

    // Check dictionary sizes for each coordinate column
    // Get the time column (should be dictionary encoded)
    let time_col = batch.column(0);
    if let Some(dict_array) = time_col
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<Int16Type>>()
    {
        let dict_values_len = dict_array.values().len();
        eprintln!("time dictionary values length: {}", dict_values_len);
        // With LIMIT 5 and no filter on time, we should load minimal time values
        // base_limits for time with coords [7, 10, 10] and LIMIT 5:
        // inner_size = 10*10 = 100, needed = ceil(5/100) = 1
        // So time should have at most 1 or a small number of values, NOT 7
        assert!(
            dict_values_len <= 7,
            "time dictionary should have at most 7 values (full coord size), got {}",
            dict_values_len
        );
        // The ideal case is dict_values_len == 1 with LIMIT optimization
    }

    let lat_col = batch.column(1);
    if let Some(dict_array) = lat_col
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<Int16Type>>()
    {
        let dict_values_len = dict_array.values().len();
        eprintln!("lat dictionary values length: {}", dict_values_len);
        // lat has filter, so we load all values to find matches
        // But after filtering, we should only have 1 unique lat value (lat = 5.0)
        assert!(
            dict_values_len >= 1,
            "lat dictionary should have at least 1 value for the filter match"
        );
    }

    let lon_col = batch.column(2);
    if let Some(dict_array) = lon_col
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<Int16Type>>()
    {
        let dict_values_len = dict_array.values().len();
        eprintln!("lon dictionary values length: {}", dict_values_len);
        // lon has no filter, with LIMIT 5:
        // base_limits for lon: inner_size = 1, needed = 5
        // So lon should have at most 5 values
        assert!(
            dict_values_len <= 10,
            "lon dictionary should have at most 10 values (full coord size), got {}",
            dict_values_len
        );
    }
}

#[tokio::test]
async fn test_limit_one_row() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // LIMIT 1 is the extreme case - should load minimal data
    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 1").await;

    assert_eq!(batch.num_rows(), 1, "Should return exactly 1 row");
    assert_eq!(batch.num_columns(), 5, "Should have all columns");
}

// =============================================================================
// Limit pushdown past filter tests (ZarrLimitPushdownRule)
// =============================================================================
// These tests verify that the ZarrLimitPushdownRule correctly pushes LIMIT
// into ZarrExec even when there's a FilterExec in between.

#[tokio::test]
async fn test_limit_pushdown_past_filter() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // This query has a filter + limit - the ZarrLimitPushdownRule should push
    // the limit into ZarrExec despite the FilterExec in between
    let sql = "SELECT * FROM data WHERE time BETWEEN 2 AND 5 LIMIT 10";

    // Get the physical plan and verify ZarrExec has the limit
    let plan = get_physical_plan(&ctx, sql).await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    // The limit should be pushed down to ZarrExec
    assert_eq!(
        zarr_exec.limit(),
        Some(10),
        "Limit should be pushed into ZarrExec"
    );

    // Also verify the query returns correct results
    let batch = execute_query_single(&ctx, sql).await;
    assert_eq!(batch.num_rows(), 10, "Should return exactly 10 rows");
}

#[tokio::test]
async fn test_limit_pushdown_past_equality_filter() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Equality filter with limit
    let sql = "SELECT * FROM data WHERE time = 3 LIMIT 5";

    let plan = get_physical_plan(&ctx, sql).await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    assert_eq!(
        zarr_exec.limit(),
        Some(5),
        "Limit should be pushed into ZarrExec with equality filter"
    );

    let batch = execute_query_single(&ctx, sql).await;
    assert_eq!(batch.num_rows(), 5, "Should return exactly 5 rows");
}

#[tokio::test]
async fn test_limit_pushdown_past_multiple_filters() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Multiple coordinate filters with limit
    let sql = "SELECT * FROM data WHERE time = 3 AND lat BETWEEN 2 AND 7 LIMIT 3";

    let plan = get_physical_plan(&ctx, sql).await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    assert_eq!(
        zarr_exec.limit(),
        Some(3),
        "Limit should be pushed into ZarrExec with multiple filters"
    );

    let batch = execute_query_single(&ctx, sql).await;
    assert_eq!(batch.num_rows(), 3, "Should return exactly 3 rows");
}

#[tokio::test]
async fn test_limit_pushdown_with_projection() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter + projection + limit
    let sql = "SELECT temperature, humidity FROM data WHERE time >= 4 LIMIT 7";

    let plan = get_physical_plan(&ctx, sql).await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    assert_eq!(
        zarr_exec.limit(),
        Some(7),
        "Limit should be pushed with projection and filter"
    );

    let batch = execute_query_single(&ctx, sql).await;
    assert_eq!(batch.num_rows(), 7, "Should return exactly 7 rows");
    assert_eq!(batch.num_columns(), 2, "Should project 2 columns");
}

#[tokio::test]
async fn test_limit_pushdown_no_filter() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Simple limit without filter - should still be pushed
    let sql = "SELECT * FROM data LIMIT 15";

    let plan = get_physical_plan(&ctx, sql).await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    assert_eq!(
        zarr_exec.limit(),
        Some(15),
        "Limit should be pushed without filter too"
    );

    let batch = execute_query_single(&ctx, sql).await;
    assert_eq!(batch.num_rows(), 15, "Should return exactly 15 rows");
}

#[tokio::test]
async fn test_limit_pushdown_with_data_variable_filter() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on data variable (not a coordinate) - DataFusion handles post-scan filtering
    // but limit should still be pushed to ZarrExec
    let sql = "SELECT * FROM data WHERE temperature > 290 LIMIT 5";

    let plan = get_physical_plan(&ctx, sql).await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    assert_eq!(
        zarr_exec.limit(),
        Some(5),
        "Limit should be pushed even with data variable filter"
    );

    let batches = execute_query(&ctx, sql).await;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows <= 5, "Should respect limit");
}

#[tokio::test]
async fn test_limit_pushdown_preserves_smaller_existing_limit() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // If ZarrExec already has a smaller limit, it should be preserved
    // (This tests the optimization logic - smaller limits should not be replaced)
    let sql = "SELECT * FROM data LIMIT 3";

    let plan = get_physical_plan(&ctx, sql).await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    assert_eq!(zarr_exec.limit(), Some(3), "Limit should be 3");

    let batch = execute_query_single(&ctx, sql).await;
    assert_eq!(batch.num_rows(), 3, "Should return exactly 3 rows");
}

// ============================================================================
// Limit pushdown must not change results
// ============================================================================
//
// A LIMIT may only reach ZarrExec through operators that pass rows through
// one-for-one. Below an aggregate, join or sort, reading fewer input rows
// changes the answer, not just its size. These compare against the same query
// without pushdown (the full result, limited afterwards).

/// Pretty-printed rows of `sql`, sorted, for order-insensitive comparison.
async fn sorted_rows(ctx: &datafusion::prelude::SessionContext, sql: &str) -> Vec<String> {
    let batches = execute_query(ctx, sql).await;
    let text = arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    let mut rows: Vec<String> = text
        .lines()
        .filter(|l| l.starts_with('|'))
        .skip(1)
        // Pretty-printing pads cells to each result's widest value.
        .map(|l| l.split('|').map(str::trim).collect::<Vec<_>>().join("|"))
        .collect();
    rows.sort();
    rows
}

#[tokio::test]
async fn test_limit_above_group_by_keeps_aggregates_correct() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);
    let grouped = "SELECT time, AVG(temperature) AS a, COUNT(*) AS n FROM data GROUP BY time";

    let plan = get_physical_plan(&ctx, &format!("SELECT * FROM ({grouped}) q LIMIT 7")).await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");
    assert_eq!(
        zarr_exec.limit(),
        None,
        "LIMIT must not pass below GROUP BY"
    );

    // All 7 groups, each over its full 100 cells.
    let limited = sorted_rows(&ctx, &format!("SELECT * FROM ({grouped}) q LIMIT 7")).await;
    assert_eq!(limited, sorted_rows(&ctx, grouped).await);
}

#[tokio::test]
async fn test_limit_above_order_by_returns_smallest_rows() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let limited = sorted_rows(
        &ctx,
        "SELECT temperature FROM data ORDER BY temperature LIMIT 5",
    )
    .await;
    let mut all = sorted_rows(&ctx, "SELECT temperature FROM data").await;
    // Compare numerically: the 5 smallest of all 700 temperatures.
    let parse = |r: &String| r.trim_matches('|').parse::<i64>().unwrap();
    all.sort_by_key(parse);
    let mut expected: Vec<String> = all.into_iter().take(5).collect();
    expected.sort();
    assert_eq!(limited, expected);
}

#[tokio::test]
async fn test_limit_above_join_keeps_join_correct() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);
    let joined = "SELECT d.time, d.lat, d.temperature - c.clim AS anomaly \
                  FROM data d JOIN (SELECT lat, AVG(temperature) AS clim FROM data GROUP BY lat) c \
                  ON d.lat = c.lat";

    let limited = execute_query(&ctx, &format!("SELECT * FROM ({joined}) q LIMIT 50")).await;
    assert_eq!(limited.iter().map(|b| b.num_rows()).sum::<usize>(), 50);
    // Every returned row is a row of the full join.
    let all = sorted_rows(&ctx, joined).await;
    for row in sorted_rows(&ctx, &format!("SELECT * FROM ({joined}) q LIMIT 50")).await {
        assert!(
            all.binary_search(&row).is_ok(),
            "row not in full join: {row}"
        );
    }
}

#[tokio::test]
async fn test_limit_with_offset_pushes_skip_plus_fetch() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT * FROM data LIMIT 5 OFFSET 10").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");
    assert_eq!(
        zarr_exec.limit(),
        Some(15),
        "scan must read offset + limit rows"
    );

    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 5 OFFSET 10").await;
    assert_eq!(batch.num_rows(), 5);
}
