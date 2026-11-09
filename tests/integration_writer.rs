//! Integration tests for the Zarr write path.
//!
//! See docs/zarr-write-roundtrip-plan.md. Phase 1 (skeleton) is checked against
//! our own reader here and against xarray in `scripts/check_skeleton.py`. Phase 2
//! (the chunk-writing sink) is checked by round-tripping a per-cell *formula*
//! `temp[t,y,x] = 1000t + 100y + x` through our reader: because the value encodes
//! the position, a stride or axis-transpose bug moves the value to the wrong cell
//! and the read-back mismatches — and the sink's offset math and zarrs' array
//! indexing are independent implementations, so this is not the symmetric-blind
//! trap a plain `read(write(x))` would be.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Float32Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use common::{create_test_context, execute_query, register_zarr_table};
use zarr_datafusion::writer::{
    create_skeleton, write_batches, write_batches_partitioned, write_chunks, CoordSpec,
    CoordValues, DataVarSpec, SkeletonSpec, WriteDataType,
};

/// Mirrors the round-trip fixture's grid: time(7) x lat(10) x lon(12), ragged
/// chunks. Distinct axis lengths are what make an axis swap detectable.
fn rt_spec() -> SkeletonSpec {
    let coords = vec![
        CoordSpec::new("time", CoordValues::Int64((0..7).collect())),
        CoordSpec::new("lat", CoordValues::Int64((0..10).collect())),
        CoordSpec::new("lon", CoordValues::Int64((0..12).collect())),
    ];
    let data_vars = vec![
        DataVarSpec::new("temperature", WriteDataType::Int64),
        DataVarSpec::new("reflectance", WriteDataType::Float32),
    ];
    SkeletonSpec::new(coords, data_vars, vec![1, 4, 5])
}

/// A scratch store path, removed if a previous run left one behind.
fn scratch(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    p.push(name);
    let _ = std::fs::remove_dir_all(&p);
    p.to_string_lossy().into_owned()
}

#[test]
fn skeleton_creates_readable_store() {
    let path = scratch("skeleton_readable.zarr");
    create_skeleton(&path, &rt_spec()).expect("create skeleton");

    let ctx = create_test_context();
    let (schema, meta) = register_zarr_table(&ctx, "sk", &path);

    // Coordinates come back in *dimension* order, not alphabetical: the reader
    // honours the v3 `dimension_names` the skeleton wrote, so it never reaches
    // the alphabetical fallback (schema_inference.rs). This is the writer and
    // reader agreeing on which axis is which — the property the whole round-trip
    // plan is built to protect.
    let coord_names: Vec<&str> = meta.coords.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(coord_names, vec!["time", "lat", "lon"]);

    // The cartesian product of the grid we asked for.
    assert_eq!(meta.total_rows, 7 * 10 * 12);

    let var_names: Vec<&str> = meta.data_vars.iter().map(|v| v.name.as_str()).collect();
    assert!(var_names.contains(&"temperature"));
    assert!(var_names.contains(&"reflectance"));

    for field in ["time", "lat", "lon", "temperature", "reflectance"] {
        assert!(
            schema.field_with_name(field).is_ok(),
            "schema should expose '{field}'"
        );
    }
}

#[tokio::test]
async fn skeleton_data_vars_read_as_fill_value() {
    let path = scratch("skeleton_fill.zarr");
    create_skeleton(&path, &rt_spec()).expect("create skeleton");

    let ctx = create_test_context();
    register_zarr_table(&ctx, "sk", &path);

    // No chunks were written, so every data cell must come back as fill_value:
    // NaN for the float variable, 0 for the int one. This is the "allocated but
    // empty" property Phase 2 will rely on to leave holes in filtered writes.
    let batches = execute_query(
        &ctx,
        "SELECT COUNT(*) AS n, \
                COUNT(reflectance) FILTER (WHERE NOT isnan(reflectance)) AS non_nan, \
                SUM(ABS(temperature)) AS abs_sum \
         FROM sk",
    )
    .await;

    let batch = &batches[0];
    let n = batch.column(0).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
    let non_nan = batch.column(1).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
    let abs_sum = batch.column(2).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();

    assert_eq!(n.value(0), 7 * 10 * 12, "full cartesian product of the grid");
    assert_eq!(non_nan.value(0), 0, "float var must be entirely NaN fill");
    assert_eq!(abs_sum.value(0), 0, "int var must be entirely 0 fill");
}

#[tokio::test]
async fn skeleton_coords_hold_their_values() {
    let path = scratch("skeleton_coords.zarr");
    create_skeleton(&path, &rt_spec()).expect("create skeleton");

    let ctx = create_test_context();
    register_zarr_table(&ctx, "sk", &path);

    // Coordinate arrays are written in full, so they are real data, not fill.
    // Distinct maxima per axis also catch a lat/lon transposition.
    let batches = execute_query(
        &ctx,
        "SELECT MAX(time) AS t, MAX(lat) AS y, MAX(lon) AS x FROM sk",
    )
    .await;
    let batch = &batches[0];
    let col = |i: usize| {
        batch
            .column(i)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0)
    };
    assert_eq!((col(0), col(1), col(2)), (6, 9, 11));
}

#[test]
fn skeleton_rejects_specs_that_would_be_unreadable() {
    // A chunk shape must cover every dimension.
    let mut spec = rt_spec();
    spec.chunks = vec![1, 4];
    let err = create_skeleton(&scratch("bad_chunks.zarr"), &spec).unwrap_err();
    assert!(err.to_string().contains("chunk shape is required"), "{err}");

    // A data variable named like a coordinate would be read back as a 1-D
    // coordinate, silently losing the variable.
    let mut spec = rt_spec();
    spec.data_vars.push(DataVarSpec::new("lat", WriteDataType::Float32));
    let err = create_skeleton(&scratch("bad_collide.zarr"), &spec).unwrap_err();
    assert!(err.to_string().contains("collides with a coordinate"), "{err}");
}

// ---------------------------------------------------------------------------
// Phase 2 — the chunk-writing sink
// ---------------------------------------------------------------------------

const NT: i64 = 7;
const NLAT: i64 = 10;
const NLON: i64 = 12;

/// Value that encodes its own grid position, so a misplaced write is detectable.
fn cell(t: i64, y: i64, x: i64) -> i64 {
    1000 * t + 100 * y + x
}

/// One RecordBatch over the full grid (or a filtered subset), with plain typed
/// columns — no dictionary encoding (Phase 2 is the plain-column path). `keep`
/// selects which cells to emit; unemitted cells must read back as fill.
fn grid_batch(keep: impl Fn(i64, i64, i64) -> bool) -> RecordBatch {
    let (mut time, mut lat, mut lon) = (vec![], vec![], vec![]);
    let (mut temp, mut refl) = (vec![], vec![]);
    for t in 0..NT {
        for y in 0..NLAT {
            for x in 0..NLON {
                if !keep(t, y, x) {
                    continue;
                }
                time.push(t);
                lat.push(y);
                lon.push(x);
                temp.push(cell(t, y, x));
                // reflectance mirrors the formula as f32, with one NaN hole.
                refl.push(if (t, y, x) == (1, 2, 3) {
                    f32::NAN
                } else {
                    cell(t, y, x) as f32
                });
            }
        }
    }
    let schema = Schema::new(vec![
        Field::new("time", DataType::Int64, false),
        Field::new("lat", DataType::Int64, false),
        Field::new("lon", DataType::Int64, false),
        Field::new("temperature", DataType::Int64, false),
        Field::new("reflectance", DataType::Float32, true),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int64Array::from(time)),
            Arc::new(Int64Array::from(lat)),
            Arc::new(Int64Array::from(lon)),
            Arc::new(Int64Array::from(temp)),
            Arc::new(Float32Array::from(refl)),
        ],
    )
    .unwrap()
}

async fn scalar_i64(ctx: &datafusion::prelude::SessionContext, sql: &str) -> i64 {
    let b = execute_query(ctx, sql).await;
    b[0].column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

async fn scalar_f32(ctx: &datafusion::prelude::SessionContext, sql: &str) -> f32 {
    let b = execute_query(ctx, sql).await;
    b[0].column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .value(0)
}

#[tokio::test]
async fn sink_full_write_roundtrips_every_cell() {
    let path = scratch("sink_full.zarr");
    create_skeleton(&path, &rt_spec()).unwrap();
    let n = write_batches(&path, &rt_spec(), [grid_batch(|_, _, _| true)]).unwrap();
    assert_eq!(n, (NT * NLAT * NLON) as u64);

    let ctx = create_test_context();
    register_zarr_table(&ctx, "t", &path);

    // Whole-cube sum: independent of any per-cell read, catches gross errors.
    let mut expected = 0i64;
    for t in 0..NT {
        for y in 0..NLAT {
            for x in 0..NLON {
                expected += cell(t, y, x);
            }
        }
    }
    assert_eq!(scalar_i64(&ctx, "SELECT SUM(temperature) FROM t").await, expected);

    // The corner of the doubly-ragged last chunk (lat 10/4, lon 12/5): the place
    // an edge-chunk sizing bug would surface.
    assert_eq!(
        scalar_i64(
            &ctx,
            "SELECT temperature FROM t WHERE time=6 AND lat=9 AND lon=11"
        )
        .await,
        cell(6, 9, 11),
    );
    // An interior cell, for a second independent position check.
    assert_eq!(
        scalar_i64(
            &ctx,
            "SELECT temperature FROM t WHERE time=3 AND lat=7 AND lon=5"
        )
        .await,
        cell(3, 7, 5),
    );
}

#[tokio::test]
async fn sink_is_order_independent() {
    // Scatter-by-index means row arrival order cannot matter (§5.6). Reverse the
    // single batch's rows and the whole-cube sum must be unchanged.
    let path = scratch("sink_shuffled.zarr");
    create_skeleton(&path, &rt_spec()).unwrap();

    let forward = grid_batch(|_, _, _| true);
    // Reverse every column by reversing the row indices via a take.
    let n = forward.num_rows();
    let idx = Int64Array::from((0..n as i64).rev().collect::<Vec<_>>());
    let reversed = RecordBatch::try_new(
        forward.schema(),
        forward
            .columns()
            .iter()
            .map(|c| arrow::compute::take(c, &idx, None).unwrap())
            .collect(),
    )
    .unwrap();

    write_batches(&path, &rt_spec(), [reversed]).unwrap();

    let ctx = create_test_context();
    register_zarr_table(&ctx, "t", &path);
    let mut expected = 0i64;
    for t in 0..NT {
        for y in 0..NLAT {
            for x in 0..NLON {
                expected += cell(t, y, x);
            }
        }
    }
    assert_eq!(scalar_i64(&ctx, "SELECT SUM(temperature) FROM t").await, expected);
}

#[tokio::test]
async fn sink_filtered_write_leaves_fill_holes() {
    // Write only the time=3 slice; every other cell must read back as fill —
    // 0 for the int variable, NaN for the float (§5.8 silent-fill semantics).
    let path = scratch("sink_filtered.zarr");
    create_skeleton(&path, &rt_spec()).unwrap();
    write_batches(&path, &rt_spec(), [grid_batch(|t, _, _| t == 3)]).unwrap();

    let ctx = create_test_context();
    register_zarr_table(&ctx, "t", &path);

    // Only the 120 written cells are non-fill for reflectance; the other 720 are
    // NaN. `SUM(CASE ...)` is used rather than `COUNT(...) FILTER (...)` on
    // purpose: the count optimiser folds `COUNT` to the grid cardinality and
    // ignores the FILTER predicate (a pre-existing optimiser issue, unrelated to
    // the sink — the written store is byte-correct, confirmed with zarr-python).
    assert_eq!(
        scalar_i64(
            &ctx,
            "SELECT SUM(CASE WHEN NOT isnan(reflectance) THEN 1 ELSE 0 END) FROM t"
        )
        .await,
        NLAT * NLON,
    );
    assert_eq!(
        scalar_i64(
            &ctx,
            "SELECT SUM(CASE WHEN isnan(reflectance) THEN 1 ELSE 0 END) FROM t"
        )
        .await,
        (NT - 1) * NLAT * NLON,
    );
    // A written cell holds its value; an unwritten one is the int fill 0.
    assert_eq!(
        scalar_i64(&ctx, "SELECT temperature FROM t WHERE time=3 AND lat=4 AND lon=6").await,
        cell(3, 4, 6),
    );
    assert_eq!(
        scalar_i64(&ctx, "SELECT temperature FROM t WHERE time=0 AND lat=4 AND lon=6").await,
        0,
    );
}

#[tokio::test]
async fn sink_roundtrips_nan_in_float() {
    let path = scratch("sink_nan.zarr");
    create_skeleton(&path, &rt_spec()).unwrap();
    write_batches(&path, &rt_spec(), [grid_batch(|_, _, _| true)]).unwrap();

    let ctx = create_test_context();
    register_zarr_table(&ctx, "t", &path);

    // The deliberately-NaN cell reads back NaN; a neighbour holds its real value.
    assert!(
        scalar_f32(&ctx, "SELECT reflectance FROM t WHERE time=1 AND lat=2 AND lon=3")
            .await
            .is_nan(),
    );
    assert_eq!(
        scalar_f32(&ctx, "SELECT reflectance FROM t WHERE time=1 AND lat=2 AND lon=4").await,
        cell(1, 2, 4) as f32,
    );
}

#[test]
fn sink_rejects_value_off_the_grid() {
    // A coordinate value with no axis index means the source and target grids
    // disagree — a loud error, never a silent drop (§5.6).
    let path = scratch("sink_offgrid.zarr");
    create_skeleton(&path, &rt_spec()).unwrap();

    let schema = Schema::new(vec![
        Field::new("time", DataType::Int64, false),
        Field::new("lat", DataType::Int64, false),
        Field::new("lon", DataType::Int64, false),
        Field::new("temperature", DataType::Int64, false),
        Field::new("reflectance", DataType::Float32, true),
    ]);
    let bad = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int64Array::from(vec![0i64])),
            Arc::new(Int64Array::from(vec![999i64])), // lat 999 is off the 0..10 axis
            Arc::new(Int64Array::from(vec![0i64])),
            Arc::new(Int64Array::from(vec![1i64])),
            Arc::new(Float32Array::from(vec![1.0f32])),
        ],
    )
    .unwrap();

    let err = write_batches(&path, &rt_spec(), [bad]).unwrap_err().to_string();
    assert!(err.contains("999") && err.contains("axis"), "{err}");
}

#[test]
fn sink_refuses_custom_fill_value() {
    // The sink writes the whole array including holes, so it can only honour the
    // default fills it knows (0 / NaN); a custom fill must be refused, not
    // silently written as a default-fill hole.
    let path = scratch("sink_customfill.zarr");
    let mut spec = rt_spec();
    spec.data_vars[0].fill_value = Some(zarrs::array::FillValue::from(-1i64));
    // Skeleton creation itself is fine; the sink is where it must be caught.
    create_skeleton(&path, &spec).unwrap();
    let err = write_batches(&path, &spec, [grid_batch(|_, _, _| true)])
        .unwrap_err()
        .to_string();
    assert!(err.contains("custom fill_value"), "{err}");
}

// ---------------------------------------------------------------------------
// Phase 5 — chunk-aligned partitioned (concurrent) writes
// ---------------------------------------------------------------------------

/// Read every cell of a data variable back, in row-major order, as f64 (NaN-aware
/// equality handles the reflectance holes).
async fn read_all_f64(path: &str, var: &str) -> Vec<f64> {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "t", path);
    // ORDER BY the coordinates to get a deterministic, comparable ordering.
    let sql = format!("SELECT {var} FROM t ORDER BY time, lat, lon");
    let batches = execute_query(&ctx, &sql).await;
    let mut out = Vec::new();
    for b in &batches {
        let col = b.column(0);
        if let Some(a) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
            out.extend((0..a.len()).map(|i| a.value(i) as f64));
        } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
            out.extend((0..a.len()).map(|i| a.value(i) as f64));
        } else if let Some(a) = col.as_any().downcast_ref::<arrow::array::Float64Array>() {
            out.extend((0..a.len()).map(|i| a.value(i)));
        }
    }
    out
}

fn nan_aware_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

#[tokio::test]
async fn partitioned_write_matches_monolithic_across_partition_counts() {
    // The chunk-aligned outer partitioning must produce byte-identical results to
    // the single-writer path, for any partition count — including counts that
    // exceed the number of outer chunks (7 time chunks here, chunk=1).
    let batch = grid_batch(|_, _, _| true);

    // Reference: single-writer path.
    let mono = scratch("part_mono.zarr");
    create_skeleton(&mono, &rt_spec()).unwrap();
    write_batches(&mono, &rt_spec(), [batch.clone()]).unwrap();
    let mono_temp = read_all_f64(&mono, "temperature").await;
    let mono_refl = read_all_f64(&mono, "reflectance").await;

    for parts in [1usize, 2, 3, 7, 16] {
        let path = scratch(&format!("part_{parts}.zarr"));
        create_skeleton(&path, &rt_spec()).unwrap();
        let n = write_batches_partitioned(&path, &rt_spec(), [batch.clone()], parts).unwrap();
        assert_eq!(n, (NT * NLAT * NLON) as u64, "row count with {parts} partitions");

        assert!(
            nan_aware_eq(&read_all_f64(&path, "temperature").await, &mono_temp),
            "temperature differs with {parts} partitions",
        );
        assert!(
            nan_aware_eq(&read_all_f64(&path, "reflectance").await, &mono_refl),
            "reflectance differs with {parts} partitions",
        );
    }
}

#[tokio::test]
async fn partitioned_write_is_faithful_to_the_formula() {
    // An independent check that the concurrent slabs land values at the right
    // cells: spot-check the doubly-ragged corner and an interior cell through the
    // reader after a many-partition write.
    let path = scratch("part_formula.zarr");
    create_skeleton(&path, &rt_spec()).unwrap();
    write_batches_partitioned(&path, &rt_spec(), [grid_batch(|_, _, _| true)], 7).unwrap();

    let ctx = create_test_context();
    register_zarr_table(&ctx, "t", &path);
    assert_eq!(
        scalar_i64(&ctx, "SELECT temperature FROM t WHERE time=6 AND lat=9 AND lon=11").await,
        cell(6, 9, 11),
    );
    assert_eq!(
        scalar_i64(&ctx, "SELECT temperature FROM t WHERE time=2 AND lat=3 AND lon=7").await,
        cell(2, 3, 7),
    );
}

#[tokio::test]
async fn partitioned_write_leaves_fill_holes() {
    // A filtered (partial) write under partitioning must still leave the unwritten
    // cells as fill — the slabs that receive no rows write all-fill chunks.
    let path = scratch("part_holes.zarr");
    create_skeleton(&path, &rt_spec()).unwrap();
    // Only the time=3 slice; with chunk=1 on time, only one outer chunk is touched.
    write_batches_partitioned(&path, &rt_spec(), [grid_batch(|t, _, _| t == 3)], 4).unwrap();

    let ctx = create_test_context();
    register_zarr_table(&ctx, "t", &path);
    assert_eq!(
        scalar_i64(&ctx, "SELECT temperature FROM t WHERE time=3 AND lat=4 AND lon=6").await,
        cell(3, 4, 6),
    );
    assert_eq!(
        scalar_i64(&ctx, "SELECT temperature FROM t WHERE time=0 AND lat=4 AND lon=6").await,
        0,
        "untouched outer chunk must read as fill",
    );
    assert_eq!(
        scalar_i64(
            &ctx,
            "SELECT SUM(CASE WHEN isnan(reflectance) THEN 1 ELSE 0 END) FROM t"
        )
        .await,
        (NT - 1) * NLAT * NLON,
    );
}

#[tokio::test]
async fn chunk_writers_fill_one_store_between_them() {
    // One writer a 4 x 5 tile of lat and lon, through all time: the chunks are 1 x 4
    // x 5, so each writer's rows fall in chunks no other writer's do. A writer that
    // wrote beyond its own chunks would erase another's cells.
    let mono = scratch("chunks_mono.zarr");
    create_skeleton(&mono, &rt_spec()).unwrap();
    write_batches(&mono, &rt_spec(), [grid_batch(|_, _, _| true)]).unwrap();

    let path = scratch("chunks_tiles.zarr");
    create_skeleton(&path, &rt_spec()).unwrap();
    let mut rows = 0;
    for y0 in (0..NLAT).step_by(4) {
        for x0 in (0..NLON).step_by(5) {
            let tile = grid_batch(|_, y, x| y / 4 == y0 / 4 && x / 5 == x0 / 5);
            rows += write_chunks(&path, &rt_spec(), [tile]).unwrap();
        }
    }
    assert_eq!(rows, (NT * NLAT * NLON) as u64);

    for var in ["temperature", "reflectance"] {
        assert!(
            nan_aware_eq(
                &read_all_f64(&path, var).await,
                &read_all_f64(&mono, var).await
            ),
            "{var} differs from one writer's",
        );
    }
}

#[tokio::test]
async fn a_chunk_writer_fills_the_rest_of_its_chunks_and_leaves_others() {
    let path = scratch("chunks_holes.zarr");
    create_skeleton(&path, &rt_spec()).unwrap();
    write_chunks(&path, &rt_spec(), [grid_batch(|_, _, _| true)]).unwrap();
    // One cell of the chunk at time 3, lat 4 to 7, lon 5 to 9.
    let one = grid_batch(|t, y, x| (t, y, x) == (3, 6, 7));
    write_chunks(&path, &rt_spec(), [one]).unwrap();

    let ctx = create_test_context();
    register_zarr_table(&ctx, "t", &path);
    let at = |t, y, x| format!("SELECT temperature FROM t WHERE time={t} AND lat={y} AND lon={x}");
    assert_eq!(scalar_i64(&ctx, &at(3, 6, 7)).await, cell(3, 6, 7));
    let rest = scalar_i64(&ctx, &at(3, 4, 5)).await;
    assert_eq!(rest, 0, "the rest of its chunk is fill");
    let across = scalar_i64(&ctx, &at(3, 4, 4)).await;
    assert_eq!(across, cell(3, 4, 4), "another chunk is kept");
    let before = scalar_i64(&ctx, &at(2, 6, 7)).await;
    assert_eq!(before, cell(2, 6, 7), "another chunk is kept");
}

#[tokio::test(flavor = "multi_thread")]
async fn writers_write_through_an_object_store_url() {
    let mono = scratch("url_mono.zarr");
    create_skeleton(&mono, &rt_spec()).unwrap();
    write_batches(&mono, &rt_spec(), [grid_batch(|_, _, _| true)]).unwrap();

    let chunks = scratch("url_chunks.zarr");
    let parts = scratch("url_parts.zarr");
    let urls = [&chunks, &parts].map(|p| format!("file://{p}"));
    tokio::task::spawn_blocking(move || {
        let [chunks, parts] = &urls;
        create_skeleton(chunks, &rt_spec()).unwrap();
        for y0 in (0..NLAT).step_by(4) {
            let tile = grid_batch(|_, y, _| y / 4 == y0 / 4);
            write_chunks(chunks, &rt_spec(), [tile]).unwrap();
        }
        create_skeleton(parts, &rt_spec()).unwrap();
        write_batches_partitioned(parts, &rt_spec(), [grid_batch(|_, _, _| true)], 3).unwrap();
    })
    .await
    .unwrap();

    for path in [&chunks, &parts] {
        for var in ["temperature", "reflectance"] {
            assert!(
                nan_aware_eq(
                    &read_all_f64(path, var).await,
                    &read_all_f64(&mono, var).await
                ),
                "{var} in {path} differs from a local write's",
            );
        }
    }
}

#[tokio::test]
async fn sink_places_rows_on_a_descending_axis() {
    // Latitude often runs from north to south.
    let mut spec = rt_spec();
    spec.coords[1] = CoordSpec::new("lat", CoordValues::Int64((0..NLAT).rev().collect()));
    let path = scratch("descending.zarr");
    create_skeleton(&path, &spec).unwrap();
    write_batches(&path, &spec, [grid_batch(|_, _, _| true)]).unwrap();

    let ctx = create_test_context();
    register_zarr_table(&ctx, "t", &path);
    let at = "SELECT temperature FROM t WHERE time=2 AND lat=3 AND lon=7";
    assert_eq!(scalar_i64(&ctx, at).await, cell(2, 3, 7));
}
