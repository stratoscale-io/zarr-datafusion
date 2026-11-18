//! Write a fully-populated store through the Phase 2 sink (Zarr write path).
//!
//! Emits the round-trip fixture's grid — time(7) x lat(10) x lon(12), ragged
//! chunks (1,4,5) — with every data cell set to a position-encoding formula
//!
//!     temperature[t,y,x] = 1000*t + 100*y + x
//!     reflectance[t,y,x] = temperature (as f32), with one NaN hole at (1,2,3)
//!
//! Pair with the outside-consumer check, which re-derives the formula with
//! zarr-python and so cannot share our reader's assumptions:
//!
//! ```bash
//! cargo run --example write_filled
//! uv run --with 'zarr>=3' --with numpy scripts/check_sink.py data/sink_demo.zarr
//! ```

use std::sync::Arc;

use arrow::array::{Float32Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use zarr_datafusion::writer::{
    create_skeleton, write_batches, CoordSpec, CoordValues, DataVarSpec, SkeletonSpec,
    WriteDataType,
};

const NT: i64 = 7;
const NLAT: i64 = 10;
const NLON: i64 = 12;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/sink_demo.zarr".to_string());
    let _ = std::fs::remove_dir_all(&path);

    let spec = SkeletonSpec::new(
        vec![
            CoordSpec::new("time", CoordValues::Int64((0..NT).collect())),
            CoordSpec::new("lat", CoordValues::Int64((0..NLAT).collect())),
            CoordSpec::new("lon", CoordValues::Int64((0..NLON).collect())),
        ],
        vec![
            DataVarSpec::new("temperature", WriteDataType::Int64),
            DataVarSpec::new("reflectance", WriteDataType::Float32),
        ],
        vec![1, 4, 5],
    );
    create_skeleton(&path, &spec)?;

    let (mut time, mut lat, mut lon, mut temp, mut refl) = (vec![], vec![], vec![], vec![], vec![]);
    for t in 0..NT {
        for y in 0..NLAT {
            for x in 0..NLON {
                time.push(t);
                lat.push(y);
                lon.push(x);
                temp.push(1000 * t + 100 * y + x);
                refl.push(if (t, y, x) == (1, 2, 3) {
                    f32::NAN
                } else {
                    (1000 * t + 100 * y + x) as f32
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
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int64Array::from(time)),
            Arc::new(Int64Array::from(lat)),
            Arc::new(Int64Array::from(lon)),
            Arc::new(Int64Array::from(temp)),
            Arc::new(Float32Array::from(refl)),
        ],
    )?;

    let n = write_batches(&path, &spec, [batch])?;
    println!("Wrote {n} rows to {path}");
    println!("\nVerify with an outside consumer:");
    println!("  uv run --with 'zarr>=3' --with numpy scripts/check_sink.py {path}");
    Ok(())
}
