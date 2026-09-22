# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

zarr-datafusion is a Rust library integrating Zarr (v2 and v3) array storage with Apache DataFusion for querying multidimensional scientific data using SQL. It flattens nD gridded data into a 2D tabular format.

## Build Commands

```bash
cargo build                      # Build the library
cargo test                       # Run all tests
cargo clippy                     # Run linter
cargo fmt                        # Format code
cargo run --bin zarr-cli         # Run interactive SQL CLI
cargo run --example query_synthetic  # Run synthetic data example
cargo run --example query_era5       # Run ERA5 climate data example
cargo run --example query_gcs        # Run GCS remote read example
```

### Icechunk stores (optional `icechunk` feature)

Icechunk (versioned Zarr) support is behind the optional `icechunk` Cargo feature,
so the default build stays lean. Build/run with the feature to read icechunk repos:

```bash
cargo build --features icechunk
cargo test  --features icechunk --test integration_icechunk   # local icechunk query tests
# Query a local icechunk store from the CLI (feature required):
cargo run --features icechunk --bin zarr-cli -- \
  -c "CREATE EXTERNAL TABLE syn STORED AS ZARR LOCATION 'data/synthetic.icechunk';
      SELECT COUNT(*) FROM syn;"

# Query a REMOTE icechunk repo (public, anonymous). LOCATION is the repo root;
# OPTIONS('group' ...) selects a subgroup. Filter every coordinate of the target
# variable so the read stays tiny (see the OOM caveat below):
cargo run --features icechunk --bin zarr-cli -- \
  -c "CREATE EXTERNAL TABLE cira STORED AS ZARR
        LOCATION 'gs://extremeweatherbench/cira-icechunk'
        OPTIONS ('group' 'FOUR_v200_GFS');
      SELECT t2 FROM cira
        WHERE init_time = TIMESTAMP '2020-09-30T12:00:00Z'
          AND lead_time = 0 AND latitude = 40.0 AND longitude = 280.0;"
# The network-gated remote test is #[ignore]'d; run it explicitly with:
#   cargo test --features icechunk --test integration_icechunk -- --ignored remote_cira
```

Detection is automatic for both local and remote:
- **Local**: `LOCATION` is a dir with `snapshots/` + a `repo` marker.
- **Remote** (`gs://` / `s3://`): probed on the object store for a `snapshots/`
  or `refs/` child under the repo prefix.

Either is opened read-only at its `main` branch tip and exposed as a zarrs async
store, reusing the normal async schema-inference/read path. Remote repos open
**anonymously** by default (target: public archives like ExtremeWeatherBench);
pass `OPTIONS('anonymous' 'false')` to use environment credentials instead. For a
repo whose data-variable chunks are virtual references into archival HDF5 on S3
(e.g. CIRA → `s3://noaa-oar-mlwp-data/`), we rewrite the persisted S3 containers to
`anonymous: true` (working around an icechunk 2.1.0 skip-signature bug — see
`examples/icechunk_read_cira.rs`). Icechunk does its own object I/O, so the CLI's
"disk bytes" stat reads 0.

**OOM caveat**: the scan flattens the selected variable's full coordinate cube
into one batch, so an unfiltered `SELECT` (or one with `ORDER BY` but no filter)
over a large store like CIRA (a ~173-billion-row surface cube) will OOM. Filter
every coordinate of the variable you read (or read coordinate columns alone with a
plain `LIMIT`, which takes the coordinate-only fast path). Note the store is
**mixed-dimensionality** (surface vars are 4-D, pressure-level vars are 5-D with
`level`), which works only because arrays carry native V3 `dimension_names`; see
`docs/design-decisions.md` §14b.

## Architecture

```
src/
├── lib.rs                       # Crate root, exports modules
├── bin/zarr_cli/
│   ├── main.rs                  # Interactive SQL REPL with history
│   └── highlight.rs             # SQL syntax highlighting
├── reader/
│   ├── mod.rs                   # Reader module exports
│   ├── schema_inference.rs      # Infer Arrow schema from Zarr metadata
│   ├── zarr_reader.rs           # Zarr reading, flattening nD→2D, Arrow conversion
│   ├── filter.rs                # Filter pushdown: parse WHERE clauses for coord filters
│   ├── stats.rs                 # I/O statistics tracking (bytes, timing)
│   ├── storage.rs               # Storage backends (local, GCS, S3)
│   ├── tracked_store.rs         # Wrapped store for tracking disk I/O
│   ├── coord.rs                 # Coordinate handling utilities
│   └── dtype.rs                 # Data type conversions
├── datasource/
│   ├── zarr.rs                  # ZarrTable: DataFusion TableProvider
│   ├── factory.rs               # TableFactory for registering Zarr tables
│   └── object_store_registry.rs # http(s)/gs/s3 stores for Parquet/CSV/JSON tables
├── optimizer/
│   ├── mod.rs                   # Optimizer module exports
│   ├── minmax_optimization.rs   # MIN()/MAX() → constant folding from stats
│   └── count_optimization.rs    # COUNT(*) → constant folding from stats
└── physical_plan/
    └── zarr_exec.rs             # ZarrExec: DataFusion ExecutionPlan

examples/
├── common/mod.rs                # Shared example utilities
├── query_synthetic.rs           # Query local synthetic Zarr data
├── query_era5.rs                # Query ERA5 climate data
└── query_gcs.rs                 # Query remote Zarr from GCS

tests/
├── common/mod.rs                # Shared test utilities
├── integration_query.rs         # Basic SQL query tests
├── integration_formats.rs       # Zarr v2/v3 format tests
├── integration_optimizer.rs     # Optimizer rule tests
├── integration_pushdown.rs      # Filter/projection pushdown tests
└── integration_error.rs         # Error handling tests
```

**Data flow**: SQL query → ZarrTable.scan() → ZarrExec.execute() → read_zarr() → RecordBatch

**Key features**:
- **Schema inference**: Automatically detects coordinates (1D) and data variables (nD) from Zarr metadata
- **Projection pushdown**: Only loads arrays needed for the query
- **Filter pushdown**: Coordinate equality filters (e.g., `time = X`) reduce data reads
- **Limit pushdown**: LIMIT clauses stop reading early
- **Optimizer rules**: MIN/MAX/COUNT queries use statistics instead of scanning
- **I/O statistics**: Track bytes read, timing breakdown, compression ratio
- **Remote storage**: GCS and S3 support via object_store
- **DictionaryArray for coordinates**: Memory-efficient encoding (~75% savings)

## Assumptions

The library assumes a specific Zarr store structure:

1. **Coordinates are 1D arrays**: Any `shape.len() == 1` array is a coordinate
2. **Data variables are nD arrays**: Dimensionality equals number of coordinates
3. **Cartesian product**: Data variables represent `coord1 × coord2 × coord3`
4. **Alphabetical ordering**: Coordinates sorted by name, data dimensions follow same order

```
weather.zarr/
├── lat/          shape: [10]          → coordinate
├── lon/          shape: [10]          → coordinate
├── time/         shape: [7]           → coordinate
├── temperature/  shape: [10, 10, 7]   → data variable (lat × lon × time)
└── humidity/     shape: [10, 10, 7]   → data variable (lat × lon × time)
```

## Python Scripts

**Always use `uv` for running Python commands.** Dependencies are specified inline via `--with` flags.

```bash
# Generate test data (see scripts/generate_data.sh for dependencies)
./scripts/generate_data.sh

# Run Python scripts directly with uv
uv run --with zarr --with numpy --with xarray --with gcsfs --with dask --with numcodecs scripts/data_gen.py

# One-off Python commands
uv run --with numpy python -c "import numpy; print(numpy.__version__)"
```

Check `scripts/generate_data.sh` for the canonical list of Python dependencies.

## Test Data

```bash
./scripts/generate_data.sh
```

Generates 8 dataset variations in `data/`:
- `synthetic_v2.zarr`, `synthetic_v2_blosc.zarr` — Zarr v2 (with/without Blosc)
- `synthetic_v3.zarr`, `synthetic_v3_blosc.zarr` — Zarr v3 (with/without Blosc)
- `era5_v2.zarr`, `era5_v2_blosc.zarr` — ERA5 climate data, Zarr v2
- `era5_v3.zarr`, `era5_v3_blosc.zarr` — ERA5 climate data, Zarr v3

Synthetic data: time(7), lat(10), lon(10), temperature(7×10×10), humidity(7×10×10). Uses seed=42.

## Key Types

- `ZarrTable` — DataFusion TableProvider for Zarr stores
- `ZarrExec` — Physical execution plan (supports local and remote async reads)
- `ZarrVersion` — Enum for v2/v3 format detection
- `CoordFilters` — Parsed coordinate filters for pushdown (e.g., `time = X`)
- `ZarrIoStats` — Thread-safe I/O statistics (bytes, timing, array counts)
- `MinMaxStatisticsRule` — Optimizer rule for MIN/MAX → constant folding
- `CountStatisticsRule` — Optimizer rule for COUNT → constant folding
- `infer_schema()` — Infers Arrow schema from Zarr v2/v3 metadata
- `detect_zarr_version()` — Detects Zarr format version from metadata files
- `read_zarr()` — Reads Zarr arrays into Arrow RecordBatch (sync, local)
- `read_zarr_async()` — Reads Zarr arrays from remote stores (async, GCS/S3)
- `DictionaryArray<Int16Type>` — Used for coordinate columns (keys=indices, values=unique coords)

## Debugging

```bash
# Run with tracing logs
RUST_LOG=debug cargo run --example query_gcs
RUST_LOG=zarr_datafusion=debug cargo run --example query_gcs

# Interactive debugging with lldb
cargo build --example query_gcs
rust-lldb target/debug/examples/query_gcs

# In lldb:
(lldb) breakpoint set -n main
(lldb) run
(lldb) n                    # step over
(lldb) s                    # step into
(lldb) frame variable       # show local variables
(lldb) p variable_name      # print variable
```
