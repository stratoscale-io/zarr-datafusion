#!/usr/bin/env bash
# Generate test data for zarr-datafusion
#
# Usage: ./scripts/generate_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Generating synthetic weather data + ERA5 data..."
uv run --python 3.12 --with zarr --with numpy --with xarray --with gcsfs --with dask --with numcodecs "$SCRIPT_DIR/data_gen.py"
