#!/usr/bin/env python3
"""Generate test Zarr datasets for zarr-datafusion.

Creates 8 dataset variations:
- Synthetic: v2, v2_blosc, v3, v3_blosc
- ERA5: v2, v2_blosc, v3, v3_blosc

Codecs use Blosc with LZ4 compression.
"""

import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("data")

# Blosc compressor with LZ4
BLOSC_LZ4 = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

# ERA5 settings
ERA5_GCS_URL = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
ERA5_VARIABLES = ["geopotential", "temperature"]
ERA5_DATE = "2025-01-01"
ERA5_NUM_TIMESTAMPS = 3
ERA5_NUM_LEVELS = 2

ZarrVersion = Literal[2, 3]


def get_store_path(
    dataset: Literal["synthetic", "era5"],
    version: ZarrVersion,
    with_codecs: bool,
) -> Path:
    """Get the output path for a dataset variation."""
    suffix = "_blosc" if with_codecs else ""
    return DATA_DIR / f"{dataset}_v{version}{suffix}.zarr"


# -----------------------------
# Synthetic Data Generation
# -----------------------------
def generate_synthetic(
    version: ZarrVersion,
    with_codecs: bool,
    nlat: int = 10,
    nlon: int = 10,
    ntime: int = 7,
    seed: int = 42,
) -> None:
    """Generate synthetic weather data.

    Args:
        version: Zarr format version (2 or 3)
        with_codecs: Whether to use Blosc/LZ4 compression
        nlat: Number of latitude points
        nlon: Number of longitude points
        ntime: Number of time steps
        seed: Random seed for reproducibility
    """
    store_path = get_store_path("synthetic", version, with_codecs)
    np.random.seed(seed)

    compressor = BLOSC_LZ4 if with_codecs else None
    codec_desc = "Blosc/LZ4" if with_codecs else "no codecs"

    if version == 3:
        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.group(store=store, overwrite=True, zarr_format=3)

        if with_codecs:
            # Zarr v3 uses compressors parameter
            from zarr.codecs import BloscCodec

            compressors = BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")
        else:
            compressors = None

        # Create coordinate arrays (each 1D array's dim is its own name)
        root.create_array(
            "lat", data=np.arange(nlat), compressors=compressors,
            dimension_names=("lat",),
        )
        root.create_array(
            "lon", data=np.arange(nlon), compressors=compressors,
            dimension_names=("lon",),
        )
        root.create_array(
            "time", data=np.arange(ntime), compressors=compressors,
            dimension_names=("time",),
        )

        # Create data variables.
        # dimension_names is the Zarr v3 equivalent of v2's _ARRAY_DIMENSIONS;
        # xarray requires it to map array axes to coordinate names.
        temperature = root.create_array(
            "temperature",
            chunks=(1, nlat, nlon),
            data=np.random.randint(-50, 60, (ntime, nlat, nlon)),
            compressors=compressors,
            dimension_names=("time", "lat", "lon"),
        )
        humidity = root.create_array(
            "humidity",
            chunks=(1, nlat, nlon),
            data=np.random.randint(10, 80, (ntime, nlat, nlon)),
            compressors=compressors,
            dimension_names=("time", "lat", "lon"),
        )
    else:
        # Zarr v2
        root = zarr.open_group(str(store_path), mode="w", zarr_format=2)

        root.create_array("lat", data=np.arange(nlat), compressor=compressor)
        root.create_array("lon", data=np.arange(nlon), compressor=compressor)
        root.create_array("time", data=np.arange(ntime), compressor=compressor)

        temperature = root.create_array(
            "temperature",
            chunks=(1, nlat, nlon),
            data=np.random.randint(-50, 60, (ntime, nlat, nlon)),
            compressor=compressor,
        )
        humidity = root.create_array(
            "humidity",
            chunks=(1, nlat, nlon),
            data=np.random.randint(10, 80, (ntime, nlat, nlon)),
            compressor=compressor,
        )

    # Add metadata
    root.attrs["title"] = "Weekly Weather Sample"
    root.attrs["conventions"] = f"Zarr v{version}"
    temperature.attrs.update({"units": "K", "long_name": "Air Temperature"})
    humidity.attrs.update({"units": "%", "long_name": "Relative Humidity"})

    print(f"  Written: {store_path} (v{version}, {codec_desc})")


# -----------------------------
# Round-trip fixture (Zarr write path)
# -----------------------------
# A dedicated fixture for the `zarr -> parquet -> zarr` write round trip. It is
# NOT the shared synthetic store, deliberately: the streaming/aggregate tests are
# tuned to that store's 100-row lat*lon plane, and reshaping it would silently
# move which branch they cover. See docs/zarr-write-roundtrip-plan.md.
#
# Two properties the shared fixture lacks, both required to validate a writer:
#
#  1. DISTINCT AXIS LENGTHS (time=7, lat=10, lon=12). The shared store has
#     lat=lon=10, and schema_inference.rs:619 falls back to alphabetical ordering
#     when coordinates share a size — so a lat/lon swap there is undetectable:
#     every shape still matches and the round trip still passes. With distinct
#     lengths a mislabelled axis is a shape error. This is what makes the round
#     trip able to fail; round trips are otherwise blind to symmetric bugs
#     (reader swaps, writer swaps back, test goes green).
#
#  2. A FLOAT VARIABLE WITH NaN. The shared store is all randint -> int64, so it
#     exercises no float, NaN, or fill_value handling at all — precisely the
#     machinery the chunk sink depends on, and what the first real target (NDVI,
#     with NaN nodata) needs.
#
# Chunks are ragged on both spatial axes (10/4 -> 4,4,2 and 12/5 -> 5,5,2) so
# partial edge chunks are covered from the start; writing a partial final chunk
# is a classic write bug.
RT_NLAT = 10
RT_NLON = 12
RT_NTIME = 7
RT_CHUNKS = (1, 4, 5)


def get_roundtrip_store_path(version: ZarrVersion) -> Path:
    """Path for a round-trip fixture variation."""
    return DATA_DIR / f"synthetic_rt_v{version}.zarr"


def generate_roundtrip(version: ZarrVersion, seed: int = 7) -> None:
    """Generate the Zarr write round-trip fixture.

    Always Blosc/LZ4 — compression is orthogonal to what this fixture tests, and
    the read side already covers the codec matrix via the 4-way synthetic stores.

    Args:
        version: Zarr format version (2 or 3)
        seed: Random seed for reproducibility
    """
    store_path = get_roundtrip_store_path(version)
    rng = np.random.default_rng(seed)

    shape = (RT_NTIME, RT_NLAT, RT_NLON)
    temperature_data = rng.integers(-50, 60, shape)

    # float32 with a deterministic scatter of NaN holes (~49 of 840 cells), placed
    # by index so they land in interior AND ragged edge chunks alike.
    reflectance_data = rng.random(shape, dtype=np.float32)
    t_idx, y_idx, x_idx = np.indices(shape)
    reflectance_data[(t_idx + y_idx + x_idx) % 17 == 0] = np.nan
    n_nan = int(np.isnan(reflectance_data).sum())

    if version == 3:
        from zarr.codecs import BloscCodec

        compressors = BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")

        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.group(store=store, overwrite=True, zarr_format=3)

        root.create_array(
            "lat", data=np.arange(RT_NLAT), compressors=compressors,
            dimension_names=("lat",),
        )
        root.create_array(
            "lon", data=np.arange(RT_NLON), compressors=compressors,
            dimension_names=("lon",),
        )
        root.create_array(
            "time", data=np.arange(RT_NTIME), compressors=compressors,
            dimension_names=("time",),
        )

        temperature = root.create_array(
            "temperature",
            chunks=RT_CHUNKS,
            data=temperature_data,
            compressors=compressors,
            dimension_names=("time", "lat", "lon"),
        )
        reflectance = root.create_array(
            "reflectance",
            chunks=RT_CHUNKS,
            data=reflectance_data,
            compressors=compressors,
            dimension_names=("time", "lat", "lon"),
            fill_value=np.nan,
        )
    else:
        # Zarr v2 — no dimension names, matching the shared v2 fixture. This is
        # the inference path (schema_inference.rs), and with distinct axis lengths
        # the size-based mapping is now unambiguous rather than an alphabetical
        # coin flip. Note xarray CANNOT open this store (it requires
        # _ARRAY_DIMENSIONS), which is why the oracle is zarr-python based.
        root = zarr.open_group(str(store_path), mode="w", zarr_format=2)

        root.create_array("lat", data=np.arange(RT_NLAT), compressor=BLOSC_LZ4)
        root.create_array("lon", data=np.arange(RT_NLON), compressor=BLOSC_LZ4)
        root.create_array("time", data=np.arange(RT_NTIME), compressor=BLOSC_LZ4)

        temperature = root.create_array(
            "temperature",
            chunks=RT_CHUNKS,
            data=temperature_data,
            compressor=BLOSC_LZ4,
        )
        reflectance = root.create_array(
            "reflectance",
            chunks=RT_CHUNKS,
            data=reflectance_data,
            compressor=BLOSC_LZ4,
            fill_value=np.nan,
        )

    root.attrs["title"] = "Write Round-Trip Fixture"
    root.attrs["conventions"] = f"Zarr v{version}"
    temperature.attrs.update({"units": "K", "long_name": "Air Temperature"})
    reflectance.attrs.update({"units": "1", "long_name": "Surface Reflectance"})

    print(
        f"  Written: {store_path} (v{version}, Blosc/LZ4, "
        f"shape={shape}, chunks={RT_CHUNKS}, {n_nan} NaN)"
    )


# -----------------------------
# ERA5 Data Download
# -----------------------------
def generate_era5_all_variations(
    gcs_url: str = ERA5_GCS_URL,
    variables: list[str] = ERA5_VARIABLES,
    date: str = ERA5_DATE,
    num_timestamps: int = ERA5_NUM_TIMESTAMPS,
    num_levels: int = ERA5_NUM_LEVELS,
) -> None:
    """Download ERA5 data once and create all 4 variations.

    Downloads to temp store in native format, loads into memory,
    then writes all variations (v2, v2_blosc, v3, v3_blosc).

    Args:
        gcs_url: GCS URL for ERA5 Zarr store
        variables: List of variables to download
        date: Date to select (YYYY-MM-DD)
        num_timestamps: Number of hourly timestamps to download
        num_levels: Number of hybrid levels (from surface)
    """
    from zarr.codecs import BloscCodec

    temp_path = DATA_DIR / "era5_temp.zarr"

    # Step 1: Download once in native format
    print("  Opening ERA5 store...")
    ds = xr.open_zarr(
        gcs_url,
        chunks="auto",
        storage_options={"token": "anon"},
    )

    time_slice = ds.time.sel(time=date)[:num_timestamps]
    subset = ds[variables].sel(
        time=time_slice,
        hybrid=ds.hybrid[-num_levels:],
    )
    print(f"  Subset shape: {dict(subset.sizes)}")

    print("  Downloading to temp store...")
    with ProgressBar():
        subset.to_zarr(str(temp_path), mode="w", zarr_format=2)

    # Step 2: Load into memory
    print("  Loading into memory...")
    data = xr.open_zarr(str(temp_path)).load()

    all_vars = list(data.data_vars) + list(data.coords)

    # Step 3: Write all 4 variations
    # v2 without codecs
    path = get_store_path("era5", 2, False)
    encoding = {var: {"compressors": None} for var in all_vars}
    data.to_zarr(str(path), mode="w", zarr_format=2, encoding=encoding)
    print(f"  Written: {path} (v2, no codecs)")

    # v2 with Blosc/LZ4
    path = get_store_path("era5", 2, True)
    encoding = {var: {"compressors": BLOSC_LZ4} for var in all_vars}
    data.to_zarr(str(path), mode="w", zarr_format=2, encoding=encoding)
    print(f"  Written: {path} (v2, Blosc/LZ4)")

    # The data was reloaded from a v2 store, so each variable carries v2-only
    # encoding (compressor/filters) that the v3 backend rejects. Strip it.
    data_v3 = data.copy(deep=False)
    for var in all_vars:
        data_v3[var].encoding = {}

    # v3 without codecs
    path = get_store_path("era5", 3, False)
    encoding = {var: {} for var in all_vars}
    data_v3.to_zarr(str(path), mode="w", zarr_format=3, encoding=encoding)
    print(f"  Written: {path} (v3, no codecs)")

    # v3 with Blosc/LZ4
    path = get_store_path("era5", 3, True)
    encoding = {
        var: {"compressors": BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")}
        for var in all_vars
    }
    data_v3.to_zarr(str(path), mode="w", zarr_format=3, encoding=encoding)
    print(f"  Written: {path} (v3, Blosc/LZ4)")

    # Step 4: Cleanup
    print("  Cleaning up temp data...")
    shutil.rmtree(temp_path)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    """Generate all test datasets."""
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Zarr version: {zarr.__version__}")
    print()

    # Generate all synthetic variations
    print("=" * 60)
    print("SYNTHETIC DATASETS")
    print("=" * 60)
    for version in [2, 3]:
        for with_codecs in [False, True]:
            generate_synthetic(version, with_codecs)
    print()

    # Round-trip fixture for the Zarr write path
    print("=" * 60)
    print("WRITE ROUND-TRIP FIXTURE")
    print("=" * 60)
    for version in [2, 3]:
        generate_roundtrip(version)
    print()

    # Download ERA5 once and create all variations
    print("=" * 60)
    print("ERA5 DATASETS")
    print("=" * 60)
    generate_era5_all_variations()
    print()

    print("=" * 60)
    print("DONE - Generated 8 dataset variations")
    print("=" * 60)
    print()
    print("Datasets created:")
    for dataset in ["synthetic", "era5"]:
        for version in [2, 3]:
            for with_codecs in [False, True]:
                path = get_store_path(dataset, version, with_codecs)
                print(f"  {path}")


if __name__ == "__main__":
    main()
