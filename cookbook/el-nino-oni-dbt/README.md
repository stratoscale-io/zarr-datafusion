# El Niño ONI as a dbt project, read straight from Zarr

The [ONI recipe](../el-nino-oni/README.md) as a dbt project: the same result,
split into tested models, with every model running live on the Zarr store
through `zarr-flightsql` and the [`dbt-zarr`](../../integrations/dbt-zarr)
adapter. Nothing is converted to Parquet and no file is written; the models
are views (and one small in-memory table) on the server.

```
dbt (laptop) ── dbt-zarr ── Flight SQL ──▶ zarr-flightsql ──▶ ERSST v5 NetCDF (VirtualiZarr)
                                                          └─▶ ERA5 Zarr (gs:// or local)
```

## Lineage

```
 source raw.ersst ──▶ stg_ersst_monthly ─┐      (var sst_source picks one)
 source raw.era5  ──▶ stg_era5_monthly  ─┤
                                         ├──▶ int_climatology (table) ─┐
 seed base_periods ──────────────────────┼─────────────────────────────┤
                                         └──▶ int_anomaly ◀────────────┘
                                                   │
 seed season_labels ─────────────────────────▶ fct_oni ◀── tests ── seed noaa_oni
```

| Model | Materialized | What it does |
|---|---|---|
| `stg_ersst_monthly` / `stg_era5_monthly` | view | The only models that read Zarr: Niño-3.4 box (5°S–5°N, 190–240°E) spatial mean per month. ERA5 is sampled at noon on the 15th. |
| `int_climatology` | table | Mean per calendar month over NOAA's rolling 30-year base periods (`base_periods` seed). Small, and read by every anomaly, so the samples behind it are read once per `dbt run`. |
| `int_anomaly` | view | Monthly anomaly against the month's own base period. |
| `fct_oni` | view | Centred 3-month mean, season label, ENSO phase. |

Tests: `not_null`/`unique` on keys, `accepted_values` on the phase, ONI within
±4 °C, and agreement with NOAA's published ONI (the `noaa_oni` seed):

- ERSST v5 (`oni_matches_noaa`): every season within 0.05 °C. ERSST is NOAA's
  own source; the largest difference is 0.02 °C, rounding.
- ERA5 (`oni_era5_tracks_noaa`): mean error ≤ 0.2 °C, r ≥ 0.95, and every
  season from 2000 on within 0.4 °C. ERA5 is an independent dataset, so it
  can't match season by season before 2000 (see the
  [cookbook](../el-nino-oni/README.md#recipe-b--era5-independent-high-res)).

## Run

Start the server from the repository root (it resolves store locations), with
a journal so the models survive a restart:

```bash
cargo run --release --features flightsql --bin zarr-flightsql -- \
  --addr 127.0.0.1:50051 --journal oni-catalog.jsonl
```

Then, from this directory:

```bash
uv pip install ../../integrations/dbt-zarr
cp profiles.yml.example profiles.yml
dbt build --profiles-dir .
dbt show --profiles-dir . --inline "select * from {{ ref('fct_oni') }} where year = 1998"
```

By default the ONI is computed from ERSST v5, reading NOAA's NetCDF over HTTPS
through the committed `data/ersst_v5.parq` reference. Staging models are views,
so every query and test reads the store again; with a remote store, make them
tables so it's read once per run:

```bash
dbt build --profiles-dir . --vars '{staging_materialized: table}'
```

Other sources:

```bash
# ERSST from a local copy of the NetCDF (scripts/download_ersst.sh, then
# scripts/virtualize_ersst.py without ERSST_SOURCE_URL)
dbt build --profiles-dir . --vars '{ersst_location: /path/to/ersst_local.parq}'

# ERA5 (default: ARCO-ERA5 on GCS; ~1 h and several GB, see the cookbook)
dbt build --profiles-dir . --vars '{sst_source: era5}'
dbt build --profiles-dir . --vars '{sst_source: era5, era5_location: data/era5_sst_local.zarr}'
```

## Results

Measured on one machine (server and data on a Linux box, dbt on a laptop over
an SSH tunnel). "Read" is bytes the server process read (`rchar`).

**Same answer as the hand-written query.** `fct_oni` is identical to the
cookbook's `oni_ersst.sql`, row for row (916 seasons, 1950–2026), and on the
ERA5 local mirror identical to `oni_all_seasons.sql`.

**`dbt build`**: 3 seeds, 4 models, 10 tests, all pass.

| ERSST read from | Staging as view | Staging as table |
|---|---|---|
| Local NetCDF copy | 5 s | |
| NOAA over HTTPS | 7 min 25 s (each of ~5 Zarr-reading steps ~80 s) | 1 min 24 s (one 80 s read) |

**Reads**, ERSST from local NetCDF:

| Query | Hand-written | dbt models |
|---|---|---|
| All seasons | 194 MB, 2.0 s (`oni_ersst.sql`) | 33 MB, 0.4 s (`fct_oni`) |
| One month of the staging series | 0.1 MB | 0.1 MB (`stg_ersst_monthly where yr = 1998 and mo = 1`) |
| One season | n/a | 33 MB (`fct_oni where year = 1998 and season = 'DJF'`) |

- The dbt version reads less for the full series: the hand-written query
  inlines its `monthly` CTE several times, so the store is scanned several
  times; dbt's `int_climatology` table leaves one scan per query.
- Filters on the staging model reach the Zarr scan, so reading one month
  costs what the hand-written query costs.
- A filter on `fct_oni` does not pass its window function (`lag`/`lead`), so
  a one-season query reads the whole series. It is small (one Niño-3.4 box per
  month); for big stores, filter in the staging model instead.

**No files written:** the repository and `data/` were unchanged after the runs
(131,629 files compared by size and modification time); the only thing written
is the server's catalog journal.

**Restart:** after restarting the server with the same journal, `fct_oni`
returns the same rows without re-running dbt (`int_climatology` is rebuilt
from its query while the journal is replayed).
