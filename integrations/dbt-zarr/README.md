# dbt-zarr

A [dbt](https://www.getdbt.com/) adapter for
[zarr-datafusion](../../README.md). dbt models run directly on Zarr stores,
with no conversion to Parquet or any other format: models become DataFusion
views on a `zarr-flightsql` server, and coordinate filters in a query on the
last model still reach the Zarr scan, so only the chunks it needs are read.

```
dbt ── dbt-zarr ── ADBC Flight SQL ──gRPC──▶ zarr-flightsql ──▶ Zarr (local / GCS / S3)
```

## Setup

Start a server from the zarr-datafusion repo, with a journal so the dbt
catalog survives restarts:

```bash
cargo run --release --features flightsql --bin zarr-flightsql -- \
  --addr 127.0.0.1:50051 --journal catalog.jsonl
```

Install the adapter and point a profile at the server:

```bash
uv pip install ./integrations/dbt-zarr
```

```yaml
# profiles.yml
my_project:
  target: dev
  outputs:
    dev:
      type: zarr
      host: 127.0.0.1
      port: 50051
      schema: analytics     # where models are created
      # database: datafusion   (default; the server's catalog)
      # tls: false
      # token: "{{ env_var('ZARR_FLIGHTSQL_TOKEN') }}"
      threads: 4
```

Register Zarr stores as sources with `zarr_source`, from `on-run-start`. The
location is resolved by the server:

```yaml
# dbt_project.yml
on-run-start:
  - "{{ zarr_source('era5', 'gs://bucket/era5.zarr', schema='raw') }}"
```

```yaml
# models/sources.yml
sources:
  - name: raw
    database: datafusion
    schema: raw
    tables:
      - name: era5
```

## What is supported

| dbt feature | How |
|---|---|
| `view` (default) | `CREATE OR REPLACE VIEW`. Stores nothing; filters pass through stacked views to the Zarr scan. The server re-creates downstream views when one is replaced. |
| `table` | `CREATE OR REPLACE TABLE … AS`: an in-memory table on the server, recomputed from its query after a restart. For small results many models read. |
| `ephemeral` | Inlined CTEs, as usual. |
| `incremental` | `append` strategy only: an in-memory table, created on the first run (or `--full-refresh`), then `INSERT … SELECT`. After a server restart it is rebuilt from its full query, as if fully refreshed. |
| seeds | One `CREATE OR REPLACE TABLE … AS SELECT … FROM (VALUES …)`, so the server needn't see the project's files. |
| tests | Generic and singular tests, `store_failures` (as in-memory tables). |
| `dbt docs generate` | Catalog from `information_schema`. |
| `snapshot`, materialized views, grants | Not supported. |

Notes:
- Columns have Arrow type names (`Int64`, `Utf8`, `Timestamp(µs, "UTC")`).
  Zarr coordinates are dictionary-encoded and reported by their value type.
- Relations are quoted, as in Postgres. Unquoted identifiers in hand-written
  SQL are lower-cased by DataFusion.
- The server has no transactions; dbt's `BEGIN`/`COMMIT` are not sent.
- Rolling-window functions block filter pushdown, so give staging models an
  explicit time range when a downstream model uses one.

## Tests

`tests/unit` runs offline. `tests/functional` is dbt's standard adapter
suite (dbt-tests-adapter) and needs a live server; each test creates and
drops its own schema:

```bash
cargo run --features flightsql --bin zarr-flightsql -- --addr 127.0.0.1:50051 &
cd integrations/dbt-zarr
ZARR_FLIGHTSQL_PORT=50051 uv run --with-editable '.[test]' pytest tests
```
