# dbt over Arrow Flight SQL — Phase-wise Implementation Plan

> Goal: run dbt projects **directly against Zarr**, with no Parquet (or any other)
> conversion step. dbt models become DataFusion views over `ZarrTable`s and are
> served by a long-lived Arrow Flight SQL server. Filters written against the
> last model in the dependency graph should still reach `ZarrExec` as
> coordinate pushdown.

## Architecture

```
┌────────────────────────── dbt process (Python) ──────────────────────────┐
│  dbt-core: parse → dependency graph → render Jinja → materialization     │
│            │                                                             │
│            ▼                                                             │
│  dbt-zarr adapter (NEW, Python)                                          │
│    Credentials · ConnectionManager · Adapter · macros/*.sql              │
│            │  DB-API cursor.execute(sql)                                 │
│            ▼                                                             │
│  adbc_driver_flightsql (off-the-shelf)                                   │
└────────────┼─────────────────────────────────────────────────────────────┘
             │  gRPC + Arrow IPC
             ▼
┌──────────── zarr-flightsql (NEW, Rust binary) ───────────────────────────┐
│  FlightSqlService ─► ONE shared SessionContext                           │
│                       ├─ ZARR table factory, optimizer rules, UDFs       │
│                       ├─ information_schema                              │
│                       ├─ late-binding views / memtables                  │
│                       └─ DDL journal (catalog survives restarts)         │
└────────────┬─────────────────────────────────────────────────────────────┘
             ▼
        Zarr store (local / GCS / S3) — the only copy of the data
```

## Current state (what exists today)

- **Session wiring lives in the CLI only.** `src/bin/zarr_cli/main.rs:218–233`
  builds the `SessionState`: `information_schema` on, `ZARR` table factory,
  `CountStatisticsRule`, `MinMaxStatisticsRule`, `ZarrLimitPushdownRule`,
  `register_zarr_functions`, `register_metric_udfs`. A server needs the same
  wiring, so it has to move into the library.
- **`arrow-flight 58.3.0` and `tonic 0.14` are already in `Cargo.lock`**, pulled
  in by the optional `distributed` feature. The Flight SQL server types are
  behind arrow-flight's `flight-sql` cargo feature (in older arrow-rs releases it
  was called `flight-sql-experimental`; check the name for 58.x).
- **DDL support today:** `CREATE EXTERNAL TABLE … STORED AS ZARR` (via
  `ZarrTableFactory`); `CREATE VIEW`, `CREATE TABLE AS`, `CREATE SCHEMA`,
  `DROP …` come from DataFusion itself.
- **Reference workload:** `cookbook/el-nino-oni/` (ONI from ERA5 SST), with the
  NOAA reference in `oni_noaa_reference.txt`. Known gotcha: `time` is
  dictionary-encoded, so date functions need `arrow_cast(time, 'Timestamp(...)')`
  first.

## Guiding principles

- **Prove pushdown first.** Running dbt with no conversion step only pays off if
  a filter on the last view still prunes Zarr chunks. Phase 0 tests exactly that
  and blocks every later phase.
- **Every model is a view by default.** No dbt materialization writes files.
  Anything kept in memory (`table` materialization) is rebuilt from Zarr after a
  restart.
- **Use existing protocols and clients.** Use the Flight SQL spec and ADBC
  as-is, and keep the dbt adapter thin. Put the logic in the server, where it is
  shared by every client (dbt, DBeaver, Superset, pandas).
- **Each phase compiles, is tested on its own, and is useful on its own.**

## Dependency graph

```
Phase 0 pushdown-through-views spike  (GATE)
        │
Phase 1 shared session builder
        │
Phase 2 zarr-flightsql server ──► Phase 3 catalog: late-binding views + DDL journal
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
               Phase 4 dbt-zarr adapter
                       │
               Phase 5 ONI dbt example (end-to-end acceptance)
                       │
               Phase 6 extensions (Zarr materialization, incremental, ingest, …)
```

---

## Phase 0 — Spike: does pushdown survive stacked views? (GATE)

**Why:** in a dbt project, the filter that matters is usually written in the
last model or by whoever queries it, several views above the scan. If
DataFusion's view expansion and filter pushdown don't carry it down to
`ZarrExec`, every query reads the whole store and the design fails.

**Work:** new test `tests/integration_views.rs` against `data/synthetic_v3.zarr`.

| Case | Shape | Expect |
|---|---|---|
| V1 | `v1 = SELECT * FROM t`; query `v1 WHERE time = X` | `CoordFilters` has `time = X`; arrays read = 1 time step |
| V2 | `v2 = SELECT time, lat, AVG(temperature) … GROUP BY time, lat`; query `v2 WHERE time = X` | filter pushed below the `Aggregate` (filter is on a group key) |
| V3 | three views stacked (stg → int → fct) like ONI | same chunk count as the hand-written single query |
| V4 | filter on a window-function output (ONI rolling mean) | **expected NOT to push down.** Record this as a limitation and add a time-range pre-filter in the staging model |
| V5 | replace upstream `v1`, then query downstream `v2` | detects stale views (see Phase 3) |

Check this with `ZarrIoStats` (arrays and bytes read) and `EXPLAIN` output,
not just result values.

**Exit:** V1–V3 prune as well as the single-query form. V4 and V5 behavior is
documented. If V2 fails, fix filter pushdown through `Aggregate` before going
further.

### Results (2026-10-29, `synthetic_v3`, `tests/integration_views.rs`)

Full scan of `temperature` = 5600 bytes (7 chunks × 800).

| Case | Reached `ZarrExec` | Bytes (view / hand-written) | Verdict |
|---|---|---|---|
| V1 plain view | `time = 3` | 800 / 800 | ✅ pushes |
| V2 aggregate, filter on group key | `time = 3` | 800 / 800 | ✅ pushes |
| V3 stg → int → fct (rename + arithmetic) | `time = 3`, `lat ∈ [2,5]` | 320 / 320 | ✅ pushes |
| V3b `arrow_cast(time, 'Int64') AS t`, filter on `t` | `time = 3` | 800 / 800 | ✅ pushes (DataFusion removes the cast) |
| V3c join with climatology view | filtered side `time = 3`, climatology side none | 6400 / 6400 | ✅ same as hand-written |
| V4 rolling window | none | 5600 (full) | ❌ blocked, as expected |
| V5 replace upstream view | — | — | ❌ downstream still sees old definition (test `#[ignore]`d) |

**V3b on ERA5 (`era5_v3.zarr`, `time` = `Dictionary(Int16, Timestamp(µs, "UTC"))`,
staging view uses the cookbook's `arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') AS ts`):**
one time step = 8,305,920 bytes, full scan = 24,917,760.

| Case | Downstream filter | Reached `ZarrExec` | Bytes | Verdict |
|---|---|---|---|---|
| E1 | `ts = TIMESTAMP '…01:00Z'` | `time Eq(µs, UTC)` | 1 step | ✅ |
| E2 | `ts BETWEEN 01:00Z AND 02:00Z` | `time Range` | 2 steps | ✅ |
| E3 | `extract(hour FROM ts) = 1` | `time DatePart(HOUR, 1)` | 1 step | ✅ |
| E4 | `hr = 1`, where `hr = CAST(extract(hour FROM ts) AS INT)` in staging | `time DatePart(HOUR, 1)` | 1 step | ✅ |
| E5 | `arrow_cast(time, 'Timestamp(Nanosecond, None)')`, `ts = TIMESTAMP '…01:00'` | `time Eq(ns, no tz)` | 1 step | ✅ |

DataFusion removes the dictionary/timestamp casts and `CoordFilters` handles
the result, including derived `CAST(extract(...))` columns that a dbt staging
model would expose.

**Gate passed.** Remaining caveats:
- V4: staging models need an explicit time-range filter (or a later rule that
  widens the time range by the window's frame size).
- V5 confirms Phase 3's `LateBindingView` is required, not optional.

---

## Phase 1 — Shared session builder

**Work:** add `src/session.rs`:

```rust
pub fn zarr_session_config() -> SessionConfig;          // information_schema on
pub fn build_session_state(config: SessionConfig) -> Result<SessionState>;
pub fn build_session_context(config: SessionConfig) -> Result<SessionContext>;
```

Callers tune the session through the plain DataFusion `SessionConfig` (e.g.
`zarr_session_config().with_batch_size(n)`) rather than a custom options type.
Move the CLI's session setup into it and switch the CLI to call it. The CLI's
behavior must not change.

**Exit:** `cargo test` passes; the CLI and the server (Phase 2) build identical
sessions.

### Status (2026-10-29, on tower): done, not committed

- `src/session.rs` wires everything the CLI did: `RemoteObjectStoreRegistry`,
  the `ZARR` table factory, `CountStatisticsRule`, `MinMaxStatisticsRule`,
  `ZarrLimitPushdownRule`, `CardinalityRule`, `COPY TO … STORED AS ZARR`,
  `zarr_describe`, and the metric UDFs.
- `zarr-cli` now calls `build_session_context(zarr_session_config())`.
- Only behavior change: a failure to register the Zarr write format was a
  printed warning and is now a startup error. Registration only inserts into
  a map, so it doesn't fail in practice.
- `tests/integration_session.rs` (3 tests) checks the registered rules and
  functions, `CREATE EXTERNAL TABLE … STORED AS ZARR` + `information_schema`,
  and that the caller's config is kept.
- Full `cargo test` passes. A `zarr-cli` smoke test (external table,
  `information_schema`, `zarr_describe`, `mae`, `COPY TO … STORED AS ZARR`)
  behaves as before.
- Not migrated yet: `examples/common`, the test helpers, and `head.rs`. Each
  sets up its own session on purpose (e.g. without some rules, or with the
  distributed planner).

---

## Phase 2 — `zarr-flightsql` server binary

**Layout:**

```
src/flightsql/
├── mod.rs          # pub fn serve(addr, ctx, opts) -> Result<()>
├── service.rs      # impl FlightSqlService for ZarrFlightSqlService
├── statement.rs    # classify SQL (query vs DDL), execute, ticket encoding
└── auth.rs         # optional static bearer token
src/bin/zarr_flightsql.rs   # CLI flags: --addr, --token, --journal, --init-sql
```

**Cargo:** new feature `flightsql = ["dep:arrow-flight", "dep:tonic"]`, with
`arrow-flight = { version = "58", features = ["flight-sql"], optional = true }`.
Add a `[[bin]] zarr-flightsql` with `required-features = ["flightsql"]`.

**Service state:** a single `Arc<SessionContext>` shared by every connection.
dbt runs several threads, each with its own connection, and they must all see
one catalog. DataFusion's in-memory catalog providers are already safe to use
from several threads.

**Statement handling.** ADBC's DB-API `cursor.execute()` sends **every**
statement, DDL included, as `CommandStatementQuery`, so:

```
get_flight_info_statement(sql)
  ├─ plan = ctx.state().create_logical_plan(sql)
  ├─ if plan is Ddl / Dml / Statement:
  │     execute NOW (exactly once), keep the (empty or count) result under a
  │     handle, return FlightInfo { schema, ticket = Handle(id) }
  └─ else (query):
        return FlightInfo { schema = plan.schema(), ticket = Sql(sql) }

do_get_statement(ticket)
  ├─ Handle(id) → stream the stored result, then remove it
  └─ Sql(sql)   → ctx.sql(sql).execute_stream() → FlightDataEncoder
```

- Encode the ticket as a small enum (`Sql(String) | Handle(Uuid)`). A
  `Sql` ticket carries no server state, at the cost of planning the query
  twice. That is cheap because planning Zarr only reads metadata.
- Also implement `do_put_statement_update` (`CommandStatementUpdate`) for
  clients that send DDL through the update path. It runs the statement and
  returns a row count.
- `get_flight_info_sql_info` / `do_get_sql_info`: the minimum `SqlInfo` set
  (server name, version, read-only = false). Some clients ask for this when they
  connect.
- Metadata commands `CommandGetCatalogs`, `GetDbSchemas`, `GetTables`,
  `GetTableTypes`: implement them as SQL over `information_schema`. dbt doesn't
  need them, because its macros query `information_schema` directly, but
  DBeaver and other JDBC tools do. **Scheduled for Phase 2b; not on dbt's
  critical path.**
- Prepared statements: **not supported yet**, and return `Unimplemented`. Seeds
  avoid parameterized inserts (Phase 4).

**Errors:** convert `DataFusionError` to `tonic::Status`
(`Plan`/`SQL` → `InvalidArgument`, `NotImplemented` → `Unimplemented`, everything
else → `Internal`), keeping the full message so it shows up in dbt's log.

**Startup:** `--init-sql <file>` runs statements such as
`CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION '…'` before the server starts
listening.

**Tests:** `tests/integration_flightsql.rs` starts the server on an ephemeral
port inside the test and uses `arrow_flight::sql::client::FlightSqlServiceClient`:
- a SELECT round trip on synthetic data matches `ctx.sql()`;
- DDL runs exactly once (create a view, then check `information_schema.tables`);
- an invalid SQL statement returns `InvalidArgument` with the DataFusion message;
- two concurrent clients see the same catalog.

Also add a Python smoke test under `scripts/` (run with `uv`):

```bash
uv run --with adbc-driver-flightsql --with pyarrow python scripts/flightsql_smoke.py
```

**Exit:** both test suites pass, and `adbc_driver_flightsql.dbapi` can run
`CREATE VIEW` and `SELECT` against the server.

### Status (2026-10-29, on tower): done

What was built differs from the sketch above in a few places:
- **No `auth.rs`:** auth is a tonic `Interceptor` (`BearerAuth` in
  `flightsql/mod.rs`) in front of every call, so no handler can forget the
  check. `do_handshake` accepts any request.
- **Tickets carry no server state:** the statement handle is either
  `q` + SQL (queries, planned again on `DoGet`) or `r` + an Arrow IPC stream
  holding the already-computed DDL/DML result, so the server never stores
  results waiting to be fetched. Statements with side effects
  (`Ddl | Dml | Copy | Statement`) run during `GetFlightInfo`.
- **Dictionary columns:** the Flight encoder expands dictionary columns
  (our coordinates), so `FlightInfo` advertises the expanded schema too.
- **Public API:** `flight_sql_service(ctx, token)` returns a tonic service to
  add to any `Server`; there is no `serve()` wrapper.
- `--init-sql` files are split with `DFParser`, so any number of statements
  per file works.

Tests:
- `tests/integration_flightsql.rs` (8 tests, `--features flightsql`):
  - a SELECT through Flight matches running it locally;
  - the advertised schema matches the streamed one for dictionary coordinates;
  - DDL runs exactly once;
  - invalid SQL returns `InvalidArgument` with the DataFusion message;
  - two clients see one catalog;
  - `execute_update` returns row counts;
  - bearer auth (missing / wrong / right token);
  - `GetSqlInfo`.
- `scripts/flightsql_smoke.py` (ADBC DB-API, run from another machine
  through an SSH tunnel) passes:
  - DDL through `cursor.execute`;
  - dbt's `information_schema` relation query (`BASE TABLE` / `VIEW`);
  - a filtered query through a view;
  - Arrow fetch;
  - errors and a wrong token reported cleanly.

  `--init-sql` was checked by hand.

Findings for Phase 4:
- Connect with `autocommit=True`. The server has no transactions, and
  without it ADBC warns that it cannot disable autocommit.
- ADBC's DB-API calls `Prepare` before every `execute`. The server returns
  `Unimplemented`, so ADBC executes directly, but every statement pays one
  extra round trip. Supporting prepared statements (Phase 6) would remove it.
- `SET` changes the shared session for every client, because there is one
  `SessionContext`. That's acceptable for dbt, but worth documenting.

---

## Phase 3 — Catalog semantics: late-binding views + DDL journal

**Problem 1, stale views.** DataFusion's `ViewTable` keeps the *resolved*
`LogicalPlan` from when the view was created, including references to the table
providers it reads. If `stg` is replaced later, an existing `fct` view still
uses the old `stg`. A full `dbt run` rebuilds models in dependency order, so
this mostly goes unnoticed. But `dbt run -s stg_nino34` alone, a single-model
run, would leave downstream views silently stale.

**Fix:** add `LateBindingView`, a `TableProvider` that stores the **SQL text**
and output schema and plans the SQL again on each `scan()`. When the planned
schema differs from the stored one, it returns a clear error telling the user to
rebuild downstream models.

- In the server's statement path, intercept `CREATE [OR REPLACE] VIEW`: parse it
  with `sqlparser` (already a dependency), check that it plans, then register a
  `LateBindingView` in place of `ViewTable`.
- Phase 0 test V5 must pass with this in place.
- Re-planning on each scan only reads metadata, so it stays cheap. The Phase 0
  pushdown tests run again over `LateBindingView`.

**Problem 2, the catalog is lost on restart.** Add a DDL journal
(`--journal path/catalog.sql`):

- Append every successful catalog-changing statement (`CREATE SCHEMA`, `CREATE
  EXTERNAL TABLE`, `CREATE [OR REPLACE] VIEW`, `CREATE [OR REPLACE] TABLE AS`,
  `DROP …`) to the journal, with fsync.
- At startup, replay the journal in order.
- Compaction: on startup or on demand, rewrite the journal as the minimal
  current state. Keep the last `CREATE` for each live object, drop objects that
  were dropped, and order statements so each object is created after the
  objects it reads from.
- `CREATE TABLE AS` (memtables) replays by recomputing from Zarr. There is no
  data snapshot, which keeps the "no conversion" principle.

**Exit:** a dbt-style sequence of statements, then a server restart, gives the
same `information_schema` and the same query results. A single-model rebuild is
visible downstream without rebuilding the downstream models.

---

## Phase 4 — `dbt-zarr` adapter (Python)

Lives in `integrations/dbt-zarr/` (own `pyproject.toml`, managed with `uv`).
Depends on `dbt-core>=1.8`, `dbt-adapters`, and `adbc-driver-flightsql`.

```
integrations/dbt-zarr/
├── pyproject.toml
├── dbt/adapters/zarr/
│   ├── __init__.py          # AdapterPlugin registration
│   ├── __version__.py
│   ├── credentials.py       # ZarrCredentials
│   ├── connections.py       # ZarrConnectionManager(SQLConnectionManager)
│   ├── impl.py              # ZarrAdapter(SQLAdapter)
│   ├── relation.py          # ZarrRelation
│   └── column.py            # ZarrColumn (Arrow type strings)
└── dbt/include/zarr/
    ├── dbt_project.yml
    └── macros/
        ├── adapters.sql
        ├── catalog.sql
        ├── sources.sql
        └── materializations/{view,table,seed,test}.sql
```

### Python classes

| Class | Responsibilities |
|---|---|
| `ZarrCredentials` | `host`, `port`, `tls`, `token`, `database` (= DataFusion catalog, default `datafusion`), `schema`. `type = "zarr"`. |
| `ZarrConnectionManager` | `open()` → `adbc_driver_flightsql.dbapi.connect("grpc://host:port", db_kwargs={…token…})`; `begin`/`commit`/`rollback` do nothing (no transactions); `exception_handler` turns ADBC errors into `DbtDatabaseError`, keeping the server's message; `get_response()` → `AdapterResponse("OK", rows_affected=…)`; `cancel()` closes the cursor. |
| `ZarrAdapter` | `date_function() = "now()"`; `convert_{text,number,boolean,datetime,date,time}_type` → DataFusion SQL types (`VARCHAR`, `DOUBLE`/`BIGINT`, `BOOLEAN`, `TIMESTAMP`, …); `is_cancelable() = True`; `quote()` uses `"`. |
| `ZarrRelation` | Quoting turned off by default. DataFusion lowercases unquoted identifiers, so model names stay lower-case. |
| `ZarrColumn` | Parses Arrow type strings from `information_schema.columns.data_type`, including `Dictionary(Int16, Float64)` used for coordinate columns. |

### Macros (adapter dispatch `zarr__*`)

| Macro | SQL |
|---|---|
| `list_schemas` | `SELECT schema_name FROM information_schema.schemata WHERE catalog_name = …` |
| `list_relations_without_caching` | `SELECT table_catalog, table_name, table_schema, CASE table_type WHEN 'VIEW' THEN 'view' ELSE 'table' END FROM information_schema.tables WHERE table_schema = …` |
| `get_columns_in_relation` | `information_schema.columns`, ordered by `ordinal_position` |
| `create_schema` / `drop_schema` | `CREATE SCHEMA IF NOT EXISTS` / `DROP SCHEMA IF EXISTS … CASCADE` |
| `drop_relation` | `DROP VIEW IF EXISTS` / `DROP TABLE IF EXISTS` depending on relation type |
| `rename_relation` | raise a compiler error: DataFusion has no rename, and our materializations never call it |
| `truncate_relation` | raise an error (not supported) |
| `get_catalog` | built on `information_schema` so `dbt docs generate` works |
| `current_timestamp` | `now()` |

### Materializations

The built-in `view` and `table` materializations use a create-temp, rename,
drop-backup sequence. They are **overridden**:

- **`view`** (default): `CREATE OR REPLACE VIEW {{ this }} AS {{ sql }}`. One
  statement, already atomic on the server. Runs hooks, updates the relation
  cache, and applies grants as a no-op.
- **`table`**: `CREATE OR REPLACE TABLE {{ this }} AS {{ sql }}` → a server
  memtable. Only for small results that many models read, like a climatology.
  Recomputed after restart through the journal.
- **`seed`**: no row inserts. Write
  `CREATE OR REPLACE EXTERNAL TABLE {{ this }} STORED AS CSV LOCATION
  '<absolute seed path>' OPTIONS ('format.has_header' 'true')`. Needs the
  server to reach the dbt project's files (same host or shared mount); see
  Phase 6 for the remote case.
- **`test`**: the default test materialization only runs a `SELECT` and
  doesn't need `rename`, so check it works unchanged and override only if
  needed.
- **`incremental`, `snapshot`**: raise a clear "not supported yet" error.

### Zarr sources

```sql
{% macro zarr_source(name, location, schema=target.schema) %}
  CREATE EXTERNAL TABLE IF NOT EXISTS {{ schema }}.{{ name }}
  STORED AS ZARR LOCATION '{{ location }}'
{% endmacro %}
```

Call it from `on-run-start` in the project. Declare the same tables in
`sources.yml` so `{{ source('era5', 'sst') }}` resolves, shows up in the
lineage graph, and supports source tests and freshness checks.

### Tests

- Unit tests: type mapping, `ZarrColumn` parsing.
- The `dbt-tests-adapter` basic suite against a server started in the test
  fixture: `BaseSimpleMaterializations` (view and table only),
  `BaseSingularTests`, `BaseGenericTests`, `BaseEmpty`, `BaseEphemeral`.
  `BaseIncremental` and `BaseSnapshot*` are marked as expected to fail.

**Exit:** `dbt debug`, `dbt run`, `dbt test`, `dbt docs generate` succeed
against `zarr-flightsql` on the synthetic dataset.

---

## Phase 5 — ONI example project (end-to-end acceptance)

`cookbook/el-nino-oni-dbt/`, a port of `cookbook/el-nino-oni/oni_all_seasons.sql`:

```
cookbook/el-nino-oni-dbt/
├── dbt_project.yml          # on-run-start: zarr_source('era5', 'data/era5_sst_local.zarr')
├── profiles.yml.example     # type: zarr, host: localhost, port: 50051
├── models/
│   ├── sources.yml
│   ├── staging/stg_nino34_monthly.sql     # view
│   ├── intermediate/int_climatology.sql   # table (memtable)
│   ├── intermediate/int_anomaly.sql       # view
│   └── marts/fct_oni.sql                  # view
├── seeds/noaa_oni.csv                     # from oni_noaa_reference.txt
└── tests/
    ├── schema.yml                         # not_null, unique(season_key), ONI ∈ [-4, 4]
    └── oni_matches_noaa.sql               # rows where |ours − noaa| > 0.3
```

**Models** (sketch):

```sql
-- stg_nino34_monthly: the only model that reads Zarr; noon-of-the-15th sample
SELECT arrow_cast(time, 'Timestamp(Nanosecond, None)') AS t,
       AVG(sea_surface_temperature - 273.15)          AS sst_c
FROM {{ source('era5', 'sst') }}
WHERE latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
  AND date_part('day',  time) = 15
  AND date_part('hour', time) = 12
GROUP BY 1

-- int_climatology: mean per calendar month over the base period
-- (NOAA's base period that shifts every 5 years goes in a small
--  seed `base_periods.csv`, joined by season year)

-- int_anomaly: sst_c - clim, per month

-- fct_oni: AVG(anomaly) OVER (ORDER BY t ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)
--          + season label + ENSO phase
```

**Acceptance checks:**

1. `dbt run && dbt test` passes, and `oni_matches_noaa` returns no rows over the
   overlapping period.
2. **No files written:** compare the directory listing before and after
   `dbt run`. The only change allowed is the server's journal.
3. **Pushdown holds:** query `fct_oni` for a single season and compare the
   arrays/bytes read (`ZarrIoStats`) with the hand-written cookbook query.
   Expect the same chunk count (staging time-range pre-filter per Phase 0 V4).
4. A `README.md` with the lineage graph (`dbt docs`), run instructions, and
   timings.

---

## Phase 6 — Extensions (after the minimal version, each optional)

| Item | What | Why |
|---|---|---|
| `zarr` materialization | A `DataSink` that writes query results to a Zarr array (`COPY … TO … STORED AS ZARR`) | Gridded outputs (e.g. a month×lat×lon climatology) stay gridded and can be read by xarray |
| `incremental` on `time` | Append to a Zarr materialization; `WHERE time > (SELECT max(time) FROM {{ this }})`, where `max` is folded to a constant by `MinMaxStatisticsRule` | Cheap updates for stores that grow over time |
| Remote seeds | Flight SQL bulk ingest (`CommandStatementIngest`) → memtable | dbt and the server on different machines |
| Prepared statements | `CommandPreparedStatementQuery` + parameter binding | JDBC tools, parameterized macros |
| Phase 2b metadata | `GetCatalogs/DbSchemas/Tables/TableTypes` | DBeaver / DataGrip / Superset browsing |
| Distributed | Server hands out multiple endpoints from `GetFlightInfo` | Reuse the `head`/`worker` setup |
| Auth / TLS | OIDC/JWT validation, TLS certs | Deployment beyond a single machine |

---

## Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| Filters don't push through `Aggregate`/views (Phase 0) | Design fails: every query reads the full store | Phase 0 blocks all later work; fix pushdown in the optimizer before continuing |
| Filters never push through window functions | Season queries on `fct_oni` scan all time steps | Put a time-range pre-filter in staging; later, possibly a rule that turns a `ROWS BETWEEN k` window into an expanded time range |
| ADBC DB-API behavior (DDL through `CommandStatementQuery`, `rowcount`) differs from assumptions | Adapter/server mismatch | The Phase 2 Python smoke test pins the behavior before writing the adapter |
| `information_schema.tables.table_type` values for custom providers | `LateBindingView` reported as `BASE TABLE` | Return `TableType::View` from the provider; test it |
| Dictionary-encoded coordinate types in `get_columns_in_relation` | dbt column type handling breaks | `ZarrColumn` unwraps `Dictionary(K, V)` to `V` |
| dbt-core changing the adapter interface | Adapter breaks on upgrade | Pin the dbt-core minor version; run `dbt-tests-adapter` in CI |
| Concurrent DDL from dbt threads | Journal entries out of order | Write journal entries in the order statements complete, behind a mutex; compaction re-sorts by dependency |
| Server can't see seed files | Seeds fail on remote deployments | Clear error message; Phase 6 bulk ingest |

## Rough size

| Phase | Size |
|---|---|
| 0 spike | ~200 lines of tests |
| 1 session builder | ~80 lines (moved code) |
| 2 server | ~400–500 lines of Rust + tests |
| 3 late-binding views + journal | ~300 lines of Rust + tests |
| 4 adapter | ~400 lines of Python + ~200 lines of Jinja macros |
| 5 ONI example | ~150 lines of SQL/YAML + README |
