# Design Decisions

This document records the pivotal design decisions behind the zarr-datafusion
query engine, in ADR (Architecture Decision Record) style: each entry captures
the decision, the rationale, and where relevant the consequences and trade-offs.
Commit hashes point at the change that introduced or settled the decision.

> **Cross-cutting theme.** Almost every optimization here rests on one insight:
> *scientific coordinate data is sorted and regular*, so coordinate filters can be
> resolved to exact position sets **before any data is read**. That is the
> assumption the generic DataFusion planner cannot exploit, and it is the
> engine's real advantage. The corresponding open gap: data-variable filters
> (`WHERE temperature > 20`) are still applied post-scan, because the coordinate
> trick does not apply to unsorted data values.

---

## Foundational data model

### 1. Flatten nD arrays into a 2D relational table (Cartesian product)
- **Commit:** `85aab29` (first working version)
- **Decision:** Represent a datacube such as `temperature[time, lat, lon]` as one
  row per grid cell: `(time, lat, lon, temperature)`.
- **Rationale:** DataFusion and SQL are inherently tabular. Flattening is what
  makes SQL over gridded scientific data possible at all. Every downstream
  decision is a consequence of this one.

### 2. Structural convention: 1D = coordinate, nD = data variable
- **Decision:** Infer roles from array shape — any 1D array is a coordinate, any
  nD array is a data variable — and order coordinates alphabetically.
- **Rationale:** Zero-config schema inference over arbitrary Zarr stores, with no
  manifest required.
- **Trade-off:** A rigid assumption set (Cartesian product, alphabetical
  dimension order). Documented explicitly in the README as a known limitation.

### 3. Coordinates as Arrow `DictionaryArray`, not materialized values
- **Commit:** `09eb9b5`
- **Decision:** Encode coordinate columns as `DictionaryArray` (keys = indices,
  values = unique coordinate values).
- **Rationale:** The Cartesian product repeats each coordinate value many times.
  Dictionary encoding yields ~75% memory savings and is the natural Arrow
  representation for low-cardinality repeated data.

### 4. Adaptive dictionary key width (Int16 → Int32 → Int64)
- **Commit:** `041e5ce`
- **Decision:** Choose the dictionary key type from coordinate cardinality;
  fail loudly if a key type is undersized.
- **Rationale:** Followed directly from a production bug — `as i16` silently
  wrapped past 32,767 distinct values and panicked `DictionaryArray::new`. The
  ceiling is on *distinct coordinate values in the selection* (e.g. ~65k hourly
  ERA5 timestamps), so batching could not avoid it.

---

## Pushdown and optimization (the performance core)

### 5. Internalize coordinate filters into the scan (`CoordFilters`)
- **Commit:** `f35d78f`
- **Decision:** Parse WHERE clauses on coordinates and push them into `ZarrExec`
  rather than leaving them to a post-scan `FilterExec`.
- **Rationale:** Coordinate filters prune *which chunks get read* — the single
  biggest I/O win — instead of reading everything and discarding rows.

### 6. LIMIT pushdown past `FilterExec`
- **Commit:** `a51537b`
- **Decision:** A custom `ZarrLimitPushdownRule` pushes `LIMIT` past a filter into
  `ZarrExec`, which DataFusion's generic planner cannot do.
- **Rationale:** Because coordinate filters are internalized into the scan
  (decision 5), the surviving rows are known up front on sorted scientific data,
  so the limit is provably sound — a domain-specific optimization the generic
  planner cannot make.

### 7. MIN/MAX/COUNT answered from statistics, skipping the scan
- **Commit:** `aba561b`
- **Decision:** `CountStatisticsRule` and `MinMaxStatisticsRule` constant-fold
  aggregate queries from Zarr metadata.
- **Rationale:** Coordinate bounds and array shapes live in metadata; `COUNT(*)`
  or `MIN(lat)` should not touch a single data chunk.

### 8. Vec-of-filters per coordinate (AND-composed)
- **Commit:** `041e5ce`
- **Decision:** Replace `HashMap<coord, single-filter>` (with fixed
  `eq > range > date_part` precedence) with `Vec<CoordFilterKind>` per coordinate,
  resolving each to a position set and intersecting them. Add `IN`
  (`coord IN (...)`) and `DatePartSet` (`EXTRACT(field) IN (...)`).
- **Rationale:** Real queries compose predicates (`day=15 AND hour=12`,
  `range AND date-part`). The old model silently dropped all but one predicate,
  falling back to slow post-scan filtering.
- **Trade-off:** `OR` is deliberately left to a post-scan `FilterExec` —
  cross-coordinate disjunction does not fit the per-coordinate selection model.

### 9. Coordinate-only queries skip Cartesian expansion (gated on LIMIT)
- **Commit:** `5eda994`
- **Decision:** `SELECT time FROM data LIMIT 10` returns the 7 unique times, not
  10 rows of a 700-row product — but only when a `LIMIT` is present.
- **Rationale:** The LIMIT gate preserves correct semantics for aggregates like
  `COUNT(*)` / `MIN(lat)`, which genuinely need the full Cartesian product.

### 10. Compact coordinate encoding for arithmetic sequences
- **Commit:** `a51537b`
- **Decision:** Store regularly-spaced coordinates as O(1) `(first, step, len)`
  parameters instead of O(N) explicit values, with early
  `filter_satisfiable_by_bounds()` rejection.
- **Rationale:** Climate grids are usually evenly spaced. This avoids
  materializing huge coordinate arrays and rejects impossible filters
  (`lat = 100` when bounds are `[0, 90]`) before any read.

### 10b. Aggregate pushdown driven by exact group cardinality (Phase 7)
- **Decision:** A `SUM`/`COUNT`/`AVG`/`MIN`/`MAX` over a `ZarrExec` — global, grouped
  on coordinate columns, or on a periodic function of one (`date_part('month', time)`)
  — is rewritten by `CardinalityRule` into a self-contained `ZarrAggregateExec`
  (`physical_plan/zarr_aggregate.rs`) that **replaces the whole `AggregateExec ←
  ZarrExec` subtree**. It drives the streaming scan and folds each batch into per-group
  accumulators, keying rows on the evaluated `GROUP BY` expressions. The decision is
  *deterministic*: the exact output-group count is computed by lattice counting
  (`group_cardinality`, exact for coordinate keys, period-bounded for periodic) and
  admitted against a group budget (`ZARR_MAX_GROUPS`); over budget, the rule declines
  and DataFusion handles it unchanged.
- **Rationale:** Because gridded data has a known regular shape, the group count is
  *computed*, not estimated — so pushdown is chosen only when the group table is
  provably bounded, never heuristically. Two payoffs beyond skipping DataFusion's
  hash-aggregate: it is the seam the exact-cardinality cost model plugs into, and — for
  periodic grouping over a dictionary-encoded coordinate, which **fails outright in
  stock DataFusion** (a `date_part` dictionary return-type assertion) — the operator
  decodes the coordinate before evaluating the key and so *enables a query the engine
  otherwise cannot run*. See `docs/exact-cardinality-optimizer.md`; verified in
  `tests/integration_agg_pushdown.rs`.

---

## Domain correctness (climate / CF semantics)

### 11. CF-time decoding to match xarray
- **Commit:** `4895294`
- **Decision:** Decode CF time units (`"hours since 1900-01-01"`) into real Arrow
  timestamps (`src/reader/cf_time.rs`).
- **Rationale:** Scientific users expect xarray-equivalent timestamps, not raw
  integer offsets. Also a prerequisite for date-part filter pushdown to be
  meaningful.

### 12. Mixed-dimensionality variables via dimension metadata
- **Commit:** `5eb6d43`
- **Decision:** Parse `_ARRAY_DIMENSIONS` (the xarray/CF convention) so 3D
  surface variables and 4D pressure-level variables coexist;
  `determine_effective_coords()` skips coordinates a projection does not need.
- **Rationale:** Real datasets such as ERA5 mix surface and multi-level data; the
  pure "all data variables share all coordinates" assumption breaks on them.

---

## Storage and I/O

### 13. `object_store` backends (local / GCS / S3) with tracked-store wrappers
- **Commits:** `74e6782`, `b5fd7e0`
- **Decision:** Use `object_store` for storage and wrap it with `TrackedStore` /
  `AsyncTrackedStore` to count compressed (disk) vs uncompressed (memory) bytes.
- **Rationale:** Cloud-native scientific data lives in buckets. The tracking layer
  powers the I/O-statistics UX and the compression-ratio signal needed for
  debugging read performance.

### 14. VirtualiZarr support with parallelized metadata fetching
- **Commit:** `8623462`
- **Decision:** Read VirtualiZarr Parquet chunk-reference stores (→ NetCDF/GRIB
  byte ranges). Cap detection to the first 5 arrays and load refs via
  `futures::join_all`.
- **Rationale:** Query archival data without rewriting it. The parallelization was
  forced by reality — ERA5's 277 arrays were doing 277 sequential HTTP probes
  (~72s → ~4s).

### 14b. Icechunk stores via upstream crates, reusing the async read path
- **Feature:** optional `icechunk` (off by default; ~56 transitive crates).
- **Decision:** Depend on upstream `icechunk` + `zarrs_icechunk`. Detect an
  icechunk repo by its on-disk layout (`snapshots/` + `repo` marker), open it
  read-only at `main`, and expose the resulting `AsyncIcechunkStore` as a
  `zarrs` `AsyncReadableListableStorage` — the same trait our object-store
  backends satisfy. It drops into `ZarrTable`'s `cached_remote` slot, so the
  existing async schema-inference and read paths need no icechunk-specific code.
- **Rationale:** icechunk resolves native + virtual (archival HDF5/NetCDF) chunks
  and versioning internally; reimplementing that in-house (as we did for
  VirtualiZarr) would be a large, redundant effort. Reusing the async store
  interface kept the integration to one new module plus a factory branch.
- **Remote + groups + mixed dimensionality:** `gs://`/`s3://` repos are detected by
  probing the object store for a `snapshots/`/`refs/` child, then opened through
  the icechunk API (anonymously by default). `OPTIONS('group' '<name>')` selects a
  subgroup, reused as the schema-inference/read prefix. For S3-backed virtual
  chunks we rewrite the persisted containers to `anonymous: true` (the icechunk
  2.1.0 skip-signature workaround, applied generically by reading the persisted
  container list). Crucially, the schema layer now reads native Zarr **V3**
  `dimension_names` (not just the V2 `_ARRAY_DIMENSIONS` attribute), so a
  mixed-dimensionality store like CIRA — 6 shared coordinates but 4-D surface and
  5-D pressure-level variables — maps each variable to its own dims. Validated
  end-to-end against the real CIRA forecast (`t2 ≈ 281.6 K` for one init/lead/grid
  cell).
- **Trade-offs / current limits:** icechunk performs its own object I/O, so the
  `AsyncTrackedStore` byte stats don't cover these reads (CLI shows 0 disk bytes).
  The flatten-to-one-batch model means an unfiltered scan over a
  huge cube OOMs — callers must filter every coordinate of the variable they read
  (a query-time cube restriction / streaming-by-variable model is the real fix).
  Non-anonymous credential plumbing beyond a simple
  flag, and writing remain follow-ups.

---

## Distributed execution

### 15. Pivot from Ballista to datafusion-distributed
- **Commit:** `ed274de`
- **Decision:** Drop Ballista; adopt the embeddable `datafusion-distributed`
  (Arrow Flight / Tonic) library behind a `distributed` feature.
- **Rationale:** Ballista pins DataFusion in lockstep and its stock Docker
  executor images **cannot run a custom `ZarrExec`** — a dead end for true
  multi-node execution. With datafusion-distributed the `worker` binary links the
  library and registers the codec directly. The salvageable composite-codec
  pattern (tag-prefix routing of `ZarrExec`/`ZarrTable`) carried over.

### 16. Outer-axis (axis-0) chunk-grid partitioning
- **Commit:** `554868e`
- **Decision:** `ZarrExec` exposes N partitions by slicing the outer chunk grid;
  the outer coordinate is found by **size-match, not `dimension_names`**.
- **Rationale:** The chunk is the natural read granularity, so slicing the outer
  chunk grid is the simplest sound parallelization unit. Size-match dodges a
  latent v3 `dimensions` reader underflow.

### 17. Single-site filter resolution on the head; workers replace their selection
- **Commits:** `1106a79`, `af07d70`
- **Decision:** The head resolves the outer-axis filter to the surviving index set
  and splits *that* across partitions; workers replace their outer selection with
  the head-resolved slice and read only that window.
- **Rationale:** One resolution site means no divergence between nodes. A single
  day (24 chunks) now spreads across all workers instead of stranding the work on
  one node.

### 18. Keep a chunk's survivors on one worker
- **Commit:** `89785d9`
- **Decision:** `split_indices` groups indices that share a chunk so no chunk is
  read by two workers, then packs the groups into balanced partitions.
- **Rationale:** Avoid redundant chunk reads and decompression across the cluster
  while keeping partitions balanced.

### 18b. Multi-axis (N-D box) partition fan-out when the outer axis under-parallelizes
- **Decision:** When splitting the outer axis alone yields fewer partitions than
  `target_partitions` (its chunk grid is coarse — e.g. a single-chunked time axis),
  the planner (`fan_out_inner` in `datasource/zarr.rs`) also splits one or more
  *inner* axes and emits the Cartesian product as **N-D box partitions**.
  `PartitionSpec` gains `extra: Vec<(coord_index, CoordSelection)>` — the inner-axis
  restrictions (`#[serde(default)]` for distributed-codec back-compat); the reader
  (`apply_partition_selection`) restricts every listed coordinate, not just the
  outer. Candidate inner axes are the unfiltered, multiply-chunked ones (mapped via
  `dimension_names`), split **most-chunked first, greedily**, until the inner budget
  (`target_partitions / p_outer`) is spent — so a cube whose axes are individually too
  coarse (e.g. lat=4 chunks, lon=2) can still fill the machine (4×2 boxes). The split
  is geometric and **metadata-only** (no coordinate reads). Total partitions never
  exceed `target_partitions` and never split a chunk across partitions.
- **Rationale:** Outer-only partitioning (§16) leaves the machine idle whenever the
  outer chunk grid is coarser than the core count — the pathological case is a
  single outer chunk (one partition, no parallelism) over an array whose inner axes
  hold thousands of chunks. Counting *total* touched chunks to shrink the count is a
  proven no-op (outer-only partitioning is already capped at surviving outer chunks,
  which is `≤` total), so the win is the opposite direction: *add* inner-axis
  parallelism. Only unfiltered inner axes are split because their surviving set isn't
  resolved at plan time and `extra` REPLACES a coordinate's selection; a filtered
  inner axis keeps its `CoordFilters`-derived selection instead. Verified
  value-transparent end-to-end (a fanned-out aggregate equals the single-partition
  one) — `integration_multiaxis.rs`.
- **Applicability (important — measured against real data):** fan-out *extracts*
  parallelism from the chunk grid; it cannot *create* it, because a chunk is the
  atomic read/decode unit (we never split one across partitions). So multi-axis
  fan-out only helps layouts that actually chunk their inner axes — e.g. the bundled
  `era5_v3` fixture (`temperature` chunked `[1,1,181,720]` → latitude 4 chunks). The
  *common* archive layout is the opposite: **one spatial chunk per timestep**.
  ARCO-ERA5 (`gs://gcp-public-data-arco-era5/…chunk-1.zarr-v3`,
  `sea_surface_temperature`) is `shape=[1323648,721,1440]` chunked `[1,721,1440]` →
  `n_chunks=[1323648,1,1]`: the time axis has 1.3M chunks (outer-only partitioning
  already saturates any `target_partitions`) while latitude/longitude are a *single*
  chunk each — so `fan_out_inner` finds no candidate and **declines**. For such stores
  H1 and H2 are inert. A related, distinct inefficiency they surface — a whole-globe
  chunk means a small spatial box (e.g. a Niño-3.4 filter) still reads all 721×1440
  cells per timestep — is a *chunk-granularity* over-read that no partition scheme can
  fix (bound by how the data was written), and is **not** the cardinality optimizer's
  concern.
- **Follow-ups:**
  - *H2 — split filtered inner axes (DEFERRED).* Would generalize
    `resolve_outer_selection[_async]` to resolve any coordinate's survivors, resolve
    the filtered inner candidates at plan time, and split the *surviving* set (via
    `split_selection`, which already handles scattered `Indices`) rather than the full
    extent — ranking axes by *surviving* chunk count, at the cost of bounded plan-time
    coordinate reads. **Deferred:** its payoff only lands when a filtered inner axis is
    *itself* multiply chunked, which the dominant single-spatial-chunk archives (see
    Applicability) are not — so the value is too narrow to prioritize. Also would need
    to verify the reader's data path handles an `Indices` selection sitting in `extra`
    on a non-outer axis.
  - *H3 — distributed box execution (deferred).* The codec already round-trips
    `PartitionSpec.extra` (unit-tested) and a worker's `execute()` path is identical to
    local, so N-D boxes *should* run across `head`/`worker` unchanged — but this is not
    yet exercised end-to-end. Add a `distributed`-gated test that plans a fanned query,
    ships box specs to workers, and asserts value-transparency vs. single-node; audit
    `distribute_specs_across_tasks`/`TaskEstimator` for any outer-only assumptions and
    that the box-aware `PartitionSpec::is_empty()` packs/pads correctly.

---

## Interface and ergonomics

### 19. `CREATE EXTERNAL TABLE ... STORED AS ZARR` via `TableProviderFactory`
- **Commit:** `2cbf5dd`
- **Decision:** Hook into DataFusion's standard external-table DDL.
- **Rationale:** Zero custom syntax — users register Zarr stores (local or cloud)
  with familiar SQL.

### 20. `zarr_describe()` UDTF for xarray-style metadata
- **Commit:** `4228ca6`
- **Decision:** A User-Defined Table Function returning extended schema info
  (dimension sizes, chunk info, coord vs data_var); the CLI `DESCRIBE` uses it.
- **Rationale:** Standard SQL `DESCRIBE` loses the multidimensional structure
  scientific users need; exposing it as a UDTF also makes it queryable.

### 21. Ship `zarr-cli` as a static musl binary
- **Commit:** `fe7b02a`
- **Decision:** Distribute the CLI as a static x86_64 musl binary via GitHub
  releases.
- **Rationale:** Runs on any x86_64 Linux regardless of the system glibc — deploy
  to a bare VM with no Rust toolchain.

---

## Notable open gaps

These are acknowledged in `TODO.md` and the roadmap, and shape current design
direction:

- **Data-variable filter pushdown** (`WHERE temperature > 20`) is still post-scan;
  the sorted-coordinate trick does not apply to unsorted data values.
- **LIMIT + partitioning soundness** — falls back to a single partition, since a
  per-partition limit is unsound.
- **Inner-axis / multi-axis partitioning** — only axis 0 is split, so a single
  timestep over the full global grid stays on one worker.
- **Aggregate pushdown** (push `SUM`/`AVG`/`COUNT` to chunk level) and **Top-K**
  (`ORDER BY x LIMIT k` without a full sort) are not yet implemented.
- **Unbounded single-batch scan** — the scan emits the whole selection as one
  RecordBatch, so a query whose *selection* is large OOMs even when the *result*
  is tiny (e.g. a true monthly mean over hourly ERA5). The generic fix is a
  streaming scan that honors `batch_size`. This, and a broader vision of a
  structural exact-cardinality optimizer, are explored in
  [Exact-cardinality optimization](exact-cardinality-optimizer.md).
