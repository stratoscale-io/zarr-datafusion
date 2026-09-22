# Pincode lookup — point extraction and feature fusion from a cloud grid

Two queries that turn a **list of postcodes into model-ready meteorology**, read
straight from the public ARCO-ERA5 store on GCS. No download, no conversion, no
staging step — the `CREATE EXTERNAL TABLE` points at the bucket and the query
runs.

| file | what it does | out | read |
|------|--------------|-----|------|
| [`pincode_temperature.sql`](pincode_temperature.sql) | 8 pincodes × 24 hourly steps → mean/max/min temperature | 8 rows | ~56 MB |
| [`met_feature_fusion.sql`](met_feature_fusion.sql) | 6 variables → 6 derived features, per city | 6 rows | ~394 MB |
| [`postcode_met_features.sql`](postcode_met_features.sql) | same 6 cities, pincodes resolved from **Parquet** ⋈ ERA5 **Zarr** | 6 rows | ~788 MB |

```bash
zarr-cli cookbook/pincode-lookup/pincode_temperature.sql
zarr-cli cookbook/pincode-lookup/met_feature_fusion.sql
zarr-cli cookbook/pincode-lookup/postcode_met_features.sql   # Parquet read from Hugging Face
```

## Tabular ⋈ gridded: pincodes from Parquet

The first two recipes hard-code coordinates in a `VALUES` list.
`postcode_met_features.sql` runs the same six cities and features as
`met_feature_fusion.sql`, but resolves each city's GPO pincode from a real
lookup table — [GeoNames postal codes](https://download.geonames.org/export/zip/)
(1.8M rows, 121 countries, CC-BY 4.0) converted to Parquet by
[`build_postcodes.sql`](build_postcodes.sql) and published at
[`Stratoscale/GeoNamesPincode`](https://huggingface.co/datasets/Stratoscale/GeoNamesPincode)
on Hugging Face — and joins it onto the ERA5 Zarr store in the same statement.
Two formats, two data models, two clouds, one query:

```sql
CREATE EXTERNAL TABLE postcodes STORED AS PARQUET
  LOCATION 'https://huggingface.co/datasets/Stratoscale/GeoNamesPincode/resolve/main/postal_codes.parquet';
CREATE EXTERNAL TABLE era5 STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';
```

```
city       postal_code  places  acc  grid_lat  grid_lon  wind_ms  blh_m  vent_coef_m2s
Delhi      110001           21    4     28.75     77.25     1.66    457        761
Hyderabad  500001            5    1     17.5      78.5      2.61    593       1550
Mumbai     400001            6    1     19.0      72.75     3.89    431       1676
Bengaluru  560001            9    4     13.25     77.5      2.61    699       1821
Chennai    600001            6    4     13.0      80.25     3.32    578       1923
Kolkata    700001           14    1     22.5      88.25     3.50    643       2251
```

Hyderabad, Mumbai, Chennai and Kolkata land on the same cell as the hand-picked
coordinates and reproduce `met_feature_fusion.sql` exactly. Delhi and Bengaluru
land one cell north: GeoNames has no official Indian pincode geometry and
geocodes by place name, and `accuracy` records how a point was derived, not how
close it is — most `560xxx` pincodes share one fallback district point ~28 km
north of the Bengaluru GPO. The query carries `places` and `accuracy` into the
output so that provenance stays visible; the lookup is only as good as the
table behind it.

## The trick: snap the lookup, not the grid

A postcode centroid never lands exactly on the 0.25° grid, and coordinate
matching is **exact equality** — there is no nearest-neighbour mode. So the
rounding happens on the lookup side of the join:

```sql
JOIN pins p
  ON e.latitude  = ROUND(p.lat/0.25)*0.25
 AND e.longitude = ROUND(p.lon/0.25)*0.25
```

Multiples of 0.25 are exact in f32/f64, so this stays a plain equality join. All
eight cities are east of the prime meridian; for a western-hemisphere point wrap
with `((lon + 360) % 360)` first, since ERA5 longitude is 0–360.

## What it costs

ARCO-ERA5 `chunk-1` stores **one timestep per chunk over the full lat×lon
plane**, so the pincode filter does not cut chunk reads — the `time` predicate
does. Adding pincodes is free; adding days is not. Both queries read a single
day (24 steps).

## Feature fusion

`met_feature_fusion.sql` reads temperature, dewpoint, wind u/v, boundary-layer
height and precipitation in one statement and derives relative humidity (Magnus),
wind speed from the vector components, and the **ventilation coefficient** —
boundary-layer height × wind speed, a standard operational air-quality metric.

It is the `ORDER BY` on purpose. Ranking on mixing height alone is misleading:

```
city         BLH    wind   ventilation
Delhi        439    1.80        790     <- worst dispersion
Hyderabad    593    2.61       1550
Mumbai       431    3.89       1676
Chennai      578    3.32       1923
Bengaluru    741    2.83       2098
Kolkata      643    3.50       2251
```

Mumbai's lid sits *below* Delhi's, yet Mumbai ventilates at more than twice the
wind speed and washes out with 8.5 mm of rain. A low lid only matters if the air
is also still — which is the argument that needs six variables in one query
rather than one.

This is dispersion **potential**, not pollution: there is no PM2.5 in ERA5. It is
one day (2021-06-01), so it is an anecdote, not a climatology — widen the `WHERE`
for the multi-year version.
