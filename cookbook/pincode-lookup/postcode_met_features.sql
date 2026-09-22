-- ============================================================================
--  Postcode met features: a Parquet lookup table joined onto a Zarr grid
-- ============================================================================
--  Same six cities and features as met_feature_fusion.sql, but the coordinates
--  are no longer hard-coded: each city's GPO pincode is resolved from a real
--  lookup table. Two storage formats, two data models, one SQL statement:
--
--    postcodes  -- TABULAR: GeoNames postal codes, Parquet on Hugging Face (HTTPS)
--    era5       -- GRIDDED: ARCO-ERA5, Zarr v3 on GCS (lat x lon x time)
--
--  Both are remote and public: nothing to download or build first. The Parquet
--  is https://huggingface.co/datasets/Stratoscale/GeoNamesPincode (1.8M rows,
--  CC-BY 4.0, (c) GeoNames), produced by build_postcodes.sql. For a
--  reproducible run, pin a commit instead of `main` in the LOCATION.
--
--  ONE POINT PER PINCODE: GeoNames has one row per place (post office), so a
--  pincode can have many rows (Delhi 110001 has 21). They are averaged into
--  one centroid; `places` and `accuracy` are kept so the provenance of every
--  coordinate stays visible in the output.
--
--  COORDINATE QUALITY: GeoNames has no official Indian pincode geometry and
--  geocodes by place name. Four cities land on the same cell as the hand-picked
--  coordinates in met_feature_fusion.sql; Bengaluru 560001 (a shared district
--  fallback point ~28 km north of the GPO) and Delhi 110001 land one cell north.
--  The lookup is only as good as the table behind it.
--
--  THE SNAP (see pincode_temperature.sql): round on the LOOKUP side so the
--  join stays exact equality. Multiples of 0.25 are exact in f32/f64.
--
--  FEATURES: temperature, relative humidity (Magnus), wind speed from the u/v
--  vector, boundary-layer height, daily precipitation, and the ventilation
--  coefficient (BLH x wind speed) -- see met_feature_fusion.sql for details.
--
--  COST: the `time` predicate drives chunk reads (one timestep = one full-globe
--  chunk per variable), not the pincode list. The Parquet side is noise next
--  to it: row-group statistics on the sorted file mean only the footer and the
--  matching row groups are fetched over HTTP.
--
--  Run:  zarr-cli cookbook/pincode-lookup/postcode_met_features.sql
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS postcodes
  STORED AS PARQUET
  LOCATION 'https://huggingface.co/datasets/Stratoscale/GeoNamesPincode/resolve/main/postal_codes.parquet';

CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

WITH wanted(city, postal_code) AS (VALUES
  ('Bengaluru','560001'),
  ('Mumbai',   '400001'),
  ('Delhi',    '110001'),
  ('Kolkata',  '700001'),
  ('Chennai',  '600001'),
  ('Hyderabad','500001')),
pins AS (                                        -- Parquet: one point per pincode
  SELECT w.city, p.postal_code,
    COUNT(*)          AS places,
    MAX(p.accuracy)   AS accuracy,
    AVG(p.latitude)   AS lat,
    AVG(p.longitude)  AS lon
  FROM postcodes p
  JOIN wanted w ON p.postal_code = w.postal_code
  WHERE p.country_code = 'IN'
  GROUP BY w.city, p.postal_code),
cells AS (                                       -- snap onto the ERA5 grid
  SELECT *,
    ROUND(lat/0.25)*0.25 AS grid_lat,
    ROUND(lon/0.25)*0.25 AS grid_lon
  FROM pins),
fused AS (                                       -- Zarr: hourly values per cell
  SELECT c.city, c.postal_code, c.places, c.accuracy, c.grid_lat, c.grid_lon,
    e."2m_temperature" - 273.15          AS temp_c,
    e."2m_dewpoint_temperature" - 273.15 AS dew_c,
    SQRT(POWER(e."10m_u_component_of_wind",2) + POWER(e."10m_v_component_of_wind",2)) AS wind_ms,
    e.boundary_layer_height              AS blh_m,
    e.total_precipitation * 1000         AS precip_mm
  FROM era5 e
  JOIN cells c
    ON e.latitude  = c.grid_lat
   AND e.longitude = c.grid_lon
  WHERE e.time >= '2021-06-01 00:00:00'
    AND e.time <  '2021-06-02 00:00:00')
SELECT city, postal_code, places, accuracy AS acc, grid_lat, grid_lon,
  ROUND(AVG(temp_c),1)    AS temp_c_mean,
  ROUND(MAX(temp_c),1)    AS temp_c_max,
  ROUND(AVG(100*EXP(17.625*dew_c/(243.04+dew_c))/EXP(17.625*temp_c/(243.04+temp_c))),0) AS rh_pct,
  ROUND(AVG(wind_ms),2)   AS wind_ms,
  ROUND(AVG(blh_m),0)     AS blh_m,
  ROUND(SUM(precip_mm),2) AS precip_mm,
  ROUND(AVG(blh_m) * AVG(wind_ms), 0) AS vent_coef_m2s   -- lid x flushing rate
FROM fused
GROUP BY city, postal_code, places, accuracy, grid_lat, grid_lon
ORDER BY vent_coef_m2s;                                  -- worst dispersion first
