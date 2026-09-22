-- ============================================================================
--  Build the global postcode lookup: GeoNames TSV -> Parquet
-- ============================================================================
--  GeoNames publishes ~1.8M postal-code rows for 121 countries as one
--  headerless, tab-separated UTF-8 file (CC-BY 4.0). This turns it into a
--  sorted, zstd-compressed Parquet file that postcode_met_features.sql joins
--  onto the ERA5 Zarr store.
--
--  Fetch first (from the repo root):
--    mkdir -p data/geonames && cd data/geonames
--    curl -O https://download.geonames.org/export/zip/allCountries.zip
--    unzip allCountries.zip allCountries.txt
--
--  One row is a PLACE, not a postcode: 1,826,904 rows cover 1,080,715 distinct
--  (country_code, postal_code) pairs. Every row is kept here; the consumer
--  collapses to one point per postcode.
--
--  `accuracy` records HOW the coordinate was derived (1 = estimated,
--  4 = matched a GeoNames place, 6 = centroid of addresses or shape), not how
--  close it is. Countries without published postal geometry (e.g. India) fall
--  back to shared district points -- check before trusting a postcode there.
--
--  Sorted by (country_code, postal_code) so row-group statistics let a
--  country/postcode filter skip most of the file.
--
--  Published at https://huggingface.co/datasets/Stratoscale/GeoNamesPincode
--
--  Run:  zarr-cli cookbook/pincode-lookup/build_postcodes.sql   (~3 s, 19 MB)
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS geonames_postal (
  country_code VARCHAR NOT NULL,
  postal_code  VARCHAR NOT NULL,
  place_name   VARCHAR,
  admin_name1  VARCHAR,    -- state / province
  admin_code1  VARCHAR,
  admin_name2  VARCHAR,    -- county / district
  admin_code2  VARCHAR,
  admin_name3  VARCHAR,    -- community / sub-district
  admin_code3  VARCHAR,
  latitude     DOUBLE NOT NULL,
  longitude    DOUBLE NOT NULL,
  accuracy     TINYINT
) STORED AS CSV
  LOCATION 'data/geonames/allCountries.txt'
  OPTIONS ('format.delimiter' E'\t', 'format.has_header' 'false');

COPY (SELECT * FROM geonames_postal ORDER BY country_code, postal_code, place_name)
  TO 'data/geonames/postal_codes.parquet'
  STORED AS PARQUET
  OPTIONS ('format.compression' 'zstd(3)');
