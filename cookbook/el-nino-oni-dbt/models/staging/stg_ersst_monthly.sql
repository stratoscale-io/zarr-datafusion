{{ config(enabled=var('sst_source') == 'ersst') }}

-- Niño-3.4 box (5°S-5°N, 170°W-120°W) spatial mean per month, from ERSST v5
-- (already monthly, already °C). This and stg_era5_monthly are the only
-- models that read Zarr; var('sst_source') enables one of them.
--
-- Every WHERE condition is on a coordinate, so it reaches the Zarr scan. The
-- year range is written here, not left to downstream models: fct_oni's
-- window function stops filters from passing through to this view.

select yr, mo, avg(sst_c) as sst_c
from (
    select
        cast(extract(year  from ts) as int) as yr,
        cast(extract(month from ts) as int) as mo,
        sst                                 as sst_c
    from (
        select
            -- time is dictionary-encoded; cast before using date functions
            arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') as ts,
            sst
        from {{ source('raw', 'ersst') }}
        where lat between  -5.0 and   5.0
          and lon between 190.0 and 240.0
    ) as box
    where extract(year from ts) between {{ var('first_year') }} and {{ var('last_year') }}
) as samples
group by yr, mo
-- guard against fill or masked cells
having avg(sst_c) between 0 and 50
