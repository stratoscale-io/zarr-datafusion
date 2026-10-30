{{ config(enabled=var('sst_source') == 'era5') }}

-- Niño-3.4 box (5°S-5°N, 170°W-120°W) from ERA5, sampled at noon on the 15th
-- of each month and averaged over space.
--
-- Every WHERE condition is on a coordinate (time via extract), so it reaches
-- the Zarr scan and only one time step per month is read. The year range is
-- written here, not left to downstream models: fct_oni's window function
-- stops filters from passing through to this view.

select yr, mo, avg(sst_c) as sst_c
from (
    select
        cast(extract(year  from ts) as int) as yr,
        cast(extract(month from ts) as int) as mo,
        sst_c
    from (
        select
            -- time is dictionary-encoded; cast before using date functions
            arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') as ts,
            sea_surface_temperature - 273.15                        as sst_c
        from {{ source('raw', 'era5') }}
        where latitude  between  -5.0 and   5.0
          and longitude between 190.0 and 240.0
    ) as box
    where extract(day  from ts) = 15
      and extract(hour from ts) = 12
      and extract(year from ts) between {{ var('first_year') }} and {{ var('last_year') }}
) as samples
group by yr, mo
-- absent chunks read back as NaN: drop those months
having avg(sst_c) between 0 and 50
