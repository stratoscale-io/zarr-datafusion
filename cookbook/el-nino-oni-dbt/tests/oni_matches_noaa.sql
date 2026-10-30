{{ config(enabled=var('sst_source') == 'ersst') }}

-- ERSST v5 is the dataset NOAA computes the ONI from, so every season should
-- match NOAA's published value to rounding.
select o.year, o.season, o.oni_c as ours, n.anom as noaa
from {{ ref('fct_oni') }} as o
join {{ ref('noaa_oni') }} as n
  on n.year = o.year and n.season = o.season
where abs(o.oni_c - n.anom) > 0.05
