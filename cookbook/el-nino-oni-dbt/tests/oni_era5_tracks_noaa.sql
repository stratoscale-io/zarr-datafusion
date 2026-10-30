{{ config(enabled=var('sst_source') == 'era5') }}

-- ERA5 is an independent SST product, so it can't match NOAA (built from
-- ERSST v5) season by season: the cookbook's residual reaches 0.95 °C in the
-- 1970s. Check the overall fit instead, and a tight bound from 2000 on.
-- Returns a row (fails) if the fit is off.
with paired as (
    select o.year, o.oni_c as ours, n.anom as noaa
    from {{ ref('fct_oni') }} as o
    join {{ ref('noaa_oni') }} as n
      on n.year = o.year and n.season = o.season
)
select
    count(*)                                                 as seasons,
    avg(abs(ours - noaa))                                    as mae,
    corr(ours, noaa)                                         as r,
    max(case when year >= 2000 then abs(ours - noaa) end)    as max_err_since_2000
from paired
having avg(abs(ours - noaa)) > 0.2
    or corr(ours, noaa) < 0.95
    or max(case when year >= 2000 then abs(ours - noaa) end) > 0.4
