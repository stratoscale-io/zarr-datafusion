-- Mean SST per calendar month over each block's 30-year base period.
-- A table: it's small (13 blocks x 12 months) and every anomaly reads it, so
-- the Zarr samples behind it are read once per `dbt run`, not per query.

select
    p.yr_lo,
    m.mo,
    avg(m.sst_c) as clim_sst
from {{ ref('base_periods') }} as p
join {{ ref('stg_' ~ var('sst_source') ~ '_monthly') }} as m
  on m.yr between p.base_lo and p.base_hi
group by p.yr_lo, m.mo
