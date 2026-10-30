-- Monthly anomaly against the climatology of the month's own base period.

select
    m.yr,
    m.mo,
    m.yr * 12 + (m.mo - 1) as t,     -- contiguous month counter
    m.sst_c - c.clim_sst   as anom
from {{ ref('stg_' ~ var('sst_source') ~ '_monthly') }} as m
join {{ ref('base_periods') }} as p
  on m.yr between p.yr_lo and p.yr_hi
join {{ ref('int_climatology') }} as c
  on c.yr_lo = p.yr_lo and c.mo = m.mo
