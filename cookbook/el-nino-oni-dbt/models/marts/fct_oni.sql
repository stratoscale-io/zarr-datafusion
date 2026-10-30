-- ONI: centred 3-month mean of the monthly anomaly, one row per season,
-- labelled by the year of its centre month (DJF 1998 = Dec 1997-Feb 1998).
-- A season needs both neighbouring months, as in the cookbook query.

with neighbours as (
    select
        yr,
        mo,
        t,
        anom,
        lag(t)     over (order by t) as prev_t,
        lag(anom)  over (order by t) as prev_anom,
        lead(t)    over (order by t) as next_t,
        lead(anom) over (order by t) as next_anom
    from {{ ref('int_anomaly') }}
),

oni as (
    select yr, mo, (prev_anom + anom + next_anom) / 3.0 as oni_raw
    from neighbours
    where prev_t = t - 1
      and next_t = t + 1
      and yr >= 1950
)

select
    o.yr                                 as year,
    o.mo                                 as centre_month,
    s.season,
    cast(o.yr as varchar) || '-' || s.season as season_key,
    round(o.oni_raw, 2)                  as oni_c,
    case when o.oni_raw >=  0.5 then 'El Niño'
         when o.oni_raw <= -0.5 then 'La Niña'
         else 'Neutral' end              as enso_phase
from oni as o
join {{ ref('season_labels') }} as s
  on s.mo = o.mo
