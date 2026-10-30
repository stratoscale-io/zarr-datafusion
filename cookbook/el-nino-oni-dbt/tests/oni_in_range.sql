-- ONI has stayed within ±3 °C since 1950; anything outside is a unit or
-- sampling error.
select * from {{ ref('fct_oni') }}
where oni_c not between -4 and 4
