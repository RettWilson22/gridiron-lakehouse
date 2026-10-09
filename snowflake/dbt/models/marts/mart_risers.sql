-- Players whose usage over their last three games is up on their earlier games, as of the
-- start of each week of the current season, with that week's projection alongside.
with risers as (
    select * from {{ ref('stg_risers') }}
),

current_season as (
    select max(season) as season from {{ ref('stg_projections') }}
)

select
    r.riser_key,
    r.season,
    r.week,
    r.player_id,
    r.player_name,
    r.position,
    r.team,
    r.is_riser,
    r.riser_rank,
    r.baseline_basis,
    r.games_recent,
    r.games_earlier,
    r.expected_ppr_recent,
    r.expected_ppr_before,
    r.expected_ppr_change,
    r.snap_share_recent,
    r.snap_share_before,
    r.snap_share_change,
    r.target_share_recent,
    r.target_share_before,
    r.target_share_change,
    r.carry_share_recent,
    r.carry_share_before,
    r.carry_share_change,
    r.red_zone_share_recent,
    r.red_zone_share_before,
    r.red_zone_share_change,
    c.proj_ppr,
    c.pos_rank_ppr,
    c.matchup
from risers r
inner join current_season s
    on r.season = s.season
left join {{ ref('mart_cheat_sheet') }} c
    on r.season = c.season and r.week = c.week and r.player_id = c.player_id
