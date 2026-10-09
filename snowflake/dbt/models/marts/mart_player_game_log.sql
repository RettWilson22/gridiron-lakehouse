-- Game-by-game results for the current and previous season, with the projection that was
-- published for the game when there is one. Used for recent form on the Start / Sit page
-- and to rescore actual stat lines under custom league settings.
with current_season as (
    select max(season) as season from {{ ref('stg_projections') }}
)

select
    w.player_week_key,
    w.season,
    w.week,
    w.player_id,
    w.player_name,
    w.position,
    w.team,
    w.opponent,
    w.fantasy_points_ppr,
    w.fantasy_points_half,
    w.fantasy_points_std,
    w.passing_yards,
    w.passing_tds,
    w.passing_interceptions,
    w.rushing_yards,
    w.rushing_tds,
    w.receptions,
    w.receiving_yards,
    w.receiving_tds,
    w.fumbles_lost,
    w.two_point_conversions,
    w.special_teams_tds,
    w.targets,
    w.carries,
    w.snap_share,
    w.target_share,
    w.carry_share,
    w.expected_ppr,
    p.proj_ppr,
    p.floor_ppr,
    p.ceiling_ppr,
    p.kind as projection_kind
from {{ ref('stg_player_week') }} w
cross join current_season c
left join {{ ref('stg_projections') }} p
    on w.season = p.season and w.week = p.week and w.player_id = p.player_id
where w.season >= c.season - 1
