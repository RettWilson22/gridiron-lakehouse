-- The cheat sheet: every projected player-week of the current season, with the upcoming
-- week flagged. Completed weeks carry the actual points next to the projection, so the
-- same table answers "who do I start this week" and "how did last week's sheet do".
with projections as (
    select * from {{ ref('stg_projections') }}
),

current_season as (
    select max(season) as season from projections
),

upcoming as (
    select season, min(week) as week
    from {{ ref('stg_team_week') }}
    where not is_final
      and season = (select season from current_season)
    group by season
),

actuals as (
    select season, week, player_id, fantasy_points_ppr, fantasy_points_half, fantasy_points_std
    from {{ ref('stg_player_week') }}
),

games as (
    select game_id, team, is_final from {{ ref('stg_team_week') }}
)

select
    p.projection_key,
    p.season,
    p.week,
    coalesce(p.season = u.season and p.week = u.week, false) as is_upcoming,
    g.is_final as game_final,
    p.kind,
    p.player_id,
    p.player_name,
    p.position,
    p.team,
    p.opponent,
    p.is_home,
    {{ dbt.concat(["case when p.is_home then 'vs ' else '@ ' end", "p.opponent"]) }} as matchup,
    p.kickoff_at,
    p.implied_points,
    p.team_spread,
    p.depth_rank,
    coalesce(p.report_status, 'None') as injury_status,
    p.report_primary_injury as injury,
    p.opp_matchup_rank,
    p.ecr_rank,
    p.proj_ppr,
    p.floor_ppr,
    p.ceiling_ppr,
    p.pos_rank_ppr,
    p.tier_ppr,
    p.start_sit_ppr,
    p.proj_half,
    p.floor_half,
    p.ceiling_half,
    p.pos_rank_half,
    p.tier_half,
    p.start_sit_half,
    p.proj_std,
    p.floor_std,
    p.ceiling_std,
    p.pos_rank_std,
    p.tier_std,
    p.start_sit_std,
    p.proj_passing_yards,
    p.proj_passing_tds,
    p.proj_passing_interceptions,
    p.proj_rushing_yards,
    p.proj_rushing_tds,
    p.proj_receptions,
    p.proj_receiving_yards,
    p.proj_receiving_tds,
    p.proj_fumbles_lost,
    p.proj_two_point_conversions,
    p.baseline_last3,
    p.baseline_season_avg,
    case when g.is_final then coalesce(a.fantasy_points_ppr, 0) end as actual_ppr,
    case when g.is_final then coalesce(a.fantasy_points_half, 0) end as actual_half,
    case when g.is_final then coalesce(a.fantasy_points_std, 0) end as actual_std,
    p.model_version,
    p.generated_at
from projections p
inner join current_season c
    on p.season = c.season
left join upcoming u
    on p.season = u.season and p.week = u.week
left join games g
    on p.game_id = g.game_id and p.team = g.team
left join actuals a
    on p.season = a.season and p.week = a.week and p.player_id = a.player_id
