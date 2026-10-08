-- Team-season leaderboard of fourth-down aggressiveness relative to the model, with the
-- league context needed to read it (season average and each team's gap to it).
with aggressiveness as (
    select * from {{ ref('stg_coach_aggressiveness') }}
),

teams as (
    select * from {{ ref('stg_team_season_summary') }}
),

joined as (
    select
        a.team_season_key,
        a.season,
        a.team,
        a.coach,
        a.neutral_fourth_downs,
        a.go_recommendations,
        a.went_for_it_when_recommended,
        a.go_rate_when_recommended,
        a.go_rate as neutral_go_rate,
        a.expected_points_lost,
        a.expected_points_lost_per_fourth_down,
        a.avg_missed_go_advantage,
        a.aggressiveness_rank,
        t.epa_per_play,
        t.go_attempts as season_go_attempts,
        t.conversion_rate as season_conversion_rate
    from aggressiveness a
    inner join teams t
        on a.team_season_key = t.team_season_key
)

select
    joined.*,
    avg(go_rate_when_recommended) over (partition by season) as league_go_rate_when_recommended,
    go_rate_when_recommended
        - avg(go_rate_when_recommended) over (partition by season) as go_rate_vs_league,
    rank() over (
        partition by season order by expected_points_lost_per_fourth_down asc
    ) as decision_quality_rank
from joined
