-- How each team's fourth-down behaviour changed season over season.
with seasons as (
    select
        t.team_season_key,
        t.season,
        t.team,
        a.coach,
        t.fourth_downs,
        t.go_attempts,
        t.go_rate,
        t.conversion_rate,
        t.epa_per_play,
        a.go_rate_when_recommended,
        a.expected_points_lost_per_fourth_down
    from {{ ref('stg_team_season_summary') }} t
    left join {{ ref('stg_coach_aggressiveness') }} a
        on t.team_season_key = a.team_season_key
)

select
    seasons.*,
    lag(go_rate) over (partition by team order by season) as prior_go_rate,
    go_rate - lag(go_rate) over (partition by team order by season) as go_rate_change,
    lag(go_rate_when_recommended) over (
        partition by team order by season
    ) as prior_go_rate_when_recommended,
    case
        when lag(coach) over (partition by team order by season) is null then false
        else coach <> lag(coach) over (partition by team order by season)
    end as coach_changed,
    count(*) over (partition by team) as seasons_in_sample
from seasons
