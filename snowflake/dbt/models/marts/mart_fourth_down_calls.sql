-- Play-level view for the app: each fourth down, what the model recommended, and how the
-- actual call compares.
select
    s.play_key,
    s.game_id,
    s.season,
    s.season_type,
    s.week,
    s.game_date,
    s.team,
    s.opponent,
    s.coach,
    s.qtr,
    s.ydstogo,
    s.yardline_100,
    s.score_differential,
    s.game_seconds_remaining,
    s.wp,
    s.decision,
    s.outcome,
    s.recommendation,
    s.p_convert,
    s.ev_go,
    s.ev_punt,
    s.ev_field_goal,
    s.go_advantage,
    s.expected_points_lost,
    s.is_neutral_situation,
    case
        when s.decision = s.recommendation then 'agreed'
        when s.recommendation = 'go' then 'too_conservative'
        when s.decision = 'go' then 'too_aggressive'
        else 'different_kick'
    end as call_quality,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score
from {{ ref('stg_fourth_down_scored') }} s
inner join {{ ref('stg_game_summary') }} g
    on s.game_id = g.game_id
