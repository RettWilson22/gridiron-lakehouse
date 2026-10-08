select
    {{ dbt.concat(["cast(season as " ~ dbt.type_string() ~ ")", "'-'", "team"]) }} as team_season_key,
    season,
    team,
    coach,
    neutral_fourth_downs,
    go_attempts,
    go_recommendations,
    went_for_it_when_recommended,
    expected_points_lost,
    avg_missed_go_advantage,
    go_rate,
    go_rate_when_recommended,
    expected_points_lost_per_fourth_down,
    aggressiveness_rank
from {{ source('gold', 'coach_aggressiveness') }}
