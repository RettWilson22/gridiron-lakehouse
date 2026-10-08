select
    {{ dbt.concat(["cast(season as " ~ dbt.type_string() ~ ")", "'-'", "team"]) }} as team_season_key,
    season,
    team,
    offensive_plays,
    epa_per_play,
    success_rate,
    fourth_downs,
    go_attempts,
    go_conversions,
    punts,
    field_goal_attempts,
    go_rate,
    conversion_rate
from {{ source('gold', 'team_season_summary') }}
