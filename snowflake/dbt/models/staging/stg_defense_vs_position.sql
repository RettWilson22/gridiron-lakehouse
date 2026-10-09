select
    {{ dbt.concat(["cast(season as " ~ dbt.type_string() ~ ")", "'-'", "cast(week as " ~ dbt.type_string() ~ ")", "'-'", "team", "'-'", "position"]) }} as defense_week_key,
    season,
    week,
    game_id,
    team as defense,
    opponent as offense,
    position,
    ppr_allowed,
    half_allowed,
    std_allowed,
    ppr_allowed_l6,
    games_l6,
    matchup_rank
from {{ source('gold', 'defense_vs_position') }}
