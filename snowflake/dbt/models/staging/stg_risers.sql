select
    {{ dbt.concat(["cast(season as " ~ dbt.type_string() ~ ")", "'-'", "cast(week as " ~ dbt.type_string() ~ ")", "'-'", "player_id"]) }} as riser_key,
    *
from {{ source('gold', 'risers') }}
