-- The leaderboard's counts must equal a recount from the play-level data.
with recount as (
    select
        season,
        team,
        count(*) as neutral_fourth_downs,
        sum(case when recommendation = 'go' then 1 else 0 end) as go_recommendations,
        sum(case when recommendation = 'go' and decision = 'go' then 1 else 0 end)
            as went_for_it_when_recommended
    from {{ ref('stg_fourth_down_scored') }}
    where season_type = 'REG' and is_neutral_situation
    group by season, team
)

select l.team_season_key
from {{ ref('mart_fourth_down_leaderboard') }} l
left join recount r
    on l.season = r.season and l.team = r.team
where r.team is null
   or l.neutral_fourth_downs <> r.neutral_fourth_downs
   or l.go_recommendations <> r.go_recommendations
   or l.went_for_it_when_recommended <> r.went_for_it_when_recommended
