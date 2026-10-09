-- Every method is scored on the same players: per season, week and position, all four
-- methods have the same row count.
select season, week, position
from {{ ref('mart_projection_scorecard') }}
group by season, week, position
having count(distinct n) > 1 or count(*) <> 4
