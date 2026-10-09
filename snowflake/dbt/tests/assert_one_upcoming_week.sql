-- The cheat sheet flags exactly one upcoming week (while the season has games left).
select count(distinct week) as upcoming_weeks
from {{ ref('mart_cheat_sheet') }}
where is_upcoming
having count(distinct week) > 1
