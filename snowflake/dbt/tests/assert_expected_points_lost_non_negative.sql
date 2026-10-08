-- The actual call can never beat the best option by the model's own valuation.
select play_key, expected_points_lost
from {{ ref('stg_fourth_down_scored') }}
where expected_points_lost < -0.000001
