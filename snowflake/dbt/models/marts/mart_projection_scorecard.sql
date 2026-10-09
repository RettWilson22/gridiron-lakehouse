-- The accuracy record: projected vs actual by season, week and position, for the model and
-- the baselines, over the same evaluation pool the Python backtest uses (players ranked in
-- FantasyPros' positional top N that week, with a value for every method). One row per
-- season, week, position and method; sums are stored so any range of weeks aggregates
-- exactly (MAE = abs_error_sum / n, RMSE = sqrt(squared_error_sum / n)).
with games as (
    select game_id, team, is_final from {{ ref('stg_team_week') }}
),

actuals as (
    select season, week, player_id, fantasy_points_ppr from {{ ref('stg_player_week') }}
),

pool as (
    select
        p.season,
        p.week,
        p.position,
        p.kind,
        p.player_id,
        p.proj_ppr,
        p.floor_ppr,
        p.ceiling_ppr,
        p.baseline_last3,
        p.baseline_season_avg,
        p.ecr_rank,
        coalesce(a.fantasy_points_ppr, 0) as actual
    from {{ ref('stg_projections') }} p
    inner join games g
        on p.game_id = g.game_id and p.team = g.team and g.is_final
    left join actuals a
        on p.season = a.season and p.week = a.week and p.player_id = a.player_id
    where p.ecr_rank is not null
      and p.ecr_rank <= case p.position
          {% for position, size in var('pool_size').items() %}
          when '{{ position }}' then {{ size }}
          {% endfor %}
      end
      and p.baseline_last3 is not null
      and p.baseline_season_avg is not null
),

methods as (
    select season, week, position, kind, player_id, actual, 'model' as method,
        proj_ppr as prediction,
        case when actual between floor_ppr and ceiling_ppr then 1 else 0 end as inside
    from pool
    union all
    select season, week, position, kind, player_id, actual, 'last3', baseline_last3, null
    from pool
    union all
    select season, week, position, kind, player_id, actual, 'season_avg', baseline_season_avg, null
    from pool
    union all
    select season, week, position, kind, player_id, actual, 'ecr', -ecr_rank, null
    from pool
),

ranked as (
    -- Average ranks for ties, as in Spearman's coefficient.
    select
        methods.*,
        rank() over (partition by season, week, position, method order by prediction)
            + (count(*) over (partition by season, week, position, method, prediction) - 1) / 2.0
            as prediction_rank,
        rank() over (partition by season, week, position, method order by actual)
            + (count(*) over (partition by season, week, position, method, actual) - 1) / 2.0
            as actual_rank
    from methods
)

select
    {{ dbt.concat([
        "cast(season as " ~ dbt.type_string() ~ ")", "'-'",
        "cast(week as " ~ dbt.type_string() ~ ")", "'-'", "position", "'-'", "method"
    ]) }} as scorecard_key,
    season,
    week,
    position,
    method,
    max(kind) as kind,
    count(*) as n,
    case when method = 'ecr' then null else sum(abs(prediction - actual)) end as abs_error_sum,
    case when method = 'ecr' then null else sum((prediction - actual) * (prediction - actual)) end
        as squared_error_sum,
    case when method = 'ecr' then null else sum(prediction - actual) end as error_sum,
    case when method = 'ecr' then null else avg(abs(prediction - actual)) end as mae,
    sum(inside) as inside_count,
    case when count(*) >= 5 then corr(prediction_rank, actual_rank) end as spearman
from ranked
group by season, week, position, method
