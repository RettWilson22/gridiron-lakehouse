-- SQL and Python agree: per test season and position, the scorecard rebuilt here from the
-- published projections reproduces the backtest metrics computed in the Databricks train
-- task (same pool, same MAE, same interval coverage). Needs the full data, so it is
-- skipped on the CI target, whose fixture holds a handful of teams.
{{ config(enabled=(target.name != 'ci')) }}

with scorecard as (
    select
        cast(season as {{ dbt.type_string() }}) as scope,
        position,
        method,
        sum(n) as n,
        sum(abs_error_sum) / sum(n) as mae,
        sum(inside_count) * 1.0 / sum(n) as interval_coverage
    from {{ ref('mart_projection_scorecard') }}
    where method <> 'ecr' and kind = 'backtest'
    group by 1, 2, 3
)

select b.scope, b.position, b.method, b.n, s.n as scorecard_n, b.mae, s.mae as scorecard_mae
from {{ ref('mart_backtest_summary') }} b
inner join scorecard s
    on b.scope = s.scope and b.position = s.position and b.method = s.method
where b.n <> s.n
   or abs(b.mae - s.mae) > 0.0001
   or (b.method = 'model' and abs(b.interval_coverage - s.interval_coverage) > 0.0001)
