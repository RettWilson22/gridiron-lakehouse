-- Walk-forward backtest results as computed in the Databricks train task, with readable
-- method names for the app.
select
    metric_key,
    scope,
    position,
    method,
    case method
        when 'model' then 'Model'
        when 'last3' then 'Last 3 games average'
        when 'season_avg' then 'Season-to-date average'
        when 'ecr' then 'FantasyPros ECR'
    end as method_label,
    n,
    weeks,
    mae,
    rmse,
    bias,
    spearman,
    interval_coverage,
    pool_coverage
from {{ ref('stg_backtest_metrics') }}
