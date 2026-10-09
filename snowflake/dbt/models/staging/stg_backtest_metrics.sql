select
    {{ dbt.concat(["scope", "'-'", "position", "'-'", "method"]) }} as metric_key,
    *
from {{ source('gold', 'backtest_metrics') }}
