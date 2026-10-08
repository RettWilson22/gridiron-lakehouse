"""Data access for the Streamlit app: Snowflake when deployed, DuckDB when run locally.

Both backends expose the same dbt marts (``marts.*``) and accept ``?`` bind parameters, so
the app's SQL is shared. Column names are normalised to lower case because Snowflake
returns upper-case names.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

HERE = Path(__file__).resolve().parent
DBT_DIR = HERE.parent / "dbt"


class Source(Protocol):
    name: str

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame: ...

    def recommend(
        self, ydstogo: float, yardline_100: float, score_diff: float, seconds_left: float
    ) -> dict[str, Any]: ...


def _lower(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(c).lower() for c in frame.columns]
    return frame


class SnowflakeSource:
    name = "Snowflake"

    def __init__(self, session: Any) -> None:
        self._session = session

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return _lower(self._session.sql(sql, params=params or []).to_pandas())

    def recommend(
        self, ydstogo: float, yardline_100: float, score_diff: float, seconds_left: float
    ) -> dict[str, Any]:
        row = self._session.sql(
            "SELECT APP.FOURTH_DOWN_RECOMMENDATION(?, ?, ?, ?) AS R",
            params=[ydstogo, yardline_100, score_diff, seconds_left],
        ).collect()[0]
        result: dict[str, Any] = json.loads(row["R"])
        return result


class DuckDBSource:
    name = "DuckDB (local)"

    def __init__(self, path: Path) -> None:
        import duckdb  # noqa: PLC0415 - local-only dependency, absent in Snowflake

        self._con = duckdb.connect(str(path), read_only=True)
        self.path = path
        spec = importlib.util.spec_from_file_location(
            "fourth_down_udf", HERE.parent / "snowpark" / "fourth_down_udf.py"
        )
        assert spec and spec.loader
        self._udf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._udf)

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return _lower(self._con.execute(sql, params or []).df())

    def recommend(
        self, ydstogo: float, yardline_100: float, score_diff: float, seconds_left: float
    ) -> dict[str, Any]:
        result: dict[str, Any] = self._udf.recommend(
            ydstogo, yardline_100, score_diff, seconds_left
        )
        return result


def local_database() -> Path:
    """Prefer the full local build, fall back to the CI fixture build."""
    configured = os.environ.get("GRIDIRON_DUCKDB")
    if configured:
        return Path(configured)
    for name in ("local.duckdb", "ci.duckdb"):
        if (DBT_DIR / name).exists():
            return DBT_DIR / name
    raise FileNotFoundError(
        "No DuckDB build found. Run `make dbt-ci` (fixture data) or `make dbt-local`."
    )


def connect() -> Source:
    try:
        from snowflake.snowpark.context import (  # noqa: PLC0415 - only exists in Snowflake
            get_active_session,
        )

        return SnowflakeSource(get_active_session())
    except Exception:  # not running inside Snowflake (or Snowpark not installed)
        return DuckDBSource(local_database())
