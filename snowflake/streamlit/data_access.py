"""Data access for the cheat sheet app, in three modes that share every query.

* Snowflake: Streamlit in Snowflake reads the dbt marts in ``GRIDIRON.MARTS`` and scores
  custom league settings with the ``APP.FANTASY_POINTS`` Python UDF.
* Local: a DuckDB build of the same dbt marts (``make dbt-local`` or ``make dbt-ci``);
  custom scoring runs the UDF's Python handler in-process.
* Public: a static snapshot of the marts as Parquet files (``streamlit_public/snapshot``),
  for the copy on Streamlit Community Cloud; queried with an in-memory DuckDB.

All modes accept ``?`` bind parameters and return lower-case column names.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Protocol

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DBT_DIR = REPO / "snowflake" / "dbt"
SNAPSHOT_DIR = REPO / "streamlit_public" / "snapshot"

# The marts the app reads. scripts/export_public_snapshot.py exports the same list.
MARTS: Final = (
    "mart_cheat_sheet",
    "mart_risers",
    "mart_projection_scorecard",
    "mart_player_game_log",
    "mart_backtest_summary",
)
# Stat-line keys understood by gridiron.scoring and the FANTASY_POINTS UDF. Copies of
# gridiron.features.COMPONENTS and gridiron.scoring.STATS, because Streamlit in Snowflake
# runs this file without the package; tests/test_scoring.py fails if they drift.
PROJECTED_STATS: Final = (
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
    "two_point_conversions",
)
ACTUAL_STATS: Final = (*PROJECTED_STATS, "special_teams_tds")


def load_helper(name: str) -> ModuleType:
    """Import a stdlib-only helper module from the ``gridiron`` package.

    Locally the package is installed; on Streamlit Community Cloud the repository's ``src``
    folder is put on the path; in Streamlit in Snowflake the module file is uploaded next
    to the app.
    """
    try:
        return importlib.import_module(f"gridiron.{name}")
    except ImportError:
        src = REPO / "src"
        if (src / "gridiron").is_dir():
            sys.path.insert(0, str(src))
            return importlib.import_module(f"gridiron.{name}")
        return importlib.import_module(name)


def _lower(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(c).lower() for c in frame.columns]
    return frame


def stat_object_sql(stats: tuple[str, ...], prefix: str) -> str:
    """``OBJECT_CONSTRUCT`` of a stat line held in columns ``<prefix><stat>``."""
    pairs = ", ".join(f"'{stat}', {prefix}{stat}" for stat in stats)
    return f"OBJECT_CONSTRUCT({pairs}, 'position', position)"


class Source(Protocol):
    name: str
    description: str

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame: ...

    def custom_points(
        self,
        table: str,
        *,
        stats: tuple[str, ...],
        prefix: str,
        where: str,
        params: list[Any],
        scoring: dict[str, Any],
    ) -> pd.DataFrame: ...


class SnowflakeSource:
    name = "snowflake"
    description = "Snowflake (live data)"

    def __init__(self, session: Any) -> None:
        self._session = session

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return _lower(self._session.sql(sql, params=params or []).to_pandas())

    def custom_points(
        self,
        table: str,
        *,
        stats: tuple[str, ...],
        prefix: str,
        where: str,
        params: list[Any],
        scoring: dict[str, Any],
    ) -> pd.DataFrame:
        """Score every row's stat line in Snowflake with the FANTASY_POINTS UDF."""
        sql = (
            f"select player_id, APP.FANTASY_POINTS({stat_object_sql(stats, prefix)}, "
            f"PARSE_JSON(?)) as custom_points from {table} where {where}"
        )
        return self.query(sql, [json.dumps(scoring), *params])


class DuckDBSource:
    name = "local"
    description = "DuckDB (local build of the dbt marts)"

    def __init__(self, path: Path | None = None) -> None:
        import duckdb  # noqa: PLC0415 - not available in Streamlit in Snowflake

        self._con = duckdb.connect(str(path) if path else ":memory:", read_only=bool(path))

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return _lower(self._con.execute(sql, params or []).df())

    def custom_points(
        self,
        table: str,
        *,
        stats: tuple[str, ...],
        prefix: str,
        where: str,
        params: list[Any],
        scoring: dict[str, Any],
    ) -> pd.DataFrame:
        """Score in Python with the same handler the Snowflake UDF runs."""
        scorer = load_helper("scoring")
        columns = ", ".join(f"{prefix}{stat} as {stat}" for stat in stats)
        rows = self.query(
            f"select player_id, position, {columns} from {table} where {where}", params
        )
        points = [scorer.fantasy_points(r, scoring) for r in rows.to_dict("records")]
        return pd.DataFrame({"player_id": rows["player_id"], "custom_points": points})


class SnapshotSource(DuckDBSource):
    name = "public"
    description = "static snapshot, refreshed by hand (public copy)"

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self._con.execute("create schema if not exists marts")
        for mart in MARTS:
            path = (directory / f"{mart}.parquet").as_posix()
            self._con.execute(f"create view marts.{mart} as select * from read_parquet('{path}')")


def local_database() -> Path | None:
    """The full local build if it exists, else the CI fixture build."""
    configured = os.environ.get("GRIDIRON_DUCKDB")
    if configured:
        return Path(configured)
    for name in ("local.duckdb", "ci.duckdb"):
        if (DBT_DIR / name).exists():
            return DBT_DIR / name
    return None


def connect(mode: str | None = None) -> Source:
    """Pick the data source: ``GRIDIRON_APP_MODE`` if set, else whatever is available."""
    mode = mode or os.environ.get("GRIDIRON_APP_MODE")
    if mode in (None, "snowflake"):
        try:
            from snowflake.snowpark.context import (  # noqa: PLC0415 - Snowflake only
                get_active_session,
            )

            return SnowflakeSource(get_active_session())
        except Exception:
            if mode == "snowflake":
                raise
    if mode == "public":
        return SnapshotSource(Path(os.environ.get("GRIDIRON_SNAPSHOT_DIR", SNAPSHOT_DIR)))
    database = local_database()
    if database is not None:
        return DuckDBSource(database)
    if SNAPSHOT_DIR.is_dir():
        return SnapshotSource(SNAPSHOT_DIR)
    raise FileNotFoundError(
        "No data found. Run `make dbt-ci` (fixture data) or `make dbt-local` (full local run)."
    )
