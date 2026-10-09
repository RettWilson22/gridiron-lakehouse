"""Rebuild the checked-in test fixtures from locally landed data and the local lakehouse.

1. ``tests/fixtures/landing``: a small slice of every landed dataset in the landing layout
   (games involving the four NFC North teams in 2024 weeks 1-4, 2025 weeks 1-4 and 2026
   weeks 1-5, where week 5 is unplayed). It covers both depth chart formats and, for the
   snapshot format, includes snapshots taken after kickoff that must never be used. Tests
   run the real pipeline over it.
2. ``tests/fixtures/gold``: the serving tables from a full local run, filtered to the same
   games. They are the column contract for the generated Snowflake DDL and the input of
   the dbt ``ci`` target and the Streamlit app test.

    python scripts/build_fixtures.py --landing-root data/landing --lakehouse-dir data/lakehouse
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gridiron.serving import SERVING_TABLES
from gridiron.transforms import PLAY_COLUMNS

TEAMS = ("DET", "GB", "MIN", "CHI")
SKILL = ("QB", "RB", "FB", "WR", "TE")
WEEKS = {2024: range(1, 5), 2025: range(1, 5), 2026: range(1, 6)}
FIXTURES = Path("tests/fixtures")


def latest_file(folder: Path) -> Path:
    files = sorted(folder.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"nothing landed in {folder}")
    return files[-1]


def write(frame: pd.DataFrame, source: Path, out_root: Path, landing_root: Path) -> None:
    target = out_root / source.relative_to(landing_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    schema = pq.read_schema(source)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table.cast(pa.schema([schema.field(c) for c in table.column_names])), target)


def fixture_games(landing_root: Path) -> pd.DataFrame:
    games = pd.read_parquet(latest_file(landing_root / "schedules"))
    wanted = pd.concat(
        [games[(games["season"] == s) & games["week"].isin(list(w))] for s, w in WEEKS.items()]
    )
    wanted = wanted[(wanted["game_type"] == "REG")]
    return wanted[wanted["home_team"].isin(TEAMS) | wanted["away_team"].isin(TEAMS)]


def select_depth_charts(
    depth: pd.DataFrame, season_games: pd.DataFrame, teams: set[str], weeks: range
) -> pd.DataFrame:
    """Weekly charts for the fixture weeks, or for snapshot charts the last snapshot before
    and the first snapshot after each kickoff (the latter must never be used)."""
    if "dt" not in depth:
        return depth[
            depth["club_code"].isin(teams)
            & depth["week"].isin(list(weeks))
            & (depth["formation"] == "Offense")
        ]
    depth = depth[depth["team"].isin(teams) & depth["pos_abb"].isin(["QB", "RB", "WR", "TE"])]
    unique = sorted(set(pd.to_datetime(depth["dt"], utc=True)))
    kickoffs = (
        pd.to_datetime(season_games["gameday"] + " " + season_games["gametime"])
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    keep: set[str] = {str(depth["dt"].max())}
    for kickoff in kickoffs:
        before = [s for s in unique if s < kickoff]
        after = [s for s in unique if s >= kickoff]
        keep |= {s.strftime("%Y-%m-%dT%H:%M:%SZ") for s in before[-1:] + after[:1]}
    return depth[depth["dt"].isin(keep)]


def build_landing(landing_root: Path, out_root: Path) -> None:
    games = fixture_games(landing_root)
    game_ids = set(games["game_id"])
    teams = set(games["home_team"]) | set(games["away_team"])
    write(games, latest_file(landing_root / "schedules"), out_root, landing_root)
    player_ids: set[str] = set()
    pfr_ids: set[str] = set()

    for season, weeks in WEEKS.items():

        def path(dataset: str, season: int = season) -> Path:
            return latest_file(landing_root / dataset / str(season))

        pbp = pd.read_parquet(path("pbp"), columns=list(PLAY_COLUMNS))
        write(pbp[pbp["game_id"].isin(game_ids)], path("pbp"), out_root, landing_root)
        for dataset in ("player_stats", "snap_counts", "ff_opportunity"):
            frame = pd.read_parquet(path(dataset))
            frame = frame[frame["game_id"].isin(game_ids)]
            if dataset == "snap_counts":
                frame = frame[frame["position"].isin(SKILL)]
            write(frame, path(dataset), out_root, landing_root)
            if dataset == "player_stats":
                player_ids |= set(frame["player_id"].dropna())
            if dataset == "snap_counts":
                pfr_ids |= set(frame["pfr_player_id"].dropna())
        for dataset in ("rosters", "injuries"):
            frame = pd.read_parquet(path(dataset))
            frame = frame[
                frame["team"].isin(teams)
                & frame["week"].isin(list(weeks))
                & (frame["game_type"] == "REG")
                & frame["position"].isin(SKILL)
            ]
            write(frame, path(dataset), out_root, landing_root)
            player_ids |= set(frame["gsis_id"].dropna())

        depth = select_depth_charts(
            pd.read_parquet(path("depth_charts")), games[games["season"] == season], teams, weeks
        )
        write(depth, path("depth_charts"), out_root, landing_root)
        player_ids |= set(depth["gsis_id"].dropna())

    players = pd.read_parquet(latest_file(landing_root / "players"))
    players = players[
        (players["gsis_id"].isin(player_ids) | players["pfr_id"].isin(pfr_ids))
        & players["position_group"].isin(["QB", "RB", "WR", "TE"])
    ]
    write(players, latest_file(landing_root / "players"), out_root, landing_root)
    crosswalk = pd.read_parquet(latest_file(landing_root / "player_ids"))
    crosswalk = crosswalk[crosswalk["gsis_id"].isin(set(players["gsis_id"]))]
    write(crosswalk, latest_file(landing_root / "player_ids"), out_root, landing_root)
    fantasypros = set(crosswalk["fantasypros_id"].dropna())
    for season in WEEKS:
        folder = landing_root / "ecr" / str(season)
        if not folder.is_dir():
            continue
        ecr = pd.read_parquet(latest_file(folder))
        ecr = ecr[ecr["id"].isin(fantasypros)]
        season_games = games[games["season"] == season]
        first = dt.date.fromisoformat(season_games["gameday"].min()) - dt.timedelta(days=7)
        last = dt.date.fromisoformat(season_games["gameday"].max())
        dates = pd.to_datetime(ecr["scrape_date"]).dt.date
        write(ecr[(dates >= first) & (dates <= last)], latest_file(folder), out_root, landing_root)


def build_gold(lakehouse_dir: Path, landing_root: Path, out_dir: Path) -> None:
    games = fixture_games(landing_root)
    game_ids = set(games["game_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SERVING_TABLES:
        # Filter with Arrow rather than pandas so nullable integer columns keep their types.
        table = pq.read_table(lakehouse_dir / f"{name}.parquet")
        if "game_id" in table.column_names:
            table = table.filter(pc.is_in(table["game_id"], pa.array(sorted(game_ids))))
        elif name == "risers":
            table = table.filter(pc.is_in(table["team"], pa.array(TEAMS)))
        pq.write_table(table.replace_schema_metadata(None), out_dir / f"{name}.parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landing-root", type=Path, default=Path("data/landing"))
    parser.add_argument("--lakehouse-dir", type=Path, default=Path("data/lakehouse"))
    args = parser.parse_args(argv)

    landing_out = FIXTURES / "landing"
    shutil.rmtree(landing_out, ignore_errors=True)
    build_landing(args.landing_root, landing_out)
    build_gold(args.lakehouse_dir, args.landing_root, FIXTURES / "gold")
    for path in sorted(FIXTURES.rglob("*.parquet")):
        print(f"{path}: {pq.read_metadata(path).num_rows} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
