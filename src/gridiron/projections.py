"""Turn model output into the published tables: projections (with ranks, tiers and
start/sit labels in every scoring format) and usage risers.

Live projections are frozen once a game kicks off: a later run (for example Saturday's,
after the final injury reports) only replaces projections for games that have not started,
so the published track record is exactly what was shown before each game.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from gridiron.features import COMPONENTS, KEY, with_baselines
from gridiron.model import FORMATS
from gridiron.tiers import POOL, assign_tiers, start_sit

PROJECTION_COLUMNS: Final = (
    "season",
    "week",
    "game_id",
    "kickoff_at",
    "player_id",
    "player_name",
    "position",
    "team",
    "opponent",
    "is_home",
    "implied_points",
    "team_spread",
    "total_line",
    "depth_rank",
    "report_status",
    "report_primary_injury",
    "opp_matchup_rank",
    *(f"proj_{c}" for c in COMPONENTS),
    *(f"{kind}_{fmt}" for fmt in FORMATS for kind in ("proj", "floor", "ceiling")),
    "baseline_last3",
    "baseline_season_avg",
    "ecr_rank",
    "kind",
    "model_version",
    "generated_at",
)

RANKING_COLUMNS: Final = tuple(
    f"{kind}_{fmt}" for fmt in FORMATS for kind in ("pos_rank", "tier", "start_sit")
)

# Risers: usage over a player's last three games against his earlier games this season
# (or last season's average when he has fewer than two earlier games this season).
RISER_RECENT_GAMES: Final = 3
RISER_MIN_EARLIER_GAMES: Final = 2
RISER_MIN_XFP_GAIN: Final = 2.5
RISER_MIN_RECENT_XFP: Final = 5.0
USAGE: Final = ("snap_share", "target_share", "carry_share", "red_zone_share", "expected_ppr")
# The risers table layout. It is the same with or without rows (no upcoming week in the
# offseason), so the serving table and its Parquet files keep one schema.
RISER_COLUMNS: Final[dict[str, str]] = {
    "season": "int64",
    "week": "int64",
    "player_id": "string",
    "player_name": "string",
    "position": "string",
    "team": "string",
    "games_recent": "int64",
    "games_earlier": "int64",
    "baseline_basis": "string",
    **{
        f"{metric}_{part}": "float64" for metric in USAGE for part in ("recent", "before", "change")
    },
    "is_riser": "bool",
    "riser_rank": "int64",
}


def upcoming_week(team_week: pd.DataFrame) -> tuple[int, int] | None:
    """The next regular-season week of the current season (the latest one in the schedule)
    with a game that has not been played, if any. Unfinished games from earlier seasons
    (a cancelled game never gets a final score) are ignored."""
    if team_week.empty:
        return None
    current = team_week[team_week["season"] == team_week["season"].max()]
    pending = current[~current["is_final"].astype(bool)]
    if pending.empty:
        return None
    first = pending.sort_values(["season", "week"]).iloc[0]
    return int(first["season"]), int(first["week"])


def projection_frame(
    frame: pd.DataFrame, kind: str, model_version: str, now: pd.Timestamp
) -> pd.DataFrame:
    """Candidate rows with model output columns, in the published column layout."""
    frame = with_baselines(frame).assign(kind=kind, model_version=model_version, generated_at=now)
    frame["kickoff_at"] = pd.to_datetime(frame["kickoff_at"], utc=True)
    frame["is_home"] = frame["is_home"].astype("boolean")
    if "ecr_rank" not in frame:
        frame["ecr_rank"] = np.nan
    return frame[list(PROJECTION_COLUMNS)]


def attach_ecr_rank(frame: pd.DataFrame, ecr: pd.DataFrame) -> pd.DataFrame:
    """Add the FantasyPros positional rank for the week, when the player was ranked at the
    position he is projected at (rankings are scraped on Fridays, so the live week only has
    one after Friday)."""
    rankings = ecr.dropna(subset=["player_id"])[[*KEY, "position", "ecr_rank"]]
    rankings = rankings.rename(columns={"position": "_ecr_position", "ecr_rank": "_ecr_rank"})
    out = frame.merge(rankings.drop_duplicates(KEY), on=KEY, how="left")
    same = out["_ecr_position"] == out["position"]
    out["ecr_rank"] = out["_ecr_rank"].where(same).astype("Int64")
    return out.drop(columns=["_ecr_position", "_ecr_rank"])


def merge_live(existing: pd.DataFrame | None, new: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Combine stored live projections with a fresh run.

    Stored rows for games that have kicked off are kept as they were. Fresh rows are only
    used for games that have not kicked off yet, and replace stored rows for those games.
    """
    fresh = new[pd.to_datetime(new["kickoff_at"], utc=True) > now]
    if existing is None or existing.empty:
        return fresh.reset_index(drop=True)
    started = pd.to_datetime(existing["kickoff_at"], utc=True) <= now
    replaced_weeks = set(map(tuple, fresh[["season", "week"]].drop_duplicates().to_numpy()))
    in_fresh_week = existing[["season", "week"]].apply(tuple, axis=1).isin(replaced_weeks)
    keep = existing[started | ~in_fresh_week]
    fresh = fresh.merge(keep[KEY], on=KEY, how="left", indicator=True)
    fresh = fresh[fresh["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([keep, fresh], ignore_index=True).sort_values(KEY).reset_index(drop=True)


def assemble(backtest: pd.DataFrame, live: pd.DataFrame | None, ecr: pd.DataFrame) -> pd.DataFrame:
    """All published projections: live weeks use only live rows, other weeks the backtest."""
    if live is None or live.empty:
        combined = backtest
    else:
        live_weeks = live[["season", "week"]].drop_duplicates()
        backtest = backtest.merge(live_weeks, on=["season", "week"], how="left", indicator=True)
        backtest = backtest[backtest["_merge"] == "left_only"].drop(columns="_merge")
        combined = pd.concat([backtest, live], ignore_index=True)
    combined = attach_ecr_rank(combined.drop(columns="ecr_rank"), ecr)
    ranked = add_rankings(combined.reset_index(drop=True))
    return ranked[[*PROJECTION_COLUMNS, *RANKING_COLUMNS]]


def add_rankings(frame: pd.DataFrame) -> pd.DataFrame:
    """Positional rank, tier and start/sit label per week in each scoring format.

    Tiers are computed among the top ``tiers.POOL`` players at the position (the pool a
    12-team league would consider); players below that get no tier.
    """
    out = frame.copy()
    for fmt in FORMATS:
        projection = f"proj_{fmt}"
        rank = out.groupby(["season", "week", "position"])[projection].rank(
            ascending=False, method="first"
        )
        out[f"pos_rank_{fmt}"] = rank.astype("int64")
        out[f"start_sit_{fmt}"] = [
            start_sit(p, int(r)) for p, r in zip(out["position"], rank, strict=True)
        ]
        tiers = pd.Series(np.nan, index=out.index)
        for (_, _, position), group in out.groupby(["season", "week", "position"]):
            pool = group[group[f"pos_rank_{fmt}"] <= POOL[str(position)]]
            tiers.loc[pool.index] = np.asarray(assign_tiers(pool[projection].tolist()), "float64")
        out[f"tier_{fmt}"] = tiers.astype("Int64")
    return out


def risers(player_week: pd.DataFrame, weeks: list[tuple[int, int]]) -> pd.DataFrame:
    """Usage over each player's last three games vs. before, as of the start of each week."""
    records = []
    games = player_week.sort_values(["player_id", "season", "week"])
    previous_season = games.groupby(["player_id", "season"])[list(USAGE)].mean().reset_index()
    previous_season["season"] += 1
    previous_season = previous_season.set_index(["player_id", "season"])
    for season, week in weeks:
        before = games[(games["season"] == season) & (games["week"] < week)]
        for player_id, history in before.groupby("player_id"):
            if len(history) < RISER_RECENT_GAMES:
                continue
            recent = history.tail(RISER_RECENT_GAMES)
            earlier = history.iloc[:-RISER_RECENT_GAMES]
            baseline: pd.Series
            if len(earlier) >= RISER_MIN_EARLIER_GAMES:
                baseline = earlier[list(USAGE)].mean()
                basis = "earlier games this season"
            elif (str(player_id), season) in previous_season.index:
                baseline = pd.Series(previous_season.loc[(str(player_id), season)])
                basis = "last season"
            else:
                continue
            last = recent.iloc[-1]
            record = {
                "season": season,
                "week": week,
                "player_id": player_id,
                "player_name": last["player_name"],
                "position": last["position"],
                "team": last["team"],
                "games_recent": len(recent),
                "games_earlier": len(earlier),
                "baseline_basis": basis,
            }
            for metric in USAGE:
                record[f"{metric}_recent"] = float(recent[metric].mean())
                record[f"{metric}_before"] = float(baseline[metric])
                record[f"{metric}_change"] = record[f"{metric}_recent"] - record[f"{metric}_before"]
            record["is_riser"] = bool(
                record["expected_ppr_change"] >= RISER_MIN_XFP_GAIN
                and record["expected_ppr_recent"] >= RISER_MIN_RECENT_XFP
            )
            records.append(record)
    if not records:
        return empty_risers()
    out = pd.DataFrame.from_records(records)
    out["riser_rank"] = (
        out.groupby(["season", "week"])["expected_ppr_change"]
        .rank(ascending=False, method="first", na_option="bottom")
        .astype("int64")
    )
    out = out.sort_values(["season", "week", "riser_rank"]).reset_index(drop=True)
    return out[list(RISER_COLUMNS)].astype(RISER_COLUMNS)


def empty_risers() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in RISER_COLUMNS.items()})
