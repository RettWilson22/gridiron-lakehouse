"""Walk-forward backtest and accuracy metrics against simple baselines and expert rankings.

Protocol: for each test season ``S`` the model is trained on every completed regular-season
week of seasons before ``S`` and then projects every week of ``S``. Features for week ``w``
only use information from before that week's games (see ``gridiron.features``), so each
projection is what the system would have published before kickoff, using a model that was
frozen before the season began. Retraining once per season rather than every week is
cheaper and, if anything, conservative: the live system retrains weekly and so also learns
from the current season's earlier weeks.

Evaluation pool: for each position-week, the players FantasyPros ranked in the top ``N``
of their position that week (``TOP_N``: two starters' worth per team in a 12-team
1 QB / 2 RB / 3 WR / 1 TE league). The pool is defined by a pre-game source that is
independent of every method being compared, and only weeks with a weekly ranking snapshot
are scored. Within the pool, rows are scored when every method has a value (rookies before
their first game have no baseline). Players who were active but did not record a stat
score zero.

Metrics per position: MAE, RMSE and bias of PPR points; Spearman rank correlation with the
actual points, computed within each week and averaged over weeks; and for the model the
share of actual scores inside its 10th-90th percentile range (80% is ideal).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np
import pandas as pd

from gridiron.features import KEY
from gridiron.model import ProjectionModel

TOP_N: Final[dict[str, int]] = {"QB": 24, "RB": 48, "WR": 72, "TE": 24}
METHODS: Final = ("model", "last3", "season_avg", "ecr")
PREDICTIONS: Final = {
    "model": "proj_ppr",
    "last3": "baseline_last3",
    "season_avg": "baseline_season_avg",
}
MIN_PLAYERS_FOR_RANK: Final = 5
ACTUAL: Final = "fantasy_points_ppr"


def with_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Baseline projections, computed from the same pre-game features.

    * last 3 games: average PPR over the player's last three games (may span seasons);
    * season to date: average over this season's earlier games, or last season's average
      before the player's first game of the season.
    """
    out = frame.copy()
    out["baseline_last3"] = out["ppr_last3"]
    out["baseline_season_avg"] = out["season_ppr_mean"].fillna(out["prev_season_ppr_mean"])
    return out


def completed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["is_final"].astype(bool)]


def walk_forward(
    frame: pd.DataFrame,
    seasons: Iterable[int],
    before_week: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Season-by-season walk-forward projections for completed weeks of ``seasons``.

    ``before_week`` optionally limits a season to weeks before the given week (used for
    the in-progress season, whose remaining weeks are projected live instead).
    """
    history = completed(frame)
    parts = []
    for season in sorted(set(seasons)):
        train = history[history["season"] < season]
        test = history[history["season"] == season]
        if before_week and season in before_week:
            test = test[test["week"] < before_week[season]]
        if test.empty or train.empty:
            continue
        model = ProjectionModel.fit(train)
        projected = with_baselines(pd.concat([test, model.predict(test)], axis=1))
        parts.append(projected.assign(kind="backtest", model_version=f"walk-forward-{season}"))
    if not parts:
        return frame.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def attach_ecr(projected: pd.DataFrame, ecr: pd.DataFrame) -> pd.DataFrame:
    rankings = ecr[["season", "week", "player_id", "position", "ecr", "ecr_rank"]].rename(
        columns={"position": "ecr_position"}
    )
    rankings = rankings.dropna(subset=["player_id"]).drop_duplicates(KEY)
    return projected.merge(rankings, on=KEY, how="left")


def evaluation_pool(projected: pd.DataFrame) -> pd.DataFrame:
    """Rows in the expert top-N pool with a value for every method (see module docstring)."""
    in_pool = (projected["ecr_position"] == projected["position"]) & (
        projected["ecr_rank"] <= projected["position"].map(TOP_N)
    )
    pool = projected[in_pool]
    return pool.dropna(subset=[*PREDICTIONS.values(), "ecr", ACTUAL])


def pool_coverage(projected: pd.DataFrame, ecr: pd.DataFrame, seasons: Iterable[int]) -> pd.Series:
    """Share of expert top-N player-weeks (in the scored weeks) that the model projected."""
    weeks = projected[["season", "week"]].drop_duplicates()
    top = ecr[ecr["season"].isin(list(seasons)) & ecr["player_id"].notna()]
    top = top[top["ecr_rank"] <= top["position"].map(TOP_N)].merge(weeks, on=["season", "week"])
    found = top.merge(projected[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
    return (found["_merge"] == "both").groupby(found["position"]).mean()


def _spearman_by_week(pool: pd.DataFrame, prediction: pd.Series) -> tuple[float, int]:
    frame = pool.assign(_pred=prediction)
    values = []
    for _, week in frame.groupby(["season", "week"]):
        if len(week) < MIN_PLAYERS_FOR_RANK:
            continue
        ranks = week[["_pred", ACTUAL]].rank()
        if ranks["_pred"].nunique() < 2 or ranks[ACTUAL].nunique() < 2:
            continue  # a rank correlation is undefined when every value ties
        values.append(ranks["_pred"].corr(ranks[ACTUAL]))
    clean = [v for v in values if not np.isnan(v)]
    return (float(np.mean(clean)) if clean else float("nan")), len(clean)


def score_methods(pool: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    actual = pool[ACTUAL].astype("float64")
    for method in METHODS:
        if method == "ecr":
            spearman, weeks = _spearman_by_week(pool, -pool["ecr"])
            rows.append({"method": method, "n": len(pool), "weeks": weeks, "spearman": spearman})
            continue
        prediction = pool[PREDICTIONS[method]].astype("float64")
        error = prediction - actual
        spearman, weeks = _spearman_by_week(pool, prediction)
        row: dict[str, float | int | str] = {
            "method": method,
            "n": len(pool),
            "weeks": weeks,
            "mae": float(error.abs().mean()),
            "rmse": float(np.sqrt((error**2).mean())),
            "bias": float(error.mean()),
            "spearman": spearman,
        }
        if method == "model":
            inside = (actual >= pool["floor_ppr"]) & (actual <= pool["ceiling_ppr"])
            row["interval_coverage"] = float(inside.mean())
        rows.append(row)
    return rows


def evaluate(projected: pd.DataFrame, ecr: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    """Backtest metrics per scope (each season, and all seasons together) and position."""
    seasons = sorted(set(seasons))
    scored = attach_ecr(projected[projected["season"].isin(seasons)], ecr)
    pool = evaluation_pool(scored)
    records = []
    scopes = [(str(s), [s]) for s in seasons] + [(f"{seasons[0]}-{seasons[-1]}", seasons)]
    for scope, scope_seasons in scopes:
        coverage = pool_coverage(scored[scored["season"].isin(scope_seasons)], ecr, scope_seasons)
        for position, group in pool[pool["season"].isin(scope_seasons)].groupby("position"):
            for row in score_methods(group):
                records.append(
                    {
                        "scope": scope,
                        "position": position,
                        **row,
                        "pool_coverage": float(coverage.get(position, np.nan)),
                    }
                )
    columns = [
        "scope",
        "position",
        "method",
        "n",
        "weeks",
        "mae",
        "rmse",
        "bias",
        "spearman",
        "interval_coverage",
        "pool_coverage",
    ]
    metrics = pd.DataFrame.from_records(records)
    for column in columns:
        if column not in metrics:
            metrics[column] = np.nan
    return metrics[columns].sort_values(["scope", "position", "method"]).reset_index(drop=True)
