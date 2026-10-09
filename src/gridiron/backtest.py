"""Walk-forward backtest and accuracy metrics against simple baselines and expert rankings.

Protocol: for each test season ``S`` the model is trained on every completed regular-season
week of seasons before ``S`` and then projects every week of ``S``. Features for week ``w``
only use information from before that week's games (see ``gridiron.features``), using a
model that was frozen before the season began. Retraining once per season rather than
every week is cheaper and conservative in one respect: the live system retrains weekly and
so also learns from the current season's earlier weeks. In another respect the backtest
sees more than a live run: its features use closing betting lines and the final injury
report, which a Tuesday or Thursday live run does not have yet.

Evaluation pool: for each position-week, the players FantasyPros ranked in the top ``N``
of their position that week (``tiers.POOL``: two starters' worth per team in a 12-team
1 QB / 2 RB / 3 WR / 1 TE league). The pool is defined by a pre-game source that is
independent of every method being compared, and only weeks with a weekly ranking snapshot
are scored (the latest scrape on or before the week's last game day). Within the pool, rows
are scored when every method has a value (rookies before their first game have no
baseline). A projected player without a stat row for the game scores zero, whether he sat
out or played without recording a stat (``features.attach_actuals``).

Metrics per position: MAE, RMSE and bias of PPR points; Spearman rank correlation with the
actual points, computed within each week and averaged over weeks; and for the model the
share of actual scores inside its 10th-90th percentile range (80% is ideal).

Intervals (``intervals``): a paired bootstrap over weeks for the model's MAE minus each
baseline's, and its rank correlation minus the experts'. Weeks are resampled with
replacement, all players of a drawn week together (players in the same week share the
same games and news, so they are not independent), and every method is scored on the same
draws. The intervals are the 2.5th and 97.5th percentiles of ``RESAMPLES`` draws.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np
import pandas as pd

from gridiron.features import KEY, with_baselines
from gridiron.model import OUTPUT_COLUMNS, ProjectionModel, fingerprint
from gridiron.tiers import POOL

METHODS: Final = ("model", "last3", "season_avg", "ecr")
PREDICTIONS: Final = {
    "model": "proj_ppr",
    "last3": "baseline_last3",
    "season_avg": "baseline_season_avg",
}
MIN_PLAYERS_FOR_RANK: Final = 5
ACTUAL: Final = "fantasy_points_ppr"
RESAMPLES: Final = 10_000
SEED: Final = 0


def completed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["is_final"].astype(bool)]


def walk_forward(
    frame: pd.DataFrame,
    seasons: Iterable[int],
    before_week: dict[int, int] | None = None,
    stored: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Season-by-season walk-forward projections for completed weeks of ``seasons``.

    ``before_week`` optionally limits a season to weeks before the given week (used for
    the in-progress season, whose remaining weeks are projected live instead).

    Each season's ``model_version`` is ``walk-forward-<season>-<fingerprint>``, where the
    fingerprint covers the rows the model trains on and the rows it projects
    (``model.fingerprint``). ``stored`` is an earlier run's output (the
    ``backtest_projections`` table): a season whose version is unchanged reuses its stored
    model outputs instead of refitting, so finished seasons are fitted once.
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
        version = f"walk-forward-{season}-{fingerprint(train, test)[:12]}"
        outputs = stored_outputs(stored, version, test)
        if outputs is None:
            outputs = ProjectionModel.fit(train).predict(test)
        projected = with_baselines(pd.concat([test, outputs], axis=1))
        parts.append(projected.assign(kind="backtest", model_version=version))
    if not parts:
        return frame.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def stored_outputs(
    stored: pd.DataFrame | None, version: str, test: pd.DataFrame
) -> pd.DataFrame | None:
    """The model outputs stored for ``version``, aligned to ``test``; None unless the stored
    rows are exactly the test rows."""
    if stored is None or "model_version" not in stored:
        return None
    rows = stored[stored["model_version"] == version]
    if len(rows) != len(test):
        return None
    keyed = rows.astype({"season": "int64", "week": "int64", "player_id": str}).set_index(KEY)
    wanted = pd.MultiIndex.from_frame(
        test[KEY].astype({"season": "int64", "week": "int64", "player_id": str})
    )
    if keyed.index.has_duplicates or not wanted.isin(keyed.index).all():
        return None
    outputs = keyed.loc[wanted, list(OUTPUT_COLUMNS)].astype("float64")
    outputs.index = test.index
    return outputs


def attach_ecr(projected: pd.DataFrame, ecr: pd.DataFrame) -> pd.DataFrame:
    rankings = ecr[["season", "week", "player_id", "position", "ecr", "ecr_rank"]].rename(
        columns={"position": "ecr_position"}
    )
    rankings = rankings.dropna(subset=["player_id"]).drop_duplicates(KEY)
    return projected.merge(rankings, on=KEY, how="left")


def evaluation_pool(projected: pd.DataFrame) -> pd.DataFrame:
    """Rows in the expert top-N pool with a value for every method (see module docstring)."""
    in_pool = (projected["ecr_position"] == projected["position"]) & (
        projected["ecr_rank"] <= projected["position"].map(POOL)
    )
    pool = projected[in_pool]
    return pool.dropna(subset=[*PREDICTIONS.values(), "ecr", ACTUAL])


def pool_coverage(projected: pd.DataFrame, ecr: pd.DataFrame, seasons: Iterable[int]) -> pd.Series:
    """Share of expert top-N player-weeks (in the scored weeks) that the model projected."""
    weeks = projected[["season", "week"]].drop_duplicates()
    top = ecr[ecr["season"].isin(list(seasons)) & ecr["player_id"].notna()]
    top = top[top["ecr_rank"] <= top["position"].map(POOL)].merge(weeks, on=["season", "week"])
    found = top.merge(projected[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
    return (found["_merge"] == "both").groupby(found["position"]).mean()


def pooled_scope(scopes: Iterable[str]) -> str:
    """The scope that pools every test season, e.g. ``"2023-2025"`` (seasons alone are
    ``"2023"``): the longest scope name, or the only season when there is one. The Streamlit
    app, which runs in Snowflake without this package, applies the same rule."""
    return max(scopes, key=lambda scope: (len(scope), scope))


def candidate_coverage(
    candidates: pd.DataFrame, player_week: pd.DataFrame, min_points: float = 10.0
) -> dict[str, float]:
    """Share of player-games with a stat row that the candidate pool included: over all
    of them, and over those with at least ``min_points`` PPR points. Only weeks present in
    ``candidates`` count."""
    weeks = candidates[["season", "week"]].drop_duplicates()
    played = player_week[[*KEY, ACTUAL]].merge(weeks, on=["season", "week"])
    found = played.merge(candidates[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
    covered = found["_merge"] == "both"
    big = found[ACTUAL] >= min_points
    return {
        "player_games": len(found),
        "share_all": float(covered.mean()),
        f"share_{min_points:g}_plus_ppr": float(covered[big].mean()),
    }


def _weekly_spearman(pool: pd.DataFrame, prediction: pd.Series) -> pd.Series:
    """Spearman correlation of ``prediction`` with the actual points in each week; NaN for a
    week with too few players or where every value ties (the correlation is undefined)."""

    def spearman(week: pd.DataFrame) -> float:
        ranks = week.rank()
        if len(week) < MIN_PLAYERS_FOR_RANK or ranks.nunique().min() < 2:
            return float("nan")
        return float(ranks["_pred"].corr(ranks[ACTUAL]))

    frame = pool.assign(_pred=prediction)
    weekly: pd.Series = frame.groupby(["season", "week"])[["_pred", ACTUAL]].apply(spearman)
    return weekly


def _spearman_by_week(pool: pd.DataFrame, prediction: pd.Series) -> tuple[float, int]:
    values = _weekly_spearman(pool, prediction).dropna()
    return (float(values.mean()) if len(values) else float("nan")), len(values)


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


def paired_bootstrap(
    pool: pd.DataFrame, resamples: int = RESAMPLES, seed: int = SEED
) -> pd.DataFrame:
    """Paired week-level bootstrap of the model against each benchmark, for one position's
    pool. One row per comparison (the model's MAE minus each baseline's, its rank
    correlation minus the experts'): the number of weeks, the estimate (from the pool as it
    is) and the 95% percentile interval."""
    actual = pool[ACTUAL].astype("float64")
    errors = pd.DataFrame(
        {
            method: (pool[column].astype("float64") - actual).abs()
            for method, column in PREDICTIONS.items()
        }
    )
    weekly = errors.groupby([pool["season"], pool["week"]])
    by_week = weekly.sum().assign(
        rows=weekly.size(),
        rho_model=_weekly_spearman(pool, pool["proj_ppr"]),
        rho_ecr=_weekly_spearman(pool, -pool["ecr"]),
    )
    weeks = len(by_week)
    rng = np.random.default_rng(seed)
    # How often each week is drawn: row 0 is the pool as it is, the rest are resamples.
    weights = np.vstack(
        [np.ones(weeks), rng.multinomial(weeks, np.full(weeks, 1 / weeks), size=resamples)]
    )

    def mae(method: str) -> np.ndarray:
        total = weights @ by_week[method].to_numpy(dtype="float64")
        return np.asarray(total / (weights @ by_week["rows"].to_numpy(dtype="float64")))

    def mean_over_weeks(column: str) -> np.ndarray:  # weeks where it is defined
        values = by_week[column].to_numpy(dtype="float64")
        defined = ~np.isnan(values)
        with np.errstate(invalid="ignore"):  # NaN when no drawn week is defined
            total = weights @ np.where(defined, values, 0.0)
            return np.asarray(total / (weights @ defined.astype("float64")))

    differences = {
        "mae_model_minus_last3": mae("model") - mae("last3"),
        "mae_model_minus_season_avg": mae("model") - mae("season_avg"),
        "spearman_model_minus_ecr": mean_over_weeks("rho_model") - mean_over_weeks("rho_ecr"),
    }
    draws = np.vstack(list(differences.values()))
    low, high = np.percentile(draws[:, 1:], [2.5, 97.5], axis=1)
    return pd.DataFrame(
        {"weeks": weeks, "estimate": draws[:, 0], "low": low, "high": high},
        index=pd.Index(list(differences), name="comparison"),
    )


def _scopes(seasons: list[int]) -> list[tuple[str, list[int]]]:
    """Each season, and all of them together when there are several."""
    scopes = [(str(s), [s]) for s in seasons]
    if len(seasons) > 1:
        scopes.append((f"{seasons[0]}-{seasons[-1]}", seasons))
    return scopes


def intervals(
    projected: pd.DataFrame,
    ecr: pd.DataFrame,
    seasons: Iterable[int],
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    """``paired_bootstrap`` per scope and position, on the same pool as ``evaluate``."""
    seasons = sorted(set(seasons))
    pool = evaluation_pool(attach_ecr(projected[projected["season"].isin(seasons)], ecr))
    parts = {
        (scope, position): paired_bootstrap(group, resamples, seed)
        for scope, scope_seasons in _scopes(seasons)
        for position, group in pool[pool["season"].isin(scope_seasons)].groupby("position")
    }
    return pd.concat(parts, names=["scope", "position"]).reset_index()


def evaluate(projected: pd.DataFrame, ecr: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    """Backtest metrics per scope (each season, and all seasons together when there are
    several) and position."""
    seasons = sorted(set(seasons))
    scored = attach_ecr(projected[projected["season"].isin(seasons)], ecr)
    pool = evaluation_pool(scored)
    records = []
    for scope, scope_seasons in _scopes(seasons):
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
