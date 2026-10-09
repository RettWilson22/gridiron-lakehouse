"""Tiers and start/sit labels for a ranked list of projections.

Standard library only, so the Streamlit app can re-tier rankings under custom scoring in
every mode (it is uploaded next to the app in Streamlit in Snowflake).

Tiers are a one-dimensional clustering of the projections: the optimal partition of the
sorted values into ``k`` contiguous groups minimising the within-group sum of squares
(Jenks natural breaks, solved exactly by dynamic programming). ``k`` is the smallest number
of tiers whose grouping explains at least ``min_explained`` of the spread, so tiers break
where the gaps between players are large relative to the spread of the position.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

# 12-team league, 1 QB / 2 RB / 3 WR / 1 TE / 1 FLEX (RB, WR or TE).
TEAMS: Final = 12
STARTERS: Final[dict[str, int]] = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
FLEX_POSITIONS: Final = ("RB", "WR", "TE")
# Positional ranks just outside the starters that are realistic flex plays.
FLEX_DEPTH: Final[dict[str, int]] = {"RB": 12, "WR": 12, "TE": 4}
# The players worth ranking at each position: two starters' worth per team (24 QBs, 48 RBs,
# 72 WRs, 24 TEs). Tiers are drawn within it, the app lists it by default, and the backtest
# is scored on the experts' top N of this size. The dbt ``pool_size`` var and the stream
# task SQL repeat the numbers; tests/test_snowflake_objects.py checks they match.
POOL: Final[dict[str, int]] = {position: 2 * n * TEAMS for position, n in STARTERS.items()}

DEFAULT_MIN_EXPLAINED: Final = 0.9
DEFAULT_MAX_TIERS: Final = 10


def _sse_table(values: Sequence[float]) -> tuple[list[float], list[float]]:
    sums, squares = [0.0], [0.0]
    for v in values:
        sums.append(sums[-1] + v)
        squares.append(squares[-1] + v * v)
    return sums, squares


def _sse(sums: list[float], squares: list[float], i: int, j: int) -> float:
    """Within-group sum of squares of values[i:j]."""
    n = j - i
    total = sums[j] - sums[i]
    return max(squares[j] - squares[i] - total * total / n, 0.0)


def optimal_breaks(values: Sequence[float], k: int) -> tuple[list[int], float]:
    """Best split of ``values`` (already sorted) into ``k`` contiguous groups.

    Returns the start index of each group and the total within-group sum of squares.
    """
    n = len(values)
    if not 1 <= k <= n:
        raise ValueError(f"cannot split {n} values into {k} groups")
    sums, squares = _sse_table(values)
    inf = float("inf")
    cost = [[inf] * (n + 1) for _ in range(k + 1)]
    split = [[0] * (n + 1) for _ in range(k + 1)]
    cost[0][0] = 0.0
    for groups in range(1, k + 1):
        for end in range(groups, n + 1):
            for start in range(groups - 1, end):
                candidate = cost[groups - 1][start] + _sse(sums, squares, start, end)
                if candidate < cost[groups][end]:
                    cost[groups][end] = candidate
                    split[groups][end] = start
    starts, end = [], n
    for groups in range(k, 0, -1):
        start = split[groups][end]
        starts.append(start)
        end = start
    return sorted(starts), cost[k][n]


def assign_tiers(
    projections: Sequence[float],
    min_explained: float = DEFAULT_MIN_EXPLAINED,
    max_tiers: int = DEFAULT_MAX_TIERS,
) -> list[int]:
    """Tier (1 = best) for each projection, in the input order."""
    n = len(projections)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: -projections[i])
    values = [float(projections[i]) for i in order]
    mean = sum(values) / n
    total = sum((v - mean) ** 2 for v in values)
    starts = [0]
    if total > 0:
        for k in range(1, min(max_tiers, n) + 1):
            starts, sse = optimal_breaks(values, k)
            if 1 - sse / total >= min_explained:
                break
    tiers = [0] * n
    tier = 0
    boundaries = set(starts)
    for position_in_order, index in enumerate(order):
        if position_in_order in boundaries:
            tier += 1
        tiers[index] = tier
    return tiers


def start_sit(position: str, rank: int) -> str:
    """'Start', 'Flex' or 'Sit' for a positional rank in a 12-team league."""
    starters = STARTERS.get(position, 0) * TEAMS
    if rank <= starters:
        return "Start"
    if position in FLEX_POSITIONS and rank <= starters + FLEX_DEPTH[position]:
        return "Flex"
    return "Sit"
