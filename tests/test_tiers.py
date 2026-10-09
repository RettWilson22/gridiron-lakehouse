from __future__ import annotations

import itertools

import pytest

from gridiron.tiers import assign_tiers, optimal_breaks, start_sit


def brute_force(values: list[float], k: int) -> float:
    best = float("inf")
    for cuts in itertools.combinations(range(1, len(values)), k - 1):
        bounds = [0, *cuts, len(values)]
        cost = 0.0
        for start, end in itertools.pairwise(bounds):
            group = values[start:end]
            mean = sum(group) / len(group)
            cost += sum((v - mean) ** 2 for v in group)
        best = min(best, cost)
    return best


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_dynamic_programme_finds_the_optimal_split(k: int) -> None:
    values = [24.1, 23.8, 21.0, 20.7, 20.5, 17.2, 16.9, 12.0, 11.5]
    _, cost = optimal_breaks(values, k)
    assert cost == pytest.approx(brute_force(values, k))


def test_tiers_break_at_large_gaps_and_keep_input_order() -> None:
    projections = [10.0, 20.0, 19.5, 9.8, 15.0, 14.6, 19.8]
    tiers = assign_tiers(projections)
    assert tiers == [3, 1, 1, 3, 2, 2, 1]


def test_tiers_are_monotone_in_the_projection() -> None:
    projections = [float(x) for x in range(30, 0, -1)]
    tiers = assign_tiers(projections)
    assert tiers == sorted(tiers)
    assert tiers[0] == 1
    assert max(tiers) <= 10


def test_degenerate_inputs() -> None:
    assert assign_tiers([]) == []
    assert assign_tiers([5.0, 5.0, 5.0]) == [1, 1, 1]
    with pytest.raises(ValueError):
        optimal_breaks([1.0], 2)


@pytest.mark.parametrize(
    ("position", "rank", "label"),
    [
        ("QB", 12, "Start"),
        ("QB", 13, "Sit"),
        ("RB", 24, "Start"),
        ("RB", 36, "Flex"),
        ("RB", 37, "Sit"),
        ("WR", 36, "Start"),
        ("WR", 48, "Flex"),
        ("TE", 12, "Start"),
        ("TE", 16, "Flex"),
        ("TE", 17, "Sit"),
    ],
)
def test_start_sit_for_a_twelve_team_league(position: str, rank: int, label: str) -> None:
    assert start_sit(position, rank) == label
