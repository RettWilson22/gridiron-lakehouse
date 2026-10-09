from __future__ import annotations

import time

import pytest

from gridiron.player_search import (
    MAX_QUERY_CHARS,
    Player,
    did_you_mean,
    index_letter,
    initials,
    last_name,
    normalize,
    parse_query,
    search,
)

PLAYERS = [
    Player("1", "Christian McCaffrey", "RB", "SF"),
    Player("2", "Jaxon Smith-Njigba", "WR", "SEA"),
    Player("3", "Amon-Ra St. Brown", "WR", "DET"),
    Player("4", "Jameson Williams", "WR", "DET"),
    Player("5", "Jared Goff", "QB", "DET"),
    Player("6", "Kenneth Walker III", "RB", "KC"),
    Player("7", "Marvin Harrison Jr.", "WR", "ARI"),
    Player("8", "A.J. Brown", "WR", "PHI"),
    Player("9", "Justin Jefferson", "WR", "MIN"),
    Player("10", "Josh Allen", "QB", "BUF"),
    Player("11", "Chris Olave", "WR", "NO"),
    Player("12", "Sam LaPorta", "TE", "DET"),
]


def names(query: str, **kwargs: object) -> list[str]:
    return [m.player.name for m in search(PLAYERS, query, **kwargs)]  # type: ignore[arg-type]


def test_normalize_ignores_case_accents_and_punctuation() -> None:
    assert normalize("  Amon-Ra  St. Brown ") == "amon ra st brown"
    assert normalize("José Ramírez") == "jose ramirez"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Amon-Ra St. Brown", {"arsb", "ab"}),
        ("Kenneth Walker III", {"kw", "kwiii", "kw3"}),
        ("A.J. Brown", {"ajb", "ab"}),
        ("Marvin Harrison Jr.", {"mh", "mhjr", "mhj"}),
    ],
)
def test_initials(name: str, expected: set[str]) -> None:
    assert expected <= initials(name)


def test_index_sorts_by_surname() -> None:
    assert last_name("Amon-Ra St. Brown") == "st brown"
    assert index_letter("Amon-Ra St. Brown") == "S"
    assert last_name("Marvin Harrison Jr.") == "harrison"
    assert index_letter("Marvin Harrison Jr.") == "H"
    assert index_letter("Josh Allen") == "A"


def test_exact_partial_and_last_name_first() -> None:
    assert names("Justin Jefferson")[0] == "Justin Jefferson"
    assert names("jeff")[0] == "Justin Jefferson"
    assert names("jefferson justin")[0] == "Justin Jefferson"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("cmc", "Christian McCaffrey"),
        ("jsn", "Jaxon Smith-Njigba"),
        ("arsb", "Amon-Ra St. Brown"),
        ("kw3", "Kenneth Walker III"),
        ("mhj", "Marvin Harrison Jr."),
        ("ajb", "A.J. Brown"),
    ],
)
def test_initials_and_nicknames(query: str, expected: str) -> None:
    assert names(query)[0] == expected


def test_typos_still_find_the_player() -> None:
    assert names("mccaffery")[0] == "Christian McCaffrey"
    assert names("jeferson")[0] == "Justin Jefferson"


def test_team_and_position_words_filter_instead_of_matching_names() -> None:
    query = parse_query("lions wr")
    assert query.teams == {"DET"} and query.positions == {"WR"} and not query.terms
    assert set(names("lions wr")) == {"Amon-Ra St. Brown", "Jameson Williams"}
    assert names("det qb") == ["Jared Goff"]
    assert names("detroit williams") == ["Jameson Williams"]


def test_ambiguous_team_codes_only_filter_on_their_own() -> None:
    assert names("no") == ["Chris Olave"]
    assert parse_query("no brown").teams == frozenset()


def test_ties_go_to_the_more_relevant_player() -> None:
    relevance = {"8": 15.0, "3": 20.0}
    assert names("brown", relevance=relevance)[:2] == ["Amon-Ra St. Brown", "A.J. Brown"]


def test_no_match_and_did_you_mean() -> None:
    assert names("zzzz") == []
    assert names("") == []
    assert did_you_mean(PLAYERS, "jared gof") == "Jared Goff"
    assert did_you_mean(PLAYERS, "qqqqqq") is None


def test_a_query_keeps_at_most_sixty_characters_and_four_name_words() -> None:
    assert parse_query("x" * 100).terms == ("x" * MAX_QUERY_CHARS,)
    assert parse_query("a b c d e f lions wr").terms == ("a", "b", "c", "d")


def test_a_huge_query_returns_quickly() -> None:
    first = ["Justin", "Jared", "Josh", "Chris", "Sam", "Amon-Ra", "Kenneth", "Marvin"]
    last = ["Jefferson", "Goff", "Allen", "Olave", "LaPorta", "Brown", "Walker", "Harrison"]
    teams = ["DET", "MIN", "BUF", "NO", "SF", "PHI", "KC", "ARI"]
    names = [f"{given} {family}" for given in first for family in last for _ in range(8)]
    universe = [  # 512 players
        Player(str(i), f"{name}{i}", "WR", teams[i % len(teams)]) for i, name in enumerate(names)
    ]
    query = ("jefferson justin mccaffery " * 2000)[:50_000]  # 50 KB pasted into the box
    started = time.perf_counter()
    search(universe, query)
    did_you_mean(universe, query)
    assert time.perf_counter() - started < 0.5
