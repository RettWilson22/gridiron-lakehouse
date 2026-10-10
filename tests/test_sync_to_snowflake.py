from __future__ import annotations

from scripts.sync_to_snowflake import shrink_problem


def test_an_ordinary_sync_passes() -> None:
    assert shrink_problem("projections", staged=28_400, existing=28_396, deleted=33) is None
    assert shrink_problem("player_week", staged=100, existing=0, deleted=0) is None


def test_an_empty_source_is_refused() -> None:
    assert shrink_problem("projections", staged=0, existing=28_396, deleted=28_396) == (
        "projections: the source returned no rows"
    )
    assert shrink_problem("player_week", staged=0, existing=0, deleted=0) is not None


def test_deleting_more_than_a_fifth_of_a_table_is_refused() -> None:
    assert shrink_problem("player_week", staged=900, existing=1000, deleted=200) is None
    assert shrink_problem("player_week", staged=799, existing=1000, deleted=201) == (
        "player_week: the sync would delete 201 of 1000 rows"
    )


def test_risers_may_empty_out_in_the_offseason() -> None:
    assert shrink_problem("risers", staged=0, existing=529, deleted=529) is None
