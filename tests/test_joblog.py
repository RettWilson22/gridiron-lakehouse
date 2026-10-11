from __future__ import annotations

import json
import logging

import pytest

from gridiron import joblog


def test_a_task_summary_is_one_json_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        joblog.summary("score", upcoming=(2026, 5), projections_rows=28_396)
    [record] = caplog.records
    assert record.name == "gridiron.jobs.score"
    prefix, payload = record.getMessage().split(" ", 1)
    assert prefix == "summary"
    assert json.loads(payload) == {"upcoming": [2026, 5], "projections_rows": 28_396}
