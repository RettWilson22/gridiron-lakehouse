"""Log output for the job tasks: progress as plain log lines, and one JSON summary per run."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


def setup() -> None:
    """Send INFO logs to stdout, where the job run page shows a task's output."""
    # force: if the runtime has already configured the root logger, a plain basicConfig
    # would do nothing.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )


def summary(task: str, **fields: Any) -> None:
    """Log the task's outcome as one line: ``summary {"field": value, ...}``."""
    logging.getLogger(f"gridiron.jobs.{task}").info("summary %s", json.dumps(fields, default=str))
