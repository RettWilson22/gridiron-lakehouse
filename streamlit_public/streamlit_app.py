"""Public copy of the cheat sheet, for Streamlit Community Cloud.

Runs the same app as Streamlit in Snowflake, reading the static snapshot in ``snapshot/``
(exported from the dbt marts by ``scripts/export_public_snapshot.py``) instead of Snowflake.
Deploy with this file as the entry point; ``requirements.txt`` next to it pins the packages.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent / "snowflake" / "streamlit"

os.environ["GRIDIRON_APP_MODE"] = "public"
os.environ.setdefault("GRIDIRON_SNAPSHOT_DIR", str(HERE / "snapshot"))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
runpy.run_path(str(APP_DIR / "streamlit_app.py"), run_name="__main__")
