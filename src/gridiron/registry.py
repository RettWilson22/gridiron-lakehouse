"""MLflow packaging for the projection model (imported by the train and score tasks only).

The fitted ``ProjectionModel`` is saved with joblib and wrapped in an MLflow pyfunc whose
input is the feature frame and whose output is the projected stat line, points, floors and
ceilings. The ``gridiron`` package is logged with the model (``code_paths``) so the model
loads anywhere the pinned scikit-learn version is installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Final

import joblib
import mlflow
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel  # type: ignore[attr-defined]  # not re-exported

import gridiron
from gridiron.features import FEATURES
from gridiron.model import ProjectionModel

INPUT_COLUMNS: Final = ("position", *FEATURES)
ARTIFACT_KEY: Final = "projection_model"


def model_input(frame: pd.DataFrame) -> pd.DataFrame:
    """The model's input columns, with every feature as float64 (the logged signature)."""
    out = frame[list(INPUT_COLUMNS)].copy()
    out[list(FEATURES)] = out[list(FEATURES)].astype("float64")
    out["position"] = out["position"].astype(str)
    return out


class ProjectionPyfunc(PythonModel):
    def load_context(self, context: Any) -> None:
        self.model: ProjectionModel = joblib.load(context.artifacts[ARTIFACT_KEY])

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> Any:
        return self.model.predict(model_input)


def log_model(model: ProjectionModel, sample: pd.DataFrame, registered_name: str | None) -> Any:
    """Log (and optionally register) the model in the active MLflow run."""
    example = model_input(sample)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{ARTIFACT_KEY}.joblib"
        joblib.dump(model, path)
        return mlflow.pyfunc.log_model(
            name=ARTIFACT_KEY,
            python_model=ProjectionPyfunc(),
            artifacts={ARTIFACT_KEY: str(path)},
            code_paths=[str(Path(gridiron.__file__).parent)],
            signature=infer_signature(example, model.predict(example)),
            input_example=example.head(3),
            registered_model_name=registered_name,
            pip_requirements=[
                f"scikit-learn=={sklearn.__version__}",
                f"pandas=={pd.__version__}",
                f"joblib=={joblib.__version__}",
            ],
        )
