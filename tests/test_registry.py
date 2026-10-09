"""The champion lookup that lets the train task skip a refit, on a local MLflow file store."""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from gridiron import registry

NAME = "spark_catalog.gridiron.fantasy_projection"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MlflowClient:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = f"file:{tmp_path / 'mlruns'}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def test_registered_name_is_catalog_schema_model() -> None:
    name = registry.registered_name("workspace", "gridiron")
    assert name == "workspace.gridiron.fantasy_projection"


def test_champion_is_only_returned_when_trained_on_the_same_data(
    client: MlflowClient, tmp_path: Path
) -> None:
    assert registry.champion_trained_on(client, NAME, "abc") is None  # no model yet
    client.create_registered_model(NAME)
    version = client.create_model_version(NAME, source=str(tmp_path)).version
    assert registry.champion_trained_on(client, NAME, "abc") is None  # no champion yet

    registry.promote(client, NAME, version, "abc")
    champion = registry.champion_trained_on(client, NAME, "abc")
    assert champion is not None and str(champion.version) == str(version)
    assert registry.champion_trained_on(client, NAME, "def") is None  # other training data
