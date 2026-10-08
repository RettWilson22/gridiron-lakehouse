from __future__ import annotations

import pytest

from gridiron.sync_config import ConfigError, SyncConfig

BASE = {
    "DATABRICKS_HOST": "https://dbc-123.cloud.databricks.com/",
    "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/abc",
    "DATABRICKS_TOKEN": "dapi-secret-token",
    "SNOWFLAKE_ACCOUNT": "org-acct",
    "SNOWFLAKE_USER": "GRIDIRON_SYNC",
    "SNOWFLAKE_PRIVATE_KEY_FILE": "/keys/rsa_key.p8",
}


def test_from_env_applies_defaults_and_normalises_host() -> None:
    config = SyncConfig.from_env(BASE)
    assert config.databricks_host == "dbc-123.cloud.databricks.com"
    assert config.source_schema == "gridiron_serving"
    assert config.snowflake_role == "GRIDIRON_LOADER"
    assert config.snowflake_connect_kwargs()["private_key_file"] == "/keys/rsa_key.p8"
    assert "password" not in config.snowflake_connect_kwargs()


def test_missing_variables_are_all_reported() -> None:
    env = {k: v for k, v in BASE.items() if k not in {"DATABRICKS_HOST", "SNOWFLAKE_USER"}}
    with pytest.raises(ConfigError) as err:
        SyncConfig.from_env(env)
    assert "DATABRICKS_HOST" in str(err.value) and "SNOWFLAKE_USER" in str(err.value)


def test_requires_some_snowflake_credential() -> None:
    env = {k: v for k, v in BASE.items() if k != "SNOWFLAKE_PRIVATE_KEY_FILE"}
    with pytest.raises(ConfigError, match="SNOWFLAKE_PRIVATE_KEY_FILE"):
        SyncConfig.from_env(env)
    config = SyncConfig.from_env({**env, "SNOWFLAKE_PASSWORD": "pw"})
    assert config.snowflake_connect_kwargs()["password"] == "pw"


def test_secrets_never_appear_in_repr_or_description() -> None:
    config = SyncConfig.from_env({**BASE, "SNOWFLAKE_PASSWORD": "hunter2"})
    for text in (repr(config), str(config.describe())):
        assert "dapi-secret-token" not in text
        assert "hunter2" not in text


def test_without_a_token_databricks_auth_comes_from_a_cli_profile() -> None:
    env = {k: v for k, v in BASE.items() if k != "DATABRICKS_TOKEN"}
    env["DATABRICKS_CONFIG_PROFILE"] = "gridiron"
    config = SyncConfig.from_env(env)

    kwargs = config.databricks_connect_kwargs()
    assert "access_token" not in kwargs
    assert callable(kwargs["credentials_provider"])
    assert config.databricks_profile == "gridiron"


def test_a_token_is_used_when_given() -> None:
    kwargs = SyncConfig.from_env(BASE).databricks_connect_kwargs()
    assert kwargs["access_token"] == BASE["DATABRICKS_TOKEN"]
    assert "credentials_provider" not in kwargs
