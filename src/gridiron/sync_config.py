"""Configuration for the Databricks -> Snowflake sync, read from the environment.

Secrets are only ever read from environment variables (or a key file path) and are masked
in ``repr`` so they cannot leak into logs. Databricks auth uses ``DATABRICKS_TOKEN`` when it is
set; otherwise it reuses a Databricks CLI profile (``databricks auth login``), so no long-lived
token has to exist at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

REQUIRED = (
    "DATABRICKS_HOST",
    "DATABRICKS_HTTP_PATH",
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
)
DEFAULTS = {
    "DATABRICKS_CONFIG_PROFILE": "DEFAULT",
    "GRIDIRON_SOURCE_CATALOG": "workspace",
    "GRIDIRON_SOURCE_SCHEMA": "gridiron_serving",
    "SNOWFLAKE_ROLE": "GRIDIRON_LOADER",
    "SNOWFLAKE_WAREHOUSE": "GRIDIRON_WH",
    "SNOWFLAKE_DATABASE": "GRIDIRON",
    "SNOWFLAKE_SCHEMA": "SYNCED",
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SyncConfig:
    databricks_host: str
    databricks_http_path: str
    source_catalog: str
    source_schema: str
    snowflake_account: str
    snowflake_user: str
    snowflake_role: str
    snowflake_warehouse: str
    snowflake_database: str
    snowflake_schema: str
    snowflake_private_key_file: str | None = None
    snowflake_password: str | None = field(default=None, repr=False)
    databricks_token: str | None = field(default=None, repr=False)
    databricks_profile: str = "DEFAULT"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> SyncConfig:
        missing = [name for name in REQUIRED if not env.get(name)]
        key_file = env.get("SNOWFLAKE_PRIVATE_KEY_FILE") or None
        password = env.get("SNOWFLAKE_PASSWORD") or None
        if not key_file and not password:
            missing.append("SNOWFLAKE_PRIVATE_KEY_FILE (or SNOWFLAKE_PASSWORD)")
        if missing:
            raise ConfigError("missing environment variables: " + ", ".join(missing))

        def get(name: str) -> str:
            return env.get(name) or DEFAULTS[name]

        return cls(
            databricks_host=env["DATABRICKS_HOST"].removeprefix("https://").rstrip("/"),
            databricks_http_path=env["DATABRICKS_HTTP_PATH"],
            databricks_token=env.get("DATABRICKS_TOKEN") or None,
            databricks_profile=get("DATABRICKS_CONFIG_PROFILE"),
            source_catalog=get("GRIDIRON_SOURCE_CATALOG"),
            source_schema=get("GRIDIRON_SOURCE_SCHEMA"),
            snowflake_account=env["SNOWFLAKE_ACCOUNT"],
            snowflake_user=env["SNOWFLAKE_USER"],
            snowflake_role=get("SNOWFLAKE_ROLE"),
            snowflake_warehouse=get("SNOWFLAKE_WAREHOUSE"),
            snowflake_database=get("SNOWFLAKE_DATABASE"),
            snowflake_schema=get("SNOWFLAKE_SCHEMA"),
            snowflake_private_key_file=key_file,
            snowflake_password=password,
        )

    def databricks_connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "server_hostname": self.databricks_host,
            "http_path": self.databricks_http_path,
        }
        if self.databricks_token:
            kwargs["access_token"] = self.databricks_token
        else:
            kwargs["credentials_provider"] = self._cli_profile_credentials
        return kwargs

    def _cli_profile_credentials(self) -> Any:
        """Auth headers from a Databricks CLI profile (OAuth login, refreshed automatically)."""
        from databricks.sdk.core import Config  # noqa: PLC0415 (optional "sync" extra)

        config = Config(profile=self.databricks_profile, host=f"https://{self.databricks_host}")
        return config.authenticate

    def snowflake_connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "account": self.snowflake_account,
            "user": self.snowflake_user,
            "role": self.snowflake_role,
            "warehouse": self.snowflake_warehouse,
            "database": self.snowflake_database,
            "schema": self.snowflake_schema,
        }
        if self.snowflake_private_key_file:
            kwargs["private_key_file"] = self.snowflake_private_key_file
        else:
            kwargs["password"] = self.snowflake_password
        return kwargs

    def describe(self) -> dict[str, str]:
        """Non-secret settings, safe to log."""
        hidden = {"databricks_token", "snowflake_password"}
        return {f.name: str(getattr(self, f.name)) for f in fields(self) if f.name not in hidden}
