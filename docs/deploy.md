# Running and deploying

A from-scratch runbook: the full flow on a laptop, then Databricks Free Edition, a Snowflake
trial account and the public copy on Streamlit Community Cloud. Commands run from the
repository root unless noted.

## What you need

* Python 3.12, Java 17 or newer (local Spark), `make`, `openssl`; then `make setup`.
* A Databricks Free Edition workspace and the Databricks CLI (the version with
  `databricks bundle`).
* A Snowflake trial account where you have ACCOUNTADMIN.
* The Snowflake CLI (`snow`) in its own virtualenv. Installing it into the project `.venv`
  breaks dbt's dependencies:

  ```bash
  python3.12 -m venv ~/.snowflake/cli-venv
  ~/.snowflake/cli-venv/bin/pip install snowflake-cli
  ```

## 1. The full flow on a laptop

No cloud account is needed. About 200 MB lands in the git-ignored `data/` folder.

```bash
make ingest           # every dataset, 2018 through the current season; re-runs skip completed seasons
make pipeline-local   # Spark: bronze, silver, gold -> data/lakehouse/*.parquet
make model-local      # backtest, live projections, risers -> data/lakehouse, artifacts/
make dbt-local        # dbt marts over data/lakehouse in DuckDB
make app              # the app on http://localhost:8501 against the DuckDB marts
make smoke-jobs       # the Databricks job entry points against a local Spark catalog
make snapshot         # export a public snapshot from data/lakehouse
```

`make fixtures` rebuilds the checked-in test fixtures (a slice of every landed dataset for
the NFC North's games in 2024 weeks 1-4, 2025 weeks 1-4 and 2026 weeks 1-5, plus the serving
tables for those games) and the generated Snowflake DDL.

## 2. Databricks (Free Edition)

```bash
databricks auth login --host https://<workspace-host>
make bundle-validate     # TARGET=prod by default
make bundle-deploy       # clears dist/, builds the wheel, deploys
make bundle-run          # runs the gridiron-refresh job once
```

The bundle (`databricks/`) creates the `gridiron` and `gridiron_serving` schemas in the
`workspace` catalog, a `landing` volume, the Lakeflow pipeline and the `gridiron-refresh`
job. The job has five tasks run in sequence: `ingest`, `transform` (the pipeline), `train`,
`score`, `publish_serving`, and is scheduled for Tuesday, Thursday and Saturday at 12:00 UTC
(the `dev` target prefixes names and pauses the schedule).

Two deployment details:

* The bundle builds the package as a wheel on deploy, and the job and pipeline install
  `dist/*.whl`, because serverless compute cannot install an editable package from
  workspace files. A wheel left over from an older build would be installed too, so
  `make bundle-deploy` clears `dist/` first.
* Job entry points call `sys.exit` only on failure: Databricks treats any `SystemExit`,
  even code 0, as a failed task.

After the first run, check the pipeline graph and its expectation metrics, the MLflow
experiment `/Users/<you>/gridiron-fantasy`, the registered model
`workspace.gridiron.fantasy_projection` (alias `champion`) and the six tables in
`workspace.gridiron_serving`. Removing a dataset from the pipeline later does not drop its
table; drop it by hand.

For the sync, note the workspace SQL warehouse's server hostname and HTTP path (SQL
Warehouses > your warehouse > Connection details).

## 3. Snowflake (trial)

Add a `snow` connection for your own user with the ACCOUNTADMIN role
(`snow connection add`, saved in `~/.snowflake/connections.toml`) and make it the default
(`snow connection set-default <name>`), or pass `-c <name>` to each command below.

### 3.1 Warehouse, database, roles and service users

```bash
snow sql -f snowflake/setup/01_warehouse_and_monitor.sql   # X-Small warehouse, resource monitor
snow sql -f snowflake/setup/02_database_and_roles.sql      # GRIDIRON database, schemas, roles, users
```

Both scripts are idempotent. The second creates five roles and two service users:

* `GRIDIRON_ADMIN` owns the database and runs the setup scripts;
* `GRIDIRON_LOADER` writes `SYNCED` only (including the temporary stage `write_pandas`
  loads through), used by the service user `GRIDIRON_SYNC`;
* `GRIDIRON_TRANSFORMER` reads `SYNCED` (or `ICEBERG`) and builds `STAGING` and `MARTS`,
  used by the service user `GRIDIRON_DBT`;
* `GRIDIRON_APP_OWNER` owns the Streamlit app. Streamlit in Snowflake runs an app's
  queries with its owner's rights, so this role can only read `MARTS`, call
  `APP.FANTASY_POINTS` and use the warehouse;
* `GRIDIRON_READER` is for viewers and analysts: `MARTS` and `APP`, not `STAGING`.

Service users cannot log in with a password. The script grants `GRIDIRON_ADMIN` and
`GRIDIRON_APP_OWNER` to the user who runs it.

### 3.2 Key pairs

Each service user gets its own RSA key pair. The sync and the dbt profile read the private
key file without a passphrase, so keep it outside the repository and readable only by you
(`*.p8` is git-ignored as well). For the sync user:

```bash
mkdir -p ~/.snowflake/keys && chmod 700 ~/.snowflake/keys && cd ~/.snowflake/keys
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -nocrypt -out gridiron_sync_key.p8
openssl rsa -in gridiron_sync_key.p8 -pubout -out gridiron_sync_key.pub
chmod 600 gridiron_sync_key.p8

# Register the public key body (the .pub file without its BEGIN/END lines)
snow sql -q "ALTER USER GRIDIRON_SYNC SET RSA_PUBLIC_KEY = '$(grep -v 'PUBLIC KEY' gridiron_sync_key.pub | tr -d '\n')'"

# Check: RSA_PUBLIC_KEY_FP in DESC USER is SHA256: followed by this value
snow sql -q "DESC USER GRIDIRON_SYNC"
openssl rsa -pubin -in gridiron_sync_key.pub -outform DER | openssl dgst -sha256 -binary | openssl enc -base64
```

Repeat with `gridiron_dbt_key` for `GRIDIRON_DBT`. Then, back in the repository, copy
`.env.example` to `.env` and fill it in: the Databricks host and HTTP path from step 2,
`SNOWFLAKE_ACCOUNT` (`<orgname>-<accountname>`), `SNOWFLAKE_USER=GRIDIRON_SYNC` with
`SNOWFLAKE_PRIVATE_KEY_FILE` pointing at its key, and `SNOWFLAKE_DBT_USER=GRIDIRON_DBT` with
`SNOWFLAKE_DBT_PRIVATE_KEY_FILE`. The dbt settings fall back to the sync user and key when
unset. While trying things out, both can also run as your own user with key-pair
authentication: the setup script grants it `GRIDIRON_ADMIN`, which inherits the loader and
transformer roles.

### 3.3 Synced tables and the first sync

```bash
snow sql -f snowflake/sync/01_synced_tables.sql   # generated DDL for the six SYNCED tables
make sync                                         # reads .env
```

`make sync` (`scripts/sync_to_snowflake.py`) reads each serving table through the
Databricks SQL warehouse as Arrow, authenticating with the Databricks CLI login unless
`DATABRICKS_TOKEN` is set. It loads the rows into a temporary staging table and MERGEs them
into `GRIDIRON.SYNCED`, touching only rows that changed and deleting rows that disappeared,
so the streams on `SYNCED.PLAYER_WEEK` and `SYNCED.PROJECTIONS` see real changes only. A
new `generated_at` on its own does not count as a change.

### 3.4 Scoring UDF, dbt, stream and task, app

```bash
snow sql -f snowflake/snowpark/01_create_udf.sql   # APP.FANTASY_POINTS, handler src/gridiron/scoring.py

set -a && . ./.env && set +a
(cd snowflake/dbt && ../../.venv/bin/dbt build --target snowflake --profiles-dir .)

snow sql -f snowflake/streams_tasks/01_projection_results_stream_task.sql
snow sql -f snowflake/streamlit/01_create_streamlit.sql
```

* dbt runs as `GRIDIRON_TRANSFORMER` and reads `SYNCED` (`gold_source: synced`).
* The stream and task script creates `APP.PROJECTION_RESULTS`, streams on
  `SYNCED.PLAYER_WEEK` and `SYNCED.PROJECTIONS` (with initial rows, so the first run
  backfills every week) and the task `APP.MAINTAIN_PROJECTION_RESULTS` (Tuesdays 14:00 UTC,
  skipped when both streams are empty), which rebuilds every week that either stream
  touched. To run it now, use the `EXECUTE TASK` line commented at the end of the script.
* The UDF script grants `GRIDIRON_APP_OWNER` usage on `APP.FANTASY_POINTS` each time it
  replaces the function.
* The app script uploads the files as `GRIDIRON_ADMIN`, then drops the app if it exists
  and creates it again as `GRIDIRON_APP_OWNER` (an app created by an earlier version of
  the script was owned by `GRIDIRON_ADMIN`).
* The app is created on the warehouse runtime (`RUNTIME_NAME = 'SYSTEM$WAREHOUSE_RUNTIME'`),
  because trial accounts cannot always start the compute pool the container runtime needs.
  `snowflake/streamlit/environment.yml` pins Streamlit 1.52.2, the newest in the Snowflake
  channel. Viewers need only `GRIDIRON_READER`.

### 3.5 Updating an existing deployment

The scripts are idempotent, so an update reruns them in the same order:
`02_database_and_roles.sql` (as ACCOUNTADMIN), `01_create_udf.sql`, the dbt build, the
stream and task script and the app script. The stream and task script keeps existing
streams and adds any new one; a new stream starts with every existing row, so the task's
next run rebuilds every week once.

## 4. The public copy (Streamlit Community Cloud)

The public app runs the same code against a static Parquet snapshot of the marts in
`streamlit_public/snapshot/`. Export it from the Databricks serving tables (same `.env`
and CLI login as the sync), check it, and commit it:

```bash
set -a && . ./.env && set +a && .venv/bin/python scripts/export_public_snapshot.py --from-databricks
make app-public
git add streamlit_public/snapshot
```

The export builds the dbt marts in a throwaway DuckDB database, copies only the columns
listed in `EXPORTED_COLUMNS` in the script (a new mart column stays out until it is added
there) and blanks the per-player FantasyPros ranks. A test checks that the committed files
have exactly those columns and no expert ranks. In Streamlit Community Cloud, create an
app from the GitHub repository, branch `main`, main file `streamlit_public/streamlit_app.py`,
Python 3.12; it installs `streamlit_public/requirements.txt`. The snapshot changes only when
it is exported and committed again.

## 5. Each week

The Databricks job runs on its own. After a run, `make sync` and the dbt build refresh
Snowflake, and the task rebuilds results for any week whose actuals changed. The sync, the
dbt build and the snapshot export are run by hand from a laptop; a scheduled GitHub Action
or dbt Projects on Snowflake could run them.

## The Iceberg path (workspaces with external storage)

The serving tables are published so that Snowflake could read them in place through Apache
Iceberg: Delta UniForm properties (`delta.enableIcebergCompatV2`,
`delta.universalFormat.enabledFormats = iceberg`, `delta.columnMapping.mode = name`,
deletion vectors off), created once and refreshed with `INSERT OVERWRITE` so their catalog
identity is stable. On Databricks Free Edition that path does not work, for two reasons
checked in the deployed workspace:

1. The Free Edition metastore uses Unity Catalog privilege model 1.0, in which the
   `EXTERNAL USE SCHEMA` privilege is not applicable, so it cannot be granted.
2. The Unity Catalog Iceberg REST endpoint returns the tables' Iceberg metadata but vends
   zero storage credentials for tables on Databricks default storage, which is where Free
   Edition keeps managed tables. Snowflake can see the tables but cannot read their files.

So the MERGE-based sync is the production path. The Iceberg SQL is kept for workspaces
whose serving schema lives on external storage, where credential vending applies; it has
not been run end to end:

```bash
snow sql -f snowflake/iceberg/01_catalog_integration.sql   # ACCOUNTADMIN; fill in the placeholders
snow sql -f snowflake/iceberg/02_iceberg_tables.sql        # generated Iceberg tables
snow sql -f snowflake/iceberg/03_iceberg_views.sql         # upper-case views in ICEBERG
(cd snowflake/dbt && ../../.venv/bin/dbt build --target snowflake --profiles-dir . --vars '{gold_source: iceberg}')
snow sql -f snowflake/streams_tasks/02_projection_results_dynamic_table.sql
```

On this path the dynamic table replaces the stream and task, because Databricks rewrites
each serving table on every run and there is no useful row-level change feed.

## Cost notes

* **Databricks Free Edition** has no charges and a daily fair-use compute quota. The job uses
  serverless compute only, one pipeline (Free Edition allows one per type) and five tasks
  run one at a time (it allows five concurrent tasks).
* **Snowflake trial**: one X-Small warehouse (1 credit per hour while running) with
  `AUTO_SUSPEND = 60` and a 30-minute statement timeout, under a resource monitor that
  notifies at 50% and 80% of 20 credits a month and suspends the warehouse at 100%. The task
  runs on that warehouse, so the monitor covers it, and its `WHEN SYSTEM$STREAM_HAS_DATA`
  clause skips runs (and warehouse starts) when nothing changed. An open Streamlit app keeps
  the warehouse running. Credits used so far are in [verification.md](verification.md).
