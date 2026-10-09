# Verification

What has actually been run, where, and what it produced. All times are UTC.

## Cloud: Databricks Free Edition and a Snowflake trial

Deployed and checked on 2026-10-08 and 2026-10-09 (Databricks Free Edition, serverless;
Snowflake trial on AWS us-east-2), following [deploy.md](deploy.md). The production job run
below started on 2026-10-09, after the 00:15 kickoff of the week 5 Thursday game.

| Step | Result |
| --- | --- |
| Databricks job `gridiron-refresh` (prod target) | 12.2 min end to end: ingest 1.5, pipeline 3.5, train 4.7, score 1.4, publish 1.0 |
| Model | `workspace.gridiron.fantasy_projection` v1 registered in Unity Catalog with alias `champion` |
| Scored | 28,396 projections (27,857 backtest + 539 live for 2026 week 5), 529 risers |
| Sync to Snowflake | 6 tables, row counts identical to Databricks (largest: `player_week`, 48,018 rows) |
| dbt on Snowflake | 55 of 55 models and tests pass, including the check that SQL reproduces the Python backtest |
| Stream + task | first run backfilled 27,857 projected-vs-actual rows in 3.1 s |
| Scoring UDF | `APP.FANTASY_POINTS` registered from `src/gridiron/scoring.py` and returning points |
| Streamlit in Snowflake | app deployed on the warehouse runtime with Streamlit 1.52.2 |
| Public copy | snapshot exported with `scripts/export_public_snapshot.py --from-databricks`, committed, deployed to Streamlit Community Cloud |
| Snowflake credits | about 1.4 for the deployment and a day of testing |

Because the job ran after Thursday's kickoff (TB at DAL), that game has no live projection
in the cloud data or in the public snapshot: projections are never created after kickoff.
The local run, made before kickoff, did project it (below). The cloud backtest has the same
sample sizes, baselines and expert correlations as the local one; the model's own numbers
differ slightly because it was trained separately ([methodology.md](methodology.md)).

Also confirmed in the cloud:

* the pipeline on serverless, including all eleven Auto Loader bronze tables and depth charts
  in both formats (weekly and timestamped snapshots);
* MLflow pyfunc logging with `code_paths`, Unity Catalog registration and the alias;
* Free Edition can download from both GitHub releases and `raw.githubusercontent.com`;
* removing datasets from the pipeline does not drop their tables (they were dropped by hand);
* every Snowflake script on the sync path: setup, synced tables, scoring UDF, stream and
  task, and the Streamlit app;
* the Iceberg path stops at credential vending on Free Edition
  ([details](deploy.md#the-iceberg-path-workspaces-with-external-storage)), so its SQL has
  not been run end to end.

Changed since that deployment and not yet deployed: the years-of-experience feature fix
(identical outputs locally, below), the `GRIDIRON_DBT` service user in
`02_database_and_roles.sql`, and small wording changes in the app.

## Local: a laptop against the real data

Apple Silicon, Python 3.12, Java 21, on 2026-10-08 and 2026-10-09, with the commands in
[deploy.md](deploy.md#1-the-full-flow-on-a-laptop).

* **Ingest** landed every dataset for 2018-2026. A re-run skipped completed seasons and
  rewrote only what had changed upstream (the schedule file) in 8 seconds.
* **Pipeline** (`make pipeline-local`, plain Spark over the same functions and expectation
  rules as the Lakeflow pipeline): 36 seconds. 400,513 plays, 151,498 box scores, 211,325
  snap counts, 364,933 depth chart rows, 43,255 weekly ECR rows; gold `player_week` 48,018
  rows, `team_week` 4,798, `defense_vs_position` 17,528, `depth_chart_week` 71,075.
  Warn-level expectations flagged 215 snap-count rows without a GSIS id, 190 ranking rows
  without one, and 44 player-games without a snap count.
* **Model** (`make model-local`): about a minute for four walk-forward models (2023, 2024,
  2025 and the 2026 backfill), the live model and scoring. 27,857 backtest projections and
  572 live projections for 2026 week 5 (two teams on bye), published at 23:54 on
  2026-10-08. Re-runs after Thursday's kickoff (00:18 and 02:08 on 2026-10-09) kept the 33
  projections for that game exactly as published and refreshed the rest.
* **Years-of-experience fix** (2026-10-09): rebuilding the 67,246 feature rows with the
  roster as of each week changed no value, and `make model-local` reproduced
  `artifacts/backtest_metrics.json` exactly (the file now also records candidate-pool
  coverage).
* **Job smoke test** (`make smoke-jobs`): the real `train`, `score` (twice) and
  `publish_serving` (twice) entry points against a local Spark catalog, with MLflow logging
  and loading the pyfunc model from a local store (registration skipped).
* **dbt**: `dbt build --target ci` passes 54 nodes on the fixtures; `make dbt-local` passes 55
  over the full local data, including the SQL-vs-Python reconciliation test.
* **Checks**: `make check` (ruff, mypy strict, generated-SQL check, dbt CI build, 131 pytest
  tests) passes; CI runs the same steps on every push to main and every pull request.
* **App**: the Streamlit app's headless tests (`tests/test_streamlit_app.py`) cover the
  local DuckDB mode and the public snapshot mode.
