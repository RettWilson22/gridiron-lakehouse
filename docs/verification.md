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

Changed since that deployment and not yet deployed, all checked locally (below):

* Databricks: the years-of-experience feature fix (identical outputs), gradient-boosting
  early stopping switched off (new numbers), refits skipped when the training data is
  unchanged (fingerprinted model versions, a tagged champion), features for the upcoming
  week only in the score task, typed empty risers, the upcoming week limited to the
  current season, new silver expectations on names, teams and positions, and MLflow
  bounded below 4.
* Snowflake: the `GRIDIRON_DBT` service user and the `GRIDIRON_APP_OWNER` role in
  `02_database_and_roles.sql` (the app is now created as that role; the reader loses
  `STAGING`, the loader gets `CREATE STAGE`), a second stream on `SYNCED.PROJECTIONS`
  for the results task, and a sync that ignores `generated_at` on its own.
* The app: one cached sheet per week, league scoring in a form, escaped third-party
  text, capped search input, and small wording changes.
* The model: floor and ceiling bands fitted on out-of-fold projections (new ranges, same
  point projections), and week-level bootstrap intervals for the backtest comparisons,
  which only the local runner writes (`artifacts/backtest_intervals.json`).

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
* **Early stopping off** (2026-10-09): before the change, `make model-local` on a copy of
  the data reproduced `artifacts/backtest_metrics.json` exactly in 84 seconds. With early
  stopping off it took 96 seconds and produced the same 27,857 backtest projections, sample
  sizes, baselines and expert numbers. Model metrics changed where early stopping had been
  on (WRs in every test season, RBs in 2024 and 2025, TEs in 2025); QB numbers are
  identical. That run (after Thursday's kickoff) kept the 33 projections for Thursday's
  game as published and refreshed the other 539 live projections with the new model.
* **Skipping refits** (2026-10-09): `make model-local` took 86 seconds with no stored
  backtest table and 19 to 21 seconds on reruns that reused all four backtest seasons,
  with identical metrics.
* **Choosing the band folds** (2026-10-09): the 2022 season, projected by a model
  trained on 2018-2021 with bands from its own training projections and from 2, 3 and 4
  out-of-fold groups, gave the coverage table in
  [methodology.md](methodology.md#model). The walk-forward step for that one season took
  24, 34, 45 and 63 seconds.
* **Out-of-fold bands and bootstrap intervals** (2026-10-09, 21:15 to 21:20): before the
  change, `make model-local` on a copy of the data reproduced
  `artifacts/backtest_metrics.json` exactly, in 80 seconds with no stored backtest table
  and 20 seconds on a rerun. After it, `make model-local` refitted all four backtest
  seasons (the fold setting is part of the fingerprint) in 271 seconds, and a rerun that
  reused all four took 60 seconds. The 27,857 backtest projections, sample sizes and every
  MAE, RMSE, bias and rank correlation were identical to before; only range coverage
  changed (2023-2025: QB 77.5% to 80.4%, RB 82.6% to 84.8%, WR 83.5% to 85.1%, TE 78.7% to
  81.5%). Recomputing the intervals from the stored projections in a separate process gave
  the same numbers as `artifacts/backtest_intervals.json`. The live run kept the 33
  projections for Thursday's game as published on 2026-10-08 and refreshed the other 539
  with the new ranges.
* **Job smoke test** (`make smoke-jobs`): the real `train` (twice), `score` (twice) and
  `publish_serving` (twice) entry points against a local Spark catalog, with a throwaway
  MLflow file store as tracking store and model registry. The first `train` registered v1
  as champion with its training fingerprint; the second reused both backtest seasons,
  found the champion trained on the same data and registered nothing. `score` loaded the
  champion through the registry, as in production. 79 seconds in all. Rerun after the
  band change (2026-10-09, 21:22 to 21:26) with the same results: the second `train`
  reused both seasons and kept v1. It took 206 seconds, with other local work running
  part of that time.
* **New silver expectations** (names, teams, positions), run over the local silver
  tables: the only rows flagged were the 88 box scores and 1 player row of "Kenneth
  Murray, Jr." (a comma in the name).
* **dbt**: `dbt build --target ci` passes 54 nodes on the fixtures; `make dbt-local` passes 55
  over the full local data, including the SQL-vs-Python reconciliation test.
* **Checks**: `make check` (ruff, mypy strict, generated-SQL check, dbt CI build, 219 pytest
  tests) passes; CI runs the same steps on every push to main and every pull request.
* **App**: the Streamlit app's 13 headless tests (`tests/test_streamlit_app.py`) cover the
  local DuckDB mode and the public snapshot mode, including the league scoring form, the
  number of queries a scoring change runs and a hostile player name rendered as text.
