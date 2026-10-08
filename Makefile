# Common tasks. Everything runs from the repository root with the project virtualenv.
PYTHON ?= python3.12
VENV   ?= .venv
BIN    := $(VENV)/bin
DBT    := cd snowflake/dbt && ../../$(BIN)/dbt
TARGET ?= dev

.PHONY: help setup lint format typecheck test sql-check dbt-ci dbt-local check \
        ingest pipeline-local train-local fixtures smoke-jobs app \
        bundle-validate bundle-deploy bundle-run sync clean

help:  ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install the project with all local extras
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[spark,ml,app,sync,dev]"

lint:  ## Ruff lint and format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Apply ruff formatting and safe fixes
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck:  ## mypy (strict)
	$(BIN)/mypy

test:  ## pytest (local Spark needs Java 17+)
	$(BIN)/pytest

sql-check:  ## Fail if generated Snowflake DDL is out of date
	$(BIN)/python scripts/generate_snowflake_sql.py --check

dbt-ci:  ## dbt build on DuckDB against the checked-in fixtures
	$(DBT) build --target ci --profiles-dir .

dbt-local:  ## dbt build on DuckDB against the full local lakehouse (data/lakehouse)
	$(DBT) build --target local --profiles-dir . --vars '{gold_dir: ../../data/lakehouse}'

check: lint typecheck sql-check dbt-ci test  ## Everything CI runs

ingest:  ## Download nflverse seasons into data/landing (idempotent)
	$(BIN)/python databricks/jobs/ingest.py --landing-dir data/landing

pipeline-local:  ## Run the bronze/silver/gold transformations locally into data/lakehouse
	$(BIN)/python scripts/run_local_pipeline.py --landing-dir data/landing --out-dir data/lakehouse

train-local:  ## Train/evaluate the model locally; writes artifacts/ and scored tables
	$(BIN)/python scripts/train_local.py --lakehouse-dir data/lakehouse

fixtures:  ## Rebuild checked-in test fixtures and generated SQL from local data
	$(BIN)/python scripts/build_fixtures.py --landing-dir data/landing
	$(BIN)/python scripts/generate_snowflake_sql.py

smoke-jobs:  ## Run the Databricks job entry points against a local Spark catalog
	$(BIN)/python scripts/smoke_databricks_jobs.py --lakehouse-dir data/lakehouse

app:  ## Run the Streamlit app locally against the DuckDB marts
	$(BIN)/streamlit run snowflake/streamlit/streamlit_app.py

bundle-validate:  ## databricks bundle validate (needs the Databricks CLI and auth)
	cd databricks && databricks bundle validate -t $(TARGET)

bundle-deploy:  ## databricks bundle deploy
	cd databricks && databricks bundle deploy -t $(TARGET)

bundle-run:  ## Run the end-to-end Databricks job
	cd databricks && databricks bundle run -t $(TARGET) gridiron_refresh

sync:  ## Fallback: copy Databricks serving tables into Snowflake (reads .env)
	set -a && . ./.env && set +a && $(BIN)/python scripts/sync_to_snowflake.py

clean:  ## Remove caches and build output (keeps data/ and .venv/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache spark-warehouse metastore_db mlruns
	rm -rf snowflake/dbt/target snowflake/dbt/logs snowflake/dbt/*.duckdb
