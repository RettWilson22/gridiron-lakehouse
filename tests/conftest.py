from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession

from gridiron.config import FIRST_SEASON
from gridiron.lakehouse import build_gold, build_silver, read_landing
from gridiron.local_spark import local_session
from gridiron.workflow import Prepared, prepare

FIXTURES = Path(__file__).parent / "fixtures"
LANDING = FIXTURES / "landing"
GOLD = FIXTURES / "gold"
REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    session = local_session("gridiron-tests", shuffle_partitions=2)
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def bronze(spark: SparkSession) -> dict[str, DataFrame]:
    return read_landing(spark, LANDING)


@pytest.fixture(scope="session")
def silver(bronze: dict[str, DataFrame]) -> dict[str, DataFrame]:
    return {name: df.cache() for name, df in build_silver(bronze, FIRST_SEASON).items()}


@pytest.fixture(scope="session")
def gold(silver: dict[str, DataFrame]) -> dict[str, DataFrame]:
    return {name: df.cache() for name, df in build_gold(silver).items()}


@pytest.fixture(scope="session")
def tables(silver: dict[str, DataFrame], gold: dict[str, DataFrame]) -> dict[str, pd.DataFrame]:
    """Every silver and gold table built from the landing fixture, as pandas."""
    return {name: df.toPandas() for name, df in {**silver, **gold}.items()}


@pytest.fixture(scope="session")
def prepared(tables: dict[str, pd.DataFrame]) -> Prepared:
    return prepare(lambda name: tables[name].copy())
