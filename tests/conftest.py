from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession

from gridiron import transforms
from gridiron.local_spark import local_session

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    session = local_session("gridiron-tests", shuffle_partitions=2)
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def sample_pdf() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "pbp_sample.parquet")


@pytest.fixture(scope="session")
def bronze(spark: SparkSession) -> DataFrame:
    return transforms.with_ingest_metadata(spark.read.parquet(str(FIXTURES / "pbp_sample.parquet")))


@pytest.fixture(scope="session")
def plays(bronze: DataFrame) -> DataFrame:
    return transforms.silver_plays(bronze).cache()


@pytest.fixture(scope="session")
def decisions(plays: DataFrame) -> DataFrame:
    return transforms.fourth_down_decisions(plays).cache()


@pytest.fixture(scope="session")
def decisions_pdf(decisions: DataFrame) -> pd.DataFrame:
    return decisions.toPandas()
