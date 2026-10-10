from __future__ import annotations

import sys

from pyspark.sql import SparkSession


def test_python_workers_run_the_driver_interpreter(spark: SparkSession) -> None:
    workers = spark.sparkContext.parallelize(range(2), 2).map(lambda _: sys.executable).collect()
    assert set(workers) == {sys.executable}
