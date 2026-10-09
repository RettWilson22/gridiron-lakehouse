"""Moving tables between Spark and pandas in the job tasks."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
from pyspark.sql import DataFrame, SparkSession


def arrow_table(frame: pd.DataFrame) -> pa.Table:
    """pandas -> Arrow, with all-null columns typed as strings (Delta has no NULL type)."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    fields = [
        pa.field(f.name, pa.string()) if pa.types.is_null(f.type) else f for f in table.schema
    ]
    return table.cast(pa.schema(fields))


def to_spark(spark: SparkSession, frame: pd.DataFrame) -> DataFrame:
    # Going through a pyarrow Table (Spark >= 4.0) avoids pandas-version-specific
    # conversion paths for nullable and Arrow-backed columns.
    return spark.createDataFrame(arrow_table(frame))


def write_table(spark: SparkSession, frame: pd.DataFrame, table: str) -> None:
    (
        to_spark(spark, frame)
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table)
    )


def read_table(spark: SparkSession, table: str) -> pd.DataFrame:
    frame: pd.DataFrame = spark.read.table(table).toPandas()
    return frame
