"""Job task: publish gold tables as UniForm (Iceberg-readable) tables for Snowflake."""

from __future__ import annotations

import argparse
import sys

from pyspark.sql import SparkSession

from gridiron.serving import SERVING_TABLES, publish_statements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True, help="schema holding the gold tables")
    parser.add_argument("--serving-schema", required=True)
    args = parser.parse_args(argv)

    spark = SparkSession.builder.getOrCreate()
    for table in SERVING_TABLES:
        for statement in publish_statements(args.catalog, args.schema, args.serving_schema, table):
            spark.sql(statement)
        print(f"published {args.catalog}.{args.serving_schema}.{table}")
    return 0


if __name__ == "__main__":
    # Databricks reports any SystemExit as a failed task, even exit code 0, so only exit
    # explicitly on failure.
    if (code := main()) != 0:
        sys.exit(code)
