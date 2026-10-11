"""Job task: publish the serving datasets as UniForm (Iceberg-readable) tables."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from gridiron import joblog
from gridiron.serving import SERVING_TABLES, publish_statements, qualified


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True, help="schema holding the gold tables")
    parser.add_argument("--serving-schema", required=True)
    args = parser.parse_args(argv)
    joblog.setup()

    spark = SparkSession.builder.getOrCreate()
    actions: dict[str, list[str]] = {"replaced": [], "refreshed": []}
    for table in SERVING_TABLES:
        target = qualified(args.catalog, args.serving_schema, table)
        source_columns = spark.read.table(qualified(args.catalog, args.schema, table)).columns
        existing = (
            spark.read.table(target).columns
            if spark.catalog.tableExists(f"{args.catalog}.{args.serving_schema}.{table}")
            else None
        )
        statements = publish_statements(
            args.catalog,
            args.schema,
            args.serving_schema,
            table,
            existing_columns=existing,
            source_columns=source_columns,
        )
        for statement in statements:
            spark.sql(statement)
        actions["replaced" if len(statements) == 1 else "refreshed"].append(table)
    joblog.summary("publish_serving", schema=args.serving_schema, **actions)


if __name__ == "__main__":
    # No sys.exit: Databricks reports any SystemExit as a failed task, even exit code 0.
    main()
