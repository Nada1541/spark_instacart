"""Scalability test: vary shuffle partitions and parallelism.

Runs the same heavy join+aggregation under several `spark.sql.shuffle.partitions`
values and writes the timings to results/scalability.csv.

This is the scalability test the spec asks for in section 5.

Run from project root:
    python src/benchmarks/scalability_test.py
"""
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyspark.sql import functions as F
from src.common.spark_session import get_spark, load_tables_parquet, RESULTS_DIR


# Vary shuffle partitions across this list
PARTITION_SETTINGS = [4, 16, 64, 200]


def heavy_query(t):
    """A representative shuffle-heavy query."""
    return (
        t["order_products_prior"]
        .join(t["orders"], "order_id")
        .join(t["products"], "product_id")
        .groupBy("product_id", "order_dow")
        .agg(F.count("*").alias("c"), F.avg("reordered").alias("rr"))
    )


def main():
    rows = []
    for n in PARTITION_SETTINGS:
        spark = get_spark(f"Scalability_{n}", shuffle_partitions=n)
        t = load_tables_parquet(spark)
        # Warm up file metadata / catalogs
        t["order_products_prior"].count()

        t0 = time.perf_counter()
        heavy_query(t).count()
        elapsed = time.perf_counter() - t0
        print(f"  shuffle.partitions={n:>3}  →  {elapsed:.2f}s")
        rows.append((n, elapsed))
        spark.stop()

    out = RESULTS_DIR / "scalability.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shuffle_partitions", "elapsed_sec"])
        for n, e in rows:
            w.writerow([n, f"{e:.3f}"])
    print(f"\n==> Written: {out}")


if __name__ == "__main__":
    main()
