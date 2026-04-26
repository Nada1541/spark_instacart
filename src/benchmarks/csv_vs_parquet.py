"""Compare query performance over CSV vs Parquet files.

Re-runs three representative queries (a filter, an aggregate, a join) over
the raw CSVs and the Parquet output, prints a side-by-side timing table.

This is the file-format comparison the rubric asks for in section 6.

Run from project root:
    python src/benchmarks/csv_vs_parquet.py
"""
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

from src.common.spark_session import (
    get_spark, load_tables, load_tables_parquet, RESULTS_DIR,
)


def time_action(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run_suite(label: str, t: dict) -> dict:
    """Run a fixed set of queries and return {query_name: seconds}."""
    out = {}

    # 1) Filter on the big fact table — Parquet's column pruning matters
    out["filter_only_one_column"] = time_action(
        lambda: t["order_products_prior"]
                 .where(F.col("reordered") == 1)
                 .select(F.count("*"))
                 .collect()
    )

    # 2) Aggregation over a few columns — Parquet's columnar scan matters
    out["agg_two_columns"] = time_action(
        lambda: t["order_products_prior"]
                 .groupBy("product_id")
                 .agg(F.count("*"), F.sum("reordered"))
                 .count()
    )

    # 3) Broadcast join — file format affects scan, not the join itself
    out["broadcast_join"] = time_action(
        lambda: t["order_products_prior"]
                 .join(t["products"].select("product_id", "aisle_id"), "product_id")
                 .join(broadcast(t["aisles"]), "aisle_id")
                 .groupBy("aisle").count()
                 .count()
    )

    print(f"\n--- {label} ---")
    for k, v in out.items():
        print(f"  {k:35s} {v:6.2f}s")
    return out


def main():
    spark = get_spark("CSVvsParquet")

    print("==> Loading CSV tables...")
    csv_tables = load_tables(spark)
    csv_results = run_suite("CSV", csv_tables)

    print("\n==> Loading Parquet tables...")
    pq_tables = load_tables_parquet(spark)
    pq_results = run_suite("Parquet", pq_tables)

    # ----- write side-by-side CSV -----
    out = RESULTS_DIR / "csv_vs_parquet.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "csv_sec", "parquet_sec", "speedup"])
        for k in csv_results:
            cs = csv_results[k]
            ps = pq_results[k]
            w.writerow([k, f"{cs:.3f}", f"{ps:.3f}", f"{cs/ps:.2f}x"])
    print(f"\n==> Written: {out}")
    spark.stop()


if __name__ == "__main__":
    main()
