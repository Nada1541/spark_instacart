"""Benchmark RDD vs DataFrame vs SQL on the same 10 queries.

Produces results/benchmark_results.csv — paste this directly into the
performance comparison table the rubric requires (25% of the grade).

Run from project root:
    python src/benchmarks/run_benchmarks.py
"""
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.spark_session import get_spark, load_tables_parquet, RESULTS_DIR

# Reuse the query builders from the DataFrame and SQL modules.
# Importing the modules directly is awkward because their filenames start
# with digits, so we use importlib.
import importlib.util
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ROOT = Path(__file__).resolve().parents[2]
df_mod  = _load_module(ROOT / "src/dataframe/01_queries_df.py",  "df_q")
sql_mod = _load_module(ROOT / "src/sql/02_queries_sql.py",       "sql_q")


def time_count(df) -> float:
    t0 = time.perf_counter()
    df.count()
    return time.perf_counter() - t0


def time_sql(spark, sql: str) -> float:
    t0 = time.perf_counter()
    spark.sql(sql).count()
    return time.perf_counter() - t0


def main():
    spark = get_spark("Benchmarks")
    t = load_tables_parquet(spark)

    # Register SQL views
    for src_name, view in [("order_products_prior", "prior"),
                           ("orders", "orders"),
                           ("products", "products"),
                           ("departments", "departments"),
                           ("aisles", "aisles")]:
        if src_name in t:
            t[src_name].createOrReplaceTempView(view)

    rows = []  # (query, api, seconds)

    # --- Q1 ---
    print("Benchmarking q1...")
    rows.append(("q1", "DataFrame", time_count(df_mod.q1_complex_filter(t))))
    rows.append(("q1", "SQL",       time_sql(spark, sql_mod.Q1)))

    # --- Q2 ---
    print("Benchmarking q2...")
    rows.append(("q2", "DataFrame", time_count(df_mod.q2_aggregations(t))))
    rows.append(("q2", "SQL",       time_sql(spark, sql_mod.Q2)))

    # --- Q3 ---
    print("Benchmarking q3...")
    rows.append(("q3", "DataFrame", time_count(df_mod.q3_multi_groupby(t))))
    rows.append(("q3", "SQL",       time_sql(spark, sql_mod.Q3)))

    # --- Q4 ---
    print("Benchmarking q4...")
    rows.append(("q4", "DataFrame", time_count(df_mod.q4_top_products_per_department(t))))
    rows.append(("q4", "SQL",       time_sql(spark, sql_mod.Q4)))

    # --- Q5 ---
    print("Benchmarking q5...")
    rows.append(("q5", "DataFrame", time_count(df_mod.q5_running_basket_size(t))))
    rows.append(("q5", "SQL",       time_sql(spark, sql_mod.Q5)))

    # --- Q6 ---
    print("Benchmarking q6...")
    rows.append(("q6", "DataFrame", time_count(df_mod.q6_power_users(t))))
    rows.append(("q6", "SQL",       time_sql(spark, sql_mod.Q6)))

    # --- Q7 (broadcast join) ---
    print("Benchmarking q7 (broadcast)...")
    rows.append(("q7", "DataFrame", time_count(df_mod.q7_broadcast_join(t))))
    rows.append(("q7", "SQL",       time_sql(spark, sql_mod.Q7)))

    # --- Q8 (sort-merge join) ---
    print("Benchmarking q8 (sort-merge)...")
    orig = spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    rows.append(("q8", "DataFrame", time_count(df_mod.q8_sortmerge_join(spark, t))))
    rows.append(("q8", "SQL",       time_sql(spark, sql_mod.Q8)))
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", orig)

    # --- Q9 caching ---
    print("Benchmarking q9 caching...")
    r9 = df_mod.q9_caching_demo(t)
    rows.append(("q9_uncached_avg", "DataFrame", sum(r9['uncached'])/3))
    rows.append(("q9_cached_avg",   "DataFrame", sum(r9['cached'])/3))

    # --- Q10 partition pruning ---
    print("Benchmarking q10 partition pruning...")
    rows.append(("q10", "DataFrame", time_count(df_mod.q10_partition_pruning(spark, t))))

    # NOTE: RDD timings live in results/rdd_timings.csv (generated separately
    # by src/rdd/03_queries_rdd.py). Run that script first if you want the
    # third column for the report. We merge both here:
    rdd_csv = RESULTS_DIR / "rdd_timings.csv"
    if rdd_csv.exists():
        with open(rdd_csv) as f:
            next(f)  # header
            for line in f:
                q, sec = line.strip().split(",")
                rows.append((q, "RDD", float(sec)))

    # ----- write the comparison table -----
    out = RESULTS_DIR / "benchmark_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "api", "elapsed_sec"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}"])
    print(f"\n==> Benchmark table written to {out}")
    print("    Paste this into report/report.md §5 (Performance Comparison Table).")

    spark.stop()


if __name__ == "__main__":
    main()
