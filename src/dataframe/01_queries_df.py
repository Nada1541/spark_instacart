import sys
import time
import io
import contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyspark.sql import DataFrame, functions as F, Window
from pyspark.sql.functions import broadcast

from src.common.spark_session import (
    get_spark, load_tables_parquet, RESULTS_DIR, DATA_DIR,
)


def save_explain(df: DataFrame, name: str, mode: str = "extended") -> None:
    out = RESULTS_DIR / f"explain_df_{name}.txt"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df.explain(mode)
    out.write_text(buf.getvalue())
    print(f"    [explain saved → results/{out.name}]")

# Q1 — Complex filtering
def q1_complex_filter(t):
    """Weekend (Sat/Sun) lunch-hour orders with > 10 items."""
    basket_sizes = (
        t["order_products_prior"]
        .groupBy("order_id").agg(F.count("*").alias("basket_size"))
    )
    return (
        t["orders"].join(basket_sizes, "order_id")
        .where(
            F.col("order_dow").isin(0, 1)
            & F.col("order_hour_of_day").between(11, 14)
            & (F.col("basket_size") > 10)
        )
        .select("order_id", "user_id", "order_dow",
                "order_hour_of_day", "basket_size", "order_ts")
    )

# Q2 — Aggregations: SUM, AVG, COUNT, MIN, MAX
def q2_aggregations(t):
    return t["orders"].agg(
        F.count("*").alias("total_orders"),
        F.countDistinct("user_id").alias("unique_users"),
        F.avg("days_since_prior_order").alias("avg_days_between"),
        F.min("order_hour_of_day").alias("earliest_hour"),
        F.max("order_hour_of_day").alias("latest_hour"),
        F.sum(F.when(F.col("order_dow").isin(0, 1), 1).otherwise(0)).alias("weekend_orders"),
    )


# Q3 — Multi-attribute grouping
def q3_multi_groupby(t):
    return (
        t["order_products_prior"]
        .join(t["orders"],      "order_id")
        .join(t["products"],    "product_id")
        .join(t["departments"], "department_id")
        .groupBy("department", "order_dow", "order_hour_of_day")
        .agg(
            F.count("*").alias("line_items"),
            F.avg("reordered").alias("reorder_rate"),
        )
    )

# Q4 — Sorting + ranking with window  
def q4_top_products_per_department(t, top_n: int = 5):
    """Top-N products per department by reorders.

    FIX: Build the aggregate first, then reference its column in the window.
    """
    reorder_counts = (
        t["order_products_prior"]
        .join(t["products"],    "product_id")
        .join(t["departments"], "department_id")
        .groupBy("department", "product_id", "product_name")
        .agg(F.sum("reordered").alias("total_reorders"))
    )
    w = Window.partitionBy("department").orderBy(F.col("total_reorders").desc())
    return (
        reorder_counts
        .withColumn("rank", F.rank().over(w))
        .where(F.col("rank") <= top_n)
        .orderBy("department", "rank")
    )

# Q5 — Window functions: running averages + cumulative sums
def q5_running_basket_size(t):
    basket = (
        t["order_products_prior"]
        .groupBy("order_id").agg(F.count("*").alias("basket_size"))
    )
    enriched = t["orders"].join(basket, "order_id")
    w = (
        Window.partitionBy("user_id")
              .orderBy("order_number")
              .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    return (
        enriched
        .withColumn("running_avg_basket", F.avg("basket_size").over(w))
        .withColumn("cumulative_items",   F.sum("basket_size").over(w))
        .select("user_id", "order_number", "order_ts",
                "basket_size", "running_avg_basket", "cumulative_items")
    )


# Q6 — Nested subquery (HAVING-style filter on a global aggregate)
def q6_power_users(t):
    basket = (
        t["order_products_prior"]
        .groupBy("order_id").agg(F.count("*").alias("basket_size"))
    )
    user_avg = (
        t["orders"].join(basket, "order_id")
        .groupBy("user_id").agg(F.avg("basket_size").alias("user_avg_basket"))
    )
    overall = user_avg.agg(F.avg("user_avg_basket").alias("global_avg"))
    return (
        user_avg.crossJoin(broadcast(overall))   # broadcast the 1-row aggregate
        .where(F.col("user_avg_basket") > F.col("global_avg"))
        .select("user_id", "user_avg_basket", "global_avg")
    )

# Q7 — Broadcast join (tiny dimension)
def q7_broadcast_join(t):
    """Line-items per aisle. `aisles` is 134 rows → forced broadcast."""
    return (
        t["order_products_prior"]
        .join(t["products"].select("product_id", "aisle_id"), "product_id")
        .join(broadcast(t["aisles"]), "aisle_id")
        .groupBy("aisle").agg(F.count("*").alias("line_items"))
        .orderBy(F.col("line_items").desc())
    )


# Q8 — Sort-merge join
def q8_sortmerge_join(spark, t):
    """32M ⋈ 3.4M with broadcasting OFF → forces SortMergeJoin in the plan."""
    orig = spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    try:
        return (
            t["order_products_prior"]
            .join(t["orders"], "order_id")
            .groupBy("order_dow")
            .agg(
                F.count("*").alias("line_items"),
                F.avg("reordered").alias("reorder_rate"),
            )
            .orderBy("order_dow")
        )
    finally:
        spark.conf.set("spark.sql.autoBroadcastJoinThreshold", orig)

# Q9 — Caching impact  
def q9_caching_demo(t):
    """Time the same aggregation 3× uncached, then 3× cached.

    FIX: cache() returns a NEW DataFrame, so we have to reassign.
    The original code threw the cached reference away.
    """
    base = (
        t["order_products_prior"]
        .join(t["orders"].select("order_id", "user_id"), "order_id")
    )

    def _time_3_runs(df):
        out = []
        for _ in range(3):
            t0 = time.perf_counter()
            df.groupBy("user_id").agg(F.count("*").alias("c")).count()
            out.append(time.perf_counter() - t0)
        return out

    uncached = _time_3_runs(base)

    cached = base.cache()         
    cached.count()                
    cached_times = _time_3_runs(cached)
    cached.unpersist()

    return {"uncached": uncached, "cached": cached_times}

# Q10 — Partition pruning (write partitioned, read with predicate)
def q10_partition_pruning(spark, t, target_dept: int = 4):
    path = DATA_DIR / "output" / "products_partitioned.parquet"
    if not path.exists():
        (
            t["products"]
            .write.mode("overwrite")
            .partitionBy("department_id")
            .parquet(str(path))
        )
    return (
        spark.read.parquet(str(path))
        .where(F.col("department_id") == target_dept)
    )

# Driver
def main():
    spark = get_spark("DF_Queries")
    t = load_tables_parquet(spark)

    queries = [
        ("q1_complex_filter",     lambda: q1_complex_filter(t)),
        ("q2_aggregations",       lambda: q2_aggregations(t)),
        ("q3_multi_groupby",      lambda: q3_multi_groupby(t)),
        ("q4_top_per_dept",       lambda: q4_top_products_per_department(t)),
        ("q5_running_basket",     lambda: q5_running_basket_size(t)),
        ("q6_power_users",        lambda: q6_power_users(t)),
        ("q7_broadcast_join",     lambda: q7_broadcast_join(t)),
        ("q8_sortmerge_join",     lambda: q8_sortmerge_join(spark, t)),
        ("q10_partition_pruning", lambda: q10_partition_pruning(spark, t)),
    ]

    for name, build in queries:
        print(f"\n=== {name} ===")
        df = build()
        save_explain(df, name)
        t0 = time.perf_counter()
        n = df.count()
        print(f"    rows={n:,}  elapsed={time.perf_counter()-t0:.2f}s")

    # Q9 is timing-based, not row-count-based
    print("\n=== q9_caching_demo ===")
    r = q9_caching_demo(t)
    print(f"    uncached: {[f'{x:.2f}s' for x in r['uncached']]}")
    print(f"    cached  : {[f'{x:.2f}s' for x in r['cached']]}")

    spark.stop()


if __name__ == "__main__":
    main()
    import time
    print("\n>>> Spark UI alive at http://localhost:4040 — Ctrl+C when done <<<")
    time.sleep(1800)   
