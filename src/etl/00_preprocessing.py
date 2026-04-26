import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyspark.sql import functions as F, Window
from src.common.spark_session import (
    get_spark, load_tables, DATA_DIR, PARQUET_DIR, CSV_FILENAMES,
)

BASE_DATE = "2017-01-01"


def synthesize_order_timestamp(orders_df):
    """Add an `order_ts` TimestampType column to orders.

    Cumulatively sums days_since_prior_order over each user's orders
    (sorted by order_number), anchors to BASE_DATE, adds hour-of-day.
    """
    w = Window.partitionBy("user_id").orderBy("order_number")
    return (
        orders_df
        .withColumn(
            "days_elapsed",
            F.sum(F.coalesce(F.col("days_since_prior_order"), F.lit(0.0))).over(w),
        )
        .withColumn(
            "order_date",
            F.expr(f"date_add(to_date('{BASE_DATE}'), cast(days_elapsed as int))"),
        )
        .withColumn(
            "order_ts",
            (F.col("order_date").cast("timestamp").cast("long")
             + F.col("order_hour_of_day") * 3600).cast("timestamp"),
        )
        .drop("days_elapsed")
    )


def run_preprocessing():
    spark = get_spark("InstacartETL", shuffle_partitions=64)
    print("==> SparkSession ready")

    t = load_tables(spark)
    print(f"==> Loaded {len(t)} CSV tables")

    orders = (
        t["orders"]
        .dropDuplicates(["order_id"])
        .fillna({"days_since_prior_order": 0.0})
    )

    orders = synthesize_order_timestamp(orders)
    print("==> Synthesized order_ts column")

    cleaned = {
        "aisles":               t["aisles"],
        "departments":          t["departments"],
        "products":             t["products"],
        "orders":               orders,
        "order_products_prior": t["order_products_prior"].dropDuplicates(),
        "order_products_train": t["order_products_train"].dropDuplicates(),
    }

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in cleaned.items():
        out_path = PARQUET_DIR / name
        if name == "order_products_prior":
            df = df.repartition(16, "order_id")
        print(f"    writing {out_path}")
        df.write.mode("overwrite").parquet(str(out_path))

    wide = (
        cleaned["order_products_prior"]
        .join(cleaned["orders"],      "order_id")
        .join(cleaned["products"],    "product_id")
        .join(cleaned["aisles"],      "aisle_id")
        .join(cleaned["departments"], "department_id")
    )
    wide_path = DATA_DIR / "output" / "instacart_wide.parquet"
    print(f"    writing denormalized wide table → {wide_path}")
    (
        wide
        .repartition(16, "user_id")
        .write.mode("overwrite").parquet(str(wide_path))
    )

    print("\n==> Preprocessing complete.")
    print(f"    Per-table Parquet:  {PARQUET_DIR}")
    print(f"    Denormalized wide:  {wide_path}")
    spark.stop()


if __name__ == "__main__":
    run_preprocessing()
