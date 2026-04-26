import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    IntegerType, StringType, DoubleType, StructField, StructType,
)

# ---------------------------------------------------------------------------
# Paths — single source of truth, imported by every other module
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
PARQUET_DIR = DATA_DIR / "output" / "parquet"     
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------
def get_spark(app_name: str = "InstacartProject",
              shuffle_partitions: int = 64,
              driver_memory: str = "4g") -> SparkSession:
    """Build (or return existing) SparkSession.

    Defaults are tuned for a single-machine local run. For a cluster, raise
    `shuffle_partitions` and `driver_memory`.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))
        .getOrCreate()
    )


SCHEMAS = {
    "aisles": StructType([
        StructField("aisle_id",   IntegerType(), False),
        StructField("aisle",      StringType(),  False),
    ]),
    "departments": StructType([
        StructField("department_id", IntegerType(), False),
        StructField("department",    StringType(),  False),
    ]),
    "products": StructType([
        StructField("product_id",    IntegerType(), False),
        StructField("product_name",  StringType(),  False),
        StructField("aisle_id",      IntegerType(), False),
        StructField("department_id", IntegerType(), False),
    ]),
    "orders": StructType([
        StructField("order_id",               IntegerType(), False),
        StructField("user_id",                IntegerType(), False),
        StructField("eval_set",               StringType(),  False),
        StructField("order_number",           IntegerType(), False),
        StructField("order_dow",              IntegerType(), False),
        StructField("order_hour_of_day",      IntegerType(), False),
        StructField("days_since_prior_order", DoubleType(),  True),
    ]),
    "order_products_prior": StructType([
        StructField("order_id",          IntegerType(), False),
        StructField("product_id",        IntegerType(), False),
        StructField("add_to_cart_order", IntegerType(), False),
        StructField("reordered",         IntegerType(), False),
    ]),
    "order_products_train": StructType([
        StructField("order_id",          IntegerType(), False),
        StructField("product_id",        IntegerType(), False),
        StructField("add_to_cart_order", IntegerType(), False),
        StructField("reordered",         IntegerType(), False),
    ]),
}

CSV_FILENAMES = {
    "aisles":               "aisles.csv",
    "departments":          "departments.csv",
    "products":             "products.csv",
    "orders":               "orders.csv",
    "order_products_prior": "order_products__prior.csv",
    "order_products_train": "order_products__train.csv",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_tables(spark: SparkSession) -> dict:
    """Load all 6 raw CSVs as DataFrames keyed by table name."""
    out = {}
    for name, fname in CSV_FILENAMES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"[WARN] {path} missing — skipping {name}")
            continue
        out[name] = (
            spark.read
            .option("header", True)
            .schema(SCHEMAS[name])
            .csv(str(path))
        )
    return out


def load_tables_parquet(spark: SparkSession) -> dict:
    """Load all 6 tables from Parquet (after ETL has run).

    Used by query files when a Parquet copy exists — much faster than CSV.
    Falls back to CSV automatically if Parquet isn't there yet.
    """
    if not PARQUET_DIR.exists():
        print(f"[INFO] {PARQUET_DIR} not found — falling back to CSV.")
        return load_tables(spark)

    return {
        name: spark.read.parquet(str(PARQUET_DIR / name))
        for name in CSV_FILENAMES
        if (PARQUET_DIR / name).exists()
    }
