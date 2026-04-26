import sys
import time
import io
import contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.spark_session import (
    get_spark, load_tables_parquet, RESULTS_DIR, DATA_DIR,
)


def save_explain(spark, query: str, name: str, mode: str = "extended"):
    out = RESULTS_DIR / f"explain_sql_{name}.txt"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spark.sql(query).explain(mode)
    out.write_text(buf.getvalue())
    print(f"    [explain saved → results/{out.name}]")


# ---------------------------------------------------------------------------
# SQL strings
# ---------------------------------------------------------------------------
Q1 = """
WITH basket AS (
  SELECT order_id, COUNT(*) AS basket_size
  FROM   prior
  GROUP  BY order_id
)
SELECT o.order_id, o.user_id, o.order_dow, o.order_hour_of_day,
       b.basket_size, o.order_ts
FROM   orders o
JOIN   basket b ON o.order_id = b.order_id
WHERE  o.order_dow IN (0, 1)
  AND  o.order_hour_of_day BETWEEN 11 AND 14
  AND  b.basket_size > 10
"""

Q2 = """
SELECT  COUNT(*)                                              AS total_orders,
        COUNT(DISTINCT user_id)                               AS unique_users,
        AVG(days_since_prior_order)                           AS avg_days_between,
        MIN(order_hour_of_day)                                AS earliest_hour,
        MAX(order_hour_of_day)                                AS latest_hour,
        SUM(CASE WHEN order_dow IN (0,1) THEN 1 ELSE 0 END)   AS weekend_orders
FROM    orders
"""

Q3 = """
SELECT  d.department, o.order_dow, o.order_hour_of_day,
        COUNT(*)         AS line_items,
        AVG(p.reordered) AS reorder_rate
FROM    prior p
JOIN    orders      o  ON p.order_id   = o.order_id
JOIN    products    pr ON p.product_id = pr.product_id
JOIN    departments d  ON pr.department_id = d.department_id
GROUP   BY d.department, o.order_dow, o.order_hour_of_day
"""

Q4 = """
WITH reorder_counts AS (
  SELECT d.department, pr.product_id, pr.product_name,
         SUM(p.reordered) AS total_reorders
  FROM   prior p
  JOIN   products    pr ON p.product_id = pr.product_id
  JOIN   departments d  ON pr.department_id = d.department_id
  GROUP  BY d.department, pr.product_id, pr.product_name
),
ranked AS (
  SELECT *,
         RANK() OVER (PARTITION BY department ORDER BY total_reorders DESC) AS rk
  FROM   reorder_counts
)
SELECT * FROM ranked WHERE rk <= 5 ORDER BY department, rk
"""

Q5 = """
WITH basket AS (
  SELECT order_id, COUNT(*) AS basket_size
  FROM   prior GROUP BY order_id
)
SELECT  o.user_id, o.order_number, o.order_ts, b.basket_size,
        AVG(b.basket_size) OVER (
            PARTITION BY o.user_id ORDER BY o.order_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                  AS running_avg_basket,
        SUM(b.basket_size) OVER (
            PARTITION BY o.user_id ORDER BY o.order_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                  AS cumulative_items
FROM    orders o JOIN basket b ON o.order_id = b.order_id
"""

Q6 = """
WITH basket AS (
  SELECT order_id, COUNT(*) AS basket_size
  FROM   prior GROUP BY order_id
),
user_avg AS (
  SELECT o.user_id, AVG(b.basket_size) AS user_avg_basket
  FROM   orders o JOIN basket b ON o.order_id = b.order_id
  GROUP  BY o.user_id
)
SELECT  user_id, user_avg_basket
FROM    user_avg
WHERE   user_avg_basket > (SELECT AVG(user_avg_basket) FROM user_avg)
"""

Q7 = """
SELECT  /*+ BROADCAST(a) */
        a.aisle, COUNT(*) AS line_items
FROM    prior p
JOIN    products pr ON p.product_id = pr.product_id
JOIN    aisles   a  ON pr.aisle_id   = a.aisle_id
GROUP   BY a.aisle
ORDER   BY line_items DESC
"""

Q8 = """
SELECT  o.order_dow,
        COUNT(*)         AS line_items,
        AVG(p.reordered) AS reorder_rate
FROM    prior p
JOIN    orders o ON p.order_id = o.order_id
GROUP   BY o.order_dow
ORDER   BY o.order_dow
"""

Q10_TEMPLATE = "SELECT * FROM products_partitioned WHERE department_id = 4"


# Q9 — caching demo (timing-based)
def q9_caching_demo(spark):
    spark.sql("DROP VIEW IF EXISTS cp")
    spark.sql("CREATE OR REPLACE TEMP VIEW cp AS "
              "SELECT p.order_id, o.user_id, p.reordered "
              "FROM prior p JOIN orders o ON p.order_id = o.order_id")

    def _time_3():
        out = []
        for _ in range(3):
            t0 = time.perf_counter()
            spark.sql("SELECT user_id, COUNT(*) FROM cp GROUP BY user_id").count()
            out.append(time.perf_counter() - t0)
        return out

    uncached = _time_3()

    spark.sql("CACHE TABLE cp")     # eager cache
    cached = _time_3()
    spark.sql("UNCACHE TABLE cp")
    return {"uncached": uncached, "cached": cached}


# Driver
def main():
    spark = get_spark("SQL_Queries")
    tables = load_tables_parquet(spark)

    view_names = {
        "order_products_prior": "prior",
        "orders":               "orders",
        "products":             "products",
        "departments":          "departments",
        "aisles":               "aisles",
    }
    for src_name, view in view_names.items():
        if src_name in tables:
            tables[src_name].createOrReplaceTempView(view)

    # ---------- run Q1–Q7 normally ----------
    for name, q in [("q1", Q1), ("q2", Q2), ("q3", Q3), ("q4", Q4),
                    ("q5", Q5), ("q6", Q6), ("q7", Q7)]:
        print(f"\n=== {name} (SQL) ===")
        save_explain(spark, q, name)
        t0 = time.perf_counter()
        n = spark.sql(q).count()
        print(f"    rows={n:,}  elapsed={time.perf_counter()-t0:.2f}s")

    # ---------- Q8 with broadcast disabled ----------
    print(f"\n=== q8 (SQL, sort-merge forced) ===")
    orig = spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    save_explain(spark, Q8, "q8")
    t0 = time.perf_counter()
    n = spark.sql(Q8).count()
    print(f"    rows={n:,}  elapsed={time.perf_counter()-t0:.2f}s")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", orig)

    # ---------- Q9 caching ----------
    print(f"\n=== q9_caching_demo (SQL) ===")
    r = q9_caching_demo(spark)
    print(f"    uncached: {[f'{x:.2f}s' for x in r['uncached']]}")
    print(f"    cached  : {[f'{x:.2f}s' for x in r['cached']]}")

    # ---------- Q10 partition pruning ----------
    print(f"\n=== q10_partition_pruning (SQL) ===")
    part_path = DATA_DIR / "output" / "products_partitioned.parquet"
    if not part_path.exists():
        (
            tables["products"]
            .write.mode("overwrite")
            .partitionBy("department_id")
            .parquet(str(part_path))
        )
    spark.sql(
        f"CREATE OR REPLACE TEMP VIEW products_partitioned "
        f"USING parquet OPTIONS (path '{part_path}')"
    )
    save_explain(spark, Q10_TEMPLATE, "q10")
    t0 = time.perf_counter()
    n = spark.sql(Q10_TEMPLATE).count()
    print(f"    rows={n:,}  elapsed={time.perf_counter()-t0:.2f}s")

    spark.stop()


if __name__ == "__main__":
    main()
    import time
    print("\n>>> Spark UI alive at http://localhost:4040 — Ctrl+C when done <<<")
    time.sleep(1800)   