import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.spark_session import get_spark, load_tables_parquet, RESULTS_DIR


def time_action(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    print(f"    [{label}] result={result}  elapsed={elapsed:.2f}s")
    return elapsed


def main():
    spark = get_spark("RDD_Queries")
    sc = spark.sparkContext
    t = load_tables_parquet(spark)

    orders   = t["orders"].rdd.cache()
    prior    = t["order_products_prior"].rdd.cache()
    products = t["products"].rdd.cache()
    aisles   = t["aisles"].rdd.cache()
    depts    = t["departments"].rdd.cache()

    for r in (orders, prior, products, aisles, depts):
        r.count()

    timings = {}

    # Q1 — complex filter
    print("\n=== q1 (RDD) ===")
    def q1():
        basket_sizes = (
            prior.map(lambda r: (r.order_id, 1))
                 .reduceByKey(lambda a, b: a + b)
                 .filter(lambda kv: kv[1] > 10)
        )
        weekend_lunch = (
            orders
            .filter(lambda r: r.order_dow in (0, 1)
                              and r.order_hour_of_day is not None
                              and 11 <= r.order_hour_of_day <= 14)
            .map(lambda r: (r.order_id, r))
        )
        return weekend_lunch.join(basket_sizes).count()
    timings["q1"] = time_action("q1", q1)

    # Q2 — aggregations (single-pass via aggregate)
    print("\n=== q2 (RDD) ===")
    def q2():
        zero = (0, 0.0, 0, 24, -1, 0)

        def seq(acc, r):
            cnt, ssum, ncnt, mn, mx, wknd = acc
            d = r.days_since_prior_order
            h = r.order_hour_of_day
            return (
                cnt + 1,
                ssum + (d if d is not None else 0.0),
                ncnt + (1 if d is not None else 0),
                min(mn, h) if h is not None else mn,
                max(mx, h) if h is not None else mx,
                wknd + (1 if r.order_dow in (0, 1) else 0),
            )

        def comb(a, b):
            return (a[0]+b[0], a[1]+b[1], a[2]+b[2],
                    min(a[3], b[3]), max(a[4], b[4]), a[5]+b[5])

        cnt, ssum, ncnt, mn, mx, wknd = orders.aggregate(zero, seq, comb)
        return {"orders": cnt, "avg_days": ssum/ncnt if ncnt else None,
                "min_hr": mn, "max_hr": mx, "weekend": wknd}
    timings["q2"] = time_action("q2", q2)

    # Q3 — multi-attribute groupby (department, dow, hour)
    print("\n=== q3 (RDD) ===")
    def q3():
            prod_dept_map = sc.broadcast(dict(
                products.map(lambda r: (r.product_id, r.department_id)).collect()
            ))
            dept_name_map = sc.broadcast(dict(
                depts.map(lambda r: (r.department_id, r.department)).collect()
            ))
            order_meta_map = sc.broadcast(dict(
                orders.map(lambda r: (r.order_id,
                                    (r.order_dow, r.order_hour_of_day))).collect()
            ))

            def emit(row):
                dept_id = prod_dept_map.value.get(row.product_id)
                order_meta = order_meta_map.value.get(row.order_id)
                if dept_id is None or order_meta is None:
                    return None 
                dept = dept_name_map.value.get(dept_id)
                if dept is None:
                    return None
                dow, hr = order_meta
                return ((dept, dow, hr), 1)

            return (
                prior.map(emit)
                    .filter(lambda x: x is not None)
                    .reduceByKey(lambda a, b: a + b)
                    .count()
            )
    timings["q3"] = time_action("q3", q3)

    # Q4 — top-5 products per department by reorders
    print("\n=== q4 (RDD) ===")
    def q4():
            prod_lookup = sc.broadcast(dict(
                products.map(lambda r: (r.product_id, (r.product_name, r.department_id))).collect()
            ))
            dept_lookup = sc.broadcast(dict(
                depts.map(lambda r: (r.department_id, r.department)).collect()
            ))

            def project(kv):
                pid, total = kv
                prod = prod_lookup.value.get(pid)
                if prod is None:
                    return None
                name, did = prod
                dept = dept_lookup.value.get(did)
                if dept is None:
                    return None
                return ((dept, name), total)

            per_prod = (
                prior.map(lambda r: (r.product_id, r.reordered))
                    .reduceByKey(lambda a, b: a + b)
                    .map(project)
                    .filter(lambda x: x is not None)
            )
            top5 = (
                per_prod
                .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
                .groupByKey()
                .mapValues(lambda vs: sorted(vs, key=lambda x: -x[1])[:5])
            )
            return top5.count()
    timings["q4"] = time_action("q4", q4)
    # Q5 — running average basket size per user 
    print("\n=== q5 (RDD) ===")
    def q5():
        basket = (
            prior.map(lambda r: (r.order_id, 1))
                 .reduceByKey(lambda a, b: a + b)        
        )
        keyed = (
            orders.map(lambda r: (r.order_id, (r.user_id, r.order_number)))
                  .join(basket)                       
                  .map(lambda x: ((x[1][0][0], x[1][0][1]), x[1][1]))
        )

        def running_avg(values):
            xs = sorted(values, key=lambda v: v[0])    
            running, total, out = 0.0, 0, []
            for n, sz in xs:
                total += 1
                running += sz
                out.append((n, sz, running / total))
            return out

        return (
            keyed.map(lambda x: (x[0][0], (x[0][1], x[1])))   
                 .groupByKey()
                 .mapValues(running_avg)
                 .count()
        )
    timings["q5"] = time_action("q5", q5)

    # Q6 — users above the global average basket size
    print("\n=== q6 (RDD) ===")
    def q6():
        basket = (
            prior.map(lambda r: (r.order_id, 1))
                 .reduceByKey(lambda a, b: a + b)
        )
        user_basket = (
            orders.map(lambda r: (r.order_id, r.user_id))
                  .join(basket)                                
                  .map(lambda x: (x[1][0], (x[1][1], 1)))        
                  .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
                  .mapValues(lambda v: v[0]/v[1])              
        )
        total, n = user_basket.map(lambda x: (x[1], 1)) \
                              .reduce(lambda a, b: (a[0]+b[0], a[1]+b[1]))
        global_avg = total / n
        return user_basket.filter(lambda x: x[1] > global_avg).count()
    timings["q6"] = time_action("q6", q6)

    # Q7 — broadcast (map-side) join with aisles
    print("\n=== q7 (RDD) ===")
    def q7():
            aisle_b = sc.broadcast(dict(aisles.map(lambda r: (r.aisle_id, r.aisle)).collect()))
            prod_aisle = sc.broadcast(dict(products.map(lambda r: (r.product_id, r.aisle_id)).collect()))

            def emit(r):
                aid = prod_aisle.value.get(r.product_id)
                if aid is None:
                    return None
                aisle_name = aisle_b.value.get(aid)
                if aisle_name is None:
                    return None
                return (aisle_name, 1)

            return (
                prior.map(emit)
                    .filter(lambda x: x is not None)
                    .reduceByKey(lambda a, b: a + b)
                    .count()
            )
    timings["q7"] = time_action("q7", q7)

    # Q8 — shuffle join prior ⋈ orders (the slow one)
    print("\n=== q8 (RDD) ===")
    def q8():
        return (
            prior.map(lambda r: (r.order_id, r.reordered))
                 .join(orders.map(lambda r: (r.order_id, r.order_dow)))   # shuffle
                 .map(lambda x: (x[1][1], (x[1][0], 1)))   # (dow, (reord, 1))
                 .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
                 .count()
        )
    timings["q8"] = time_action("q8", q8)

    # Q9 — caching demo (RDD)
    print("\n=== q9_caching (RDD) ===")
    base = prior.map(lambda r: (r.order_id, 1)).reduceByKey(lambda a, b: a + b)
    # uncached
    uncached = []
    for _ in range(3):
        t0 = time.perf_counter()
        base.count()
        uncached.append(time.perf_counter() - t0)
    base.cache(); base.count()
    cached = []
    for _ in range(3):
        t0 = time.perf_counter()
        base.count()
        cached.append(time.perf_counter() - t0)
    base.unpersist()
    print(f"    uncached: {[f'{x:.2f}s' for x in uncached]}")
    print(f"    cached  : {[f'{x:.2f}s' for x in cached]}")
    timings["q9_uncached_avg"] = sum(uncached) / len(uncached)
    timings["q9_cached_avg"]   = sum(cached)   / len(cached)

    # Q10 — partition emulation (partitionBy on a pair RDD)
    print("\n=== q10_partition (RDD) ===")
    def q10():
        partitioned = (
            products.map(lambda r: (r.department_id, r))
                    .partitionBy(8)
                    .cache()
        )
        partitioned.count()
        return partitioned.filter(lambda kv: kv[0] == 4).count()
    timings["q10"] = time_action("q10", q10)

    # Save timings
    out = RESULTS_DIR / "rdd_timings.csv"
    with open(out, "w") as f:
        f.write("query,elapsed_sec\n")
        for k, v in timings.items():
            f.write(f"{k},{v:.4f}\n")
    print(f"\n==> RDD timings written to {out}")

    spark.stop()


if __name__ == "__main__":
    main()
