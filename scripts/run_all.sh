#!/usr/bin/env bash
# Run the full pipeline end-to-end. Execute from project root:
#   bash scripts/run_all.sh

set -euo pipefail

echo "============================================================"
echo "  Instacart Spark — full pipeline"
echo "============================================================"

# 1. ETL: CSV -> cleaned + timestamped Parquet
echo "[1/7] ETL"
python src/etl/00_preprocessing.py

# 2. DataFrame queries (writes results/explain_df_*.txt)
echo "[2/7] DataFrame queries"
python src/dataframe/01_queries_df.py

# 3. SQL queries (writes results/explain_sql_*.txt)
echo "[3/7] SQL queries"
python src/sql/02_queries_sql.py

# 4. RDD queries (writes results/rdd_timings.csv)
echo "[4/7] RDD queries"
python src/rdd/03_queries_rdd.py

# 5. Cross-API benchmark table
echo "[5/7] Benchmark comparison"
python src/benchmarks/run_benchmarks.py

# 6. CSV vs Parquet + scalability
echo "[6/7] File-format + scalability tests"
python src/benchmarks/csv_vs_parquet.py
python src/benchmarks/scalability_test.py

# 7. Market basket analysis (FPGrowth) — pattern mining from the problem
#    statement. Slow; can be skipped with SKIP_ML=1.
if [ "${SKIP_ML:-0}" = "0" ]; then
  echo "[7/7] Market basket (FPGrowth)"
  python src/ml/04_market_basket.py
else
  echo "[7/7] Skipping FPGrowth (SKIP_ML=1)"
fi

echo
echo "Done. All artefacts in results/:"
ls -1 results/
