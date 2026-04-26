#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <python_file> [extra spark-submit args...]"
  exit 1
fi

SCRIPT="$1"
shift  

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


MASTER="${SPARK_MASTER:-local[*]}"
DEPLOY_MODE="${SPARK_DEPLOY_MODE:-client}"

DRIVER_MEMORY="${DRIVER_MEMORY:-4g}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-4g}"
EXECUTOR_CORES="${EXECUTOR_CORES:-2}"
NUM_EXECUTORS="${NUM_EXECUTORS:-4}"

SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-64}"

echo "==> Submitting $SCRIPT"
echo "    master     = $MASTER"
echo "    deploy     = $DEPLOY_MODE"
echo "    drv mem    = $DRIVER_MEMORY"
echo "    exec mem   = $EXECUTOR_MEMORY"
echo "    exec cores = $EXECUTOR_CORES"
echo "    # execs    = $NUM_EXECUTORS"
echo

cd "$PROJECT_ROOT"

spark-submit \
  --master "$MASTER" \
  --deploy-mode "$DEPLOY_MODE" \
  --driver-memory "$DRIVER_MEMORY" \
  --executor-memory "$EXECUTOR_MEMORY" \
  --executor-cores "$EXECUTOR_CORES" \
  --num-executors "$NUM_EXECUTORS" \
  --conf "spark.sql.shuffle.partitions=$SHUFFLE_PARTITIONS" \
  --conf "spark.sql.adaptive.enabled=false" \
  --conf "spark.sql.autoBroadcastJoinThreshold=10485760" \
  --conf "spark.eventLog.enabled=true" \
  --conf "spark.eventLog.dir=file://$PROJECT_ROOT/results/spark-events" \
  --py-files "$PROJECT_ROOT/src/common/spark_session.py" \
  "$SCRIPT" \
  "$@"
