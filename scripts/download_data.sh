#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(realpath "$SCRIPT_DIR/../data")"

cd "$DATA_DIR"

echo "==> Downloading instacart-market-basket-analysis from Kaggle"
kaggle competitions download -c instacart-market-basket-analysis

echo "==> Unzipping outer archive"
unzip -o instacart-market-basket-analysis.zip
rm -f instacart-market-basket-analysis.zip

for f in *.csv.gz; do
  [ -e "$f" ] || continue
  echo "    decompressing $f"
  gunzip -f "$f"
done

if compgen -G "*.7z" > /dev/null; then
  for f in *.7z; do
    echo "    extracting $f"
    7z x -y "$f"
  done
fi

echo
echo "==> CSV files now present:"
ls -lh *.csv
