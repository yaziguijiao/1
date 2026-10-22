#!/usr/bin/env bash
# 一键复现：依次执行全部分析脚本，输出即 README 中引用的全部数字。
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

for f in analysis/00_profile.py analysis/01_analysis.py analysis/02_threshold.py \
         analysis/03_verify.py analysis/04_huadong.py analysis/05_ttl_path.py \
         analysis/06_final.py; do
  echo
  echo "########## $f ##########"
  "$PY" "$f"
done
