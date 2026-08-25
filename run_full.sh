#!/usr/bin/env bash
# Full run: the 10 task sets behind the reported speedup figure.
#
# The baseline is capped at 1800 s per task set by default. The paper uses a
# 10,000 s cap; 1800 s keeps the run within a reviewer's patience while still
# letting nine of the ten task sets finish outright. Task sets that hit the cap
# yield a LOWER BOUND on speedup and are excluded from the median.
# Set NPTEST_TIMEOUT=10000 to use the paper's cap.
#
# Usage:  ./run_full.sh
# Env:    GPU_BIN, NPTEST_BIN   override binary locations
#         NPTEST_TIMEOUT                   baseline cap in seconds (default 1800)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

GPU_BIN="${GPU_BIN:-$HERE/bin/saga}"
NPTEST_BIN="${NPTEST_BIN:-$HERE/baseline/build/nptest}"
NPTEST_TIMEOUT="${NPTEST_TIMEOUT:-1800}"

source "$HERE/tools/preflight.sh" || {
    echo; echo "No usable GPU -- cannot run the solver on this machine."; exit 2; }

echo
echo "=== FULL: dataset/full_n20_m5_u30 (10 task sets, n=20, m=5, U'=30%) ==="
python3 tools/compare.py \
    --dataset dataset/full_n20_m5_u30 \
    --cores 5 \
    --variant rtss24 \
    --gpu-bin "$GPU_BIN" \
    --nptest-bin "$NPTEST_BIN" \
    --work "$HERE/work/full" \
    --out "$HERE/results_full.csv" \
    --nptest-timeout "$NPTEST_TIMEOUT" \
    --reference "$HERE/reference_results/a100/results_full.json"
