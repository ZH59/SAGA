#!/usr/bin/env bash
# Kick-the-tires run: one task set, a few minutes at most.
#
# Confirms the GPU solver and the nptest baseline both build, run, agree on the
# schedulability verdict, and produce matching per-job response times.
#
# Usage:  ./run_smoke.sh
# Env:    GPU_BIN, NPTEST_BIN   override binary locations
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

GPU_BIN="${GPU_BIN:-$HERE/bin/saga}"
NPTEST_BIN="${NPTEST_BIN:-$HERE/baseline/build/nptest}"

source "$HERE/tools/preflight.sh" || {
    echo; echo "No usable GPU -- cannot run the solver on this machine."; exit 2; }

echo
echo "=== SMOKE: 1 task set from dataset/full_n20_m5_u30 (n=20, m=5, U'=30%) ==="
python3 tools/compare.py \
    --dataset dataset/full_n20_m5_u30 \
    --limit 1 \
    --cores 5 \
    --variant rtss24 \
    --gpu-bin "$GPU_BIN" \
    --nptest-bin "$NPTEST_BIN" \
    --work "$HERE/work/smoke" \
    --out "$HERE/results_smoke.csv" \
    --nptest-timeout 600
