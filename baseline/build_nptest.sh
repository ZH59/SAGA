#!/usr/bin/env bash
# Build the CPU baseline: nptest, the SAG-org exact schedulability tester.
#
# The paper compares against the RTSS'24 EDD analyzer of Srinivasan et al., which
# lives in the unified SAG-org repository. We pin the exact commit the paper used.
#
# Usage:  baseline/build_nptest.sh [build-dir]
# Env:    NPTEST_SOURCE=/path/to/schedule_abstraction-main
#             use a local checkout instead of cloning (offline evaluation)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$HERE/baseline/build}"
SRC_DIR="$HERE/baseline/schedule_abstraction-main"

REPO_URL="https://github.com/SAG-org/schedule_abstraction-main.git"
PINNED_COMMIT="47a45b69cbc90d0d0fd36d7c9ac8edaf041a6f05"

# --- obtain the source -------------------------------------------------------
if [[ -n "${NPTEST_SOURCE:-}" ]]; then
    echo "Using local nptest source: $NPTEST_SOURCE"
    SRC_DIR="$NPTEST_SOURCE"
elif [[ -d "$SRC_DIR/.git" ]]; then
    echo "Reusing existing clone at $SRC_DIR"
else
    echo "Cloning $REPO_URL (needs network access)..."
    git clone --recursive "$REPO_URL" "$SRC_DIR"
    git -C "$SRC_DIR" checkout --quiet "$PINNED_COMMIT"
    git -C "$SRC_DIR" submodule update --init --recursive
fi

ACTUAL="$(git -C "$SRC_DIR" rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "nptest source commit: $ACTUAL"
if [[ "$ACTUAL" != "$PINNED_COMMIT" && "$ACTUAL" != "unknown" ]]; then
    echo "WARNING: expected pinned commit $PINNED_COMMIT" >&2
    echo "         results may differ from the reported reference numbers." >&2
fi

# The vendored lib/yaml-cpp submodule is used automatically when yaml-cpp is not
# installed system-wide, so no extra dependency is needed.
if [[ ! -e "$SRC_DIR/lib/yaml-cpp/CMakeLists.txt" ]]; then
    echo "ERROR: lib/yaml-cpp is empty -- clone was not recursive." >&2
    echo "       Run: git -C '$SRC_DIR' submodule update --init --recursive" >&2
    exit 1
fi

# --- build (sequential: the paper's baseline configuration) ------------------
# PARALLEL_RUN stays OFF. The parallel build has known data races (confirmed
# under ThreadSanitizer) and is not what the reported numbers used.
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DPARALLEL_RUN=OFF
cmake --build "$BUILD_DIR" --target nptest -j "$(nproc)"

echo
echo "Built: $BUILD_DIR/nptest"
"$BUILD_DIR/nptest" --help 2>&1 | head -3
