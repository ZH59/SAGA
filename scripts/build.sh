#!/usr/bin/env bash
# Build the SAGA GPU solver (expand_test_v5).
#
# Emits native SASS for sm_70/75/80/90 (whatever the local nvcc supports) plus
# forward-compat PTX, so the binary runs on any NVIDIA GPU from Volta onward.
#
# Usage:  scripts/build.sh [build-dir]
# Env:    CUDACXX=/path/to/nvcc   override CUDA compiler autodetection
#         JOBS=N                  parallel compile jobs (default: nproc, max 16)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$HERE/build}"
JOBS="${JOBS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}"

# --- locate a CUDA 12.x nvcc -------------------------------------------------
# CUDA 11.5 miscompiles this source (libstdc++-11 std::function SFINAE), so 12.x
# is required rather than merely preferred.
find_nvcc() {
    if [[ -n "${CUDACXX:-}" && -x "${CUDACXX}" ]]; then echo "$CUDACXX"; return; fi
    local c
    for c in $(ls -d /usr/local/cuda-12.* /usr/local/cuda 2>/dev/null | sort -rV); do
        [[ -x "$c/bin/nvcc" ]] && { echo "$c/bin/nvcc"; return; }
    done
    command -v nvcc 2>/dev/null || true
}

NVCC="$(find_nvcc)"
if [[ -z "$NVCC" ]]; then
    echo "ERROR: no nvcc found. Install the CUDA Toolkit (12.x) or set CUDACXX." >&2
    exit 1
fi

VER="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
MAJOR="${VER%%.*}"
echo "Using nvcc $VER at $NVCC"
if [[ "$MAJOR" -lt 12 ]]; then
    echo "ERROR: CUDA $VER is too old; this artifact needs CUDA 12.x." >&2
    echo "       Found only $NVCC. Set CUDACXX to a 12.x nvcc." >&2
    exit 1
fi

# --- configure + build -------------------------------------------------------
export CUDACXX="$NVCC"
cmake -S "$HERE/src/framework_v5" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER="$NVCC"
cmake --build "$BUILD_DIR" --target expand_test_v5 -j "$JOBS"

BIN="$BUILD_DIR/expand_test_v5"
echo
echo "Built: $BIN"
if command -v cuobjdump >/dev/null 2>&1 || [[ -x "$(dirname "$NVCC")/cuobjdump" ]]; then
    CUOBJ="$(command -v cuobjdump || echo "$(dirname "$NVCC")/cuobjdump")"
    echo "Embedded GPU code:"
    "$CUOBJ" --list-elf "$BIN" 2>/dev/null | sed -n 's/.*\.sm_\([0-9]*\)\..*/  native SASS sm_\1/p' | sort -u
    "$CUOBJ" --list-ptx "$BIN" 2>/dev/null | sed -n 's/.*\.compute_\([0-9]*\)\..*/  PTX (JIT) compute_\1/p' | sort -u
fi
