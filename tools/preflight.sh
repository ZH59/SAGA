#!/usr/bin/env bash
# Detect the GPU and adapt the run to it.  Meant to be SOURCED, not executed:
#
#     source scripts/preflight.sh
#
# Sets (and exports) SAG_V5_HOST_SPILL_GB when the GPU has less memory than the
# A100 the reference numbers were measured on, so a smaller card spills to
# pinned host memory instead of stopping at a layer it cannot hold.
#
# Sets SAGA_GPU_IS_A100=1 when the detected device matches the reference
# hardware, so callers can decide whether wall-time comparison is meaningful.
#
# Returns (not exits) non-zero when no usable CUDA GPU is present, so a caller
# can fall back to --use-reference-data rather than dying.

# --- reusable helpers (also exercised directly by tests/test_arch_compat.sh) --
# A cubin built for sm_Xy runs on a device of compute capability X.z when z >= y
# (same major version). A different major version never matches.
saga_arch_compatible() {            # <space-separated have list> <want>
    local have="$1" want="$2" a ok=1
    for a in $have; do
        if [[ "${a:0:1}" == "${want:0:1}" && "$a" -le "$want" ]]; then ok=0; fi
    done
    return $ok
}

# Echo the sm_ numbers the given binary carries, one per line.
saga_binary_archs() {               # <path to binary>
    local bin="$1"
    if command -v cuobjdump >/dev/null 2>&1; then
        cuobjdump --list-elf "$bin" 2>/dev/null | sed -n 's/.*sm_\([0-9]*\).*/\1/p' | sort -u
    elif [[ -f "$bin.arch" ]]; then
        grep -oE '^[0-9]+$' "$bin.arch" | sort -u
    fi
}

# Tests source this file only to reach the helpers above.
if [[ -n "${SAGA_PREFLIGHT_DEFINE_ONLY:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi

SAGA_GPU_NAME=""
SAGA_GPU_CC=""
SAGA_GPU_VRAM_MIB=""
SAGA_GPU_IS_A100=0

# --- glibc floor -------------------------------------------------------------
# bin/saga imports dlopen@GLIBC_2.34, so it cannot start on an older glibc. Catch
# that here rather than letting the loader fail with a bare version error.
SAGA_GLIBC_MIN="2.34"
_glibc=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
if [[ -n "$_glibc" ]]; then
    if [[ "$(printf '%s\n' "$SAGA_GLIBC_MIN" "$_glibc" | sort -V | head -1)" != "$SAGA_GLIBC_MIN" ]]; then
        echo "PREFLIGHT: ERROR -- this system has glibc $_glibc; bin/saga needs $SAGA_GLIBC_MIN or newer."
        echo "           Distributions that satisfy it include Ubuntu 22.04+, Debian 12+,"
        echo "           RHEL/Rocky/Alma 9+, and Fedora 35+."
        echo "           Contact the authors for a build against an older glibc."
        return 1 2>/dev/null || exit 1
    fi
    echo "PREFLIGHT: glibc $_glibc (>= $SAGA_GLIBC_MIN required)"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "PREFLIGHT: nvidia-smi not found -- no NVIDIA driver on this machine."
    echo "           The GPU solver cannot run here."
    echo "           You can still inspect the authors' measurements in"
    echo "           reference_results/a100/ and re-run the analysis over them."
    return 1 2>/dev/null || exit 1
fi

_q=$(nvidia-smi --query-gpu=name,compute_cap,memory.total \
                --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ -z "$_q" ]]; then
    echo "PREFLIGHT: nvidia-smi found no GPU (driver installed but no device?)."
    return 1 2>/dev/null || exit 1
fi

SAGA_GPU_NAME=$(echo "$_q" | cut -d, -f1 | sed 's/^ *//;s/ *$//')
SAGA_GPU_CC=$(echo "$_q"   | cut -d, -f2 | sed 's/^ *//;s/ *$//')
SAGA_GPU_VRAM_MIB=$(echo "$_q" | cut -d, -f3 | sed 's/^ *//;s/ *$//')

# Validate the parse before acting on it. An unexpected nvidia-smi output would
# otherwise cascade into nonsense ("compute capability ,", "no GPU code for sm_")
# instead of one clear message.
if [[ ! "$SAGA_GPU_CC" =~ ^[0-9]+\.[0-9]+$ || ! "$SAGA_GPU_VRAM_MIB" =~ ^[0-9]+$ ]]; then
    echo "PREFLIGHT: ERROR -- could not parse nvidia-smi output."
    echo "           got: '$_q'"
    echo "           expected: '<name>, <major.minor>, <MiB>'"
    echo "           Run 'nvidia-smi --query-gpu=name,compute_cap,memory.total"
    echo "           --format=csv,noheader,nounits' to see what your driver reports."
    return 1 2>/dev/null || exit 1
fi

echo "PREFLIGHT: $SAGA_GPU_NAME  (compute capability $SAGA_GPU_CC, ${SAGA_GPU_VRAM_MIB} MiB VRAM)"

# --- minimum architecture ----------------------------------------------------
# controller.cu uses cooperative_groups::this_grid().sync(), which is SM_70+.
_cc_major=${SAGA_GPU_CC%%.*}
if [[ -n "$_cc_major" && "$_cc_major" -lt 7 ]]; then
    echo "PREFLIGHT: ERROR -- compute capability $SAGA_GPU_CC is below the 7.0 minimum."
    echo "           The solver uses cooperative grid synchronisation (SM_70+)."
    return 1 2>/dev/null || exit 1
fi

# --- memory sizing -----------------------------------------------------------
# Host spill is OFF in the solver by default. Without it, a layer that exceeds the
# per-buffer budget makes the solver stop and report TRUNCATED -- no answer at
# all. That happens on this dataset even on an 80 GiB A100 (taskset_005), so
# spill is enabled unconditionally here, and more generously on small cards.
if [[ "$SAGA_GPU_VRAM_MIB" -lt 24000 ]]; then
    export SAG_V5_HOST_SPILL_GB="${SAG_V5_HOST_SPILL_GB:-16}"
    echo "PREFLIGHT: VRAM below 24 GiB -- enabling host spill"
else
    export SAG_V5_HOST_SPILL_GB="${SAG_V5_HOST_SPILL_GB:-32}"
    echo "PREFLIGHT: enabling host spill (required for this dataset)"
fi
echo "           SAG_V5_HOST_SPILL_GB=$SAG_V5_HOST_SPILL_GB (needs that much free system RAM)"

# --- does the shipped binary contain code for this GPU? ----------------------
# Turns the opaque runtime failure "no kernel image is available for execution on
# the device" into an actionable message.
#
# The architecture list comes from cuobjdump when the CUDA Toolkit happens to be
# installed, and otherwise from bin/saga.arch, a manifest generated at package
# time. The manifest matters: this artifact needs no CUDA Toolkit, so cuobjdump is
# absent on a typical evaluator's machine and a cuobjdump-only check would
# silently do nothing there.
_saga_bin="${GPU_BIN:-$PWD/bin/saga}"
_have=$(saga_binary_archs "$_saga_bin")
_want="${SAGA_GPU_CC/./}"                       # 8.0 -> 80

if [[ -z "$_have" ]]; then
    echo "PREFLIGHT: NOTE -- cannot determine which GPU architectures the binary"
    echo "           carries (no cuobjdump and no ${_saga_bin##*/}.arch manifest)."
    echo "           Skipping the architecture check."
elif saga_arch_compatible "$_have" "$_want"; then
    echo "PREFLIGHT: binary carries GPU code compatible with sm_${_want}" \
         "(has: $(echo $_have | sed 's/ /, sm_/g; s/^/sm_/'))."
else
    echo "PREFLIGHT: ERROR -- the binary has no GPU code for sm_${_want}."
    echo "           It contains: $(echo $_have | sed 's/ /, sm_/g; s/^/sm_/')"
    echo "           This build cannot run on this GPU."
    echo "           (compute capability 10.x / Blackwell needs a build made with"
    echo "           CUDA 12.8 or later -- contact the authors.)"
    return 1 2>/dev/null || exit 1
fi

case "$SAGA_GPU_NAME" in
    *A100*) SAGA_GPU_IS_A100=1 ;;
esac

if [[ "$SAGA_GPU_IS_A100" -eq 1 ]]; then
    echo "PREFLIGHT: A100 detected -- wall times are comparable to the reference results."
else
    echo "PREFLIGHT: NOTE -- this is not an A100."
    echo "           The run should COMPLETE and the WCRT correctness check should PASS."
    echo "           Wall times and speedup WILL differ from the reported numbers;"
    echo "           those were measured on an NVIDIA A100 80GB PCIe."
fi

export SAGA_GPU_NAME SAGA_GPU_CC SAGA_GPU_VRAM_MIB SAGA_GPU_IS_A100
return 0 2>/dev/null || true
