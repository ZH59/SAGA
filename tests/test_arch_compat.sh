#!/usr/bin/env bash
# Tests the GPU-architecture compatibility rule used by tools/preflight.sh.
#
# It sources preflight.sh with SAGA_PREFLIGHT_DEFINE_ONLY=1 and calls
# saga_arch_compatible directly, so this exercises the SHIPPED implementation
# rather than a copy of it that could drift out of step.
#
# NVIDIA rule: a cubin built for sm_Xy runs on a device of compute capability X.z
# when z >= y (same major version). A different major version never matches.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SAGA_PREFLIGHT_DEFINE_ONLY=1
# shellcheck source=../tools/preflight.sh
source "$HERE/tools/preflight.sh"

if ! declare -F saga_arch_compatible >/dev/null; then
    echo "FAIL: tools/preflight.sh did not define saga_arch_compatible"
    exit 1
fi
if ! declare -F saga_binary_archs >/dev/null; then
    echo "FAIL: tools/preflight.sh did not define saga_binary_archs"
    exit 1
fi

pass=0; fail=0
check() {   # check <desc> <have> <want> <expect: yes|no>
    local got="no"
    saga_arch_compatible "$2" "$3" && got="yes"
    if [[ "$got" == "$4" ]]; then
        pass=$((pass+1)); echo "  ok   $1"
    else
        fail=$((fail+1)); echo "  FAIL $1 (have='$2' want=$3 expected=$4 got=$got)"
    fi
}

echo "cubin compatibility rule (preflight's own implementation):"
check "A100 (80) runs on the sm_80 cubin"           "70 75 80 90" 80  yes
check "RTX 3090 (86) runs on the sm_80 cubin"       "70 75 80 90" 86  yes
check "L40S (89) runs on the sm_80 cubin"           "70 75 80 90" 89  yes
check "H100 (90) runs on the sm_90 cubin"           "70 75 80 90" 90  yes
check "V100 (70) runs on the sm_70 cubin"           "70 75 80 90" 70  yes
check "T4 (75) runs on the sm_75 cubin"             "70 75 80 90" 75  yes
check "Blackwell (100) NOT covered by sm_70..90"    "70 75 80 90" 100 no
check "RTX 50xx (120) NOT covered by sm_70..90"     "70 75 80 90" 120 no
check "V100 (70) NOT covered by an sm_80+ build"    "80 90"       70  no
check "T4 (75) NOT covered by an sm_80+ build"      "80 90"       75  no
check "sm_90 cubin does not run on 8.x"             "90"          86  no
check "empty architecture list matches nothing"     ""            80  no

# The manifest must let the check work without the CUDA Toolkit, which is the
# normal situation for an evaluator: this artifact ships no toolkit dependency.
echo
echo "architecture manifest:"
if [[ -f "$HERE/bin/saga.arch" ]]; then
    echo "  ok   bin/saga.arch is present"
    pass=$((pass+1))
    archs=$(grep -oE '^[0-9]+$' "$HERE/bin/saga.arch" | sort -u | tr '\n' ' ')
    if [[ -n "$archs" ]]; then
        echo "  ok   manifest lists architectures: $archs"; pass=$((pass+1))
    else
        echo "  FAIL manifest contains no architecture numbers"; fail=$((fail+1))
    fi
else
    echo "  FAIL bin/saga.arch missing -- the guard cannot work without cuobjdump"
    fail=$((fail+1))
fi

echo "  ---"
echo "  passed=$pass failed=$fail"
[[ "$fail" -eq 0 ]]
