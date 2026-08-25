#!/usr/bin/env bash
# Run every test in this artifact. No GPU required.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rc=0

echo "=== unit tests: compare.py parsing and comparison logic ==="
PYTHONPATH="$HERE/tools" python3 -m unittest discover -s "$HERE/tests" -p 'test_*.py' -v || rc=1

echo
echo "=== unit test: GPU architecture compatibility rule ==="
bash "$HERE/tests/test_arch_compat.sh" || rc=1

echo
echo "=== system test: dataset integrity ==="
python3 "$HERE/tests/test_dataset_integrity.py" || rc=1

echo
[[ $rc -eq 0 ]] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $rc
