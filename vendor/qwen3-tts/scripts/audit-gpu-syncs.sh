#!/usr/bin/env bash
# Audit GPU→CPU synchronization points in the hot path.
#
# Every `to_vec1()` call forces the GPU to flush its queue and copy data back
# to the CPU. Minimizing these is key to keeping the GPU saturated.
#
# Usage: bash scripts/audit-gpu-syncs.sh
#        make audit-gpu-syncs

set -euo pipefail

echo "=== GPU→CPU sync points (to_vec1 calls in src/) ==="
echo

grep -rn 'to_vec1' src/ --include='*.rs' || echo "(none found)"

echo
echo "Total: $(grep -rc 'to_vec1' src/ --include='*.rs' | awk -F: '{s+=$2} END {print s}') calls"
