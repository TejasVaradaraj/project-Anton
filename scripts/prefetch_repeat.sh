#!/usr/bin/env bash
# Repeat offload and prefetch in one session, and try earlier warm-ups: does more lead time help?
#
#   PATH=/tmp/old/bin:$PATH scripts/prefetch_repeat.sh results/final/scenarios
set -uo pipefail

OUT=${1:?results folder}
S='uv run --frozen --no-dev python scripts/week6_scenarios.py --results-dir'
$S "$OUT" --only B --suffix _repeat
$S "$OUT" --only C --suffix _repeat
$S "$OUT" --only C --suffix _lead2 --lead-s 2
$S "$OUT" --only C --suffix _lead5 --lead-s 5
echo "=== REPEAT_FINISHED"
