#!/usr/bin/env bash
# The GPU runs left after the restore timings: correctness at exact chunk lengths, the prefetch
# comparison, then the final sweep. Finished work is skipped, so this can simply be run again.
#
#   PATH=/tmp/old/bin:$PATH scripts/finish_runs.sh results/final
set -uo pipefail

OUT=${1:?results folder}
cleanup() {
    pkill -9 -f '[v]llm serve'
    pkill -9 -f 'VLLM::'
    pkill -9 -f '[l]mcache server'
    sleep 5
}

echo "=== exact chunk correctness"
P=$OUT/correctness_exact/prompts.jsonl
mkdir -p "$OUT/correctness_exact"
[ -s "$P" ] || uv run --frozen --no-dev --with tokenizers python scripts/make_text_prompts.py \
    --parquet /tmp/swe/default/train/0000.parquet --tokenizer /tmp/tok/tokenizer.json --out "$P" \
    --lengths 15872 31744 32000
RECORD_ARGS='--split-chunk 256' bash scripts/correctness_matrix.sh "$OUT/correctness_exact" "$P" \
    gpu_only mp cpu
cleanup

echo "=== scenarios"
uv run --frozen --no-dev python scripts/week6_scenarios.py --results-dir "$OUT/scenarios" ||
    echo "=== scenarios FAILED"
cleanup

echo "=== final sweep"
uv run --frozen --no-dev python scripts/week7_final.py --results-dir "$OUT/sweep" ||
    echo "=== final sweep FAILED"
cleanup
echo "=== ALL_FINISHED"
