#!/usr/bin/env bash
# All GPU runs on the verified versions: restore timing per tier, the prefetch comparison, then the
# final sweep. Each step carries on if another fails. Run from the repo root on a GPU machine, with
# vLLM 0.25.1 and LMCache 0.5.1 on PATH:
#
#   scripts/final_runs.sh results/final
set -uo pipefail

OUT=${1:?results folder}
A='uv run --frozen --no-dev anton'
mkdir -p "$OUT"

tier() { # name config [extra args]
    local name=$1 config=configs/servers/$2.yaml
    shift 2
    echo "=== tier $name"
    rm -rf /content/lmcache_disk
    mkdir -p /content/lmcache_disk
    $A server start --config "$config" &&
        $A tiers run --server "$config" --tier configs/tiers/week4.yaml --run-id "tiers_$name" \
            --results-dir "$OUT" "$@" || echo "=== tier $name FAILED"
    $A server stop --config "$config"
    pkill -9 -f 'vllm serve'
    pkill -9 -f 'lmcache server'
    sleep 5
}

tier gpu_only a100_small_gpu_only
tier mp a100_small_mp
tier cpu a100_small_cpu
tier disk a100_small_disk --drop-cache-dir /content/lmcache_disk
$A tiers compare --run "GPU only=$OUT/tiers_gpu_only" --run "Multi-process=$OUT/tiers_mp" \
    --run "In-process CPU=$OUT/tiers_cpu" --run "In-process disk=$OUT/tiers_disk" --out "$OUT/tiers"

echo "=== scenarios"
uv run --frozen --no-dev python scripts/week6_scenarios.py --results-dir "$OUT/scenarios" ||
    echo "=== scenarios FAILED"
pkill -9 -f 'vllm serve'
pkill -9 -f 'lmcache server'

echo "=== final sweep"
uv run --frozen --no-dev python scripts/week7_final.py --results-dir "$OUT/sweep" ||
    echo "=== final sweep FAILED"
pkill -9 -f 'vllm serve'
pkill -9 -f 'lmcache server'
echo "=== ALL_FINISHED"
