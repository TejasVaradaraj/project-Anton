#!/usr/bin/env bash
# Run the correctness check on several server setups, one at a time, on a GPU machine.
#
#   scripts/correctness_matrix.sh OUT_DIR PROMPTS_FILE_OR_random SETUP [SETUP ...]
#
# Setups: gpu_only, cpu, disk, mp (multi-process), mp60 (multi-process, 60 s for saves to finish).
# A setup whose results file already exists is skipped, so a stopped run can continue.
set -uo pipefail

OUT=$1
PROMPTS=$2
shift 2
A='uv run --frozen --no-dev anton'
DISK_DIR=${DISK_DIR:-/content/lmcache_disk}
mkdir -p "$OUT"

for setup in "$@"; do
    if [ -s "$OUT/$setup.jsonl" ]; then
        echo "=== $setup already recorded"
        continue
    fi
    echo "=== $setup"
    settle=5
    lmcache_pid=
    case "$setup" in
    mp | mp60)
        config=configs/servers/a100_small_mp.yaml
        [ "$setup" = mp60 ] && settle=60
        lmcache server --host localhost --port 5555 --l1-size-gb 100 --eviction-policy LRU \
            --http-port 8091 >"$OUT/$setup.lmcache.log" 2>&1 &
        lmcache_pid=$!
        sleep 40
        ;;
    *) config=configs/servers/a100_small_$setup.yaml ;;
    esac
    # Left-over saved data would make "fresh" requests restores instead.
    rm -rf "$DISK_DIR"
    mkdir -p "$DISK_DIR"

    prompts_arg=()
    [ "$PROMPTS" = random ] || prompts_arg=(--prompts "$PROMPTS")
    if $A server start --config "$config"; then
        # Record to a temporary label's file, so a crash can't look like a finished setup.
        $A correctness record --server "$config" --label "$setup.partial" --out "$OUT" \
            --settle "$settle" "${prompts_arg[@]}" ${RECORD_ARGS:-} &&
            mv "$OUT/$setup.partial.jsonl" "$OUT/$setup.jsonl" ||
            echo "=== $setup FAILED during recording"
    else
        echo "=== $setup FAILED to start"
    fi
    $A server stop --config "$config"
    cp "logs/$(basename "$config" .yaml).log" "$OUT/$setup.vllm.log" 2>/dev/null
    [ -n "$lmcache_pid" ] && kill "$lmcache_pid"
    pkill -9 -f 'vllm serve' 2>/dev/null
    pkill -9 -f EngineCore 2>/dev/null
    pkill -9 -f 'lmcache server' 2>/dev/null
    sleep 5
done

args=()
for setup in "$@"; do
    [ -s "$OUT/$setup.jsonl" ] && args+=("$setup=$OUT/$setup.jsonl")
done
[ ${#args[@]} -gt 0 ] && uv run --frozen --no-dev python scripts/correctness_summary.py "$OUT/summary.csv" "${args[@]}"
echo "=== FINISHED"
