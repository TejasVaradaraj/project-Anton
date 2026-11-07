"""Summarize a trace replay: latency, throughput, and goodput under a latency target."""

from pathlib import Path
from typing import Any

import pandas as pd

from project_anton.harness.runlog import read_jsonl, write_json

# Goodput counts only requests that meet these targets. These are from the project plan.
DEFAULT_SLO = {"ttft_s": 2.0, "tpot_s": 0.1}
# Below this many samples a percentile is really just one of the largest values.
MIN_SAMPLES = {0.95: 20, 0.99: 100}
HIT_THRESHOLD = 0.9  # Share of the previous prompt found in cache for a turn to count as a hit
COUNTERS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
)


def load_replay(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(path))
    if df.empty:
        return df
    df["valid"] = df["invalid_reasons"].map(lambda reasons: not reasons)
    return df


def _percentile_ms(values: pd.Series, q: float) -> float | None:
    if len(values) < MIN_SAMPLES.get(q, 1):
        return None
    return float(values.quantile(q) * 1000)


def _latency(valid: pd.DataFrame) -> dict[str, Any]:
    ttft = valid["ttft_s"].dropna()
    tpot = valid["tpot_s"].dropna()
    cached = valid[valid["previous_prompt_tokens"].notna()]
    reused = cached["cached_tokens"].dropna() / cached["previous_prompt_tokens"]
    return {
        "n": len(valid),
        "ttft_p50_ms": _percentile_ms(ttft, 0.5),
        "ttft_p95_ms": _percentile_ms(ttft, 0.95),
        "ttft_p99_ms": _percentile_ms(ttft, 0.99),
        "tpot_p50_ms": _percentile_ms(tpot, 0.5),
        # Near 1 means each turn reused the whole previous prompt from the cache.
        "previous_prompt_cached_p50": float(reused.median()) if not reused.empty else None,
        # Eviction hits some turns first, so the median is the last thing to move.
        "previous_prompt_cached_mean": float(reused.mean()) if not reused.empty else None,
        "previous_prompt_cached_p10": float(reused.quantile(0.1)) if not reused.empty else None,
    }


def summarize_replay(df: pd.DataFrame, slo: dict[str, float] | None = None) -> dict[str, Any]:
    slo = slo or DEFAULT_SLO
    valid = df[df["valid"]]
    meets_slo = (valid["ttft_s"] <= slo["ttft_s"]) & (
        valid["tpot_s"].isna() | (valid["tpot_s"] <= slo["tpot_s"])
    )
    summary: dict[str, Any] = {
        "requests_total": len(df),
        "requests_valid": len(valid),
        "invalid_reasons": df["invalid_reasons"].explode().dropna().value_counts().to_dict(),
        "slo": slo,
        "slo_met_fraction": float(meets_slo.mean()) if len(valid) else None,
        "max_concurrent_at_send": int(df["concurrent_at_send"].max()),
        "overall": _latency(valid),
    }
    if len(valid):
        ends = valid["sent_s"] + valid["e2e_s"]
        duration = float(ends.max() - valid["sent_s"].min())
        tokens = valid["completion_tokens"]
        summary |= {
            "duration_s": duration,
            "output_tokens_per_s": float(tokens.sum() / duration) if duration > 0 else None,
            "requests_per_s": len(valid) / duration if duration > 0 else None,
            "goodput_requests_per_s": float(meets_slo.sum() / duration) if duration > 0 else None,
            "goodput_output_tokens_per_s": (
                float(tokens[meets_slo].sum() / duration) if duration > 0 else None
            ),
        }
    # The turn right after a tool pause is what this project is trying to speed up.
    after_tool = valid[valid["tool"].notna()]
    summary["after_tool"] = _latency(after_tool)
    summary["by_tool"] = {tool: _latency(group) for tool, group in after_tool.groupby("tool")}
    # Slow hits point to queueing, and slow misses to eviction. Offload can only fix the second.
    if not after_tool.empty:
        found = (after_tool["cached_tokens"].fillna(0)) / after_tool["previous_prompt_tokens"]
        summary["cache_hit"] = _latency(after_tool[found >= HIT_THRESHOLD])
        summary["cache_miss"] = _latency(after_tool[found < HIT_THRESHOLD])
    return summary


def peak_kv_usage(metrics_path: Path) -> float | None:
    """Highest vLLM KV usage seen. It looks like this counts running requests only, so it can
    stay low while finished sessions still sit in the prefix cache. It isn't "how full"."""
    values = [
        value
        for sample in read_jsonl(metrics_path)
        for name, value in sample["metrics"].items()
        if name.startswith("vllm:kv_cache_usage_perc")
    ]
    return max(values) if values else None


def counter_changes(metrics_path: Path) -> dict[str, float]:
    """How much vLLM's cache counters grew during this run. They count from server start."""
    samples = read_jsonl(metrics_path)
    if not samples:
        return {}
    first, last = samples[0]["metrics"], samples[-1]["metrics"]
    changes = {}
    for prefix in COUNTERS:
        for name, value in last.items():
            if name.startswith(prefix + "{") or name == prefix:
                short = prefix.removeprefix("vllm:").removesuffix("_total")
                changes[short] = value - first.get(name, 0.0)
    return changes


def analyze_replay(run_dir: Path, slo: dict[str, float] | None = None) -> dict[str, Any]:
    df = load_replay(run_dir / "raw" / "requests.jsonl")
    if df.empty:
        raise ValueError(f"no requests recorded in {run_dir}")
    summary = {"run_id": str(df["run_id"].iloc[0]), **summarize_replay(df, slo)}
    metrics = run_dir / "raw" / "sysmon" / "vllm_metrics.jsonl"
    summary["peak_kv_usage_running"] = peak_kv_usage(metrics)
    summary["counters"] = counter_changes(metrics)
    write_json(run_dir / "replay_summary.json", summary)
    return summary
