"""Summarize a trace replay: latency, throughput, and goodput under a latency target."""

from pathlib import Path
from typing import Any

import pandas as pd

from project_anton.harness.runlog import read_jsonl, write_json

# Goodput counts only requests that meet these targets. These are from the project plan.
DEFAULT_SLO = {"ttft_s": 2.0, "tpot_s": 0.1}
# Below this many samples a percentile is really just one of the largest values.
MIN_SAMPLES = {0.95: 20, 0.99: 100}


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


def analyze_replay(run_dir: Path, slo: dict[str, float] | None = None) -> dict[str, Any]:
    df = load_replay(run_dir / "raw" / "requests.jsonl")
    if df.empty:
        raise ValueError(f"no requests recorded in {run_dir}")
    summary = {"run_id": str(df["run_id"].iloc[0]), **summarize_replay(df, slo)}
    summary["peak_kv_usage_running"] = peak_kv_usage(
        run_dir / "raw" / "sysmon" / "vllm_metrics.jsonl"
    )
    write_json(run_dir / "replay_summary.json", summary)
    return summary
