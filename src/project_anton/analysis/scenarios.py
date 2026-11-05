"""Compare whole-system setups on the same workload: baseline, offload, prefetch, ideal."""

import json
from pathlib import Path

import pandas as pd

from project_anton.harness.runlog import read_jsonl


def scenario_row(name: str, run_dir: Path) -> dict:
    summary = json.loads((run_dir / "replay_summary.json").read_text())
    after = summary["after_tool"]
    records = read_jsonl(run_dir / "raw" / "requests.jsonl")
    sessions: dict[str, list[dict]] = {}
    for r in records:
        if r["status"] == "ok":
            sessions.setdefault(r["session_id"], []).append(r)
    # Agent task time includes tool pauses, so this is what an agent's user would notice.
    task_times = [
        max(r["sent_s"] + r["e2e_s"] for r in rs) - min(r["sent_s"] for r in rs)
        for rs in sessions.values()
    ]
    row = {
        "scenario": name,
        "requests_valid": summary["requests_valid"],
        "requests_total": summary["requests_total"],
        "previous_cached_mean": after["previous_prompt_cached_mean"],
        "after_tool_ttft_p50_ms": after["ttft_p50_ms"],
        "after_tool_ttft_p95_ms": after["ttft_p95_ms"],
        "after_tool_ttft_p99_ms": after["ttft_p99_ms"],
        "tpot_p50_ms": after["tpot_p50_ms"],
        "slo_met_fraction": summary["slo_met_fraction"],
        "goodput_tokens_per_s": summary.get("goodput_output_tokens_per_s"),
        "agent_task_s_p50": float(pd.Series(task_times).median()) if task_times else None,
        "external_cache_hits": (summary.get("counters") or {}).get("external_prefix_cache_hits"),
    }
    stats_path = run_dir / "proxy_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        for key in (
            "warmups_sent",
            "cancelled_tool_ended_first",
            "late_still_running",
            "completed_before_request",
            "early_by_p50_s",
        ):
            row[key] = stats.get(key)
    return row


def compare_scenarios(runs: dict[str, Path], out: Path) -> pd.DataFrame:
    table = pd.DataFrame([scenario_row(name, path) for name, path in runs.items()])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f")
    return table
