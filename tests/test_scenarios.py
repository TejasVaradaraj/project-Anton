import json

import pandas as pd
import pytest

from project_anton.analysis.scenarios import compare_scenarios


def write_run(run_dir, ttft_p95: float, proxy: dict | None = None) -> None:
    (run_dir / "raw").mkdir(parents=True)
    after = {
        "previous_prompt_cached_mean": 0.9,
        "ttft_p50_ms": 100.0,
        "ttft_p95_ms": ttft_p95,
        "ttft_p99_ms": None,
        "tpot_p50_ms": 12.0,
    }
    summary = {
        "requests_valid": 4,
        "requests_total": 4,
        "after_tool": after,
        "slo_met_fraction": 1.0,
        "goodput_output_tokens_per_s": 50.0,
        "counters": {"external_prefix_cache_hits": 7.0},
    }
    (run_dir / "replay_summary.json").write_text(json.dumps(summary))
    rows = [
        {"session_id": "s0", "status": "ok", "sent_s": 0.0, "e2e_s": 1.0},
        {"session_id": "s0", "status": "ok", "sent_s": 9.0, "e2e_s": 1.0},
        {"session_id": "s1", "status": "ok", "sent_s": 5.0, "e2e_s": 1.0},
        {"session_id": "s1", "status": "http_error", "sent_s": 99.0, "e2e_s": None},
    ]
    (run_dir / "raw" / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if proxy:
        (run_dir / "proxy_stats.json").write_text(json.dumps(proxy))


def test_compare_scenarios(tmp_path):
    write_run(tmp_path / "a", 900.0)
    write_run(tmp_path / "c", 300.0, proxy={"warmups_sent": 5, "late_still_running": 1})
    out = tmp_path / "out" / "s.csv"
    table = compare_scenarios({"baseline": tmp_path / "a", "prefetch": tmp_path / "c"}, out)
    assert table["scenario"].tolist() == ["baseline", "prefetch"]
    # Session times are 10 s and 1 s. The failed request doesn't count.
    assert table["agent_task_s_p50"].iloc[0] == pytest.approx(5.5)
    assert table["warmups_sent"].iloc[1] == 5
    assert pd.isna(table["warmups_sent"].iloc[0])
    assert out.exists()
