"""Replay, monitoring, sweeps and analysis against the fake server."""

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fake_vllm import FakeVllm

from project_anton.analysis.kv_math import ModelShape
from project_anton.analysis.replay import (
    analyze_replay,
    counter_changes,
    peak_kv_usage,
    summarize_replay,
)
from project_anton.harness.replay import (
    ReplaySweep,
    finished_sessions,
    grid_points,
    load_replay_sweep,
    run_grid,
    run_replay,
)
from project_anton.harness.runlog import read_jsonl
from project_anton.harness.server import ServerConfig
from project_anton.harness.sysmon import SystemMonitor, parse_vllm_metrics
from project_anton.harness.ttft_bench import PreflightError, TooManyFailures
from project_anton.traces.generators import TraceConfig, generate_trace

SHAPE = ModelShape(layers=32, kv_heads=8, head_dim=128)
SMALL = {
    "prefix_tokens": 320,
    "first_turn_tokens": [32, 64],
    "steps": [3, 3],
    "tool_output_tokens": [32, 64],
    "response_tokens": [2, 3],
    "tools": {"read_file": {"weight": 1.0, "median_s": 0.05, "sigma": 0.01}},
}


def make_trace(**overrides) -> TraceConfig:
    values = dict(name="t", kind="agent", sessions=3, arrival_rate_per_s=100.0, params=SMALL)
    return TraceConfig(**(values | overrides))


def cfg_for(server: FakeVllm) -> ServerConfig:
    port = int(server.base_url.rsplit(":", 1)[1])
    return ServerConfig("fake", server.model, port, {}, (0, 128_000), SHAPE)


def replay_on(server: FakeVllm, trace: TraceConfig, run_dir: Path, run_id="r1") -> Path:
    async def go():
        async with server:
            return await run_replay(cfg_for(server), trace, run_id, run_dir, log=lambda _: None)

    return asyncio.run(go())


def test_replays_every_turn_of_every_session(tmp_path):
    records = read_jsonl(replay_on(FakeVllm(), make_trace(), tmp_path))
    assert len(records) == 9
    assert all(not r["invalid_reasons"] for r in records)
    assert finished_sessions(records) == {"s0", "s1", "s2"}


def test_exact_lengths_reach_the_server(tmp_path):
    server = FakeVllm()
    trace = make_trace(sessions=1)
    records = read_jsonl(replay_on(server, trace, tmp_path))
    session = generate_trace(trace)[0]
    assert [r["target_prompt_tokens"] for r in records] == [
        len(q["prompt"]) for q in server.requests
    ]
    assert [q["max_tokens"] for q in server.requests] == [t.response_tokens for t in session.turns]


def test_later_turns_reuse_the_previous_prompt_from_cache(tmp_path):
    records = read_jsonl(replay_on(FakeVllm(), make_trace(sessions=1), tmp_path))
    for r in records[1:]:
        floor = r["previous_prompt_tokens"] // 16 * 16
        assert r["cached_tokens"] == floor


def test_shared_prefix_is_cached_for_later_sessions(tmp_path):
    trace = make_trace(sessions=2, arrival_rate_per_s=1.0)
    records = read_jsonl(replay_on(FakeVllm(), trace, tmp_path))
    firsts = {r["session_id"]: r for r in records if r["turn"] == 0}
    assert firsts["s0"]["cached_tokens"] == 0 or firsts["s1"]["cached_tokens"] == 320


def test_tool_pause_is_waited_for(tmp_path):
    records = read_jsonl(replay_on(FakeVllm(), make_trace(sessions=1), tmp_path))
    gaps = [b["sent_s"] - a["sent_s"] for a, b in zip(records, records[1:], strict=False)]
    assert all(gap >= 0.03 for gap in gaps)
    assert records[1]["tool"] == "read_file"


def test_sessions_overlap_when_server_is_slow(tmp_path):
    server = FakeVllm(token_delay_s=0.05)
    records = read_jsonl(replay_on(server, make_trace(sessions=4), tmp_path))
    assert max(r["concurrent_at_send"] for r in records) >= 2


def test_prompt_too_long_is_caught_before_sending(tmp_path):
    server = FakeVllm(max_model_len=100)
    with pytest.raises(PreflightError):
        replay_on(server, make_trace(), tmp_path)
    assert server.requests == []


def test_failed_turn_ends_session_and_repeated_failures_stop_run(tmp_path):
    server = FakeVllm(fail_mode="http_500")
    trace = make_trace(sessions=8, arrival_rate_per_s=20.0)
    with pytest.raises(TooManyFailures, match="--run-id r1"):
        replay_on(server, trace, tmp_path)
    records = read_jsonl(tmp_path / "raw" / "requests.jsonl")
    assert all(r["turn"] == 0 for r in records)


def test_resume_reruns_only_unfinished_session_with_new_prompts(tmp_path):
    trace = make_trace()
    path = replay_on(FakeVllm(), trace, tmp_path)
    records = read_jsonl(path)
    lost = records[-1]["session_id"]
    kept = records[:-1]
    path.write_text("".join(json.dumps(r) + "\n" for r in kept))

    server = FakeVllm()
    replay_on(server, trace, tmp_path)
    assert len(server.requests) == 3

    after = read_jsonl(path)
    assert finished_sessions(after) == {"s0", "s1", "s2"}
    old = {r["prompt_digest"] for r in after if r["session_id"] == lost and r["attempt"] == 0}
    new = {r["prompt_digest"] for r in after if r["session_id"] == lost and r["attempt"] == 1}
    assert len(new) == 3 and not old & new


def test_resume_with_changed_trace_is_refused(tmp_path):
    replay_on(FakeVllm(), make_trace(), tmp_path)
    with pytest.raises(ValueError, match="differs"):
        replay_on(FakeVllm(), make_trace(seed=9), tmp_path)


def test_resume_when_finished_sends_nothing(tmp_path):
    trace = make_trace()
    replay_on(FakeVllm(), trace, tmp_path)
    server = FakeVllm()
    replay_on(server, trace, tmp_path)
    assert server.requests == []


# --- Analysis ---


def rows(n: int, ttft: float, tpot: float = 0.01, tool: str | None = "grep") -> list[dict]:
    return [
        {
            "run_id": "r",
            "valid": True,
            "invalid_reasons": [],
            "ttft_s": ttft,
            "tpot_s": tpot,
            "e2e_s": 1.0,
            "sent_s": float(i),
            "completion_tokens": 10,
            "tool": tool,
            "concurrent_at_send": 1,
            "previous_prompt_tokens": 100,
            "cached_tokens": 96,
        }
        for i in range(n)
    ]


def test_goodput_counts_only_requests_meeting_targets():
    df = pd.DataFrame(rows(6, ttft=0.5) + rows(4, ttft=3.0))
    summary = summarize_replay(df)
    assert summary["slo_met_fraction"] == pytest.approx(0.6)
    # Sent over 0-5 s and each lasts 1 s, so 6 s in total.
    assert summary["duration_s"] == pytest.approx(6.0)
    assert summary["requests_per_s"] == pytest.approx(10 / 6)
    assert summary["goodput_requests_per_s"] == pytest.approx(1.0)
    assert summary["goodput_output_tokens_per_s"] == pytest.approx(10.0)


def test_slow_tpot_also_fails_the_target():
    summary = summarize_replay(pd.DataFrame(rows(4, ttft=0.1, tpot=0.5)))
    assert summary["slo_met_fraction"] == 0


def test_percentiles_withheld_for_small_samples():
    assert summarize_replay(pd.DataFrame(rows(10, 0.1)))["overall"]["ttft_p95_ms"] is None
    assert summarize_replay(pd.DataFrame(rows(25, 0.1)))["overall"]["ttft_p95_ms"] == pytest.approx(
        100
    )


def test_after_tool_excludes_first_turns():
    df = pd.DataFrame(rows(3, 0.1, tool=None) + rows(2, 0.2, tool="grep"))
    summary = summarize_replay(df)
    assert summary["after_tool"]["n"] == 2
    assert set(summary["by_tool"]) == {"grep"}


def test_analyze_replay_writes_summary(tmp_path):
    replay_on(FakeVllm(), make_trace(), tmp_path)
    summary = analyze_replay(tmp_path)
    assert summary["requests_valid"] == 9
    assert summary["after_tool"]["n"] == 6
    assert summary["after_tool"]["previous_prompt_cached_p50"] > 0.9
    assert json.loads((tmp_path / "replay_summary.json").read_text())["run_id"] == "r1"


def test_peak_kv_usage_reads_real_metric_name(tmp_path):
    path = tmp_path / "m.jsonl"
    samples = [
        {"t": 1, "metrics": {'vllm:kv_cache_usage_perc{engine="0"}': 0.1, "vllm:other": 9.0}},
        {"t": 2, "metrics": {'vllm:kv_cache_usage_perc{engine="0"}': 0.3}},
    ]
    path.write_text("".join(json.dumps(x) + "\n" for x in samples))
    assert peak_kv_usage(path) == 0.3
    assert peak_kv_usage(tmp_path / "missing.jsonl") is None


def test_counter_changes_are_measured_from_the_first_sample(tmp_path):
    path = tmp_path / "m.jsonl"
    name = 'vllm:external_prefix_cache_hits_total{engine="0"}'
    samples = [
        {"t": 1, "metrics": {name: 100.0, "vllm:prefix_cache_hits_total": 5.0}},
        {"t": 2, "metrics": {name: 350.0, "vllm:prefix_cache_hits_total": 9.0, "vllm:other": 1.0}},
    ]
    path.write_text("".join(json.dumps(x) + "\n" for x in samples))
    assert counter_changes(path) == {"external_prefix_cache_hits": 250.0, "prefix_cache_hits": 4.0}
    assert counter_changes(tmp_path / "missing.jsonl") == {}


def test_cached_mean_and_p10_reveal_eviction_the_median_hides():
    data = rows(10, 0.1)
    for row, cached in zip(data, [0, 96, 96, 96, 96, 96, 96, 96, 96, 96], strict=True):
        row["cached_tokens"] = cached
    after = summarize_replay(pd.DataFrame(data))["after_tool"]
    assert after["previous_prompt_cached_p50"] == pytest.approx(0.96)
    assert after["previous_prompt_cached_mean"] == pytest.approx(0.864)
    assert after["previous_prompt_cached_p10"] < 0.96


# --- Monitoring ---


def test_parse_vllm_metrics_keeps_labels_and_drops_others():
    text = (
        "# HELP vllm:x help\n"
        'vllm:gpu_cache_usage_perc{model_name="m"} 0.5\n'
        "vllm:bad notanumber\n"
        "python_gc_total 3.0\n"
    )
    assert parse_vllm_metrics(text) == {'vllm:gpu_cache_usage_perc{model_name="m"}': 0.5}


def test_monitor_samples_metrics_runs_tools_and_skips_missing(tmp_path):
    loop_forever = [sys.executable, "-c", "import time\nprint('tick', flush=True)\ntime.sleep(60)"]
    commands = {"fake_tool": loop_forever, "missing": ["no-such-tool-anywhere"]}

    async def go():
        async with FakeVllm() as server:
            monitor = SystemMonitor(tmp_path, server.base_url, interval_s=0.05, commands=commands)
            async with monitor:
                await asyncio.sleep(0.4)
            return monitor

    monitor = asyncio.run(go())
    assert monitor.procs["fake_tool"].poll() is not None  # It was stopped, not left running
    assert "tick" in (tmp_path / "fake_tool.log").read_text()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status == {"skipped": ["missing"], "started": ["fake_tool"], "exited_early": {}}
    samples = read_jsonl(tmp_path / "vllm_metrics.jsonl")
    assert len(samples) >= 2
    assert "vllm:num_requests_running" in samples[0]["metrics"]


def test_tool_that_quits_early_is_recorded_and_does_not_break_the_run(tmp_path):
    # macOS iostat rejects the Linux flags and exits at once, which crashed a real test run.
    commands = {"quits": [sys.executable, "-c", "raise SystemExit(2)"]}

    async def go():
        async with SystemMonitor(
            tmp_path, "http://127.0.0.1:1", interval_s=0.05, commands=commands
        ):
            await asyncio.sleep(0.3)

    asyncio.run(go())
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["exited_early"] == {"quits": 2}


def test_monitor_survives_server_with_no_metrics(tmp_path):
    async def go():
        async with SystemMonitor(tmp_path, "http://127.0.0.1:1", interval_s=0.05, commands={}):
            await asyncio.sleep(0.1)

    asyncio.run(go())
    assert (tmp_path / "status.json").exists()


# --- Sweeps ---


def test_grid_points_are_stable_and_complete():
    sweep = ReplaySweep("s", "x", {"sessions": [1, 2], "arrival_rate_per_s": [0.5, 1]})
    points = grid_points(sweep)
    assert len(points) == 4
    assert len({run_id for run_id, _ in points}) == 4
    assert points == grid_points(sweep)
    assert "arrival_rate_per_s-0p5" in points[0][0]


def test_grid_skips_finished_points(tmp_path):
    trace_path = tmp_path / "trace.yaml"
    trace_path.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "kind": "agent",
                "sessions": 1,
                "arrival_rate_per_s": 100.0,
                "params": SMALL,
            }
        )
    )
    sweep = ReplaySweep("g", str(trace_path), {"sessions": [1, 2]})
    results = tmp_path / "results"

    async def go(server):
        async with server:
            return await run_grid(cfg_for(server), sweep, results, log=lambda _: None)

    first = FakeVllm()
    assert len(asyncio.run(go(first))) == 2
    assert len(first.requests) == 9  # 1 session, then 2, at 3 turns each

    second = FakeVllm()
    asyncio.run(go(second))
    assert second.requests == []

    # A point that didn't finish runs again, and only that point.
    (results / "g_sessions-2" / "done.json").unlink()
    third = FakeVllm()
    asyncio.run(go(third))
    assert third.requests == []  # Its sessions had all finished, so resume has nothing to send
    assert (results / "g_sessions-2" / "done.json").exists()


def test_committed_replay_sweeps_load():
    paths = list(Path("configs/replay").glob("*.yaml"))
    assert paths
    for path in paths:
        sweep = load_replay_sweep(path)
        assert Path(sweep.trace).exists()
        assert grid_points(sweep)
