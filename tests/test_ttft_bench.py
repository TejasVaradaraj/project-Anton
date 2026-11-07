"""The whole pipeline against the fake server: plan, run, resume, analyze."""

import asyncio
import json

import pandas as pd
import pytest
from fake_vllm import FakeVllm

from project_anton.analysis.kv_math import ModelShape
from project_anton.analysis.ttft import (
    analyze_run,
    fit_prefill,
    load_requests,
    prefill_by_length,
    summarize,
)
from project_anton.harness.runlog import read_jsonl
from project_anton.harness.server import ServerConfig
from project_anton.harness.ttft_bench import (
    PreflightError,
    SweepConfig,
    TooManyFailures,
    build_plan,
    finished_pairs,
    invalid_reasons,
    run_sweep,
)

SHAPE = ModelShape(layers=32, kv_heads=8, head_dim=128)


def make_sweep(**overrides) -> SweepConfig:
    values = dict(name="t", prompt_lengths=(160, 320), reps=2, max_tokens=3, warmup_requests=1)
    return SweepConfig(**(values | overrides))


def cfg_for(server: FakeVllm, model: str = "fake/model") -> ServerConfig:
    port = int(server.base_url.rsplit(":", 1)[1])
    return ServerConfig("fake", model, port, {}, (0, 128_000), SHAPE)


def sweep_on(server: FakeVllm, sweep: SweepConfig, run_dir, run_id="run1", model="fake/model"):
    async def go():
        async with server:
            return await run_sweep(
                cfg_for(server, model), sweep, run_id, run_dir, log=lambda _: None
            )

    return asyncio.run(go())


def base_record(**overrides) -> dict:
    record = dict(
        status="ok",
        phase="cold",
        prompt_tokens=100,
        target_prompt_tokens=100,
        completion_tokens=3,
        target_max_tokens=3,
        ttft_s=0.1,
        cached_tokens=0,
    )
    return record | overrides


# --- Plan ---


def test_plan_covers_everything_once_and_is_seeded():
    sweep = make_sweep(reps=3)
    plan = build_plan(sweep)
    assert len({p.pair_id for p in plan}) == 6
    assert plan == build_plan(sweep)


def test_plan_is_shuffled_so_length_is_not_tied_to_time():
    sweep = make_sweep(prompt_lengths=(100, 200, 400, 800), reps=5)
    lengths = [p.prompt_tokens for p in build_plan(sweep)]
    assert lengths != sorted(lengths)


# --- Validity ---


def test_good_record_is_valid():
    assert invalid_reasons(base_record()) == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"status": "truncated"}, "truncated"),
        ({"prompt_tokens": 99}, "prompt_length_mismatch"),
        ({"prompt_tokens": None}, "prompt_length_mismatch"),
        ({"completion_tokens": 2}, "completion_length_mismatch"),
        ({"ttft_s": None}, "no_tokens"),
        ({"cached_tokens": 96}, "cold_prompt_was_cached"),
        ({"phase": "warm", "cached_tokens": 0}, "warm_prompt_missed_cache"),
    ],
)
def test_bad_records_are_flagged(change, reason):
    assert reason in invalid_reasons(base_record(**change))


def test_unknown_cached_tokens_is_not_held_against_a_record():
    assert invalid_reasons(base_record(cached_tokens=None)) == []


# --- Full run ---


def test_cold_then_warm_with_correct_cache_reporting(tmp_path):
    server = FakeVllm()
    path = sweep_on(server, make_sweep(), tmp_path)
    records = read_jsonl(path)
    assert [r["phase"] for r in records].count("warmup") == 1
    pairs = [r for r in records if r["phase"] != "warmup"]
    assert len(pairs) == 8
    assert all(not r["invalid_reasons"] for r in pairs)
    for cold, warm in zip(pairs[::2], pairs[1::2], strict=True):
        assert (cold["phase"], warm["phase"]) == ("cold", "warm")
        assert cold["prompt_digest"] == warm["prompt_digest"]
        assert cold["cached_tokens"] == 0
        assert warm["cached_tokens"] == warm["target_prompt_tokens"]


def test_cold_prompts_are_all_different(tmp_path):
    records = read_jsonl(sweep_on(FakeVllm(), make_sweep(), tmp_path))
    digests = [r["prompt_digest"] for r in records if r["phase"] == "cold"]
    assert len(set(digests)) == len(digests)


def test_second_run_on_same_server_is_still_cold(tmp_path):
    """Same seed, same live server, new run_id: the prompts must not be cached yet."""
    server = FakeVllm()

    async def go():
        async with server:
            cfg = cfg_for(server)
            quiet = lambda _: None  # noqa: E731
            await run_sweep(cfg, make_sweep(), "run1", tmp_path / "a", log=quiet)
            return await run_sweep(cfg, make_sweep(), "run2", tmp_path / "b", log=quiet)

    records = read_jsonl(asyncio.run(go()))
    assert all(r["cached_tokens"] == 0 for r in records if r["phase"] == "cold")


def test_analysis_recovers_the_injected_prefill_rate(tmp_path):
    """Fake server at 20 us/token (50,000 tokens/s). The pipeline should find that rate."""
    server = FakeVllm(prefill_s_per_token=20e-6)
    sweep = make_sweep(prompt_lengths=(2000, 4000, 8000), reps=3)
    sweep_on(server, sweep, tmp_path)

    analysis = analyze_run(tmp_path)
    assert analysis["prefill_fit"]["prefill_tokens_per_s"] == pytest.approx(50_000, rel=0.15)
    # 128 KiB/token x 50,000 tokens/s is about 6.55 GB/s at every length.
    for row in analysis["by_prompt_length"]:
        assert row["break_even_storage_gbytes_per_s"] == pytest.approx(6.55, rel=0.15)
    assert analysis["requests_valid"] == analysis["requests_total"]

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "ttft_vs_prompt.png").stat().st_size > 0
    summary = summarize(load_requests(tmp_path / "raw" / "requests.jsonl"))
    cold = summary[summary.phase == "cold"].set_index("prompt_tokens").ttft_p50_ms
    warm = summary[summary.phase == "warm"].set_index("prompt_tokens").ttft_p50_ms
    assert cold[8000] > cold[2000]
    assert warm[8000] < cold[8000] / 2


def test_small_samples_report_no_p95(tmp_path):
    sweep_on(FakeVllm(), make_sweep(), tmp_path)
    summary = summarize(load_requests(tmp_path / "raw" / "requests.jsonl"))
    assert summary["ttft_p95_ms"].isna().all()


def fake_records(cold_by_length: dict[int, float], warm_s: float = 0.01) -> pd.DataFrame:
    rows = []
    for length, cold_s in cold_by_length.items():
        rows.append({"phase": "cold", "target_prompt_tokens": length, "ttft_s": cold_s})
        rows.append({"phase": "warm", "target_prompt_tokens": length, "ttft_s": warm_s})
    return pd.DataFrame(rows).assign(valid=True)


def test_superlinear_ttft_gets_a_warning_despite_high_r_squared():
    # The shape of the real A100 run: cost per token rises with length.
    df = fake_records(
        {1000: 0.078, 2000: 0.141, 4000: 0.286, 8000: 0.607, 16000: 1.375, 32000: 3.475}
    )
    fit = fit_prefill(df)
    assert fit["r_squared"] > 0.95
    assert "warning" in fit


def test_linear_ttft_gets_no_warning():
    assert "warning" not in fit_prefill(fake_records({1000: 0.11, 2000: 0.21, 4000: 0.41}))


def test_break_even_uses_time_saved_at_each_length():
    rows = prefill_by_length(fake_records({1000: 0.11, 2000: 0.21}), SHAPE)
    # 1000 tokens x 128 KiB = 0.131072 GB, and 0.10 s saved.
    assert rows[0]["break_even_storage_gbytes_per_s"] == pytest.approx(1.31072)
    assert rows[0]["cold_to_warm_ratio"] == pytest.approx(11)
    assert rows[1]["break_even_storage_gbytes_per_s"] == pytest.approx(1.31072)


def test_no_break_even_when_cache_saves_no_time():
    assert prefill_by_length(fake_records({1000: 0.005}), SHAPE) == []


def test_no_prefill_rate_from_one_length(tmp_path):
    sweep_on(FakeVllm(), make_sweep(prompt_lengths=(160,)), tmp_path)
    assert fit_prefill(load_requests(tmp_path / "raw" / "requests.jsonl")) is None


# --- Preflight ---


def test_preflight_rejects_prompts_longer_than_server_allows(tmp_path):
    server = FakeVllm(max_model_len=200)
    with pytest.raises(PreflightError, match="max_model_len"):
        sweep_on(server, make_sweep(), tmp_path)
    assert server.requests == []


def test_preflight_rejects_wrong_model(tmp_path):
    with pytest.raises(PreflightError, match="config wants"):
        sweep_on(FakeVllm(), make_sweep(), tmp_path, model="other/model")


# --- Failure and resume ---


def test_stops_after_repeated_failures(tmp_path):
    server = FakeVllm(fail_mode="http_500")
    sweep = make_sweep(reps=10, warmup_requests=0, max_consecutive_failures=3)
    with pytest.raises(TooManyFailures, match="--run-id run1"):
        sweep_on(server, sweep, tmp_path)
    assert len(server.requests) == 3


def test_resume_skips_finished_and_retries_failed_with_new_prompt(tmp_path):
    sweep = make_sweep(warmup_requests=0, reps=3, max_consecutive_failures=99)
    path = sweep_on(FakeVllm(), sweep, tmp_path)
    records = read_jsonl(path)

    # Pretend the last pair's warm request never arrived, as in a dropped session.
    records = records[:-1]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    lost_pair = records[-1]["pair_id"]
    assert lost_pair not in finished_pairs(records)

    server = FakeVllm()
    sweep_on(server, sweep, tmp_path)
    assert len(server.requests) == 2  # only the unfinished pair, cold and warm

    after = read_jsonl(path)
    retried = [r for r in after if r["pair_id"] == lost_pair and r["attempt"] == 1]
    assert [r["phase"] for r in retried] == ["cold", "warm"]
    first_cold = next(r for r in after if r["pair_id"] == lost_pair and r["attempt"] == 0)
    assert retried[0]["prompt_digest"] != first_cold["prompt_digest"]
    assert len(finished_pairs(after)) == 6


def test_resume_with_changed_config_is_refused(tmp_path):
    sweep_on(FakeVllm(), make_sweep(), tmp_path)
    with pytest.raises(ValueError, match="differs"):
        sweep_on(FakeVllm(), make_sweep(max_tokens=9), tmp_path)


def test_finished_pairs_needs_both_phases_in_one_attempt():
    records = [
        {"pair_id": "a", "attempt": 0, "phase": "cold", "status": "ok"},
        {"pair_id": "a", "attempt": 1, "phase": "warm", "status": "ok"},
        {"pair_id": "b", "attempt": 0, "phase": "cold", "status": "ok"},
        {"pair_id": "b", "attempt": 0, "phase": "warm", "status": "http_error"},
        {"pair_id": "c", "attempt": 2, "phase": "cold", "status": "ok"},
        {"pair_id": "c", "attempt": 2, "phase": "warm", "status": "ok"},
        {"pair_id": None, "attempt": 0, "phase": "warmup", "status": "ok"},
    ]
    assert finished_pairs(records) == {"c"}
