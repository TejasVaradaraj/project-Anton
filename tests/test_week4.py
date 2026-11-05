"""Tier timing, disk baseline parsing and correctness comparison, against the fake server."""

import asyncio
import json
import os
from dataclasses import replace

import pytest
from fake_vllm import FakeVllm

from project_anton.analysis.kv_math import ModelShape
from project_anton.analysis.tiers import compare_tiers, load_run, summarize_tier
from project_anton.correctness.compare import compare_runs
from project_anton.harness.runlog import read_jsonl
from project_anton.harness.server import ServerConfig, load_server_config
from project_anton.harness.storage import JOBS, fio_command, parse_fio
from project_anton.harness.tiers import (
    TierConfig,
    build_plan,
    drop_page_cache,
    load_tier_config,
    parse_external_hits,
    run_tiers,
)

SHAPE = ModelShape(layers=32, kv_heads=8, head_dim=128)
TIER = TierConfig(
    "t", (320, 640), reps=2, max_tokens=3, filler_tokens=320, filler_prompts=3, settle_s=0
)


def cfg_for(server: FakeVllm) -> ServerConfig:
    port = int(server.base_url.rsplit(":", 1)[1])
    return ServerConfig("fake", server.model, port, {}, (0, 128_000), SHAPE)


def tiers_on(server: FakeVllm, run_dir, tier=TIER, **kwargs):
    async def go():
        async with server:
            return await run_tiers(
                cfg_for(server), tier, "r", run_dir, log=lambda _: None, **kwargs
            )

    return read_jsonl(asyncio.run(go()))


def small_cache(**kwargs) -> FakeVllm:
    # 40 blocks is 640 tokens, so 960 filler tokens push out any earlier prompt.
    return FakeVllm(prefill_s_per_token=50e-6, capacity_blocks=40, **kwargs)


def test_plan_is_complete_and_shuffled():
    tier = TierConfig("t", (1, 2, 3, 4), reps=5)
    plan = build_plan(tier)
    assert len(set(plan)) == 20
    assert plan != sorted(plan)


def test_each_item_is_cold_warm_then_evicted_with_same_prompt(tmp_path):
    records = tiers_on(small_cache(), tmp_path)
    assert len(records) == 12
    for start in range(0, 12, 3):
        trio = records[start : start + 3]
        assert [r["phase"] for r in trio] == ["cold", "warm", "evicted"]
        assert len({r["prompt_digest"] for r in trio}) == 1
        assert trio[0]["cached_tokens"] == 0
        assert trio[1]["cached_tokens"] > 0
        assert not any(r["invalid_reasons"] for r in trio)


def test_without_offload_eviction_recomputes(tmp_path):
    records = tiers_on(small_cache(), tmp_path)
    evicted = [r for r in records if r["phase"] == "evicted"]
    assert all(r["cached_tokens"] == 0 for r in evicted)
    assert all(r["external_hits_delta"] == 0 for r in evicted)


def test_offload_is_seen_and_its_load_time_measured(tmp_path):
    # Fake tier loads 10x faster than recomputing.
    server = small_cache(offload_load_s_per_token=5e-6)
    records = tiers_on(server, tmp_path)
    evicted = [r for r in records if r["phase"] == "evicted"]
    assert all(r["external_hits_delta"] > 0 for r in evicted)

    df, config = load_run(tmp_path)
    table = summarize_tier(df, ModelShape(**config["server"]["kv_shape"]), "fake")
    assert table["served_from_tier"].all()
    row = table[table["prompt_tokens"] == 640].iloc[0]
    assert row["warm_ttft_ms"] < row["evicted_ttft_ms"] < row["cold_ttft_ms"]
    # 128 KiB/token at 5 us/token is about 26 GB/s. Timer overhead makes the measured rate lower.
    assert 5 < row["effective_gbytes_per_s"] <= 30


def test_tier_comparison_writes_table_and_plot(tmp_path):
    tiers_on(small_cache(), tmp_path / "a")
    tiers_on(small_cache(offload_load_s_per_token=5e-6), tmp_path / "b")
    table = compare_tiers({"none": tmp_path / "a", "fake": tmp_path / "b"}, tmp_path / "out")
    assert set(table["tier"]) == {"none", "fake"}
    assert not table[table["tier"] == "none"]["served_from_tier"].any()
    assert (tmp_path / "out" / "tiers.png").stat().st_size > 0


def test_resume_skips_finished_items(tmp_path):
    tiers_on(small_cache(), tmp_path)
    again = small_cache()
    tiers_on(again, tmp_path)
    assert again.requests == []


def test_changed_tier_config_is_refused(tmp_path):
    tiers_on(small_cache(), tmp_path)
    with pytest.raises(ValueError, match="differs"):
        tiers_on(small_cache(), tmp_path, tier=replace(TIER, reps=9))


def test_parse_external_hits():
    text = (
        "# HELP vllm:external_prefix_cache_hits_total x\n"
        'vllm:external_prefix_cache_hits_total{engine="0",model_name="m"} 512.0\n'
        "vllm:external_prefix_cache_hits_created 9.0\n"
    )
    assert parse_external_hits(text) == 512.0
    assert parse_external_hits("nothing") == 0


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="macOS has no posix_fadvise")
def test_drop_page_cache_visits_nested_files(tmp_path):
    (tmp_path / "gpu0").mkdir()
    (tmp_path / "gpu0" / "a.pt").write_bytes(b"x" * 100)
    assert drop_page_cache(tmp_path) == 1


def test_committed_tier_configs_load():
    tier = load_tier_config("configs/tiers/week4.yaml")
    assert max(tier.prompt_lengths) < 32768
    # Filler must exceed the small cache, or nothing gets evicted.
    assert tier.filler_tokens * tier.filler_prompts > 65_536


@pytest.mark.parametrize("name", ["a100_small_gpu_only", "a100_small_cpu", "a100_small_disk"])
def test_small_cache_servers_differ_only_in_offload(name):
    cfg = load_server_config(f"configs/servers/{name}.yaml")
    assert cfg.vllm_args["kv-cache-memory"] == 8 * 2**30
    assert cfg.vllm_args["enable-prompt-tokens-details"] is True


# --- Disk baseline ---


def test_fio_always_bypasses_page_cache(tmp_path):
    for job in JOBS.values():
        assert "--direct=1" in fio_command(job, tmp_path / "f", "1G", 5)


def test_parse_fio_reads_the_side_that_ran():
    side = {
        "bw_bytes": 2_000_000_000,
        "iops": 2000.0,
        "lat_ns": {"mean": 450_000},
        "clat_ns": {"percentile": {"50.000000": 400_000, "99.000000": 900_000}},
    }
    empty = {"bw_bytes": 0, "iops": 0.0}
    read = parse_fio(json.dumps({"jobs": [{"read": side, "write": empty}]}))
    write = parse_fio(json.dumps({"jobs": [{"read": empty, "write": side}]}))
    assert read == write
    assert read["gbytes_per_s"] == 2.0
    assert read["latency_mean_us"] == 450
    assert read["completion_p50_us"] == 400
    assert read["completion_p99_us"] == 900


# --- Correctness ---


def result(phase="evicted", tokens=("a", "b"), logprobs=(-0.5, -1.0), length=100, i=0) -> dict:
    return {
        "length": length,
        "i": i,
        "phase": phase,
        "tokens": list(tokens),
        "logprobs": list(logprobs),
    }


def test_identical_runs_match():
    report = compare_runs([result()], [result()])
    assert report == {
        "compared": 1,
        "token_mismatches": 0,
        "max_logprob_diff": 0.0,
        "by_phase": {"evicted": {"n": 1, "token_mismatches": 0, "max_logprob_diff": 0.0}},
    }


def test_wrong_token_is_caught():
    report = compare_runs([result()], [result(tokens=("a", "WRONG"))])
    assert report["token_mismatches"] == 1
    assert report["by_phase"]["evicted"]["token_mismatches"] == 1


def test_logprob_drift_is_measured_per_phase():
    base = [result(), result(phase="fresh")]
    other = [result(logprobs=(-0.5, -1.25)), result(phase="fresh")]
    report = compare_runs(base, other)
    assert report["max_logprob_diff"] == pytest.approx(0.25)
    assert report["by_phase"]["fresh"]["max_logprob_diff"] == 0
    assert report["by_phase"]["evicted"]["max_logprob_diff"] == pytest.approx(0.25)


def test_requests_missing_from_baseline_are_not_compared():
    assert compare_runs([result(i=1)], [result(i=2)])["compared"] == 0
