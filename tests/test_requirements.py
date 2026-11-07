import pandas as pd
import pytest

from project_anton.analysis.requirements import (
    LLAMA_8B,
    agent_workloads,
    measured_requirements,
    measured_restores,
    projected_70b,
    requirement_row,
    saved_seconds,
)

BY_LENGTH = [
    {"prompt_tokens": 1000, "cold_ttft_s": 0.2, "warm_ttft_s": 0.1},
    {"prompt_tokens": 3000, "cold_ttft_s": 0.6, "warm_ttft_s": 0.1},
]


def test_saved_time_interpolates_between_measurements():
    assert saved_seconds(1000, BY_LENGTH) == pytest.approx(0.1)
    assert saved_seconds(2000, BY_LENGTH) == pytest.approx(0.3)


@pytest.mark.parametrize("context", [999, 3001])
def test_refuses_to_extrapolate_beyond_measurements(context):
    with pytest.raises(ValueError, match="outside"):
        saved_seconds(context, BY_LENGTH)


def test_requirement_arithmetic():
    row = requirement_row("w", 1000, 0.1, LLAMA_8B, 0.5)
    # 1000 tokens x 128 KiB = 131,072,000 bytes, loaded in 50 ms.
    assert row["kv_gbytes"] == pytest.approx(0.131072)
    assert row["bandwidth_gbytes_per_s"] == pytest.approx(2.62144)
    assert row["chunk_mib"] == 32
    # 2.62144 GB/s divided by 32 MiB chunks.
    assert row["chunk_iops"] == pytest.approx(78.125)
    # 1000 tokens is 4 chunks, sharing 50 ms.
    assert row["latency_per_chunk_ms"] == pytest.approx(12.5)


def test_tighter_budget_needs_faster_storage():
    table = measured_requirements(BY_LENGTH, {"w": 2000})
    assert table["load_budget"].tolist() == [1.0, 0.5, 0.1]
    assert table["bandwidth_gbytes_per_s"].is_monotonic_increasing


def test_70b_needs_2_5x_the_bytes_per_token():
    table = projected_70b({"w": 10000}, (5000.0,))
    row = table.iloc[0]
    assert row["kv_gbytes"] == pytest.approx(10000 * 320 * 1024 / 1e9)
    assert row["time_saved_ms"] == pytest.approx(2000)


def test_agent_workloads_use_real_percentiles():
    calibration = {"final_context_tokens": {"p50": 11788.2, "p75": 28335.0, "p90": 30337.0}}
    assert agent_workloads(calibration)["Agent, median session"] == 11788


def test_measured_restores_compare_delivered_with_break_even():
    tiers = pd.DataFrame(
        [
            {
                "tier": "Multi-process",
                "prompt_tokens": 1000,
                "external_hits_p50": 1000,
                "cold_ttft_ms": 1100.0,
                "warm_ttft_ms": 100.0,
                "load_ms": 100.0,
            },
            {
                "tier": "GPU only",
                "prompt_tokens": 1000,
                "external_hits_p50": 0,
                "cold_ttft_ms": 1100.0,
                "warm_ttft_ms": 100.0,
                "load_ms": 1000.0,
            },
        ]
    )
    table = measured_restores(tiers, "Multi-process")
    assert len(table) == 1
    row = table.iloc[0]
    # 1000 tokens x 128 KiB = 0.131 GB.
    assert row["delivered_gbytes_per_s"] == pytest.approx(1.31072)
    assert row["break_even_gbytes_per_s"] == pytest.approx(0.131072)
