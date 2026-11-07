"""Turn measurements into storage requirements: bandwidth, IOPS and latency per workload.

The inputs are measured, but these requirements are calculated from them. They assume a restore is
sequential whole-chunk reads, and they ignore storage queueing under load.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project_anton.analysis.kv_math import ModelShape, kv_bytes_per_token

CHUNK_TOKENS = 256  # LMCache's chunk size in these runs
# How much of the time a restore saves may be spent on the restore itself.
LOAD_BUDGETS = (1.0, 0.5, 0.1)
LLAMA_8B = ModelShape(layers=32, kv_heads=8, head_dim=128)
LLAMA_70B = ModelShape(layers=80, kv_heads=8, head_dim=128)


def saved_seconds(context_tokens: int, by_length: list[dict[str, Any]]) -> float:
    """Measured time a cache hit saves at this length. Interpolated between measured lengths."""
    lengths = [row["prompt_tokens"] for row in by_length]
    saved = [row["cold_ttft_s"] - row["warm_ttft_s"] for row in by_length]
    if not lengths[0] <= context_tokens <= lengths[-1]:
        raise ValueError(f"{context_tokens} is outside the measured {lengths[0]}-{lengths[-1]}")
    return float(np.interp(context_tokens, lengths, saved))


def requirement_row(
    label: str, context_tokens: int, saved_s: float, shape: ModelShape, budget: float
) -> dict[str, Any]:
    token_bytes = kv_bytes_per_token(shape)
    kv_bytes = context_tokens * token_bytes
    chunk_bytes = CHUNK_TOKENS * token_bytes
    chunks = int(np.ceil(context_tokens / CHUNK_TOKENS))
    load_s = saved_s * budget
    bandwidth = kv_bytes / load_s
    return {
        "workload": label,
        "context_tokens": context_tokens,
        "kv_gbytes": kv_bytes / 1e9,
        "time_saved_ms": saved_s * 1000,
        "load_budget": budget,
        "load_time_ms": load_s * 1000,
        "bandwidth_gbytes_per_s": bandwidth / 1e9,
        "chunk_mib": chunk_bytes / 2**20,
        "chunk_iops": bandwidth / chunk_bytes,
        # If chunks were read one at a time, each would need to take no longer than this.
        "latency_per_chunk_ms": load_s / chunks * 1000,
    }


def measured_requirements(
    by_length: list[dict[str, Any]], workloads: dict[str, int], shape: ModelShape = LLAMA_8B
) -> pd.DataFrame:
    rows = []
    for label, context in workloads.items():
        saved = saved_seconds(context, by_length)
        rows += [requirement_row(label, context, saved, shape, budget) for budget in LOAD_BUDGETS]
    return pd.DataFrame(rows)


def projected_70b(
    workloads: dict[str, int], assumed_prefill_tokens_per_s: tuple[float, ...]
) -> pd.DataFrame:
    """A projection, not a measurement: nothing here ran on 70B or H200.

    Prefill speed on 4x H200 is unknown, so it's an assumed range. Each row is what storage would
    need if that assumption held, with the load taking all of the time saved.
    """
    rows = []
    for label, context in workloads.items():
        for rate in assumed_prefill_tokens_per_s:
            row = requirement_row(label, context, context / rate, LLAMA_70B, 1.0)
            rows.append({"assumed_prefill_tokens_per_s": rate, **row})
    return pd.DataFrame(rows)


def agent_workloads(calibration: dict[str, Any]) -> dict[str, int]:
    final = calibration["final_context_tokens"]
    return {
        "Agent, median session": round(final["p50"]),
        "Agent, p75 session": round(final["p75"]),
        "Agent, p90 session": round(final["p90"]),
    }


def build_requirements(week1_analysis: Path, calibration: Path, out_dir: Path) -> pd.DataFrame:
    by_length = json.loads(week1_analysis.read_text())["by_prompt_length"]
    workloads = agent_workloads(json.loads(calibration.read_text()))
    measured = measured_requirements(by_length, workloads)
    out_dir.mkdir(parents=True, exist_ok=True)
    measured.to_csv(out_dir / "requirements_8b.csv", index=False, float_format="%.3f")
    projected_70b(workloads, (2000.0, 5000.0, 10000.0)).to_csv(
        out_dir / "projection_70b.csv", index=False, float_format="%.3f"
    )
    return measured


def measured_restores(tiers: pd.DataFrame, tier: str, shape: ModelShape = LLAMA_8B) -> pd.DataFrame:
    """What a restore needed, against what a tier delivered, from one tier comparison table."""
    token_bytes = kv_bytes_per_token(shape)
    rows = []
    for row in tiers[tiers["tier"] == tier].itertuples():
        restored_gbytes = row.external_hits_p50 * token_bytes / 1e9
        saved_ms = row.cold_ttft_ms - row.warm_ttft_ms
        rows.append(
            {
                "tier": tier,
                "prompt_tokens": row.prompt_tokens,
                "restored_gbytes": restored_gbytes,
                "recompute_ms": saved_ms,
                "restore_ms": row.load_ms,
                "delivered_gbytes_per_s": restored_gbytes / row.load_ms * 1000,
                "break_even_gbytes_per_s": restored_gbytes / saved_ms * 1000,
            }
        )
    return pd.DataFrame(rows)


def build_final_requirements(tiers_csv: Path, calibration: Path, out_dir: Path) -> pd.DataFrame:
    """The requirements table from the final runs, with restores that passed correctness."""
    tiers = pd.read_csv(tiers_csv)
    gpu_only = tiers[tiers["tier"] == "GPU only"]
    by_length = [
        {
            "prompt_tokens": row.prompt_tokens,
            "cold_ttft_s": row.cold_ttft_ms / 1000,
            "warm_ttft_s": row.warm_ttft_ms / 1000,
        }
        for row in gpu_only.itertuples()
    ]
    workloads = agent_workloads(json.loads(calibration.read_text()))
    out_dir.mkdir(parents=True, exist_ok=True)
    table = measured_requirements(by_length, workloads)
    table.to_csv(out_dir / "requirements_final_8b.csv", index=False, float_format="%.3f")
    restores = measured_restores(tiers, "Multi-process")
    restores.to_csv(out_dir / "multi_process_restores.csv", index=False, float_format="%.3f")
    return table
