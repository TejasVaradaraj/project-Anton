"""Compare storage tiers: measured load time and the bandwidth it implies."""

import json
import os
from pathlib import Path
from typing import Any

# Set before importing matplotlib. See analysis/ttft.py.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from project_anton.analysis.kv_math import ModelShape, cache_bytes  # noqa: E402
from project_anton.harness.runlog import read_jsonl  # noqa: E402

PHASES = ("cold", "warm", "evicted")
MAX_PLAUSIBLE_BYTES_PER_S = 32e9  # PCIe 4.0 x16, the A100's link, tops out near this


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.DataFrame(read_jsonl(run_dir / "raw" / "requests.jsonl"))
    if df.empty:
        raise ValueError(f"no requests recorded in {run_dir}")
    df["valid"] = df["invalid_reasons"].map(lambda reasons: not reasons)
    return df, json.loads((run_dir / "config.json").read_text())


def summarize_tier(df: pd.DataFrame, shape: ModelShape, tier: str) -> pd.DataFrame:
    """One row per prompt length.

    effective_gbytes_per_s is KV size / extra time compared with warm, so it covers every cost of
    getting the cache back, not only the storage read. It is a measured delivery rate.
    """
    valid = df[df["valid"]]
    rows = []
    for length, group in valid.groupby("target_prompt_tokens"):
        medians = group.groupby("phase")["ttft_s"].median()
        if not set(PHASES) <= set(medians.index):
            continue
        cold, warm, evicted = (float(medians[p]) for p in PHASES)
        hits = group[group["phase"] == "evicted"]["external_hits_delta"]
        load_s = evicted - warm
        size = cache_bytes(int(length), shape)
        rows.append(
            {
                "tier": tier,
                "prompt_tokens": int(length),
                "n_evicted": int((group["phase"] == "evicted").sum()),
                "cold_ttft_ms": cold * 1000,
                "warm_ttft_ms": warm * 1000,
                "evicted_ttft_ms": evicted * 1000,
                "evicted_p90_ms": float(
                    np.percentile(group[group["phase"] == "evicted"]["ttft_s"], 90) * 1000
                ),
                "load_ms": load_s * 1000,
                "saved_vs_cold_ms": (cold - evicted) * 1000,
                # vLLM's counter also counts tokens already on the GPU, so check the speed too.
                # A rate above what the GPU's PCIe link can carry means nothing really loaded.
                "served_from_tier": bool(hits.median() > 0)
                and load_s > 0
                and size / load_s <= MAX_PLAUSIBLE_BYTES_PER_S,
                "external_hits_p50": float(hits.median()),
                "effective_gbytes_per_s": size / load_s / 1e9 if load_s > 0 else None,
                "break_even_gbytes_per_s": size / (cold - warm) / 1e9 if cold > warm else None,
            }
        )
    return pd.DataFrame(rows)


def plot_tiers(table: pd.DataFrame, path: Path) -> None:
    """TTFT by tier at each length. Log scale, because the values span about 100x."""
    lengths = sorted(table["prompt_tokens"].unique())
    fig, axes = plt.subplots(1, len(lengths), figsize=(3.2 * len(lengths), 3.8), dpi=150)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#fcfcfb")
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    for ax, length in zip(axes, lengths, strict=True):
        data = table[table["prompt_tokens"] == length]
        labels = ["Warm (GPU)", "Cold"] + list(data["tier"])
        warm = float(data["warm_ttft_ms"].median())
        cold = float(data["cold_ttft_ms"].median())
        values = [warm, cold] + list(data["evicted_ttft_ms"])
        ax.set_facecolor("#fcfcfb")
        bars = ax.bar(range(len(values)), values, color="#c3c2b7", width=0.7)
        for bar, color in zip(bars[2:], colors, strict=False):
            bar.set_color(color)
        ax.set_yscale("log")
        ax.set_xticks(range(len(values)), labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{length:,} tokens", loc="left", fontsize=10, color="#0b0b0b")
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#52514e", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("TTFT (ms, log scale)", color="#52514e")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def compare_tiers(run_dirs: dict[str, Path], out_dir: Path) -> pd.DataFrame:
    tables = []
    for tier, run_dir in run_dirs.items():
        df, config = load_run(run_dir)
        shape = ModelShape(**config["server"]["kv_shape"])
        tables.append(summarize_tier(df, shape, tier))
    table = pd.concat(tables, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "tiers.csv", index=False, float_format="%.4f")
    plot_tiers(table, out_dir / "tiers.png")
    return table
