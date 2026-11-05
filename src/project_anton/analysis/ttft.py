"""Turn a run's raw requests into a summary table, a prefill estimate and a plot."""

import json
import os
from pathlib import Path
from typing import Any

# Set before importing matplotlib, which reads this on import. Colab sets it to a
# notebook-only backend that this virtualenv doesn't have, and the import then crashes.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from project_anton.analysis.kv_math import ModelShape, break_even_bandwidth  # noqa: E402
from project_anton.harness.runlog import read_jsonl, write_json  # noqa: E402

# A p95 from fewer samples is really just the largest value, so it isn't reported.
MIN_SAMPLES_FOR_P95 = 20
PHASE_STYLE = {"cold": "#2a78d6", "warm": "#eb6834"}
PHASE_LABEL = {"cold": "Cold (not cached)", "warm": "Warm (prefix cached)"}


def load_requests(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(path))
    if df.empty:
        return df
    df["valid"] = df["invalid_reasons"].map(lambda reasons: not reasons)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per phase and prompt length, using valid cold/warm requests.

    Counts of invalid requests stay in the table, so problems stay visible.
    """
    rows = []
    measured = df[df["phase"].isin(["cold", "warm"])]
    for (phase, length), group in measured.groupby(["phase", "target_prompt_tokens"]):
        valid = group[group["valid"]]
        row: dict[str, Any] = {
            "phase": phase,
            "prompt_tokens": int(length),
            "n_valid": len(valid),
            "n_invalid": len(group) - len(valid),
        }
        if valid.empty:
            rows.append(row)
            continue
        ttft_ms = valid["ttft_s"] * 1000
        tpot_ms = valid["tpot_s"].dropna() * 1000
        row |= {
            "ttft_p50_ms": ttft_ms.median(),
            "ttft_min_ms": ttft_ms.min(),
            "ttft_max_ms": ttft_ms.max(),
            "ttft_p95_ms": (
                ttft_ms.quantile(0.95) if len(valid) >= MIN_SAMPLES_FOR_P95 else float("nan")
            ),
            "tpot_p50_ms": tpot_ms.median() if not tpot_ms.empty else float("nan"),
            "e2e_p50_ms": valid["e2e_s"].median() * 1000,
        }
        cached = valid["cached_tokens"].dropna()
        row["cached_fraction_p50"] = (
            (cached / length).median() if not cached.empty else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["phase", "prompt_tokens"]).reset_index(drop=True)


def fit_prefill(df: pd.DataFrame) -> dict[str, Any] | None:
    """Fit cold TTFT = overhead + tokens / prefill_rate.

    Returns None without at least two distinct lengths, or if longer prompts weren't
    slower. In that case the data doesn't support a rate, so none is reported.
    """
    cold = df[(df["phase"] == "cold") & df["valid"]]
    if cold["target_prompt_tokens"].nunique() < 2:
        return None
    x = cold["target_prompt_tokens"].to_numpy(dtype=float)
    y = cold["ttft_s"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 0:
        return None
    total = ((y - y.mean()) ** 2).sum()
    residual = ((y - (slope * x + intercept)) ** 2).sum()
    return {
        "prefill_tokens_per_s": float(1 / slope),
        "overhead_s": float(intercept),
        "r_squared": float(1 - residual / total) if total > 0 else None,
        "n_points": len(cold),
    }


def plot_ttft(df: pd.DataFrame, path: Path, title: str) -> None:
    """Median TTFT lines, with every individual request shown as a faint dot."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for phase, color in PHASE_STYLE.items():
        data = df[(df["phase"] == phase) & df["valid"]]
        if data.empty:
            continue
        ax.scatter(
            data["target_prompt_tokens"], data["ttft_s"], s=14, color=color, alpha=0.3, linewidths=0
        )
        medians = data.groupby("target_prompt_tokens")["ttft_s"].median()
        ax.plot(
            medians.index,
            medians.values,
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=6,
            markeredgecolor="#fcfcfb",
            markeredgewidth=1,
            label=PHASE_LABEL[phase],
        )
        ax.annotate(
            phase.capitalize(),
            (medians.index[-1], medians.values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#52514e",
        )
    ax.set_xlabel("Prompt length (tokens)", color="#52514e")
    ax.set_ylabel("Time to first token (s)", color="#52514e")
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.margins(x=0.08)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor="#52514e")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def analyze_run(run_dir: Path) -> dict[str, Any]:
    """Write summary.csv, analysis.json and the plot into the run folder."""
    df = load_requests(run_dir / "raw" / "requests.jsonl")
    if df.empty:
        raise ValueError(f"no requests recorded in {run_dir}")
    config = json.loads((run_dir / "config.json").read_text())

    summary = summarize(df)
    summary.to_csv(run_dir / "summary.csv", index=False, float_format="%.3f")

    prefill = fit_prefill(df)
    analysis: dict[str, Any] = {
        "run_id": config["run_id"],
        "model": config["server"]["model"],
        "requests_total": len(df),
        "requests_valid": int(df["valid"].sum()),
        "invalid_reasons": (df["invalid_reasons"].explode().dropna().value_counts().to_dict()),
        "prefill_fit": prefill,
    }
    if prefill is not None:
        shape = ModelShape(**config["server"]["kv_shape"])
        bandwidth = break_even_bandwidth(shape, prefill["prefill_tokens_per_s"])
        analysis["break_even_storage_gbytes_per_s"] = bandwidth / 1e9
    write_json(run_dir / "analysis.json", analysis)

    title = f"TTFT vs prompt length: {config['server']['model']}"
    plot_ttft(df, run_dir / "ttft_vs_prompt.png", title)
    return analysis
