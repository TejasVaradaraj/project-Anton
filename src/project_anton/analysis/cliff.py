"""Compare replay sweeps point by point, to show where GPU-only serving breaks down."""

import json
import os
from pathlib import Path
from typing import Any

# Set before importing matplotlib. See analysis/ttft.py.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
PANELS = (
    ("previous_cached_mean", "Previous prompt found in cache", (0, 1.05)),
    ("after_tool_ttft_p95_ms", "Post-tool TTFT p95 (ms)", (0, None)),
    ("goodput_tokens_per_s", "Goodput (output tokens/s)", (0, None)),
)


def _point(sweep: str, run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "replay_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    trace = json.loads((run_dir / "config.json").read_text())["trace"]
    after = summary["after_tool"]
    counters = summary.get("counters") or {}
    return {
        "sweep": sweep,
        "sessions": trace["sessions"],
        "arrival_rate_per_s": trace["arrival_rate_per_s"],
        "requests_valid": summary["requests_valid"],
        "requests_total": summary["requests_total"],
        "peak_concurrent": summary["max_concurrent_at_send"],
        "previous_cached_mean": after["previous_prompt_cached_mean"],
        "previous_cached_p10": after["previous_prompt_cached_p10"],
        "after_tool_ttft_p50_ms": after["ttft_p50_ms"],
        "after_tool_ttft_p95_ms": after["ttft_p95_ms"],
        "after_tool_ttft_p99_ms": after["ttft_p99_ms"],
        "tpot_p50_ms": after["tpot_p50_ms"],
        "slo_met_fraction": summary["slo_met_fraction"],
        "output_tokens_per_s": summary.get("output_tokens_per_s"),
        "goodput_tokens_per_s": summary.get("goodput_output_tokens_per_s"),
        "external_cache_hits": counters.get("external_prefix_cache_hits"),
        "duration_s": summary.get("duration_s"),
    }


def load_sweep(results_dir: Path, sweep: str) -> pd.DataFrame:
    """Every finished point of one sweep, in order of load."""
    rows = [_point(sweep, d) for d in sorted(results_dir.glob(f"{sweep}_*")) if d.is_dir()]
    rows = [r for r in rows if r is not None]
    if not rows:
        raise ValueError(f"no finished points for {sweep} in {results_dir}")
    return pd.DataFrame(rows).sort_values("sessions").reset_index(drop=True)


def plot_comparison(table: pd.DataFrame, path: Path) -> None:
    """One small panel per measure, because they have different units."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    sweeps = list(table["sweep"].unique())
    for ax, (column, title, ylim) in zip(axes, PANELS, strict=True):
        ax.set_facecolor("#fcfcfb")
        for sweep, color in zip(sweeps, COLORS, strict=False):
            data = table[table["sweep"] == sweep].dropna(subset=[column])
            ax.plot(
                data["sessions"],
                data[column],
                color=color,
                linewidth=1.5,
                marker="o",
                markersize=6,
                markeredgecolor="#fcfcfb",
                markeredgewidth=1,
                label=sweep,
            )
        ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("Agent sessions", color="#52514e")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#52514e", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
    axes[0].legend(frameon=False, fontsize=9, labelcolor="#52514e", loc="lower left")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def compare_sweeps(results_dir: Path, sweeps: list[str], out_dir: Path) -> pd.DataFrame:
    table = pd.concat([load_sweep(results_dir, s) for s in sweeps], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison.csv", index=False, float_format="%.4f")
    plot_comparison(table, out_dir / "cliff.png")
    return table
