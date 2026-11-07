"""Final GPU runs: the chunk alignment test, then a repeat sweep near the cliff.

Run from the repo root on a GPU machine:

    uv run python scripts/week7_final.py --results-dir results/week7
"""

import argparse
import subprocess
import sys
from pathlib import Path

from project_anton.harness.server import require_verified_offload


def anton(*args: object) -> None:
    subprocess.run([sys.executable, "-m", "project_anton.cli", *map(str, args)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    require_verified_offload()
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    server = "configs/servers/a100_small_mp.yaml"
    anton("server", "start", "--config", server)
    try:
        anton(
            "tiers",
            "run",
            "--server",
            server,
            "--tier",
            "configs/tiers/alignment.yaml",
            "--run-id",
            "alignment_cpu",
            "--results-dir",
            out,
        )
    finally:
        anton("server", "stop", "--config", server)
    anton(
        "tiers",
        "compare",
        "--run",
        f"Multi-process={out / 'alignment_cpu'}",
        "--out",
        out / "alignment",
    )

    # Every setup gets the same GPU cache size, so the only difference is offload. The size leaves
    # room for the 9 GB of GPU memory that the multi-process LMCache server uses.
    for sweep, server in (
        ("final_gpu_only", "configs/servers/a100_final_gpu_only.yaml"),
        ("final_mp", "configs/servers/a100_final_mp.yaml"),
    ):
        anton("server", "start", "--config", server)
        try:
            anton(
                "replay",
                "sweep",
                "--server",
                server,
                "--sweep",
                f"configs/replay/{sweep}.yaml",
                "--results-dir",
                out,
            )
        finally:
            anton("server", "stop", "--config", server)
    anton(
        "replay",
        "compare",
        "--sweep",
        "final_gpu_only",
        "--sweep",
        "final_mp",
        "--results-dir",
        out,
        "--out",
        out / "comparison",
    )


if __name__ == "__main__":
    main()
