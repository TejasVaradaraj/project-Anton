"""Final GPU runs: the chunk alignment test, then a repeat sweep near the cliff.

Run from the repo root on a GPU machine:

    uv run python scripts/week7_final.py --results-dir results/week7
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# LMCache restores changed model output in Week 4 (see README). Results from these runs would not
# mean anything until that's resolved, so they must be asked for explicitly.
if os.environ.get("ANTON_ALLOW_UNVERIFIED_OFFLOAD") != "1":
    sys.exit("Refusing to run: LMCache restores failed the correctness check.")


def anton(*args: object) -> None:
    subprocess.run([sys.executable, "-m", "project_anton.cli", *map(str, args)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    server = "configs/servers/a100_small_cpu.yaml"
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
        "tiers", "compare", "--run", f"CPU RAM={out / 'alignment_cpu'}", "--out", out / "alignment"
    )

    for sweep, server in (
        ("week7_gpu_only", "configs/servers/a100_llama31_8b.yaml"),
        ("week7_lmcache", "configs/servers/a100_llama31_8b_lmcache.yaml"),
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
        "week7_gpu_only",
        "--sweep",
        "week7_lmcache",
        "--results-dir",
        out,
        "--out",
        out / "comparison",
    )


if __name__ == "__main__":
    main()
