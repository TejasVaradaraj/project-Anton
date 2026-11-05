"""Run the prefetch comparison on one workload: baseline, offload, prefetch, and ideal.

Needs a GPU machine with vLLM and LMCache installed. Run from the repo root:

    uv run python scripts/week6_scenarios.py --results-dir results/week6
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TRACE = "configs/traces/agent_pressure.yaml"
PROXY_URL = "http://127.0.0.1:8100"
SCENARIOS = (
    ("A GPU only", "a100_small_gpu_only", False),
    ("B offload", "a100_small_cpu", False),
    ("C offload + prefetch", "a100_small_cpu", True),
    ("D ideal large GPU cache", "a100_llama31_8b", False),
)


def anton(*args: object) -> None:
    subprocess.run([sys.executable, "-m", "project_anton.cli", *map(str, args)], check=True)


def wait_for(url: str, timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    compare = []
    for name, server, prefetch in SCENARIOS:
        run_id = "scenario_" + name.split()[0]
        run_dir = args.results_dir / run_id
        config = f"configs/servers/{server}.yaml"
        anton("server", "start", "--config", config)
        proxy = None
        try:
            replay = ["replay", "run", "--server", config, "--trace", TRACE]
            replay += ["--run-id", run_id, "--results-dir", args.results_dir]
            if prefetch:
                command = [sys.executable, "-m", "project_anton.cli", "proxy", "serve"]
                proxy = subprocess.Popen(command)
                wait_for(f"{PROXY_URL}/health")
                replay += ["--proxy", PROXY_URL]
            anton(*replay)
            if proxy is not None:
                with urllib.request.urlopen(f"{PROXY_URL}/stats", timeout=10) as response:
                    stats = json.load(response)
                (run_dir / "proxy_stats.json").write_text(json.dumps(stats, indent=2))
        finally:
            if proxy is not None:
                proxy.terminate()
                proxy.wait(timeout=30)
            anton("server", "stop", "--config", config)
        anton("replay", "analyze", run_dir)
        compare += ["--run", f"{name}={run_dir}"]
    anton("replay", "scenarios", *compare, "--out", args.results_dir / "scenarios.csv")


if __name__ == "__main__":
    main()
