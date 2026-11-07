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

from project_anton.harness.server import require_verified_offload

TRACE = "configs/traces/agent_pressure.yaml"
PROXY_URL = "http://127.0.0.1:8100"


def scenarios(offload: str) -> tuple[tuple[str, str, bool], ...]:
    return (
        ("A GPU only", "a100_small_gpu_only", False),
        ("B offload", offload, False),
        ("C offload + prefetch", offload, True),
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


def run_scenario(
    name: str, server: str, prefetch: bool, results_dir: Path, suffix: str = "", lead_s: float = 0.5
) -> Path:
    run_id = "scenario_" + name.split()[0] + suffix
    run_dir = results_dir / run_id
    config = f"configs/servers/{server}.yaml"
    anton("server", "start", "--config", config)
    proxy = None
    try:
        replay = ["replay", "run", "--server", config, "--trace", TRACE]
        replay += ["--run-id", run_id, "--results-dir", results_dir]
        if prefetch:
            command = [sys.executable, "-m", "project_anton.cli", "proxy", "serve"]
            command += ["--lead", str(lead_s)]
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
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--offload", default="a100_small_mp", help="server config for B and C")
    parser.add_argument("--only", default="ABCD", help="scenario letters to run")
    parser.add_argument("--suffix", default="", help="added to run IDs, for repeat runs")
    parser.add_argument("--lead-s", type=float, default=0.5, help="how early warm-ups are sent")
    args = parser.parse_args()
    require_verified_offload()

    compare = []
    failed = []
    chosen = [item for item in scenarios(args.offload) if item[0][0] in args.only]
    for name, server, prefetch in chosen:
        # One crash shouldn't lose the other scenarios. Finished ones are skipped on a rerun.
        try:
            run_dir = run_scenario(
                name, server, prefetch, args.results_dir, args.suffix, args.lead_s
            )
        except subprocess.CalledProcessError as exc:
            print(f"{name} failed: {exc}", file=sys.stderr)
            failed.append(name)
            continue
        compare += ["--run", f"{name}={run_dir}"]
    out = args.results_dir / f"scenarios{args.suffix}.csv"
    anton("replay", "scenarios", *compare, "--out", out)
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
