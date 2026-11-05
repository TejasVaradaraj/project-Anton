"""Command line: `anton server`, `anton ttft` and `anton replay`."""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

from project_anton.analysis.cliff import compare_sweeps
from project_anton.analysis.replay import analyze_replay
from project_anton.analysis.ttft import analyze_run
from project_anton.harness.replay import load_replay_sweep, run_grid, run_replay
from project_anton.harness.runlog import capture_environment, make_run_id, write_json
from project_anton.harness.server import (
    load_server_config,
    parse_kv_capacity,
    start_server,
    stop_server,
    wait_until_healthy,
)
from project_anton.harness.sysmon import SystemMonitor
from project_anton.harness.ttft_bench import load_sweep_config, run_sweep
from project_anton.traces.generators import load_trace_config

LOG_DIR = Path("logs")


def _paths(name: str) -> tuple[Path, Path]:
    return LOG_DIR / f"{name}.log", LOG_DIR / f"{name}.pid"


def server_start(args) -> int:
    cfg = load_server_config(args.config)
    log_path, pid_path = _paths(cfg.name)
    proc = start_server(cfg, log_path, pid_path)
    print(f"started {cfg.model} pid={proc.pid}, log at {log_path}")
    try:
        wait_until_healthy(cfg.base_url, proc, timeout_s=args.timeout)
    except (RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("".join(log_path.read_text(errors="replace").splitlines(True)[-30:]), file=sys.stderr)
        stop_server(pid_path)
        return 1
    print(f"healthy at {cfg.base_url}")
    print(f"kv capacity: {parse_kv_capacity(log_path.read_text(errors='replace'))}")
    return 0


def server_stop(args) -> int:
    cfg = load_server_config(args.config)
    stopped = stop_server(_paths(cfg.name)[1])
    print("stopped" if stopped else "nothing was running")
    return 0


def server_status(args) -> int:
    cfg = load_server_config(args.config)
    log_path, _ = _paths(cfg.name)
    try:
        wait_until_healthy(cfg.base_url, timeout_s=args.timeout, poll_s=0.5)
    except TimeoutError as exc:
        print(exc)
        return 1
    print(f"healthy at {cfg.base_url}")
    if log_path.exists():
        print(f"kv capacity: {parse_kv_capacity(log_path.read_text(errors='replace'))}")
    return 0


def _record_env(cfg, run_dir: Path) -> None:
    env = capture_environment()
    log_path, _ = _paths(cfg.name)
    if log_path.exists():
        env["kv_capacity"] = parse_kv_capacity(log_path.read_text(errors="replace"))
    # One file per start, so a resumed run also records what it ran on.
    existing = len(list(run_dir.glob("env*.json")))
    write_json(run_dir / ("env.json" if not existing else f"env.resume{existing}.json"), env)
    if env["git"]["dirty"]:
        print("WARNING: uncommitted changes, so this git SHA won't fully reproduce the run")


def _save_server_log(cfg, run_dir: Path) -> None:
    log_path, _ = _paths(cfg.name)
    if log_path.exists():
        (run_dir / "raw").mkdir(parents=True, exist_ok=True)
        shutil.copy(log_path, run_dir / "raw" / "server.log")


def ttft_run(args) -> int:
    cfg = load_server_config(args.server)
    sweep = load_sweep_config(args.sweep)
    run_id = args.run_id or make_run_id(f"{cfg.name}_{sweep.name}")
    run_dir = Path(args.results_dir) / run_id
    _record_env(cfg, run_dir)
    try:
        asyncio.run(run_sweep(cfg, sweep, run_id, run_dir))
    finally:
        _save_server_log(cfg, run_dir)
    print(f"results in {run_dir}")
    return 0


def ttft_analyze(args) -> int:
    print(json.dumps(analyze_run(Path(args.run_dir)), indent=2))
    return 0


def replay_run(args) -> int:
    cfg = load_server_config(args.server)
    trace = load_trace_config(args.trace)
    run_id = args.run_id or make_run_id(f"{cfg.name}_{trace.name}")
    run_dir = Path(args.results_dir) / run_id
    _record_env(cfg, run_dir)

    async def go():
        async with SystemMonitor(run_dir / "raw" / "sysmon", cfg.base_url):
            await run_replay(cfg, trace, run_id, run_dir)

    try:
        asyncio.run(go())
    finally:
        _save_server_log(cfg, run_dir)
    print(f"results in {run_dir}")
    return 0


def replay_analyze(args) -> int:
    print(json.dumps(analyze_replay(Path(args.run_dir)), indent=2))
    return 0


def replay_sweep(args) -> int:
    cfg = load_server_config(args.server)
    sweep = load_replay_sweep(args.sweep)
    results_dir = Path(args.results_dir)
    finished = asyncio.run(
        run_grid(cfg, sweep, results_dir, before_run=lambda d: _record_env(cfg, d))
    )
    for run_id in finished:
        _save_server_log(cfg, results_dir / run_id)
        analyze_replay(results_dir / run_id)
    print(f"{len(finished)} points finished in {results_dir}")
    return 0


def replay_compare(args) -> int:
    table = compare_sweeps(Path(args.results_dir), args.sweep, Path(args.out))
    print(table.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anton", description=__doc__)
    groups = parser.add_subparsers(required=True)

    server = groups.add_parser("server", help="manage vllm serve").add_subparsers(required=True)
    for name, func, timeout in (
        ("start", server_start, 1800),
        ("stop", server_stop, 0),
        ("status", server_status, 5),
    ):
        sub = server.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--timeout", type=float, default=timeout)
        sub.set_defaults(func=func)

    ttft = groups.add_parser("ttft", help="TTFT benchmark").add_subparsers(required=True)
    run = ttft.add_parser("run")
    run.add_argument("--server", required=True)
    run.add_argument("--sweep", required=True)
    run.add_argument("--run-id", help="reuse an ID to resume a stopped run")
    run.add_argument("--results-dir", default="results")
    run.set_defaults(func=ttft_run)

    analyze = ttft.add_parser("analyze")
    analyze.add_argument("run_dir")
    analyze.set_defaults(func=ttft_analyze)

    replay = groups.add_parser("replay", help="replay traces").add_subparsers(required=True)
    run = replay.add_parser("run")
    run.add_argument("--server", required=True)
    run.add_argument("--trace", required=True)
    run.add_argument("--run-id", help="reuse an ID to resume a stopped run")
    run.add_argument("--results-dir", default="results")
    run.set_defaults(func=replay_run)

    analyze = replay.add_parser("analyze")
    analyze.add_argument("run_dir")
    analyze.set_defaults(func=replay_analyze)

    grid = replay.add_parser("sweep", help="run a grid; finished points are skipped")
    grid.add_argument("--server", required=True)
    grid.add_argument("--sweep", required=True)
    grid.add_argument("--results-dir", default="results")
    grid.set_defaults(func=replay_sweep)

    compare = replay.add_parser("compare", help="tabulate and plot finished sweeps")
    compare.add_argument("--sweep", action="append", required=True, help="repeat for each sweep")
    compare.add_argument("--results-dir", default="results")
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=replay_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
