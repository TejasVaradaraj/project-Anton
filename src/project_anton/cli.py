"""Command line: `anton server`, `anton ttft` and `anton replay`."""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

from project_anton.analysis.cliff import compare_sweeps
from project_anton.analysis.replay import analyze_replay
from project_anton.analysis.requirements import build_requirements
from project_anton.analysis.scenarios import compare_scenarios
from project_anton.analysis.tiers import compare_tiers
from project_anton.analysis.ttft import analyze_run
from project_anton.correctness.compare import compare_files, record
from project_anton.harness.replay import load_replay_sweep, run_grid, run_replay
from project_anton.harness.runlog import capture_environment, make_run_id, write_json
from project_anton.harness.server import (
    load_server_config,
    parse_kv_capacity,
    start_server,
    stop_server,
    wait_until_healthy,
)
from project_anton.harness.storage import run_baseline
from project_anton.harness.sysmon import SystemMonitor
from project_anton.harness.tiers import load_tier_config, run_tiers
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
            await run_replay(cfg, trace, run_id, run_dir, proxy_url=args.proxy)

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


def replay_scenarios(args) -> int:
    runs = dict(item.split("=", 1) for item in args.run)
    table = compare_scenarios({k: Path(v) for k, v in runs.items()}, Path(args.out))
    print(table.to_string(index=False))
    return 0


def requirements(args) -> int:
    table = build_requirements(Path(args.week1), Path(args.calibration), Path(args.out))
    print(table.to_string(index=False))
    return 0


def proxy_serve(args) -> int:
    import uvicorn

    from project_anton.proxy.app import create_app

    app = create_app(args.upstream, lead_s=args.lead, prefetch=not args.no_prefetch)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def tiers_run(args) -> int:
    cfg = load_server_config(args.server)
    tier = load_tier_config(args.tier)
    run_id = args.run_id or make_run_id(f"{cfg.name}_{tier.name}")
    run_dir = Path(args.results_dir) / run_id
    _record_env(cfg, run_dir)
    drop_dir = Path(args.drop_cache_dir) if args.drop_cache_dir else None

    async def go():
        async with SystemMonitor(run_dir / "raw" / "sysmon", cfg.base_url):
            await run_tiers(cfg, tier, run_id, run_dir, drop_cache_dir=drop_dir)

    try:
        asyncio.run(go())
    finally:
        _save_server_log(cfg, run_dir)
    print(f"results in {run_dir}")
    return 0


def tiers_compare(args) -> int:
    runs = dict(item.split("=", 1) for item in args.run)
    table = compare_tiers({k: Path(v) for k, v in runs.items()}, Path(args.out))
    print(table.to_string(index=False))
    return 0


def storage_baseline(args) -> int:
    results = run_baseline(Path(args.dir), size=args.size, runtime_s=args.runtime)
    write_json(Path(args.out), results)
    print(json.dumps(results, indent=2))
    return 0


def correctness_record(args) -> int:
    cfg = load_server_config(args.server)
    prompts = Path(args.prompts) if args.prompts else None
    print(asyncio.run(record(cfg, args.label, Path(args.out), args.settle, prompts)))
    return 0


def correctness_compare(args) -> int:
    others = dict(item.split("=", 1) for item in args.run)
    report = compare_files(Path(args.baseline), {k: Path(v) for k, v in others.items()})
    write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
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
    run.add_argument("--proxy", help="send requests through the prefetch proxy at this URL")
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

    scenarios = replay.add_parser("scenarios", help="compare setups on one workload")
    scenarios.add_argument("--run", action="append", required=True, help="name=run folder")
    scenarios.add_argument("--out", required=True)
    scenarios.set_defaults(func=replay_scenarios)

    need = groups.add_parser("requirements", help="storage requirements table")
    need.add_argument("--week1", required=True, help="a Week 1 analysis.json")
    need.add_argument("--calibration", default="configs/traces/calibration/swe_agent.json")
    need.add_argument("--out", required=True)
    need.set_defaults(func=requirements)

    proxy = groups.add_parser("proxy", help="agent-aware prefetch proxy").add_subparsers(
        required=True
    )
    serve = proxy.add_parser("serve")
    serve.add_argument("--upstream", default="http://127.0.0.1:8000")
    serve.add_argument("--port", type=int, default=8100)
    serve.add_argument("--lead", type=float, default=0.5, help="seconds before predicted tool end")
    serve.add_argument("--no-prefetch", action="store_true", help="relay only, for comparison")
    serve.set_defaults(func=proxy_serve)

    tiers = groups.add_parser("tiers", help="storage tier timing").add_subparsers(required=True)
    run = tiers.add_parser("run")
    run.add_argument("--server", required=True)
    run.add_argument("--tier", required=True)
    run.add_argument("--run-id", help="reuse an ID to resume a stopped run")
    run.add_argument("--results-dir", default="results")
    run.add_argument("--drop-cache-dir", help="drop this folder's page cache before each read")
    run.set_defaults(func=tiers_run)

    compare = tiers.add_parser("compare")
    compare.add_argument("--run", action="append", required=True, help="tier name=run folder")
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=tiers_compare)

    storage = groups.add_parser("storage", help="disk baseline").add_subparsers(required=True)
    base = storage.add_parser("baseline")
    base.add_argument("--dir", required=True)
    base.add_argument("--out", required=True)
    base.add_argument("--size", default="8G")
    base.add_argument("--runtime", type=int, default=20)
    base.set_defaults(func=storage_baseline)

    check = groups.add_parser("correctness", help="compare outputs").add_subparsers(required=True)
    rec = check.add_parser("record")
    rec.add_argument("--server", required=True)
    rec.add_argument("--label", required=True)
    rec.add_argument("--out", required=True)
    rec.add_argument("--settle", type=float, default=5.0, help="seconds to let saves finish")
    rec.add_argument("--prompts", help="prompts file from make_text_prompts.py; default random")
    rec.set_defaults(func=correctness_record)

    cmp_ = check.add_parser("compare")
    cmp_.add_argument("--baseline", required=True)
    cmp_.add_argument("--run", action="append", required=True, help="label=results file")
    cmp_.add_argument("--out", required=True)
    cmp_.set_defaults(func=correctness_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Notebooks read this through a pipe, where Python would otherwise hold output until exit.
    sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
