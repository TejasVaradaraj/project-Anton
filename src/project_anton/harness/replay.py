"""Replay a trace against a live server, and run resumable sweeps over load.

Sessions arrive on the trace's schedule whether or not the server is keeping up, so overload
shows up as delay. Within a session each request waits for the last, as a real agent's does.
"""

import asyncio
import itertools
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import yaml

from project_anton.harness.client import CompletionRequest, stream_completion
from project_anton.harness.prompts import prompt_digest
from project_anton.harness.runlog import append_jsonl, read_jsonl, write_json
from project_anton.harness.server import ServerConfig
from project_anton.harness.sysmon import SystemMonitor
from project_anton.harness.ttft_bench import TooManyFailures, check_same_config, preflight
from project_anton.traces.generators import (
    Session,
    TraceConfig,
    generate_trace,
    load_trace_config,
    session_prompts,
)

MAX_CONSECUTIVE_FAILURES = 5
REQUEST_TIMEOUT_S = 600


@dataclass(frozen=True)
class ReplaySweep:
    name: str
    trace: str
    grid: dict[str, list[Any]]


def load_replay_sweep(path: str | Path) -> ReplaySweep:
    return ReplaySweep(**yaml.safe_load(Path(path).read_text()))


def invalid_reasons(record: dict[str, Any]) -> list[str]:
    if record["status"] != "ok":
        return [record["status"]]
    reasons = []
    if record["prompt_tokens"] != record["target_prompt_tokens"]:
        reasons.append("prompt_length_mismatch")
    if record["completion_tokens"] != record["target_max_tokens"]:
        reasons.append("completion_length_mismatch")
    if record["ttft_s"] is None:
        reasons.append("no_tokens")
    return reasons


def finished_sessions(records: list[dict[str, Any]]) -> set[str]:
    """Sessions with every turn successful within one attempt."""
    ok: dict[tuple[str, int], int] = {}
    needed: dict[str, int] = {}
    for r in records:
        needed[r["session_id"]] = r["session_turns"]
        if r["status"] == "ok":
            key = (r["session_id"], r["attempt"])
            ok[key] = ok.get(key, 0) + 1
    return {sid for (sid, _), count in ok.items() if count == needed[sid]}


async def run_replay(
    cfg: ServerConfig, trace: TraceConfig, run_id: str, run_dir: Path, log=print
) -> Path:
    """Replay the trace. Rerunning with the same run_id runs only unfinished sessions."""
    config = {"run_id": run_id, "server": asdict(cfg), "trace": asdict(trace)}
    config_path = run_dir / "config.json"
    if config_path.exists():
        check_same_config(config_path, config)
    else:
        write_json(config_path, config)

    path = run_dir / "raw" / "requests.jsonl"
    previous = read_jsonl(path)
    attempts: dict[str, set[int]] = {}
    for r in previous:
        attempts.setdefault(r["session_id"], set()).add(r["attempt"])
    done = finished_sessions(previous)

    sessions = [s for s in generate_trace(trace) if s.session_id not in done]
    if not sessions:
        log(f"{run_id}: nothing to do, all sessions finished")
        return path
    log(f"{run_id}: {len(sessions)} sessions to run, {len(done)} already finished")

    # The run ID keeps this run's prompts, including the shared prefix, out of the cache
    # that an earlier run on the same server may have left behind.
    key = f"{trace.seed}/{run_id}"
    needed = max(
        s.prefix_tokens + sum(t.new_tokens + t.response_tokens for t in s.turns) for s in sessions
    )
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    state = {"failures": 0, "in_flight": 0}
    abort = asyncio.Event()

    async with httpx.AsyncClient(
        base_url=cfg.base_url, timeout=REQUEST_TIMEOUT_S, limits=limits
    ) as client:
        await preflight(client, cfg, needed)
        loop = asyncio.get_running_loop()
        start = loop.time()
        first_arrival = min(s.arrival_s for s in sessions)

        async def play(session: Session) -> None:
            await asyncio.sleep(session.arrival_s - first_arrival)
            attempt = max(attempts.get(session.session_id, {-1})) + 1
            prompts = session_prompts(session, cfg.token_id_range, key, attempt)
            previous_prompt_tokens = None
            for index, (turn, prompt) in enumerate(zip(session.turns, prompts, strict=True)):
                await asyncio.sleep(turn.pause_before_s)
                if abort.is_set():
                    return
                sent_s = loop.time() - start
                state["in_flight"] += 1
                concurrent = state["in_flight"]
                request = CompletionRequest(cfg.model, prompt, turn.response_tokens, trace.seed)
                result = await stream_completion(client, request)
                state["in_flight"] -= 1
                record = {
                    "run_id": run_id,
                    "session_id": session.session_id,
                    "session_turns": len(session.turns),
                    "attempt": attempt,
                    "turn": index,
                    "tool": turn.tool,
                    "pause_before_s": turn.pause_before_s,
                    "sent_s": sent_s,
                    "concurrent_at_send": concurrent,
                    "target_prompt_tokens": len(prompt),
                    "target_max_tokens": turn.response_tokens,
                    "previous_prompt_tokens": previous_prompt_tokens,
                    "prompt_digest": prompt_digest(prompt),
                    **result.to_dict(),
                }
                record["invalid_reasons"] = invalid_reasons(record)
                append_jsonl(path, record)
                previous_prompt_tokens = len(prompt)
                if result.status != "ok":
                    state["failures"] += 1
                    log(f"{session.session_id} turn {index}: {result.status}: {result.error}")
                    if state["failures"] >= MAX_CONSECUTIVE_FAILURES:
                        abort.set()
                    return  # Later turns depend on this one
                state["failures"] = 0

        await asyncio.gather(*(play(s) for s in sessions))
    if abort.is_set():
        raise TooManyFailures(f"stopped after repeated failures. Rerun with --run-id {run_id}.")
    return path


def _slug(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "p", str(value))


def grid_points(sweep: ReplaySweep) -> list[tuple[str, dict[str, Any]]]:
    """Every combination in the grid, with a stable run ID for each."""
    names = sorted(sweep.grid)
    points = []
    for values in itertools.product(*(sweep.grid[n] for n in names)):
        overrides = dict(zip(names, values, strict=True))
        slug = "_".join(f"{n}-{_slug(v)}" for n, v in overrides.items())
        points.append((f"{sweep.name}_{slug}", overrides))
    return points


async def run_grid(
    cfg: ServerConfig, sweep: ReplaySweep, results_dir: Path, before_run=None, log=print
) -> list[str]:
    """Run each point once. Finished points are skipped, so a dropped session can continue.

    Points share one server, so a later point can start with the cache still holding entries
    from an earlier one. Its prompts won't match them, but they do take up space until evicted.
    """
    base = load_trace_config(sweep.trace)
    finished = []
    for run_id, overrides in grid_points(sweep):
        run_dir = results_dir / run_id
        marker = run_dir / "done.json"
        if marker.exists():
            log(f"{run_id}: already finished, skipping")
            finished.append(run_id)
            continue
        trace = replace(base, **overrides)
        if before_run is not None:
            before_run(run_dir)
        async with SystemMonitor(run_dir / "raw" / "sysmon", cfg.base_url):
            await run_replay(cfg, trace, run_id, run_dir, log=log)
        write_json(marker, {"run_id": run_id})
        finished.append(run_id)
    return finished
