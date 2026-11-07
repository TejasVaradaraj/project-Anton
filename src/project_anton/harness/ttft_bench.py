"""Week 1 experiment: TTFT vs prompt length, cold vs warm prefix cache.

Each pair sends a never-seen prompt (cold), then repeats it at once (warm). The gap
between the two is what the prefix cache saves. Requests run one at a time, so this
measures the cost of one request and not queueing under load.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from project_anton.harness.client import CompletionRequest, stream_completion
from project_anton.harness.prompts import make_prompt, prompt_digest
from project_anton.harness.runlog import append_jsonl, read_jsonl, write_json
from project_anton.harness.server import ServerConfig


class PreflightError(RuntimeError):
    pass


class TooManyFailures(RuntimeError):
    pass


@dataclass(frozen=True)
class SweepConfig:
    name: str
    prompt_lengths: tuple[int, ...]
    reps: int
    max_tokens: int = 32
    warmup_requests: int = 2
    seed: int = 0
    request_timeout_s: float = 600
    max_consecutive_failures: int = 5


@dataclass(frozen=True)
class Pair:
    pair_id: str
    prompt_tokens: int
    rep: int


def load_sweep_config(path: str | Path) -> SweepConfig:
    raw = yaml.safe_load(Path(path).read_text())
    raw["prompt_lengths"] = tuple(raw["prompt_lengths"])
    return SweepConfig(**raw)


def build_plan(sweep: SweepConfig) -> list[Pair]:
    """All pairs in a seeded shuffle, so length isn't tied to time in the run.

    In sorted order, a slow drift in the machine would look like a length effect.
    """
    pairs = [
        Pair(f"L{length}_r{rep}", length, rep)
        for length in sweep.prompt_lengths
        for rep in range(sweep.reps)
    ]
    random.Random(sweep.seed).shuffle(pairs)
    return pairs


def invalid_reasons(record: dict[str, Any]) -> list[str]:
    """Why this record shouldn't be trusted as a measurement. Empty means valid."""
    if record["status"] != "ok":
        return [record["status"]]
    reasons = []
    if record["prompt_tokens"] != record["target_prompt_tokens"]:
        reasons.append("prompt_length_mismatch")
    if record["completion_tokens"] != record["target_max_tokens"]:
        reasons.append("completion_length_mismatch")
    if record["ttft_s"] is None:
        reasons.append("no_tokens")
    cached = record["cached_tokens"]
    if cached is not None:
        if record["phase"] == "cold" and cached > 0:
            reasons.append("cold_prompt_was_cached")
        if record["phase"] == "warm" and cached == 0:
            reasons.append("warm_prompt_missed_cache")
    return reasons


async def preflight(client: httpx.AsyncClient, cfg: ServerConfig, sweep: SweepConfig) -> None:
    """Catch setup mistakes before any GPU time is spent on the sweep."""
    try:
        (await client.get("/health")).raise_for_status()
        response = await client.get("/v1/models")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise PreflightError(f"server at {client.base_url} isn't ready: {exc}") from exc
    models = {m["id"]: m for m in response.json().get("data", [])}
    if cfg.model not in models:
        raise PreflightError(f"server has {sorted(models)}, config wants {cfg.model}")
    limit = models[cfg.model].get("max_model_len")
    needed = max(sweep.prompt_lengths) + sweep.max_tokens
    if limit is not None and needed > limit:
        raise PreflightError(f"sweep needs {needed} tokens, server max_model_len is {limit}")


def finished_pairs(records: list[dict[str, Any]]) -> set[str]:
    """Pairs whose cold and warm requests both completed in the same attempt."""
    ok: dict[tuple[str, int], set[str]] = {}
    for r in records:
        if r["pair_id"] and r["status"] == "ok":
            ok.setdefault((r["pair_id"], r["attempt"]), set()).add(r["phase"])
    return {pair_id for (pair_id, _), phases in ok.items() if {"cold", "warm"} <= phases}


async def run_sweep(
    cfg: ServerConfig,
    sweep: SweepConfig,
    run_id: str,
    run_dir: Path,
    log=print,
) -> Path:
    """Run the sweep. Rerunning with the same run_id continues where it stopped."""
    requests_path = run_dir / "raw" / "requests.jsonl"
    config = {"run_id": run_id, "server": asdict(cfg), "sweep": asdict(sweep)}
    config_path = run_dir / "config.json"
    if config_path.exists():
        _check_same_config(config_path, config)
    else:
        write_json(config_path, config)

    previous = read_jsonl(requests_path)
    done = finished_pairs(previous)
    todo = [p for p in build_plan(sweep) if p.pair_id not in done]
    if not todo:
        log(f"{run_id}: nothing to do, all pairs finished")
        return requests_path
    log(f"{run_id}: {len(todo)} pairs to run, {len(done)} already finished")

    attempts: dict[str, int] = {}
    for r in previous:
        if r["phase"] == "cold":
            attempts[r["pair_id"]] = attempts.get(r["pair_id"], 0) + 1

    async with httpx.AsyncClient(base_url=cfg.base_url, timeout=sweep.request_timeout_s) as client:
        await preflight(client, cfg, sweep)

        async def send(phase, pair, attempt, prompt, target_len, rep) -> dict[str, Any]:
            request = CompletionRequest(cfg.model, prompt, sweep.max_tokens, sweep.seed)
            result = await stream_completion(client, request)
            record = {
                "run_id": run_id,
                "pair_id": pair,
                "phase": phase,
                "rep": rep,
                "attempt": attempt,
                "target_prompt_tokens": target_len,
                "target_max_tokens": sweep.max_tokens,
                "prompt_digest": prompt_digest(prompt),
                **result.to_dict(),
            }
            record["invalid_reasons"] = invalid_reasons(record)
            append_jsonl(requests_path, record)
            return record

        # The first requests after startup are slowed by one-time setup. Record them,
        # but keep them out of the analysis.
        warmup_len = min(sweep.prompt_lengths)
        for i in range(sweep.warmup_requests):
            key = f"{sweep.seed}/{run_id}/warmup/{i}/{len(previous)}"
            prompt = make_prompt(warmup_len, cfg.token_id_range, key)
            await send("warmup", None, 0, prompt, warmup_len, i)

        failures = 0
        for index, pair in enumerate(todo, start=1):
            attempt = attempts.get(pair.pair_id, 0)
            # The run_id and attempt make this prompt new to the server, even if an
            # earlier run or a failed attempt has already cached a similar one.
            key = f"{sweep.seed}/{run_id}/{pair.pair_id}/{attempt}"
            prompt = make_prompt(pair.prompt_tokens, cfg.token_id_range, key)
            cold = await send("cold", pair.pair_id, attempt, prompt, pair.prompt_tokens, pair.rep)
            warm = None
            if cold["status"] == "ok":
                warm = await send(
                    "warm", pair.pair_id, attempt, prompt, pair.prompt_tokens, pair.rep
                )

            if cold["status"] == "ok" and warm["status"] == "ok":
                failures = 0
                log(
                    f"[{index}/{len(todo)}] {pair.pair_id}: cold {cold['ttft_s']:.3f}s "
                    f"warm {warm['ttft_s']:.3f}s"
                )
            else:
                failures += 1
                bad = warm if cold["status"] == "ok" else cold
                log(f"[{index}/{len(todo)}] {pair.pair_id}: FAILED {bad['status']}: {bad['error']}")
                if failures >= sweep.max_consecutive_failures:
                    raise TooManyFailures(
                        f"{failures} failures in a row, stopping. Rerun with --run-id {run_id} "
                        "once the server is fixed."
                    )
    return requests_path


def _check_same_config(path: Path, config: dict[str, Any]) -> None:
    """Refuse to mix results from two different setups in one run folder."""

    def comparable(obj: dict[str, Any]) -> dict[str, Any]:
        # The port can change between restarts without affecting measurements.
        return {**obj, "server": {k: v for k, v in obj["server"].items() if k != "port"}}

    # A JSON round trip makes tuples and lists compare equal.
    if comparable(json.loads(path.read_text())) != comparable(json.loads(json.dumps(config))):
        raise ValueError(f"{path} differs from this config. Use a new run ID.")
