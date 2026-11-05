"""Measure how fast each storage tier gives a KV cache back to the GPU.

Each prompt is sent cold, then warm, then filler prompts push it out of the GPU cache and it is
sent again. With no offload, that last request recomputes. With offload, the gap between it and
the warm request is the time to load the cache from that tier.
"""

import asyncio
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from project_anton.harness.client import CompletionRequest, stream_completion
from project_anton.harness.prompts import make_prompt, prompt_digest
from project_anton.harness.runlog import append_jsonl, read_jsonl, write_json
from project_anton.harness.server import ServerConfig
from project_anton.harness.ttft_bench import TooManyFailures, check_same_config, preflight

EXTERNAL_HITS = re.compile(
    r"^vllm:external_prefix_cache_hits_total(\{[^}]*\})? ([0-9.e+-]+)$", re.M
)
MAX_CONSECUTIVE_FAILURES = 5


@dataclass(frozen=True)
class TierConfig:
    name: str
    prompt_lengths: tuple[int, ...]
    reps: int
    max_tokens: int = 16
    filler_tokens: int = 16000
    filler_prompts: int = 8
    settle_s: float = 5.0  # Lets asynchronous saves to the tier finish
    seed: int = 0


def load_tier_config(path: str | Path) -> TierConfig:
    raw = yaml.safe_load(Path(path).read_text())
    raw["prompt_lengths"] = tuple(raw["prompt_lengths"])
    return TierConfig(**raw)


def build_plan(tier: TierConfig) -> list[tuple[int, int]]:
    plan = [(length, rep) for length in tier.prompt_lengths for rep in range(tier.reps)]
    random.Random(tier.seed).shuffle(plan)
    return plan


def parse_external_hits(metrics_text: str) -> float:
    return sum(float(m.group(2)) for m in EXTERNAL_HITS.finditer(metrics_text))


def drop_page_cache(directory: Path) -> int:
    """Ask the OS to forget cached copies of a tier's files, so reads must reach the disk.

    Otherwise a "disk" read can be served from RAM. This needs no root, but the OS may keep pages
    in use, so check it against the disk log. Returns how many files were dropped.
    """
    dropped = 0
    os.sync()
    for path in directory.rglob("*"):
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                dropped += 1
            finally:
                os.close(fd)
    return dropped


def invalid_reasons(record: dict[str, Any]) -> list[str]:
    if record["status"] != "ok":
        return [record["status"]]
    reasons = []
    if record["prompt_tokens"] != record["target_prompt_tokens"]:
        reasons.append("prompt_length_mismatch")
    if record["completion_tokens"] != record["target_max_tokens"]:
        reasons.append("completion_length_mismatch")
    cached = record["cached_tokens"]
    if record["phase"] == "cold" and cached:
        reasons.append("cold_prompt_was_cached")
    if record["phase"] == "warm" and cached == 0:
        reasons.append("warm_prompt_missed_cache")
    return reasons


async def run_tiers(
    cfg: ServerConfig,
    tier: TierConfig,
    run_id: str,
    run_dir: Path,
    drop_cache_dir: Path | None = None,
    log=print,
) -> Path:
    config = {"run_id": run_id, "server": asdict(cfg), "tier": asdict(tier)}
    config_path = run_dir / "config.json"
    if config_path.exists():
        check_same_config(config_path, config)
    else:
        write_json(config_path, config)

    path = run_dir / "raw" / "requests.jsonl"
    previous = read_jsonl(path)
    done = {r["item"] for r in previous if r["phase"] == "evicted" and r["status"] == "ok"}
    attempts = sum(1 for r in previous if r["phase"] == "cold")
    failures = 0

    async with httpx.AsyncClient(base_url=cfg.base_url, timeout=600) as client:
        await preflight(client, cfg, max(tier.prompt_lengths) + tier.max_tokens)

        async def external_hits() -> float:
            response = await client.get("/metrics")
            return parse_external_hits(response.text)

        async def send(prompt: list[int], max_tokens: int):
            request = CompletionRequest(cfg.model, prompt, max_tokens, tier.seed)
            return await stream_completion(client, request)

        for length, rep in build_plan(tier):
            item = f"L{length}_r{rep}"
            if item in done:
                continue
            attempts += 1
            key = f"{tier.seed}/{run_id}/{item}/{attempts}"
            prompt = make_prompt(length, cfg.token_id_range, key)
            ok = True
            for phase in ("cold", "warm", "evicted"):
                if phase == "evicted":
                    await asyncio.sleep(tier.settle_s)
                    for i in range(tier.filler_prompts):
                        filler_key = f"{key}/filler/{i}"
                        filler = make_prompt(tier.filler_tokens, cfg.token_id_range, filler_key)
                        await send(filler, 1)
                    await asyncio.sleep(tier.settle_s)
                    if drop_cache_dir is not None and drop_cache_dir.exists():
                        await asyncio.to_thread(drop_page_cache, drop_cache_dir)
                before = await external_hits()
                result = await send(prompt, tier.max_tokens)
                after = await external_hits()
                record = {
                    "run_id": run_id,
                    "item": item,
                    "phase": phase,
                    "rep": rep,
                    "target_prompt_tokens": length,
                    "target_max_tokens": tier.max_tokens,
                    "prompt_digest": prompt_digest(prompt),
                    "external_hits_delta": after - before,
                    **result.to_dict(),
                }
                record["invalid_reasons"] = invalid_reasons(record)
                append_jsonl(path, record)
                if result.status != "ok":
                    ok = False
                    log(f"{item} {phase}: {result.status}: {result.error}")
                    break
                log(f"{item} {phase}: {result.ttft_s:.3f}s ext={after - before:.0f}")
            failures = 0 if ok else failures + 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise TooManyFailures(f"stopped after repeated failures. Rerun with {run_id}.")
    return path
