"""Check that restoring an offloaded KV cache gives the same answers as computing it.

A wrong restore wouldn't raise an error. It would quietly produce different text, so this compares
greedy tokens and their logprobs. Two runs with no offload set the noise floor: floating-point
results can differ slightly between runs, and offload only has to stay within that.
"""

import asyncio
from pathlib import Path
from typing import Any

import httpx

from project_anton.correctness.text_prompts import load_prompts
from project_anton.harness.prompts import make_prompt
from project_anton.harness.runlog import append_jsonl, read_jsonl
from project_anton.harness.server import ServerConfig

FILLER_TOKENS = 16000
FILLER_PROMPTS = 8
LENGTHS = (1000, 4000, 16000, 30000)
PROMPTS_PER_LENGTH = 3
MAX_TOKENS = 32


async def _complete(client: httpx.AsyncClient, cfg: ServerConfig, prompt: list[int], n: int):
    response = await client.post(
        "/v1/completions",
        json={
            "model": cfg.model,
            "prompt": prompt,
            "max_tokens": n,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "logprobs": 1,
        },
    )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return {
        "tokens": logprobs.get("tokens"),
        "logprobs": logprobs.get("token_logprobs"),
    }


async def record(
    cfg: ServerConfig,
    label: str,
    out_dir: Path,
    settle_s: float = 5.0,
    prompts_file: Path | None = None,
) -> Path:
    """Send the same prompts on any server: once fresh, again cached, again after eviction."""
    path = out_dir / f"{label}.jsonl"
    # A fixed key, not the run ID: every server must see identical prompts.
    prompts = (
        load_prompts(prompts_file)
        if prompts_file
        else {
            (length, i): make_prompt(length, cfg.token_id_range, f"correctness/{length}/{i}")
            for length in LENGTHS
            for i in range(PROMPTS_PER_LENGTH)
        }
    )
    async with httpx.AsyncClient(base_url=cfg.base_url, timeout=600) as client:
        for (length, i), prompt in prompts.items():
            for phase in ("fresh", "cached", "evicted"):
                if phase == "evicted":
                    await asyncio.sleep(settle_s)
                    for f in range(FILLER_PROMPTS):
                        key = f"correctness/filler/{label}/{length}/{i}/{f}"
                        filler = make_prompt(FILLER_TOKENS, cfg.token_id_range, key)
                        await _complete(client, cfg, filler, 1)
                    await asyncio.sleep(settle_s)
                result = await _complete(client, cfg, prompt, MAX_TOKENS)
                append_jsonl(path, {"length": length, "i": i, "phase": phase, **result})
    return path


def compare_runs(baseline: list[dict], other: list[dict]) -> dict[str, Any]:
    """Compare matching requests of two runs."""
    index = {(r["length"], r["i"], r["phase"]): r for r in baseline}
    compared = token_mismatches = 0
    worst_logprob = 0.0
    by_phase: dict[str, dict[str, Any]] = {}
    for r in other:
        base = index.get((r["length"], r["i"], r["phase"]))
        if base is None:
            continue
        compared += 1
        same = base["tokens"] == r["tokens"]
        token_mismatches += not same
        diffs = [abs(a - b) for a, b in zip(base["logprobs"], r["logprobs"], strict=False)]
        worst = max(diffs, default=0.0)
        worst_logprob = max(worst_logprob, worst)
        phase = by_phase.setdefault(
            r["phase"], {"n": 0, "token_mismatches": 0, "max_logprob_diff": 0.0}
        )
        phase["n"] += 1
        phase["token_mismatches"] += not same
        phase["max_logprob_diff"] = max(phase["max_logprob_diff"], worst)
    return {
        "compared": compared,
        "token_mismatches": token_mismatches,
        "max_logprob_diff": worst_logprob,
        "by_phase": by_phase,
    }


def self_check(run: list[dict]) -> dict[str, Any]:
    """Compare each prompt's later requests with its first, within one server run.

    This is the sharpest test: same server, same prompt, greedy decoding. Only the cache state
    differs, so the answers should match without needing any noise floor.
    """
    fresh = {(r["length"], r["i"]): r for r in run if r["phase"] == "fresh"}
    report: dict[str, Any] = {}
    for r in run:
        if r["phase"] == "fresh":
            continue
        base = fresh[(r["length"], r["i"])]
        phase = report.setdefault(
            r["phase"], {"n": 0, "token_mismatches": 0, "first_token_mismatches": 0}
        )
        phase["n"] += 1
        phase["token_mismatches"] += base["tokens"] != r["tokens"]
        phase["first_token_mismatches"] += base["tokens"][:1] != r["tokens"][:1]
    return report


def compare_files(baseline: Path, others: dict[str, Path]) -> dict[str, Any]:
    base = read_jsonl(baseline)
    report: dict[str, Any] = {"self_check": {"baseline": self_check(base)}}
    for label, path in others.items():
        run = read_jsonl(path)
        report[label] = compare_runs(base, run)
        report["self_check"][label] = self_check(run)
    return report
