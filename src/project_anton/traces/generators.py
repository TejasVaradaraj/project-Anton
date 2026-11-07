"""Seeded workloads: agent, multi-turn chat, and long-context.

A trace describes sessions as token counts and pauses, with no text. The default numbers are
assumptions from the project plan. They haven't been calibrated against real agent logs yet.
"""

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from project_anton.harness.prompts import make_prompt

DEFAULTS: dict[str, dict[str, Any]] = {
    "agent": {
        "prefix_tokens": 3000,
        "first_turn_tokens": [200, 1000],
        "steps": [10, 40],
        "tool_output_tokens": [500, 3000],
        "response_tokens": [50, 300],
        # Pause times follow a log-normal: mostly short, with a few very long.
        "tools": {
            "read_file": {"weight": 0.45, "median_s": 0.5, "sigma": 0.5},
            "grep": {"weight": 0.30, "median_s": 1.0, "sigma": 0.6},
            "run_tests": {"weight": 0.25, "median_s": 20.0, "sigma": 0.8},
        },
    },
    "multi_turn": {
        "prefix_tokens": 200,
        "first_turn_tokens": [50, 500],
        "steps": [3, 10],
        "tool_output_tokens": [20, 300],
        "response_tokens": [100, 500],
        "tools": {"user": {"weight": 1.0, "median_s": 10.0, "sigma": 0.8}},
    },
    "long_context": {
        "median_prompt_tokens": 16000,
        "prompt_sigma": 0.5,
        "response_tokens": [50, 300],
    },
}


@dataclass(frozen=True)
class TraceConfig:
    name: str
    kind: str
    sessions: int
    arrival_rate_per_s: float
    seed: int = 0
    max_context_tokens: int = 32000
    pause_scale: float = 1.0  # Below 1 shortens pauses, for quick tests
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Turn:
    new_tokens: int  # Added to the context just before this request
    response_tokens: int
    pause_before_s: float
    tool: str | None


@dataclass(frozen=True)
class Session:
    session_id: str
    arrival_s: float
    prefix_tokens: int
    turns: tuple[Turn, ...]


def load_trace_config(path: str | Path) -> TraceConfig:
    return TraceConfig(**yaml.safe_load(Path(path).read_text()))


def _rng(*key: object) -> np.random.Generator:
    digest = hashlib.sha256("/".join(map(str, key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _draw(rng: np.random.Generator, spec: list[int] | dict[str, list[float]]) -> int:
    """Draw from [low, high], or from a table of measured percentiles."""
    if isinstance(spec, dict):
        return max(1, round(float(np.interp(rng.random(), spec["probs"], spec["values"]))))
    return int(rng.integers(spec[0], spec[1] + 1))


def _conversation(cfg: TraceConfig, params: dict[str, Any], rng, index: int) -> Session:
    tools = params["tools"]
    names = sorted(tools)
    weights = np.array([tools[n]["weight"] for n in names], dtype=float)
    weights /= weights.sum()

    prefix = params["prefix_tokens"]
    context = prefix
    turns = []
    for step in range(_draw(rng, params["steps"])):
        response = _draw(rng, params["response_tokens"])
        if step == 0:
            tool, pause = None, 0.0
            new = _draw(rng, params["first_turn_tokens"])
        else:
            tool = str(rng.choice(names, p=weights))
            spec = tools[tool]
            pause = float(rng.lognormal(math.log(spec["median_s"]), spec["sigma"]))
            pause *= cfg.pause_scale
            new = _draw(rng, spec.get("output_tokens", params.get("tool_output_tokens")))
        if context + new + response > cfg.max_context_tokens:
            break  # The session ends when it runs out of context, as a real one must
        turns.append(Turn(new, response, pause, tool))
        context += new + response
    return Session(f"s{index}", 0.0, prefix, tuple(turns))


def _long_context(cfg: TraceConfig, params: dict[str, Any], rng, index: int) -> Session:
    response = _draw(rng, params["response_tokens"])
    limit = cfg.max_context_tokens - response
    drawn = rng.lognormal(math.log(params["median_prompt_tokens"]), params["prompt_sigma"])
    prompt = int(min(max(drawn, 16), limit))
    return Session(f"s{index}", 0.0, 0, (Turn(prompt, response, 0.0, None),))


def generate_trace(cfg: TraceConfig) -> list[Session]:
    """Build sessions, with Poisson arrivals.

    Each session has its own random stream. So asking for more sessions keeps the earlier ones
    identical, and points in a sweep differ only in load.
    """
    if cfg.kind not in DEFAULTS:
        raise ValueError(f"unknown trace kind {cfg.kind!r}")
    if cfg.arrival_rate_per_s <= 0:
        raise ValueError("arrival_rate_per_s must be positive")
    params = DEFAULTS[cfg.kind] | cfg.params
    build = _long_context if cfg.kind == "long_context" else _conversation

    arrivals = _rng(cfg.kind, cfg.seed, "arrivals")
    now = 0.0
    sessions = []
    for index in range(cfg.sessions):
        session = build(cfg, params, _rng(cfg.kind, cfg.seed, "session", index), index)
        if session.turns:
            sessions.append(Session(session.session_id, now, session.prefix_tokens, session.turns))
        now += float(arrivals.exponential(1 / cfg.arrival_rate_per_s))
    return sessions


def session_prompts(
    session: Session, token_id_range: tuple[int, int], key: str, attempt: int = 0
) -> Iterator[list[int]]:
    """Yield each turn's full prompt.

    Each prompt holds everything before it plus the trace's own response tokens, not what the
    model generated, so runs repeat exactly. The prefix is shared across sessions, like an
    agent's system prompt. The attempt number makes a retried session new to the cache.
    """
    context: list[int] = []
    if session.prefix_tokens:
        context += make_prompt(session.prefix_tokens, token_id_range, f"{key}/prefix")
    base = f"{key}/{session.session_id}/{attempt}"
    for index, turn in enumerate(session.turns):
        context += make_prompt(turn.new_tokens, token_id_range, f"{base}/{index}/new")
        yield list(context)
        context += make_prompt(turn.response_tokens, token_id_range, f"{base}/{index}/response")
