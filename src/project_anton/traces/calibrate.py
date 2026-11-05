"""Measure real agent runs, and fit the numbers an agent trace needs.

Reads SWE-agent trajectories (nebius/SWE-agent-trajectories on Hugging Face). They record what was
said, not when, so this calibrates token counts and step counts but not pause times.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FENCE = re.compile(r"```(?:\w*\n)?(.*?)```", re.S)
COMMAND_WORD = re.compile(r"[A-Za-z_]+")
PERCENTILES = (10, 25, 50, 75, 90, 99)
DISTRIBUTION_PROBS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)

READ = {"open", "goto", "scroll_down", "scroll_up", "cat", "head", "tail", "less", "more", "sed"}
SEARCH = {"search_dir", "search_file", "find_file", "grep", "find", "ls", "tree", "pwd", "cd"}
EDIT = {"edit", "create", "rm", "mv", "cp", "mkdir", "touch", "chmod", "echo", "git"}
FINISH = {"submit", "exit"}

CountTokens = Callable[[list[str]], list[int]]


def command_of(text: str | None) -> str | None:
    """The command an agent ran: the first word of its last fenced block."""
    blocks = FENCE.findall(text or "")
    if not blocks or not blocks[-1].strip():
        return None
    return blocks[-1].strip()


def tool_of(command: str | None) -> str | None:
    """Group a command into a tool type. None means there's no tool output to measure."""
    if command is None:
        return None
    match = COMMAND_WORD.match(command)
    word = match.group() if match else ""  # ./reproduce.sh has no leading word, and is a run
    if word in FINISH:
        return None
    if word in READ:
        return "read"
    if word in SEARCH:
        return "search"
    if word in EDIT:
        return "edit"
    # Everything else runs code: python, pytest, pip, or a repo's own command line.
    return "run"


@dataclass
class Measurements:
    trajectories: int = 0
    system_tokens: list[int] = field(default_factory=list)
    first_turn_tokens: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    response_tokens: list[int] = field(default_factory=list)
    final_context_tokens: list[int] = field(default_factory=list)
    tool_output_tokens: dict[str, list[int]] = field(default_factory=dict)
    exit_status: Counter = field(default_factory=Counter)


def measure(trajectories: Iterable[dict[str, Any]], count_tokens: CountTokens) -> Measurements:
    """Count tokens in each message. Each row needs `trajectory` and `exit_status`."""
    result = Measurements()
    for row in trajectories:
        messages = row["trajectory"]
        system = next((m for m in messages if m["role"] == "system"), None)
        talk = [m for m in messages if m["role"] != "system" and m.get("text")]
        if system is None or len(talk) < 2 or talk[0]["role"] != "user":
            continue

        counts = count_tokens([system.get("system_prompt") or ""] + [m["text"] for m in talk])
        result.trajectories += 1
        result.exit_status[row.get("exit_status")] += 1
        result.system_tokens.append(counts[0])
        result.first_turn_tokens.append(counts[1])
        result.final_context_tokens.append(sum(counts))

        steps = 0
        pending_tool = None
        for message, tokens in zip(talk[1:], counts[2:], strict=True):
            if message["role"] == "ai":
                steps += 1
                result.response_tokens.append(tokens)
                pending_tool = tool_of(command_of(message["text"]))
            elif pending_tool is not None:
                result.tool_output_tokens.setdefault(pending_tool, []).append(tokens)
                pending_tool = None
        result.steps.append(steps)
    return result


def fit_distribution(values: list[int]) -> dict[str, list[float]]:
    """A table of measured percentiles to sample from.

    Real sizes don't fit a simple curve. Reads are bimodal: an error message or a whole file
    window. The top end stops at p99.9, so one rare draw can't use up a session's context.
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("no values to fit")
    points = np.percentile(data, [p * 100 for p in DISTRIBUTION_PROBS])
    return {"probs": list(DISTRIBUTION_PROBS), "values": [float(v) for v in points]}


def percentiles(values: list[int]) -> dict[str, float]:
    points = np.percentile(np.asarray(values, dtype=float), PERCENTILES)
    return {f"p{p}": float(v) for p, v in zip(PERCENTILES, points, strict=True)}


def summarize(result: Measurements) -> dict[str, Any]:
    """Everything measured, plus the parameters to put in a trace config."""
    total_outputs = sum(len(v) for v in result.tool_output_tokens.values())
    tools = {
        name: {
            "share": len(values) / total_outputs,
            "n": len(values),
            "output_tokens": percentiles(values),
            "fit": fit_distribution(values),
        }
        for name, values in sorted(result.tool_output_tokens.items())
    }
    return {
        "trajectories": result.trajectories,
        "exit_status": dict(result.exit_status.most_common()),
        "system_tokens": percentiles(result.system_tokens),
        "first_turn_tokens": percentiles(result.first_turn_tokens),
        "steps": percentiles(result.steps),
        "response_tokens": percentiles(result.response_tokens),
        "final_context_tokens": percentiles(result.final_context_tokens),
        "tools": tools,
        "fit": {
            "prefix_tokens": int(np.median(result.system_tokens)),
            "first_turn_tokens": fit_distribution(result.first_turn_tokens),
            "steps": fit_distribution(result.steps),
            "response_tokens": fit_distribution(result.response_tokens),
        },
    }
