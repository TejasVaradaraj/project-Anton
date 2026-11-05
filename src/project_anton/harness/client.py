"""Streaming client for vLLM's /v1/completions that measures TTFT and TPOT.

TTFT (time to first token): send to first generated token. Mostly prefill, the
work of reading the prompt.
TPOT (time per output token): average gap between tokens after the first.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

Clock = Callable[[], int]
ERROR_BODY_LIMIT = 500


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    prompt_token_ids: list[int]
    max_tokens: int
    seed: int = 0

    def body(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": self.prompt_token_ids,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "seed": self.seed,
            # Generate exactly max_tokens, so every run does the same work.
            "ignore_eos": True,
            "stream": True,
            # Per-chunk token counts: a chunk can hold 0, 1 or several tokens.
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }


@dataclass
class StreamResult:
    status: str  # ok | http_error | stream_error | transport_error | truncated
    send_wall_time_s: float
    error: str | None = None
    http_status: int | None = None
    ttft_s: float | None = None
    tpot_s: float | None = None
    e2e_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None  # needs --enable-prompt-tokens-details
    finish_reason: str | None = None
    token_count_source: str | None = None  # usage | chunks
    server_metrics: dict[str, Any] | None = None
    chunk_offsets_ns: list[int] = field(default_factory=list)
    chunk_cum_tokens: list[int | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_timing(
    send_ns: int,
    event_times_ns: list[int],
    first_event_tokens: int,
    total_tokens: int | None,
) -> tuple[float | None, float | None, float | None]:
    """Return (ttft_s, tpot_s, e2e_s) from the times of chunks that carried tokens.

    TPOT skips the tokens in the first chunk, since those arrive with TTFT and have
    no gap to measure. It is None when no gap exists, such as for a single token.
    """
    if not event_times_ns:
        return None, None, None
    ttft = (event_times_ns[0] - send_ns) / 1e9
    e2e = (event_times_ns[-1] - send_ns) / 1e9
    tpot = None
    if total_tokens is not None and total_tokens > first_event_tokens:
        tpot = (event_times_ns[-1] - event_times_ns[0]) / 1e9 / (total_tokens - first_event_tokens)
    return ttft, tpot, e2e


def parse_sse_line(line: str) -> str | None:
    """Return the payload of an SSE `data:` line, or None for anything else."""
    if not line.startswith("data:"):
        return None
    return line[len("data:") :].strip()


async def stream_completion(
    client: httpx.AsyncClient,
    request: CompletionRequest,
    clock: Clock = time.perf_counter_ns,
) -> StreamResult:
    """Send one streaming request. Failures come back in the result, never as exceptions,
    so one bad request can't end a long sweep."""
    # Serialize first: 16k token IDs take real time to encode, and that's client
    # overhead the server's TTFT shouldn't include.
    payload = json.dumps(request.body()).encode()
    headers = {
        "Content-Type": "application/json",
        # Compression can hold chunks back in a buffer, which would distort timing.
        "Accept-Encoding": "identity",
    }

    wall_time = time.time()
    event_times: list[int] = []
    cum_tokens: list[int | None] = []
    usage: dict[str, Any] = {}
    finish_reason = None
    server_metrics = None
    last_cum = 0
    saw_done = False

    send_ns = clock()
    result = StreamResult(status="ok", send_wall_time_s=wall_time)
    try:
        async with client.stream(
            "POST", "/v1/completions", content=payload, headers=headers
        ) as response:
            result.http_status = response.status_code
            if response.status_code != 200:
                body = await response.aread()
                result.status = "http_error"
                result.error = body.decode(errors="replace")[:ERROR_BODY_LIMIT]
                return result

            async for line in response.aiter_lines():
                # Take the time before parsing, so JSON decoding isn't counted.
                now = clock()
                data = parse_sse_line(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    saw_done = True
                    break
                chunk = json.loads(data)
                # vLLM reports generation failures inside a 200 stream.
                if "error" in chunk or chunk.get("object") == "error":
                    result.status = "stream_error"
                    result.error = data[:ERROR_BODY_LIMIT]
                    break

                chunk_usage = chunk.get("usage") or {}
                usage.update(chunk_usage)
                if chunk.get("metrics"):
                    server_metrics = chunk["metrics"]

                # The closing usage chunk has no choices: it's a summary, not a token.
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason

                cum = chunk_usage.get("completion_tokens")
                # Text can be empty while a token still arrived (half of a multi-byte
                # character), and a finish chunk can have neither.
                has_new_token = bool(choice.get("text")) or (cum is not None and cum > last_cum)
                if cum is not None:
                    last_cum = max(last_cum, cum)
                if has_new_token:
                    event_times.append(now)
                    cum_tokens.append(cum)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        result.status = "transport_error"
        result.error = f"{type(exc).__name__}: {exc}"[:ERROR_BODY_LIMIT]

    if result.status == "ok" and not saw_done:
        result.status = "truncated"
        result.error = "stream ended without [DONE]"

    result.prompt_tokens = usage.get("prompt_tokens")
    result.completion_tokens = usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details") or {}
    result.cached_tokens = details.get("cached_tokens")
    result.finish_reason = finish_reason
    result.server_metrics = server_metrics
    result.chunk_offsets_ns = [t - send_ns for t in event_times]
    result.chunk_cum_tokens = cum_tokens

    if result.completion_tokens is not None:
        total, result.token_count_source = result.completion_tokens, "usage"
    else:
        # Weaker fallback: this assumes one token per chunk.
        total, result.token_count_source = len(event_times), "chunks"
    first_tokens = cum_tokens[0] if cum_tokens and cum_tokens[0] else 1
    result.ttft_s, result.tpot_s, result.e2e_s = compute_timing(
        send_ns, event_times, first_tokens, total
    )
    return result
