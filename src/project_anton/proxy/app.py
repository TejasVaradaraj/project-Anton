"""An agent-aware proxy that brings a KV cache back to the GPU just before a tool finishes.

Agents tell the proxy when a tool starts. It predicts when the tool will end from past durations
of that tool, then sends the agent's last prompt with max_tokens=1 shortly before. That loads the
cache back from wherever it was offloaded, so the agent's real request finds it on the GPU.
"""

import asyncio
import json
import statistics
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

AGENT_HEADER = "x-agent-id"
MIN_HISTORY = 3  # Too few past durations can't predict anything


@dataclass
class AgentState:
    model: str | None = None
    last_prompt: list[int] | None = None
    tool: str | None = None
    tool_started: float | None = None
    warmup: asyncio.Task | None = None
    warmup_finished: float | None = None
    warmup_sending: bool = False


@dataclass
class Stats:
    tool_starts: int = 0
    no_prediction: int = 0
    warmups_scheduled: int = 0
    warmups_sent: int = 0
    warmups_failed: int = 0
    # The tool ended before its warmup was due, so the warmup was never sent.
    cancelled_tool_ended_first: int = 0
    # The warmup was still running when the agent's real request arrived.
    late_still_running: int = 0
    # Seconds between a warmup finishing and the real request. Long gaps risk re-eviction.
    early_by_s: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k != "early_by_s"}
        out["completed_before_request"] = len(self.early_by_s)
        out["early_by_p50_s"] = statistics.median(self.early_by_s) if self.early_by_s else None
        return out


class Predictor:
    """Median past duration per tool. The median resists the odd very long tool run."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = defaultdict(list)

    def record(self, tool: str, duration_s: float) -> None:
        self.history[tool].append(duration_s)

    def predict(self, tool: str) -> float | None:
        past = self.history[tool]
        return statistics.median(past) if len(past) >= MIN_HISTORY else None


def create_app(upstream: str, lead_s: float = 0.5, prefetch: bool = True) -> FastAPI:
    agents: dict[str, AgentState] = defaultdict(AgentState)
    predictor = Predictor()
    stats = Stats()
    clients: dict[str, httpx.AsyncClient] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        clients["up"] = httpx.AsyncClient(base_url=upstream, timeout=600, limits=limits)
        yield
        await clients["up"].aclose()

    app = FastAPI(lifespan=lifespan)

    async def send_warmup(state: AgentState, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        stats.warmups_sent += 1
        state.warmup_sending = True
        body = {
            "model": state.model,
            "prompt": state.last_prompt,
            "max_tokens": 1,
            "temperature": 0,
        }
        try:
            response = await clients["up"].post("/v1/completions", json=body)
            response.raise_for_status()
            if state.warmup is asyncio.current_task():
                state.warmup_finished = time.monotonic()
        except httpx.HTTPError:
            stats.warmups_failed += 1

    @app.post("/tool_start")
    async def tool_start(request: Request) -> JSONResponse:
        payload = await request.json()
        state = agents[payload["agent_id"]]
        state.tool = payload["tool"]
        state.tool_started = time.monotonic()
        state.warmup_finished = None
        stats.tool_starts += 1
        predicted = predictor.predict(state.tool)
        if prefetch and predicted is not None and state.last_prompt:
            stats.warmups_scheduled += 1
            state.warmup = asyncio.create_task(send_warmup(state, max(0.0, predicted - lead_s)))
        elif predicted is None:
            stats.no_prediction += 1
        return JSONResponse({"predicted_s": predicted})

    def note_request_arrived(state: AgentState) -> None:
        now = time.monotonic()
        if state.tool is not None and state.tool_started is not None:
            predictor.record(state.tool, now - state.tool_started)
        warmup = state.warmup
        if warmup is not None:
            if state.warmup_finished is not None:
                stats.early_by_s.append(now - state.warmup_finished)
            elif not warmup.done():
                # Never cancel one that's sent: aborting a request while its KV cache is loading
                # crashes the vLLM engine (assert req_id in self.requests).
                if state.warmup_sending:
                    stats.late_still_running += 1
                else:
                    stats.cancelled_tool_ended_first += 1
                    warmup.cancel()
        state.tool = state.tool_started = state.warmup = state.warmup_finished = None
        state.warmup_sending = False

    @app.post("/v1/completions")
    async def completions(request: Request) -> StreamingResponse:
        raw = await request.body()
        agent_id = request.headers.get(AGENT_HEADER)
        if agent_id:
            body = json.loads(raw)
            state = agents[agent_id]
            note_request_arrived(state)
            state.model = body["model"]
            state.last_prompt = body["prompt"]

        upstream_request = clients["up"].build_request(
            "POST",
            "/v1/completions",
            content=raw,
            headers={"Content-Type": "application/json", "Accept-Encoding": "identity"},
        )
        response = await clients["up"].send(upstream_request, stream=True)

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(
            relay(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        response = await clients["up"].get("/v1/models")
        return JSONResponse(response.json(), status_code=response.status_code)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/stats")
    async def get_stats() -> JSONResponse:
        return JSONResponse(stats.to_dict())

    return app
