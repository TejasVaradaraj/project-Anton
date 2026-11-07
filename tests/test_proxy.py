"""The prefetch proxy on real sockets, between the replayer and the fake server."""

import asyncio
import socket

import httpx
import pytest
import uvicorn
from fake_vllm import FakeVllm
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from project_anton.analysis.kv_math import ModelShape
from project_anton.harness.replay import run_replay
from project_anton.harness.runlog import read_jsonl
from project_anton.harness.server import ServerConfig
from project_anton.proxy.app import MIN_HISTORY, Predictor, create_app
from project_anton.traces.generators import TraceConfig

SHAPE = ModelShape(layers=32, kv_heads=8, head_dim=128)
PARAMS = {
    "prefix_tokens": 320,
    "first_turn_tokens": [32, 64],
    "steps": [8, 8],
    "tool_output_tokens": [32, 64],
    "response_tokens": [2, 3],
    "tools": {"read": {"weight": 1.0, "median_s": 0.2, "sigma": 0.01}},
}
TRACE = TraceConfig("t", "agent", sessions=1, arrival_rate_per_s=10.0, params=PARAMS)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RunningProxy:
    def __init__(self, upstream: str, **kwargs) -> None:
        self.port = free_port()
        config = uvicorn.Config(
            create_app(upstream, **kwargs), host="127.0.0.1", port=self.port, log_level="error"
        )
        self.server = uvicorn.Server(config)
        self.url = f"http://127.0.0.1:{self.port}"

    async def __aenter__(self) -> "RunningProxy":
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.01)
        return self

    async def __aexit__(self, *exc) -> None:
        self.server.should_exit = True
        await self.task


def replay_through_proxy(tmp_path, fake: FakeVllm, trace=TRACE, **proxy_kwargs):
    async def go():
        async with fake:
            port = int(fake.base_url.rsplit(":", 1)[1])
            cfg = ServerConfig("fake", fake.model, port, {}, (0, 128_000), SHAPE)
            async with RunningProxy(fake.base_url, **proxy_kwargs) as proxy:
                await run_replay(cfg, trace, "r", tmp_path, log=lambda _: None, proxy_url=proxy.url)
                async with httpx.AsyncClient(base_url=proxy.url) as client:
                    return (await client.get("/stats")).json()

    stats = asyncio.run(go())
    return stats, read_jsonl(tmp_path / "raw" / "requests.jsonl")


def test_predictor_needs_history_then_uses_the_median():
    predictor = Predictor()
    for _ in range(MIN_HISTORY - 1):
        predictor.record("read", 1.0)
    assert predictor.predict("read") is None
    predictor.record("read", 100.0)
    assert predictor.predict("read") == 1.0
    assert predictor.predict("other") is None


def test_requests_pass_through_unchanged_and_complete(tmp_path):
    fake = FakeVllm()
    _, records = replay_through_proxy(tmp_path, fake, prefetch=False)
    assert len(records) == 8
    assert not any(r["invalid_reasons"] for r in records)
    assert all(r["ttft_s"] is not None for r in records)


def test_no_prefetch_mode_sends_no_extra_requests(tmp_path):
    fake = FakeVllm()
    stats, records = replay_through_proxy(tmp_path, fake, prefetch=False)
    assert stats["tool_starts"] == 7
    assert stats["warmups_sent"] == 0
    assert len(fake.requests) == len(records)


def test_prefetch_starts_once_history_exists_and_sends_short_requests(tmp_path):
    fake = FakeVllm()
    stats, records = replay_through_proxy(tmp_path, fake, lead_s=0.05)
    assert stats["tool_starts"] == 7
    # The first tool runs build history, so later ones get a warmup.
    assert stats["no_prediction"] == MIN_HISTORY
    assert stats["warmups_scheduled"] == 7 - MIN_HISTORY
    assert stats["warmups_sent"] >= 1
    warmups = [q for q in fake.requests if q["max_tokens"] == 1]
    assert len(warmups) == stats["warmups_sent"]
    assert stats["warmups_failed"] == 0
    assert not any(r["invalid_reasons"] for r in records)


def test_warmup_prompt_is_a_prefix_of_the_next_real_prompt(tmp_path):
    fake = FakeVllm()
    replay_through_proxy(tmp_path, fake, lead_s=0.05)
    real = [q for q in fake.requests if q["max_tokens"] != 1]
    warmups = [q for q in fake.requests if q["max_tokens"] == 1]
    assert warmups
    for warmup in warmups:
        size = len(warmup["prompt"])
        assert any(q["prompt"][:size] == warmup["prompt"] and len(q["prompt"]) > size for q in real)


def test_warmup_restores_evicted_cache_before_the_real_request(tmp_path):
    """A small cache, with a competing session, so prompts are evicted during tool pauses."""
    trace = TraceConfig("t", "agent", sessions=4, arrival_rate_per_s=20.0, params=PARAMS)

    def run(folder, **kwargs):
        fake = FakeVllm(
            prefill_s_per_token=20e-6, capacity_blocks=60, offload_load_s_per_token=2e-6
        )
        stats, records = replay_through_proxy(folder, fake, trace=trace, **kwargs)
        return stats, fake, records

    stats, fake, _ = run(tmp_path / "with", lead_s=0.05)
    assert stats["warmups_sent"] > 0
    assert fake.external_hits > 0


def test_unknown_upstream_failure_is_counted_not_fatal(tmp_path):
    fake = FakeVllm(fail_mode="http_500")

    async def go():
        async with fake:
            async with RunningProxy(fake.base_url, prefetch=False) as proxy:
                async with httpx.AsyncClient(base_url=proxy.url) as client:
                    response = await client.post(
                        "/v1/completions", json={"model": "m", "prompt": [1], "max_tokens": 1}
                    )
                    return response.status_code

    assert asyncio.run(go()) == 500


@pytest.mark.parametrize("path", ["/health", "/v1/models", "/stats"])
def test_helper_endpoints_respond(path):
    async def go():
        async with FakeVllm() as fake:
            async with RunningProxy(fake.base_url) as proxy:
                async with httpx.AsyncClient(base_url=proxy.url) as client:
                    return (await client.get(path)).status_code

    assert asyncio.run(go()) == 200


def test_real_request_does_not_abort_a_warmup_in_flight():
    # Aborting a request while its KV cache loads crashed the vLLM engine in a real run.
    warmup_ended = []
    app = FastAPI()

    @app.post("/v1/completions")
    async def completions(request: Request) -> JSONResponse:
        body = await request.json()
        if body["max_tokens"] == 1:
            await asyncio.sleep(0.4)
            warmup_ended.append(await request.is_disconnected())
        return JSONResponse({"choices": [{"text": "x"}]})

    async def go():
        port = free_port()
        upstream = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
        upstream_task = asyncio.create_task(upstream.serve())
        while not upstream.started:
            await asyncio.sleep(0.01)
        async with RunningProxy(f"http://127.0.0.1:{port}") as proxy:
            async with httpx.AsyncClient(base_url=proxy.url) as client:
                body = {"model": "m", "prompt": [1, 2], "max_tokens": 4}
                headers = {"X-Agent-Id": "a"}
                for _ in range(MIN_HISTORY + 1):
                    await client.post("/tool_start", json={"agent_id": "a", "tool": "read"})
                    await asyncio.sleep(0.02)
                    await client.post("/v1/completions", json=body, headers=headers)
                await asyncio.sleep(0.6)
                stats = (await client.get("/stats")).json()
        upstream.should_exit = True
        await upstream_task
        return stats

    stats = asyncio.run(go())
    assert stats["late_still_running"] >= 1
    assert warmup_ended
    assert not any(warmup_ended)
