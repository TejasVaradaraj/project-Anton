"""A small fake vLLM server on a real socket, with a cost model we control.

Tests can check that measured numbers match the delays put in here. Its output is
test data and must never be reported as a performance measurement.
"""

import asyncio
import json
from dataclasses import dataclass, field

BLOCK_SIZE = 16  # vLLM caches prefixes in whole blocks


@dataclass
class FakeVllm:
    prefill_s_per_token: float = 0.0
    token_delay_s: float = 0.0
    model: str = "fake/model"
    max_model_len: int = 32768
    stream_options_supported: bool = True
    fail_mode: str | None = None  # http_500 | stream_error | truncate
    empty_first_text: bool = False
    first_chunk_tokens: int = 1
    trailing_finish_chunk_delay_s: float | None = None
    prompt_tokens_details: bool = True
    external_hits: float = 0.0
    capacity_blocks: int | None = None  # When set, oldest blocks are evicted beyond this
    offload_load_s_per_token: float | None = None  # When set, evicted blocks load at this cost
    requests: list[dict] = field(default_factory=list)
    _cached: set = field(default_factory=set)
    _offloaded: set = field(default_factory=set)
    _order: list = field(default_factory=list)
    _writers: set = field(default_factory=set)
    _server: asyncio.Server | None = None

    async def __aenter__(self) -> "FakeVllm":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc) -> None:
        self._server.close()
        # Close open connections by hand: Server.close_clients() needs Python 3.13.
        for writer in list(self._writers):
            writer.close()
        await self._server.wait_closed()

    @property
    def base_url(self) -> str:
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def _handle(self, reader, writer) -> None:
        self._writers.add(writer)
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode().split("\r\n")
                method, path, _ = lines[0].split(" ")
                pairs = (x.split(":", 1) for x in lines[1:] if x)
                headers = {k.lower(): v.strip() for k, v in pairs}
                length = int(headers.get("content-length", 0))
                body = await reader.readexactly(length) if length else b""
                if method == "GET" and path == "/health":
                    self._send_json(writer, 200, {})
                elif method == "GET" and path == "/v1/models":
                    data = [{"id": self.model, "max_model_len": self.max_model_len}]
                    self._send_json(writer, 200, {"data": data})
                elif method == "GET" and path == "/metrics":
                    text = (
                        "# HELP vllm:kv_cache_usage_perc usage\n"
                        f'vllm:kv_cache_usage_perc{{model_name="{self.model}"}} 0.25\n'
                        f"vllm:num_requests_running {len(self.requests)}.0\n"
                        "vllm:external_prefix_cache_hits_total "
                        f"{self.external_hits}\n"
                        "python_gc_objects_collected_total 7.0\n"
                    ).encode()
                    writer.write(
                        f"HTTP/1.1 200 OK\r\nContent-Length: {len(text)}\r\n\r\n".encode() + text
                    )
                elif method == "POST" and path == "/v1/completions":
                    await self._completions(writer, json.loads(body))
                else:
                    self._send_json(writer, 404, {"error": "not found"})
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self._writers.discard(writer)
            writer.close()

    def _send_json(self, writer, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        writer.write(
            f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )

    async def _send_chunk(self, writer, obj: dict | str) -> None:
        text = obj if isinstance(obj, str) else json.dumps(obj)
        line = f"data: {text}\n\n".encode()
        writer.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
        await writer.drain()

    @staticmethod
    def _block_hashes(prompt: list[int]) -> list[int]:
        """One chained hash per whole block, like vLLM. Linear time, so that the fake's
        own CPU use doesn't show up as extra TTFT on long prompts."""
        hashes, current = [], 0
        for end in range(BLOCK_SIZE, len(prompt) + 1, BLOCK_SIZE):
            current = hash((current, tuple(prompt[end - BLOCK_SIZE : end])))
            hashes.append(current)
        return hashes

    async def _completions(self, writer, req: dict) -> None:
        self.requests.append(req)
        if self.fail_mode == "http_500":
            self._send_json(writer, 500, {"error": "boom"})
            return

        prompt = req["prompt"]
        max_tokens = req["max_tokens"]
        hashes = self._block_hashes(prompt)
        hits = 0
        for block_hash in hashes:
            if block_hash not in self._cached:
                break
            hits += 1
        cached = hits * BLOCK_SIZE
        load_s = 0.0
        if self.offload_load_s_per_token is not None and hits == 0 and hashes[0] in self._offloaded:
            restored = sum(1 for h in hashes if h in self._offloaded)
            cached = restored * BLOCK_SIZE
            load_s = cached * self.offload_load_s_per_token
            self.external_hits += cached
        self._offloaded.update(hashes)
        self._cached.update(hashes)
        self._order.extend(hashes)
        if self.capacity_blocks is not None:
            while len(self._cached) > self.capacity_blocks and self._order:
                self._cached.discard(self._order.pop(0))

        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )
        await writer.drain()
        await asyncio.sleep((len(prompt) - cached) * self.prefill_s_per_token + load_s)

        if self.fail_mode == "stream_error":
            await self._send_chunk(writer, {"object": "error", "message": "engine died"})
            await self._send_chunk(writer, "[DONE]")
            writer.write(b"0\r\n\r\n")
            return

        usage_opts = req.get("stream_options") or {}
        continuous = self.stream_options_supported and usage_opts.get("continuous_usage_stats")
        include = self.stream_options_supported and usage_opts.get("include_usage")

        sent = 0
        while sent < max_tokens:
            step = self.first_chunk_tokens if sent == 0 else 1
            step = min(step, max_tokens - sent)
            sent += step
            text = "" if (self.empty_first_text and sent == step) else "x" * step
            chunk = {
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "length" if sent == max_tokens else None,
                    }
                ]
            }
            if continuous:
                chunk["usage"] = {"prompt_tokens": len(prompt), "completion_tokens": sent}
            await self._send_chunk(writer, chunk)
            if self.fail_mode == "truncate" and sent >= max_tokens // 2:
                writer.write(b"0\r\n\r\n")
                return
            if sent < max_tokens:
                await asyncio.sleep(self.token_delay_s)

        if self.trailing_finish_chunk_delay_s is not None:
            await asyncio.sleep(self.trailing_finish_chunk_delay_s)
            chunk = {"choices": [{"index": 0, "text": "", "finish_reason": "length"}]}
            if continuous:
                chunk["usage"] = {"prompt_tokens": len(prompt), "completion_tokens": sent}
            await self._send_chunk(writer, chunk)

        if include:
            usage = {"prompt_tokens": len(prompt), "completion_tokens": sent}
            if self.prompt_tokens_details:
                usage["prompt_tokens_details"] = {"cached_tokens": cached}
            await self._send_chunk(writer, {"choices": [], "usage": usage})
        await self._send_chunk(writer, "[DONE]")
        writer.write(b"0\r\n\r\n")
