import asyncio

import httpx
import pytest
from fake_vllm import FakeVllm

from project_anton.harness.client import (
    CompletionRequest,
    compute_timing,
    parse_sse_line,
    stream_completion,
)

PROMPT = list(range(1000, 1100))


def run_one(server: FakeVllm, max_tokens: int = 5, prompt=PROMPT, timeout=10.0):
    async def go():
        async with server:
            async with httpx.AsyncClient(base_url=server.base_url, timeout=timeout) as client:
                req = CompletionRequest("fake/model", prompt, max_tokens)
                return await stream_completion(client, req)

    return asyncio.run(go())


# --- Timing arithmetic ---


def test_compute_timing_exact():
    ttft, tpot, e2e = compute_timing(0, [100_000_000, 200_000_000, 300_000_000], 1, 3)
    assert ttft == pytest.approx(0.1)
    assert tpot == pytest.approx(0.1)  # 200 ms across 2 gaps
    assert e2e == pytest.approx(0.3)


def test_compute_timing_excludes_tokens_in_first_chunk():
    # 5 tokens, 3 in the first chunk: 200 ms covers the remaining 2.
    _, tpot, _ = compute_timing(0, [100_000_000, 200_000_000, 300_000_000], 3, 5)
    assert tpot == pytest.approx(0.1)


def test_compute_timing_single_token_has_no_tpot():
    ttft, tpot, _ = compute_timing(0, [100_000_000], 1, 1)
    assert ttft == pytest.approx(0.1)
    assert tpot is None


def test_compute_timing_no_events():
    assert compute_timing(0, [], 1, 0) == (None, None, None)


# --- SSE parsing ---


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("data: {}", "{}"),
        ("data:{}", "{}"),
        ("data: [DONE]", "[DONE]"),
        ("", None),
        (": keep-alive", None),
        ("event: x", None),
    ],
)
def test_parse_sse_line(line, expected):
    assert parse_sse_line(line) == expected


# --- Request shape ---


def test_request_sends_token_ids_and_fixed_output_length():
    server = FakeVllm()
    run_one(server)
    sent = server.requests[0]
    assert sent["prompt"] == PROMPT
    assert sent["stream"] is True and sent["ignore_eos"] is True and sent["temperature"] == 0
    assert sent["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}


# --- Full streams over a real socket ---


def test_success_counts_and_cached_tokens():
    result = run_one(FakeVllm(), max_tokens=5)
    assert result.status == "ok"
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 5
    assert result.cached_tokens == 0
    assert result.finish_reason == "length"
    assert result.token_count_source == "usage"
    assert result.chunk_cum_tokens == [1, 2, 3, 4, 5]


def test_measures_injected_delays():
    # Lower bounds are firm, since sleep never ends early. Upper bounds are loose for busy CI.
    result = run_one(FakeVllm(prefill_s_per_token=0.001, token_delay_s=0.02), max_tokens=6)
    assert 0.1 <= result.ttft_s < 0.5
    assert 0.02 <= result.tpot_s < 0.1
    assert result.e2e_s == pytest.approx(result.chunk_offsets_ns[-1] / 1e9)
    assert result.e2e_s >= result.ttft_s


def test_usage_chunk_and_done_are_not_tokens():
    result = run_one(FakeVllm(token_delay_s=0.01), max_tokens=3)
    assert len(result.chunk_offsets_ns) == 3


def test_late_finish_chunk_without_a_token_does_not_stretch_timing():
    result = run_one(FakeVllm(trailing_finish_chunk_delay_s=0.3), max_tokens=3)
    assert len(result.chunk_offsets_ns) == 3
    assert result.e2e_s < 0.3
    assert result.finish_reason == "length"


def test_empty_text_chunk_with_new_token_still_counts():
    # e.g. half of a multi-byte character: no text yet, but the token was generated.
    result = run_one(FakeVllm(empty_first_text=True, token_delay_s=0.01), max_tokens=3)
    assert len(result.chunk_offsets_ns) == 3
    assert result.chunk_cum_tokens[0] == 1


def test_multi_token_first_chunk_uses_server_count():
    server = FakeVllm(first_chunk_tokens=3, token_delay_s=0.02)
    result = run_one(server, max_tokens=5)
    assert result.completion_tokens == 5
    assert len(result.chunk_offsets_ns) == 3
    # Only 2 tokens came after the first chunk, so the time is divided by 2, not 4.
    gap_s = (result.chunk_offsets_ns[-1] - result.chunk_offsets_ns[0]) / 1e9
    assert result.tpot_s == pytest.approx(gap_s / 2)
    assert result.chunk_cum_tokens == [3, 4, 5]


def test_falls_back_to_chunk_count_without_usage():
    result = run_one(FakeVllm(stream_options_supported=False), max_tokens=4)
    assert result.status == "ok"
    assert result.token_count_source == "chunks"
    assert result.completion_tokens is None
    assert result.tpot_s is not None


def test_cached_tokens_missing_when_server_flag_off():
    result = run_one(FakeVllm(prompt_tokens_details=False))
    assert result.status == "ok" and result.cached_tokens is None


def test_unicode_line_separators_in_generated_text_do_not_break_parsing():
    # A real sweep lost sessions to this, since httpx's aiter_lines splits on these characters.
    result = run_one(FakeVllm(line_separator_text=True), max_tokens=4)
    assert result.status == "ok"
    assert result.completion_tokens == 4
    assert len(result.chunk_offsets_ns) == 4


# --- Failures come back as results ---


def test_http_error():
    result = run_one(FakeVllm(fail_mode="http_500"))
    assert result.status == "http_error"
    assert result.http_status == 500
    assert "boom" in result.error
    assert result.ttft_s is None


def test_error_inside_200_stream_is_not_ok():
    result = run_one(FakeVllm(fail_mode="stream_error"))
    assert result.status == "stream_error"
    assert result.http_status == 200
    assert "engine died" in result.error


def test_truncated_stream_is_not_ok():
    result = run_one(FakeVllm(fail_mode="truncate"), max_tokens=10)
    assert result.status == "truncated"


def test_connection_refused_is_reported():
    async def go():
        transport = httpx.AsyncHTTPTransport()
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:1", transport=transport, timeout=1.0
        ) as client:
            return await stream_completion(client, CompletionRequest("m", PROMPT, 5))

    result = asyncio.run(go())
    assert result.status == "transport_error"
    assert "ConnectError" in result.error


def test_read_timeout_is_reported():
    server = FakeVllm(prefill_s_per_token=0.01)
    result = run_one(server, timeout=0.05)
    assert result.status == "transport_error"
    assert "Timeout" in result.error


def test_result_serializes_to_json():
    import json

    assert json.loads(json.dumps(run_one(FakeVllm()).to_dict()))["status"] == "ok"
