import pytest

from project_anton.analysis.kv_math import (
    GIB,
    ModelShape,
    break_even_bandwidth,
    cache_bytes,
    kv_bytes_per_token,
    max_concurrency,
)

KIB = 1024
LLAMA_3B = ModelShape(layers=28, kv_heads=8, head_dim=128)
LLAMA_8B = ModelShape(layers=32, kv_heads=8, head_dim=128)
LLAMA_70B = ModelShape(layers=80, kv_heads=8, head_dim=128)


@pytest.mark.parametrize(("shape", "kib"), [(LLAMA_3B, 112), (LLAMA_8B, 128), (LLAMA_70B, 320)])
def test_bytes_per_token_matches_plan(shape, kib):
    assert kv_bytes_per_token(shape) == kib * KIB


def test_matches_vllm_log_from_real_a100_run():
    """Pinned to the numbers vLLM reported on the Colab A100 run (see SETUP.md)."""
    assert cache_bytes(461_216, LLAMA_8B) / GIB == pytest.approx(56.3, abs=0.01)
    assert max_concurrency(461_216, 32_768) == pytest.approx(14.08, abs=0.005)


def test_break_even_bandwidth():
    # 128 KiB/token at 10,000 tokens/s is 1,280,000 KiB/s.
    assert break_even_bandwidth(LLAMA_8B, 10_000) == 128 * KIB * 10_000


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_inputs(bad):
    with pytest.raises(ValueError):
        max_concurrency(100, bad)
    with pytest.raises(ValueError):
        break_even_bandwidth(LLAMA_8B, bad)
