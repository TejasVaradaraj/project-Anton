"""KV cache size arithmetic.

The KV cache is the model's saved work for each token it has read. Its size per
token depends only on the model's shape and number format, so these are
calculations, not measurements.
"""

from dataclasses import dataclass

GIB = 2**30


@dataclass(frozen=True)
class ModelShape:
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # bf16 and fp16 both use 2 bytes per number


def kv_bytes_per_token(shape: ModelShape) -> int:
    """Bytes per token: one key and one value (the leading 2), per layer and head."""
    return 2 * shape.layers * shape.kv_heads * shape.head_dim * shape.dtype_bytes


def cache_bytes(tokens: int, shape: ModelShape) -> int:
    return tokens * kv_bytes_per_token(shape)


def max_concurrency(cache_tokens: int, context_tokens: int) -> float:
    """How many full-length contexts fit at once. A capacity limit, not a load test."""
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    return cache_tokens / context_tokens


def break_even_bandwidth(shape: ModelShape, prefill_tokens_per_s: float) -> float:
    """Storage bytes/s at which loading a KV cache takes as long as recomputing it.

    Faster than this and loading wins. It ignores storage latency and transfer
    overhead, so treat it as a lower bound on the storage speed needed.
    """
    if prefill_tokens_per_s <= 0:
        raise ValueError("prefill_tokens_per_s must be positive")
    return kv_bytes_per_token(shape) * prefill_tokens_per_s
