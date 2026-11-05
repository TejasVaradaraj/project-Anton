"""Build prompts as exact-length lists of token IDs.

Sending token IDs instead of text means the server sees exactly the length we
asked for, with no tokenizer in the way. The IDs are random, so two prompts with
different keys share no prefix and can't hit each other's prefix cache.
"""

import hashlib

import numpy as np


def make_prompt(length: int, token_id_range: tuple[int, int], key: str) -> list[int]:
    """Return `length` token IDs in [low, high). The same key gives the same prompt."""
    low, high = token_id_range
    if length <= 0:
        raise ValueError("length must be positive")
    if not 0 <= low < high:
        raise ValueError(f"bad token_id_range: {token_id_range}")
    # sha256, not hash(): Python's hash() of a string changes on every run.
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).integers(low, high, size=length).tolist()


def prompt_digest(token_ids: list[int]) -> str:
    """Short fingerprint saved with each result, so runs can be compared later."""
    return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()[:16]
