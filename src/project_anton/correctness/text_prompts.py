"""Build correctness prompts from real text, as exact-length lists of token IDs.

Random token IDs are unusually sensitive to tiny numeric changes, because the model has no clear
next token. Real text shows whether a wrong restore also changes answers on normal input.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

from project_anton.harness.runlog import append_jsonl, read_jsonl


def build_text_prompts(
    texts: Iterable[str],
    encode: Callable[[str], list[int]],
    lengths: Iterable[int],
    per_length: int,
) -> list[dict]:
    """Return prompts of exact lengths, each from text no other prompt uses.

    Each starts with its own header, because real documents often open with the same boilerplate.
    A shared opening would hit the prefix cache, and "fresh" requests wouldn't be fresh.
    """
    remaining = iter(texts)
    prompts = []
    for length in lengths:
        for i in range(per_length):
            text = f"Session {length}-{i}.\n\n"
            tokens = encode(text)
            while len(tokens) < length:
                try:
                    text += next(remaining) + "\n\n"
                except StopIteration:
                    raise ValueError(f"ran out of text at length {length}, prompt {i}") from None
                tokens = encode(text)
            prompts.append({"length": length, "i": i, "tokens": tokens[:length]})
    return prompts


def write_prompts(path: Path, prompts: list[dict]) -> None:
    for prompt in prompts:
        append_jsonl(path, prompt)


def load_prompts(path: Path) -> dict[tuple[int, int], list[int]]:
    """Read a prompts file, refusing one whose prompts aren't the length they claim."""
    prompts = {}
    for row in read_jsonl(path):
        if len(row["tokens"]) != row["length"]:
            raise ValueError(f"prompt {row['length']}/{row['i']} has {len(row['tokens'])} tokens")
        prompts[(row["length"], row["i"])] = row["tokens"]
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts
