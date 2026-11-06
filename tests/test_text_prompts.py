import pytest

from project_anton.correctness.text_prompts import (
    build_text_prompts,
    load_prompts,
    write_prompts,
)


def encode(text: str) -> list[int]:
    return [hash_word(w) for w in text.split()]


def hash_word(word: str) -> int:
    return sum(map(ord, word))


def test_prompts_have_exact_lengths():
    prompts = build_text_prompts(["a b c d e f"] * 20, encode, [3, 8], 2)
    assert [(p["length"], p["i"]) for p in prompts] == [(3, 0), (3, 1), (8, 0), (8, 1)]
    assert all(len(p["tokens"]) == p["length"] for p in prompts)


def test_prompts_share_no_opening_even_with_identical_text():
    # Real trajectories open with the same system prompt, which would hit the prefix cache.
    prompts = build_text_prompts(["same text here"] * 20, lambda t: t.split(), [6], 3)
    assert len({tuple(p["tokens"][:2]) for p in prompts}) == 3


def test_each_prompt_uses_new_text():
    prompts = build_text_prompts(["one two", "three four", "five six"], lambda t: t.split(), [4], 2)
    assert "one" in prompts[0]["tokens"]
    assert "one" not in prompts[1]["tokens"]
    assert "three" in prompts[1]["tokens"]


def test_running_out_of_text_is_an_error():
    with pytest.raises(ValueError, match="ran out"):
        build_text_prompts(["short"], encode, [100], 1)


def test_prompts_file_round_trip(tmp_path):
    path = tmp_path / "p.jsonl"
    write_prompts(path, [{"length": 2, "i": 0, "tokens": [5, 6]}])
    assert load_prompts(path) == {(2, 0): [5, 6]}


def test_prompt_of_wrong_length_is_refused(tmp_path):
    path = tmp_path / "p.jsonl"
    write_prompts(path, [{"length": 3, "i": 0, "tokens": [5, 6]}])
    with pytest.raises(ValueError, match="2 tokens"):
        load_prompts(path)


def test_empty_prompts_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no prompts"):
        load_prompts(tmp_path / "missing.jsonl")
