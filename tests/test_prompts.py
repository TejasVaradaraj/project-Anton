import pytest

from project_anton.harness.prompts import make_prompt, prompt_digest

RANGE = (0, 128_000)


def test_exact_length_and_range():
    prompt = make_prompt(16_000, RANGE, "a")
    assert len(prompt) == 16_000
    assert min(prompt) >= 0 and max(prompt) < 128_000


def test_ids_are_plain_ints_for_json():
    assert type(make_prompt(3, RANGE, "a")[0]) is int


def test_same_key_is_identical():
    assert make_prompt(100, RANGE, "a") == make_prompt(100, RANGE, "a")


def test_different_keys_share_no_cacheable_prefix():
    # vLLM's prefix cache matches from the first block, so a different start means a miss.
    assert make_prompt(100, RANGE, "a")[:16] != make_prompt(100, RANGE, "b")[:16]


def test_digest_tracks_content():
    a, b = make_prompt(100, RANGE, "a"), make_prompt(100, RANGE, "b")
    assert prompt_digest(a) == prompt_digest(list(a))
    assert prompt_digest(a) != prompt_digest(b)


@pytest.mark.parametrize(
    ("length", "token_range"), [(0, RANGE), (-1, RANGE), (5, (10, 10)), (5, (-1, 10))]
)
def test_rejects_bad_input(length, token_range):
    with pytest.raises(ValueError):
        make_prompt(length, token_range, "a")
