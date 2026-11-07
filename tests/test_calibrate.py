import pytest

from project_anton.traces.calibrate import (
    command_of,
    fit_distribution,
    measure,
    percentiles,
    summarize,
    tool_of,
)


def count_words(texts: list[str]) -> list[int]:
    return [len(t.split()) for t in texts]


def run(command: str, extra: str = "thinking") -> str:
    return f"{extra}\n```\n{command}\n```"


def trajectory(*messages, exit_status="submitted", system="one two three") -> dict:
    return {
        "exit_status": exit_status,
        "trajectory": [
            {"role": "system", "system_prompt": system, "text": None},
            *[{"role": role, "text": text} for role, text in messages],
        ],
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (run("open foo.py"), "open foo.py"),
        ("```\nfirst\n```\nthen\n```\nsecond one\n```", "second one"),
        ("```bash\nls -la\n```", "ls -la"),
        ("no fence here", None),
        ("```\n\n```", None),
        (None, None),
    ],
)
def test_command_of(text, expected):
    assert command_of(text) == expected


@pytest.mark.parametrize(
    ("command", "tool"),
    [
        ("open a.py", "read"),
        ("scroll_down", "read"),
        ("search_dir foo", "search"),
        ("find.", "search"),  # Real data has spaces stripped like this
        ("ls..", "search"),
        ("edit 1:2", "edit"),
        ("mkdir.git", "edit"),
        ("python a.py", "run"),
        ("pytest", "run"),
        ("./reproduce.sh", "run"),
        ("khal list", "run"),
        ("submit", None),
        (None, None),
    ],
)
def test_tool_of(command, tool):
    assert tool_of(command) == tool


def test_measure_attributes_output_to_the_command_before_it():
    rows = [
        trajectory(
            ("user", "the task words"),
            ("ai", run("open a")),
            ("user", "file text of five words"),
            ("ai", run("python a")),
            ("user", "ok"),
            ("ai", run("submit")),
            ("user", "ignored output after submit"),
        )
    ]
    result = measure(rows, count_words)
    assert result.trajectories == 1
    assert result.system_tokens == [3]
    assert result.first_turn_tokens == [3]
    assert result.steps == [3]
    assert result.tool_output_tokens["read"] == [5]
    assert result.tool_output_tokens["run"] == [1]
    assert "submit" not in result.tool_output_tokens
    # 3 fence markers plus words in each response: "thinking", "```", command..., "```".
    assert len(result.response_tokens) == 3
    assert result.final_context_tokens == [
        3
        + 3
        + result.response_tokens[0]
        + 5
        + result.response_tokens[1]
        + 1
        + result.response_tokens[2]
        + 4
    ]


def test_response_with_no_command_has_no_tool_output():
    rows = [trajectory(("user", "task"), ("ai", "just talk"), ("user", "reminder"))]
    result = measure(rows, count_words)
    assert result.steps == [1]
    assert result.tool_output_tokens == {}


@pytest.mark.parametrize(
    "row",
    [
        {"exit_status": "x", "trajectory": []},
        trajectory(("user", "task only")),
        trajectory(("ai", "starts with ai"), ("user", "then user")),
    ],
)
def test_malformed_rows_are_skipped(row):
    assert measure([row], count_words).trajectories == 0


def test_fit_keeps_a_bimodal_shape():
    # Like real reads: many short errors and many full file windows, nothing in between.
    values = [50] * 300 + [1100] * 700
    fit = fit_distribution(values)
    by_prob = dict(zip(fit["probs"], fit["values"], strict=True))
    assert by_prob[0.25] == 50
    assert by_prob[0.5] == 1100


def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        fit_distribution([])


def test_fit_tops_out_below_the_extreme_maximum():
    fit = fit_distribution(list(range(1, 10_001)) + [10_000_000])
    assert fit["values"][-1] < 100_000


def test_percentiles_keys():
    assert percentiles([1, 2, 3])["p50"] == 2


def test_summarize_shares_sum_to_one_and_fit_is_present():
    rows = [
        trajectory(
            ("user", "task words here"),
            ("ai", run("open a")),
            ("user", "some output"),
            ("ai", run("python a")),
            ("user", "more output here"),
        ),
    ] * 3
    summary = summarize(measure(rows, count_words))
    assert summary["trajectories"] == 3
    assert sum(t["share"] for t in summary["tools"].values()) == pytest.approx(1)
    assert summary["fit"]["prefix_tokens"] == 3
    assert set(summary["fit"]) == {"prefix_tokens", "first_turn_tokens", "steps", "response_tokens"}
    assert summary["exit_status"] == {"submitted": 3}
