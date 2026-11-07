import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from project_anton.traces.generators import (
    TraceConfig,
    _draw,
    generate_trace,
    load_trace_config,
    session_prompts,
)

RANGE = (0, 128_000)
AGENT = TraceConfig("t", "agent", sessions=50, arrival_rate_per_s=2.0)


def test_same_seed_gives_identical_trace():
    assert generate_trace(AGENT) == generate_trace(AGENT)


def test_different_seed_differs():
    assert generate_trace(AGENT) != generate_trace(replace(AGENT, seed=1))


def test_more_sessions_keeps_earlier_ones_identical():
    small = generate_trace(replace(AGENT, sessions=10))
    assert generate_trace(AGENT)[:10] == small


def test_agent_shape_matches_plan():
    for session in generate_trace(AGENT):
        assert session.prefix_tokens == 3000
        assert len(session.turns) <= 40
        first, *rest = session.turns
        assert first.tool is None and first.pause_before_s == 0
        for turn in rest:
            assert 500 <= turn.new_tokens <= 3000
            assert 50 <= turn.response_tokens <= 300
            assert turn.tool in {"read_file", "grep", "run_tests"}
            assert turn.pause_before_s > 0


def test_tool_pauses_differ_by_tool():
    trace = generate_trace(replace(AGENT, sessions=200))
    pauses: dict[str, list[float]] = {}
    for session in trace:
        for turn in session.turns[1:]:
            pauses.setdefault(turn.tool, []).append(turn.pause_before_s)
    assert np.median(pauses["run_tests"]) == pytest.approx(20, rel=0.2)
    assert np.median(pauses["read_file"]) == pytest.approx(0.5, rel=0.2)


def test_poisson_arrival_rate():
    trace = generate_trace(replace(AGENT, sessions=2000))
    gaps = np.diff([s.arrival_s for s in trace])
    assert (gaps >= 0).all()
    assert gaps.mean() == pytest.approx(0.5, rel=0.1)
    # In a Poisson process the gaps vary about as much as their mean.
    assert gaps.std() == pytest.approx(gaps.mean(), rel=0.15)


@pytest.mark.parametrize("kind", ["agent", "multi_turn", "long_context"])
def test_every_kind_stays_within_context_limit(kind):
    cfg = TraceConfig("t", kind, sessions=100, arrival_rate_per_s=1.0, max_context_tokens=8000)
    trace = generate_trace(cfg)
    assert trace
    for session in trace:
        total = session.prefix_tokens + sum(t.new_tokens + t.response_tokens for t in session.turns)
        assert total <= 8000


def test_pause_scale_shortens_pauses_only():
    full = generate_trace(replace(AGENT, sessions=3))
    fast = generate_trace(replace(AGENT, sessions=3, pause_scale=0.1))
    assert fast[0].turns[1].pause_before_s == pytest.approx(full[0].turns[1].pause_before_s * 0.1)
    assert fast[0].turns[1].new_tokens == full[0].turns[1].new_tokens


def test_params_override_defaults():
    trace = generate_trace(replace(AGENT, sessions=3, params={"prefix_tokens": 100}))
    assert trace[0].prefix_tokens == 100


def test_context_too_small_for_any_turn_gives_no_session():
    assert generate_trace(replace(AGENT, max_context_tokens=100)) == []


@pytest.mark.parametrize("bad", [{"kind": "nope"}, {"arrival_rate_per_s": 0}], ids=["kind", "rate"])
def test_rejects_bad_config(bad):
    with pytest.raises(ValueError):
        generate_trace(replace(AGENT, **bad))


# --- Prompts ---


def test_each_prompt_extends_the_last_with_trace_response_tokens():
    session = generate_trace(AGENT)[0]
    prompts = list(session_prompts(session, RANGE, "k"))
    assert len(prompts) == len(session.turns)
    for earlier, later, turn_before, turn in zip(
        prompts, prompts[1:], session.turns, session.turns[1:], strict=False
    ):
        assert later[: len(earlier)] == earlier
        assert len(later) == len(earlier) + turn_before.response_tokens + turn.new_tokens


def test_sessions_share_prefix_then_diverge():
    a, b = generate_trace(AGENT)[:2]
    first_a = next(session_prompts(a, RANGE, "k"))
    first_b = next(session_prompts(b, RANGE, "k"))
    assert first_a[:3000] == first_b[:3000]
    assert first_a[3000:3016] != first_b[3000:3016]


def test_new_key_changes_the_prefix():
    session = generate_trace(AGENT)[0]
    assert (
        next(session_prompts(session, RANGE, "a"))[:16]
        != next(session_prompts(session, RANGE, "b"))[:16]
    )


def test_retry_keeps_prefix_but_is_otherwise_new():
    session = generate_trace(AGENT)[0]
    first = next(session_prompts(session, RANGE, "k", attempt=0))
    retry = next(session_prompts(session, RANGE, "k", attempt=1))
    assert first[:3000] == retry[:3000]
    assert first[3000:3016] != retry[3000:3016]


def test_prompts_repeat_exactly():
    session = generate_trace(AGENT)[0]
    assert list(session_prompts(session, RANGE, "k")) == list(session_prompts(session, RANGE, "k"))


@pytest.mark.parametrize("name", ["agent", "multi_turn", "long_context"])
def test_committed_trace_configs_generate(name):
    cfg = load_trace_config(f"configs/traces/{name}.yaml")
    assert cfg.kind == name
    assert generate_trace(cfg)


# --- Calibration ---


def calibrated() -> tuple[TraceConfig, dict]:
    cfg = load_trace_config("configs/traces/agent.yaml")
    real = json.loads(Path("configs/traces/calibration/swe_agent.json").read_text())
    return cfg, real


def test_percentile_table_is_sampled_faithfully():
    # Half the draws fall below the p50 entry, and no draw exceeds the top of the table.
    table = {"probs": [0.0, 0.5, 1.0], "values": [10, 100, 1000]}
    rng = np.random.default_rng(0)
    draws = [_draw(rng, table) for _ in range(5000)]
    assert np.median(draws) == pytest.approx(100, rel=0.1)
    assert min(draws) >= 10 and max(draws) <= 1000


def test_uniform_bounds_still_work():
    rng = np.random.default_rng(0)
    assert {_draw(rng, [3, 4]) for _ in range(50)} == {3, 4}


def test_tool_with_own_output_table_uses_it():
    tools = {
        "big": {"weight": 1.0, "median_s": 1.0, "sigma": 0.1, "output_tokens": [900, 900]},
    }
    trace = generate_trace(replace(AGENT, sessions=2, params={"tools": tools}))
    assert all(t.new_tokens == 900 for s in trace for t in s.turns[1:])


def test_agent_config_matches_calibration_file():
    cfg, real = calibrated()
    assert cfg.params["prefix_tokens"] == real["fit"]["prefix_tokens"]
    assert cfg.params["steps"]["values"] == pytest.approx(real["fit"]["steps"]["values"], abs=1)
    for name, tool in real["tools"].items():
        spec = cfg.params["tools"][name]
        assert spec["weight"] == pytest.approx(tool["share"], abs=0.001)
        assert spec["output_tokens"]["values"] == pytest.approx(tool["fit"]["values"], abs=1)


def test_simulated_sessions_resemble_real_ones():
    """The trace should reproduce real sessions it wasn't directly given, like final context."""
    cfg, real = calibrated()
    trace = generate_trace(replace(cfg, sessions=3000))
    final = [
        s.prefix_tokens + sum(t.new_tokens + t.response_tokens for t in s.turns) for s in trace
    ]
    steps = [len(s.turns) for s in trace]
    for p in (10, 50, 90):
        assert np.percentile(final, p) == pytest.approx(
            real["final_context_tokens"][f"p{p}"], rel=0.10
        )
    assert np.median(steps) == pytest.approx(real["steps"]["p50"], rel=0.15)


def test_old_plan_numbers_do_not_resemble_real_sessions():
    """Why calibration was needed. If this ever fails, the check above has lost its meaning."""
    cfg, real = calibrated()
    old = load_trace_config("configs/traces/agent_plan_assumptions.yaml")
    trace = generate_trace(replace(old, sessions=500))
    final = [
        s.prefix_tokens + sum(t.new_tokens + t.response_tokens for t in s.turns) for s in trace
    ]
    assert np.median(final) > 2 * real["final_context_tokens"]["p50"]
