import json
from pathlib import Path

import pandas as pd
import pytest

from project_anton.analysis.cliff import compare_sweeps, load_sweep


def write_point(results: Path, sweep: str, sessions: int, cached: float, finished=True) -> None:
    run_dir = results / f"{sweep}_sessions-{sessions}"
    run_dir.mkdir(parents=True)
    trace = {"sessions": sessions, "arrival_rate_per_s": 1.0}
    (run_dir / "config.json").write_text(json.dumps({"trace": trace}))
    if not finished:
        return
    after = {
        "previous_prompt_cached_mean": cached,
        "previous_prompt_cached_p10": cached / 2,
        "ttft_p50_ms": 100.0,
        "ttft_p95_ms": 1000.0 / cached,
        "ttft_p99_ms": None,
        "tpot_p50_ms": 12.0,
    }
    summary = {
        "requests_valid": 10,
        "requests_total": 10,
        "max_concurrent_at_send": 3,
        "after_tool": after,
        "slo_met_fraction": cached,
        "output_tokens_per_s": 50.0,
        "goodput_output_tokens_per_s": 50.0 * cached,
        "counters": {"external_prefix_cache_hits": 7.0},
        "duration_s": 60.0,
    }
    (run_dir / "replay_summary.json").write_text(json.dumps(summary))


def test_sweep_is_sorted_numerically_not_by_folder_name(tmp_path):
    for sessions in (96, 8, 16):
        write_point(tmp_path, "s", sessions, 0.9)
    assert load_sweep(tmp_path, "s")["sessions"].tolist() == [8, 16, 96]


def test_unfinished_points_are_left_out(tmp_path):
    write_point(tmp_path, "s", 8, 0.9)
    write_point(tmp_path, "s", 16, 0.9, finished=False)
    assert load_sweep(tmp_path, "s")["sessions"].tolist() == [8]


def test_missing_sweep_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no finished points"):
        load_sweep(tmp_path, "nope")


def test_compare_writes_table_and_plot(tmp_path):
    results = tmp_path / "results"
    for sessions, cached in ((8, 1.0), (32, 0.6)):
        write_point(results, "gpu", sessions, cached)
        write_point(results, "lm", sessions, min(1.0, cached + 0.3))
    table = compare_sweeps(results, ["gpu", "lm"], tmp_path / "out")
    assert len(table) == 4
    assert set(table["sweep"]) == {"gpu", "lm"}
    assert table["external_cache_hits"].notna().all()
    assert len(pd.read_csv(tmp_path / "out" / "comparison.csv")) == 4
    assert (tmp_path / "out" / "cliff.png").stat().st_size > 0
