"""Drive the real command line against the fake server, running in a background thread."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml
from fake_vllm import FakeVllm

from project_anton.cli import main


@pytest.fixture
def fake_server():
    server = FakeVllm(prefill_s_per_token=10e-6, token_delay_s=0.001)
    ready = threading.Event()
    loop = asyncio.new_event_loop()
    stop = loop.create_future()

    async def serve():
        async with server:
            ready.set()
            await stop

    thread = threading.Thread(target=loop.run_until_complete, args=(serve(),))
    thread.start()
    assert ready.wait(10)
    yield server
    loop.call_soon_threadsafe(stop.set_result, None)
    thread.join(10)
    loop.close()


def write_configs(tmp_path: Path, server: FakeVllm) -> tuple[Path, Path]:
    server_cfg = tmp_path / "server.yaml"
    server_cfg.write_text(
        yaml.safe_dump(
            {
                "name": "fake",
                "model": server.model,
                "port": int(server.base_url.rsplit(":", 1)[1]),
                "vllm_args": {},
                "token_id_range": [0, 1000],
                "kv_shape": {"layers": 32, "kv_heads": 8, "head_dim": 128},
            }
        )
    )
    sweep_cfg = tmp_path / "sweep.yaml"
    sweep_cfg.write_text(
        yaml.safe_dump(
            {"name": "t", "prompt_lengths": [1000, 4000], "reps": 2, "max_tokens": 4},
        )
    )
    return server_cfg, sweep_cfg


def test_run_analyze_and_resume(tmp_path, monkeypatch, fake_server, capsys):
    monkeypatch.chdir(tmp_path)
    server_cfg, sweep_cfg = write_configs(tmp_path, fake_server)
    run = ["ttft", "run", "--server", str(server_cfg), "--sweep", str(sweep_cfg)]

    assert main([*run, "--run-id", "r1"]) == 0
    run_dir = tmp_path / "results" / "r1"
    assert (run_dir / "config.json").exists()
    assert json.loads((run_dir / "env.json").read_text())["python"]
    assert (run_dir / "raw" / "requests.jsonl").exists()

    count = len(fake_server.requests)
    assert main([*run, "--run-id", "r1"]) == 0
    assert len(fake_server.requests) == count  # all finished, so nothing was re-sent
    assert (run_dir / "env.resume1.json").exists()

    capsys.readouterr()
    assert main(["ttft", "analyze", str(run_dir)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["prefill_fit"]["prefill_tokens_per_s"] > 0
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "analysis.json").exists()
    assert (run_dir / "ttft_vs_prompt.png").exists()


def test_status_reports_unreachable_server(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "server.yaml"
    cfg.write_text(
        "name: x\nmodel: m\nport: 1\ntoken_id_range: [0, 10]\n"
        "kv_shape: {layers: 1, kv_heads: 1, head_dim: 1}\n"
    )
    assert main(["server", "status", "--config", str(cfg), "--timeout", "0.1"]) == 1
    assert "not healthy" in capsys.readouterr().out


def test_stop_with_nothing_running(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "server.yaml"
    cfg.write_text(
        "name: x\nmodel: m\ntoken_id_range: [0, 10]\n"
        "kv_shape: {layers: 1, kv_heads: 1, head_dim: 1}\n"
    )
    assert main(["server", "stop", "--config", str(cfg)]) == 0
    assert "nothing was running" in capsys.readouterr().out


def test_replay_run_and_analyze(tmp_path, monkeypatch, fake_server, capsys):
    monkeypatch.chdir(tmp_path)
    server_cfg, _ = write_configs(tmp_path, fake_server)
    trace = tmp_path / "trace.yaml"
    trace.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "kind": "long_context",
                "sessions": 2,
                "arrival_rate_per_s": 100.0,
                "max_context_tokens": 2000,
                "params": {"median_prompt_tokens": 500},
            }
        )
    )
    args = ["replay", "run", "--server", str(server_cfg), "--trace", str(trace)]
    assert main([*args, "--run-id", "r"]) == 0
    run_dir = tmp_path / "results" / "r"
    assert json.loads((run_dir / "config.json").read_text())["trace"]["kind"] == "long_context"
    assert (run_dir / "raw" / "sysmon" / "status.json").exists()

    capsys.readouterr()
    assert main(["replay", "analyze", str(run_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["requests_valid"] == 2


def test_progress_is_not_held_back_when_piped():
    # A real sweep showed nothing for 30 minutes, because piped output is buffered.
    code = "import sys, time; from project_anton.cli import main; main(['--help'])"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")}
    source = Path(__file__).parent.parent / "src" / "project_anton" / "cli.py"
    assert "line_buffering=True" in source.read_text()
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert "usage: anton" in result.stdout
