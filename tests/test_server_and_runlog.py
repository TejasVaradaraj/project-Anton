import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from project_anton.analysis.kv_math import ModelShape
from project_anton.harness.runlog import append_jsonl, capture_environment, read_jsonl
from project_anton.harness.server import (
    ServerConfig,
    build_command,
    lmcache_server_command,
    load_server_config,
    parse_kv_capacity,
    require_verified_offload,
    start_server,
    stop_server,
    uses_lmcache_server,
    wait_until_healthy,
)
from project_anton.harness.ttft_bench import load_sweep_config

CONFIGS = Path(__file__).parent.parent / "configs"
SHAPE = ModelShape(layers=32, kv_heads=8, head_dim=128)


def make_cfg(port=8000, **args) -> ServerConfig:
    return ServerConfig("t", "org/model", port, args, (0, 100), SHAPE)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- Commands and configs ---


def test_build_command_flags():
    cmd = build_command(make_cfg(**{"dtype": "half", "enable-x": True, "enable-y": False}))
    assert cmd[:3] == ["vllm", "serve", "org/model"]
    assert cmd[cmd.index("--dtype") + 1] == "half"
    assert "--enable-x" in cmd
    assert "--enable-y" not in cmd


def test_server_binds_to_localhost_only():
    cmd = build_command(make_cfg())
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


@pytest.mark.parametrize("key", ["hf-token", "api-key"])
def test_config_with_secret_argument_is_refused(tmp_path, key):
    path = tmp_path / "c.yaml"
    path.write_text(
        f"name: x\nmodel: m\ntoken_id_range: [0, 10]\nvllm_args: {{{key}: abc}}\n"
        "kv_shape: {layers: 1, kv_heads: 1, head_dim: 1}\n"
    )
    with pytest.raises(ValueError, match="environment"):
        load_server_config(path)


@pytest.mark.parametrize("path", sorted((CONFIGS / "servers").glob("*.yaml")), ids=lambda p: p.stem)
def test_committed_server_configs_load(path):
    cfg = load_server_config(path)
    assert cfg.vllm_args["enable-prompt-tokens-details"] is True
    assert cfg.token_id_range[0] < cfg.token_id_range[1]


@pytest.mark.parametrize("path", sorted((CONFIGS / "sweeps").glob("*.yaml")), ids=lambda p: p.stem)
def test_committed_sweeps_fit_their_server(path):
    sweep = load_sweep_config(path)
    server = load_server_config(CONFIGS / "servers" / f"{path.stem.removeprefix('week1_')}.yaml")
    assert max(sweep.prompt_lengths) + sweep.max_tokens <= server.vllm_args["max-model-len"]


def test_committed_configs_exist():
    assert list((CONFIGS / "servers").glob("*.yaml"))
    assert list((CONFIGS / "sweeps").glob("*.yaml"))


@pytest.mark.parametrize("key", ["HF_TOKEN", "DD_API_KEY", "MY_SECRET", "db_password"])
def test_config_with_secret_env_is_refused(tmp_path, key):
    path = tmp_path / "c.yaml"
    path.write_text(
        f"name: x\nmodel: m\ntoken_id_range: [0, 10]\nenv: {{{key}: abc}}\n"
        "kv_shape: {layers: 1, kv_heads: 1, head_dim: 1}\n"
    )
    with pytest.raises(ValueError, match="environment"):
        load_server_config(path)


def test_lmcache_config_passes_connector_and_env():
    cfg = load_server_config(CONFIGS / "servers" / "a100_llama31_8b_lmcache.yaml")
    assert cfg.env["LMCACHE_LOCAL_CPU"] == "True"
    command = build_command(cfg)
    assert "LMCacheConnectorV1" in command[command.index("--kv-transfer-config") + 1]
    # Only offload may differ from the plain server, or the comparison means nothing.
    plain = load_server_config(CONFIGS / "servers" / "a100_llama31_8b.yaml")
    assert cfg.model == plain.model
    assert {k: v for k, v in cfg.vllm_args.items() if k != "kv-transfer-config"} == plain.vllm_args


def test_started_server_gets_config_env_and_inherits_the_rest(tmp_path, monkeypatch):
    monkeypatch.setenv("INHERITED", "yes")
    script = "import os; print(os.environ['INHERITED'], os.environ['FROM_CONFIG'])"
    cfg = ServerConfig("t", "m", 1, {}, (0, 10), SHAPE, env={"FROM_CONFIG": "set"})
    log = tmp_path / "s.log"
    proc = start_server(cfg, log, tmp_path / "s.pid", command=[sys.executable, "-c", script])
    proc.wait(timeout=15)
    assert log.read_text().strip() == "yes set"


# --- Log parsing ---


def test_parse_kv_capacity_real_format():
    # Format string from vLLM 0.33.0 kv_cache_utils.py, values from the A100 run.
    line = (
        "INFO 11-03 10:00:00 [kv_cache_utils.py:1148] GPU KV cache size: 461,216 tokens, "
        "Maximum concurrency for 32,768 tokens per request: 14.08x"
    )
    assert parse_kv_capacity(line) == {
        "pool": "GPU",
        "cache_tokens": 461_216,
        "context_tokens": 32_768,
        "max_concurrency": 14.08,
    }


def test_parse_kv_capacity_absent():
    assert parse_kv_capacity("nothing here") is None


# --- Process management, with a tiny HTTP server in place of vLLM ---


def test_start_wait_stop_real_process(tmp_path):
    port = free_port()
    log, pid = tmp_path / "s.log", tmp_path / "s.pid"
    cmd = [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"]
    # http.server returns 404 for /health unless the file exists.
    (tmp_path / "health").write_text("ok")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        proc = start_server(make_cfg(port), log, pid, command=cmd)
    finally:
        os.chdir(cwd)
    try:
        wait_until_healthy(f"http://127.0.0.1:{port}", proc, timeout_s=15, poll_s=0.1)
    finally:
        assert stop_server(pid) is True
    assert not pid.exists()
    proc.wait(timeout=5)


def test_stop_with_nothing_running(tmp_path):
    assert stop_server(tmp_path / "missing.pid") is False


def test_wait_reports_server_that_died(tmp_path):
    proc = start_server(
        make_cfg(free_port()),
        tmp_path / "s.log",
        tmp_path / "s.pid",
        command=[sys.executable, "-c", "raise SystemExit(3)"],
    )
    with pytest.raises(RuntimeError, match="code 3"):
        wait_until_healthy(f"http://127.0.0.1:{free_port()}", proc, timeout_s=15, poll_s=0.05)


def test_wait_times_out():
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        wait_until_healthy(f"http://127.0.0.1:{free_port()}", timeout_s=0.2, poll_s=0.05)
    assert time.monotonic() - start < 10


# --- Secrets and raw logs ---


def test_environment_capture_excludes_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETVALUE")
    monkeypatch.setenv("DD_API_KEY", "dd_SECRETVALUE")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    text = json.dumps(capture_environment())
    assert "SECRETVALUE" not in text
    assert json.loads(text)["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_cli_imports_with_colab_matplotlib_backend():
    # Colab sets this, and a real run crashed on import because of it.
    env = {
        **os.environ,
        "MPLBACKEND": "module://matplotlib_inline.backend_inline",
        "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
    }
    subprocess.run([sys.executable, "-c", "import project_anton.cli"], env=env, check=True)


def test_jsonl_survives_half_written_last_line(tmp_path):
    path = tmp_path / "r.jsonl"
    append_jsonl(path, {"a": 1})
    with path.open("a") as f:
        f.write('{"a": 2, "b"')
    assert read_jsonl(path) == [{"a": 1}]


def test_jsonl_raises_on_corruption_mid_file(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"a": 1}\nbroken\n{"a": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_jsonl_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path / "none.jsonl") == []


def test_lmcache_server_only_for_multi_process_configs():
    mp = load_server_config(CONFIGS / "servers" / "a100_small_mp.yaml")
    assert uses_lmcache_server(mp)
    assert lmcache_server_command(mp)[mp_size_index(mp) :][:1] == ["100"]
    assert not uses_lmcache_server(load_server_config(CONFIGS / "servers" / "a100_small_cpu.yaml"))


def mp_size_index(cfg: ServerConfig) -> int:
    return lmcache_server_command(cfg).index("--l1-size-gb") + 1


@pytest.mark.parametrize(
    ("version", "allow", "refused"),
    [
        ("0.25.1", None, False),
        ("0.33.0", None, True),
        ("0.33.0", "1", False),
    ],
)
def test_offload_runs_need_verified_versions(monkeypatch, version, allow, refused):
    output = subprocess.CompletedProcess([], 0, stdout=f"warning line\n{version}\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: output)
    if allow:
        monkeypatch.setenv("ANTON_ALLOW_UNVERIFIED_OFFLOAD", allow)
    else:
        monkeypatch.delenv("ANTON_ALLOW_UNVERIFIED_OFFLOAD", raising=False)
    if refused:
        with pytest.raises(SystemExit, match="Refusing"):
            require_verified_offload()
    else:
        require_verified_offload()
