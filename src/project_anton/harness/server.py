"""Start, check and stop `vllm serve` from a YAML config."""

import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from project_anton.analysis.kv_math import ModelShape

# Secrets reach vLLM through the environment. They must never sit in a config file.
FORBIDDEN_ARGS = ("hf-token", "api-key")
SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|API_?KEY", re.I)

KV_CAPACITY_RE = re.compile(
    r"(?P<pool>\S+) KV cache size: (?P<tokens>[\d,]+) tokens, "
    r"Maximum concurrency for (?P<context>[\d,]+) tokens per request: (?P<concurrency>[\d.]+)x"
)


LMCACHE_PORT = 5555
# Restores match a normal recompute on this pair, with LMCache 0.5.1. See results/week4_text.
VERIFIED_VLLM = "0.25.1"


@dataclass(frozen=True)
class ServerConfig:
    name: str
    model: str
    port: int
    vllm_args: dict[str, Any]
    token_id_range: tuple[int, int]
    kv_shape: ModelShape
    env: dict[str, str] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def load_server_config(path: str | Path) -> ServerConfig:
    raw = yaml.safe_load(Path(path).read_text())
    args = raw.get("vllm_args") or {}
    for key in args:
        if key in FORBIDDEN_ARGS:
            raise ValueError(f"{key} must come from the environment, not from {path}")
    env = {key: str(value) for key, value in (raw.get("env") or {}).items()}
    for key in env:
        if SECRET_NAME.search(key):
            raise ValueError(f"{key} must come from the environment, not from {path}")
    low, high = raw["token_id_range"]
    return ServerConfig(
        name=raw["name"],
        model=raw["model"],
        port=raw.get("port", 8000),
        vllm_args=args,
        token_id_range=(low, high),
        kv_shape=ModelShape(**raw["kv_shape"]),
        env=env,
    )


def build_command(cfg: ServerConfig) -> list[str]:
    cmd = ["vllm", "serve", cfg.model, "--host", "127.0.0.1", "--port", str(cfg.port)]
    for key, value in cfg.vllm_args.items():
        if value is True:
            cmd.append(f"--{key}")
        elif value is not False and value is not None:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def parse_kv_capacity(log_text: str) -> dict[str, Any] | None:
    """Read vLLM's startup line, e.g. 'GPU KV cache size: 461,216 tokens, ...'."""
    match = KV_CAPACITY_RE.search(log_text)
    if match is None:
        return None
    return {
        "pool": match["pool"],
        "cache_tokens": int(match["tokens"].replace(",", "")),
        "context_tokens": int(match["context"].replace(",", "")),
        "max_concurrency": float(match["concurrency"]),
    }


def wait_until_healthy(
    base_url: str,
    proc: subprocess.Popen | None = None,
    timeout_s: float = 1800,
    poll_s: float = 2.0,
) -> None:
    """Wait for /health. The timeout is long because the first start downloads the model."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} during startup")
        try:
            if httpx.get(f"{base_url}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"{base_url} not healthy after {timeout_s}s")


def start_server(
    cfg: ServerConfig, log_path: Path, pid_path: Path, command: list[str] | None = None
) -> subprocess.Popen:
    """Launch in its own process group, so it outlives the command that started it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            command or build_command(cfg),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, **cfg.env},
        )
    pid_path.write_text(str(proc.pid))
    return proc


def uses_lmcache_server(cfg: ServerConfig) -> bool:
    """Multi-process offload needs a separate `lmcache server` that owns the storage."""
    return "kv-offloading-backend" in cfg.vllm_args


def lmcache_server_command(cfg: ServerConfig) -> list[str]:
    # Colab already uses LMCache's default HTTP port, 8080.
    size = str(cfg.vllm_args["kv-offloading-size"])
    return [
        "lmcache",
        "server",
        "--host",
        "localhost",
        "--port",
        str(LMCACHE_PORT),
        "--l1-size-gb",
        size,
        "--eviction-policy",
        "LRU",
        "--http-port",
        "8091",
    ]


def wait_for_port(port: int, proc: subprocess.Popen, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"nothing listening on port {port}")


def require_verified_offload() -> None:
    """Refuse offload runs on versions whose restores failed the correctness check."""
    output = subprocess.run(["vllm", "--version"], capture_output=True, text=True, check=True)
    version = output.stdout.strip().splitlines()[-1]
    if version != VERIFIED_VLLM and os.environ.get("ANTON_ALLOW_UNVERIFIED_OFFLOAD") != "1":
        sys.exit(
            f"Refusing to run: restores are only verified on vLLM {VERIFIED_VLLM}, not {version}."
        )


def stop_server(pid_path: Path, grace_s: float = 30) -> bool:
    """Stop the whole process group, freeing the GPU. Returns False if nothing was running."""
    if not pid_path.exists():
        return False
    pid = int(pid_path.read_text())
    pid_path.unlink()
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            # Reap it if it's our own child; otherwise a dead child still looks alive.
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)
    os.killpg(pid, signal.SIGKILL)
    return True
