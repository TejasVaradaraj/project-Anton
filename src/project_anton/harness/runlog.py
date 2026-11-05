"""Record what ran: run folders, environment snapshots, and raw result files."""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allowlist on purpose. Copying os.environ would save HF_TOKEN into the results.
SAFE_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "KAGGLE_KERNEL_RUN_TYPE", "COLAB_GPU")
RECORDED_PACKAGES = ("httpx", "numpy", "pandas", "pyarrow", "matplotlib")


def _run(cmd: list[str], timeout: float = 30) -> str | None:
    """Return a command's trimmed stdout, or None if it's missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def git_info() -> dict[str, Any]:
    status = _run(["git", "status", "--porcelain"])
    return {
        "sha": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        # A dirty tree means this SHA alone can't reproduce the run.
        "dirty": bool(status) if status is not None else None,
    }


def capture_environment() -> dict[str, Any]:
    packages = {}
    for name in RECORDED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    query = "--query-gpu=name,memory.total,driver_version"
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "git": git_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "gpus": _run(["nvidia-smi", query, "--format=csv,noheader"]),
        # vLLM is installed outside the harness's virtualenv, so ask its CLI.
        "vllm_version": _run(["vllm", "--version"], timeout=120),
        "env": {k: os.environ[k] for k in SAFE_ENV_VARS if k in os.environ},
    }


def make_run_id(name: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    sha = (git_info()["sha"] or "nogit")[:7]
    return f"{stamp}_{name}_{sha}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append and force to disk, so a dropped Colab/Kaggle session loses at most one line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records, tolerating a half-written last line from a crash."""
    if not path.exists():
        return []
    records = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return records
