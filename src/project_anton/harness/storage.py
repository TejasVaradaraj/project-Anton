"""Baseline a disk with fio, bypassing the page cache, so tier results have a reference point."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

JOBS = {
    "seq_write": {"rw": "write", "bs": "1M", "iodepth": "16"},
    "seq_read": {"rw": "read", "bs": "1M", "iodepth": "16"},
    "rand_read_4k": {"rw": "randread", "bs": "4k", "iodepth": "32"},
    "rand_read_1m": {"rw": "randread", "bs": "1M", "iodepth": "16"},
}


def fio_command(job: dict[str, str], path: Path, size: str, runtime_s: int) -> list[str]:
    return [
        "fio",
        "--name=anton",
        f"--filename={path}",
        f"--rw={job['rw']}",
        f"--bs={job['bs']}",
        f"--iodepth={job['iodepth']}",
        f"--size={size}",
        f"--runtime={runtime_s}",
        "--time_based",
        "--direct=1",  # Without this, reads can come from RAM and look far too fast
        "--ioengine=libaio",
        "--group_reporting",
        "--output-format=json",
    ]


def parse_fio(output: str) -> dict[str, float]:
    job = json.loads(output)["jobs"][0]
    side = job["read"] if job["read"]["bw_bytes"] > 0 else job["write"]
    percentiles = side.get("clat_ns", {}).get("percentile", {})
    return {
        "gbytes_per_s": side["bw_bytes"] / 1e9,
        "iops": side["iops"],
        "latency_p50_us": percentiles.get("50.000000", 0) / 1000,
        "latency_p99_us": percentiles.get("99.000000", 0) / 1000,
    }


def run_baseline(directory: Path, size: str = "8G", runtime_s: int = 20) -> dict[str, Any]:
    if shutil.which("fio") is None:
        raise RuntimeError("fio isn't installed")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "anton_fio.dat"
    results: dict[str, Any] = {"directory": str(directory), "size": size}
    try:
        for name, job in JOBS.items():
            cmd = fio_command(job, path, size, runtime_s)
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            results[name] = parse_fio(out.stdout)
    finally:
        path.unlink(missing_ok=True)
    return results
