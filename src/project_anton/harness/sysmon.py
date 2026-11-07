"""Record what the machine was doing during a run: GPU, disk, network, and vLLM's own metrics.

A missing tool is noted and skipped, because Colab and Kaggle don't all have the same ones.
Monitoring must never be the reason a benchmark run fails.
"""

import asyncio
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from project_anton.harness.runlog import append_jsonl, write_json

DEFAULT_COMMANDS = {
    "gpu": ["nvidia-smi", "dmon", "-s", "pucmt", "-o", "DT", "-d", "1"],
    "disk": ["iostat", "-xmt", "1"],
    "network": ["sar", "-n", "DEV", "1"],
}
METRIC_PREFIX = "vllm:"


def parse_vllm_metrics(text: str) -> dict[str, float]:
    """Pick vLLM's values out of Prometheus text. Lines look like `vllm:name{labels} 1.0`."""
    metrics = {}
    for line in text.splitlines():
        if not line.startswith(METRIC_PREFIX):
            continue
        name, _, value = line.rpartition(" ")
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


class SystemMonitor:
    def __init__(
        self,
        out_dir: Path,
        base_url: str,
        interval_s: float = 1.0,
        commands: dict[str, list[str]] | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.base_url = base_url
        self.interval_s = interval_s
        self.commands = DEFAULT_COMMANDS if commands is None else commands
        self.procs: dict[str, subprocess.Popen] = {}
        self.skipped: list[str] = []
        self._poller: asyncio.Task | None = None

    async def __aenter__(self) -> "SystemMonitor":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for name, cmd in self.commands.items():
            if shutil.which(cmd[0]) is None:
                self.skipped.append(name)
                continue
            with (self.out_dir / f"{name}.log").open("ab") as log:
                self.procs[name] = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
        self._poller = asyncio.create_task(self._poll_metrics())
        return self

    async def __aexit__(self, *exc) -> None:
        self._poller.cancel()
        try:
            await self._poller
        except asyncio.CancelledError:
            pass
        exited = {}
        for name, proc in self.procs.items():
            # A tool that already quit (say, flags this OS doesn't accept) can't be signalled:
            # macOS raises PermissionError for that. Record it, so a thin log has an explanation.
            if proc.poll() is not None:
                exited[name] = proc.returncode
                continue
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait()
        status: dict[str, Any] = {
            "skipped": self.skipped,
            "started": sorted(self.procs),
            "exited_early": exited,
        }
        write_json(self.out_dir / "status.json", status)

    async def _poll_metrics(self) -> None:
        path = self.out_dir / "vllm_metrics.jsonl"
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5) as client:
            while True:
                try:
                    response = await client.get("/metrics")
                    if response.status_code == 200:
                        record = {"t": time.time(), "metrics": parse_vllm_metrics(response.text)}
                        append_jsonl(path, record)
                except httpx.HTTPError:
                    pass  # A busy server can miss a sample
                await asyncio.sleep(self.interval_s)
