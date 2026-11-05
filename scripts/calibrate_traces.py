"""Measure SWE-agent trajectories and write the calibration file. Needs no GPU.

Download some shards of nebius/SWE-agent-trajectories and a tokenizer.json, then run:

    uv run --with tokenizers python scripts/calibrate_traces.py \
        --parquet-dir DIR --tokenizer tokenizer.json --out configs/traces/calibration/swe_agent.json
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from project_anton.traces.calibrate import measure, summarize

COLUMNS = ["instance_id", "model_name", "exit_status", "trajectory"]


def rows(files: list[Path]) -> Iterator[dict]:
    for path in files:
        yield from pq.read_table(path, columns=COLUMNS).to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    def count_tokens(texts: list[str]) -> list[int]:
        return [len(e.ids) for e in tokenizer.encode_batch(texts, add_special_tokens=False)]

    files = sorted(args.parquet_dir.glob("*.parquet"))
    summary = summarize(measure(rows(files), count_tokens))
    output = {
        "source": "nebius/SWE-agent-trajectories (cc-by-4.0)",
        "shards": [f.name for f in files],
        "token_counts": "public Qwen2.5 tokenizer, as an approximation of Llama's",
        "not_calibrated": "pause times: the dataset records no timing",
        **summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
