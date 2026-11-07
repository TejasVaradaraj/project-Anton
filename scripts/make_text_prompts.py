"""Write real-text prompts for the correctness check, from SWE-agent trajectories. Needs no GPU.

Download a shard of nebius/SWE-agent-trajectories and the model's tokenizer.json, then run:

    uv run --with tokenizers python scripts/make_text_prompts.py \
        --parquet SHARD.parquet --tokenizer tokenizer.json --out results/week4_text/prompts.jsonl
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from project_anton.correctness.compare import LENGTHS
from project_anton.correctness.text_prompts import build_text_prompts, write_prompts


def trajectory_texts(path: Path) -> Iterator[str]:
    for row in pq.read_table(path, columns=["trajectory"]).to_pylist():
        messages = row["trajectory"] or []
        yield "\n\n".join(m.get("system_prompt") or m.get("text") or "" for m in messages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-length", type=int, default=5)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompts = build_text_prompts(
        trajectory_texts(args.parquet),
        lambda text: tokenizer.encode(text).ids,
        args.lengths,
        args.per_length,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.unlink(missing_ok=True)
    write_prompts(args.out, prompts)
    print(f"wrote {len(prompts)} prompts to {args.out}")


if __name__ == "__main__":
    main()
