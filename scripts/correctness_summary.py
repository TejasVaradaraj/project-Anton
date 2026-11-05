"""Tabulate how a cache hit or restore changes output, compared with the same prompt sent fresh.

Usage: python scripts/correctness_summary.py OUT.csv name=results.jsonl [name=results.jsonl ...]
"""

import csv
import statistics
import sys
from pathlib import Path

from project_anton.harness.runlog import read_jsonl


def summarize(name: str, path: Path) -> list[dict]:
    records = {(r["length"], r["i"], r["phase"]): r for r in read_jsonl(path)}
    prompts = [(length, i) for length, i, phase in records if phase == "fresh"]
    rows = []
    for phase in ("cached", "evicted"):
        changes = []
        changed = 0
        for length, i in prompts:
            fresh, other = records[(length, i, "fresh")], records[(length, i, phase)]
            # Only the first token: after any divergence, later tokens aren't comparable.
            changes.append(abs(other["logprobs"][0] - fresh["logprobs"][0]))
            changed += other["tokens"] != fresh["tokens"]
        rows.append(
            {
                "setup": name,
                "phase": phase,
                "prompts": len(prompts),
                "outputs_changed": changed,
                "first_token_logprob_change_median": round(statistics.median(changes), 4),
                "first_token_logprob_change_max": round(max(changes), 4),
            }
        )
    return rows


def main() -> None:
    out, *runs = sys.argv[1:]
    rows = []
    for item in runs:
        name, path = item.split("=", 1)
        rows += summarize(name, Path(path))
    with Path(out).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
