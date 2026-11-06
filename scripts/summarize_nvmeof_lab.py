"""Summarize the fio files from scripts/nvmeof_lab.sh into one table. Usage: FOLDER"""

import csv
import json
import re
import sys
from pathlib import Path

from project_anton.harness.storage import parse_fio

folder = Path(sys.argv[1])
rows = []
for path in sorted((folder / "raw").glob("*.json")):
    path_name, _, test = re.fullmatch(r"(.+?)_(chunk|lat|restore)_(.+)", path.stem).group(1, 2, 3)
    rows.append({"path": path_name, "test": test, **parse_fio(path.read_text())})
with (folder / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
