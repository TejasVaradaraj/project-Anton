"""Checks that catch a broken or leaky notebook without needing a GPU session."""

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
SECRET_RE = re.compile(r"hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|[0-9a-f]{32}")


def code_cells(path: Path) -> list[str]:
    cells = json.loads(path.read_text())["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def test_notebooks_exist():
    assert NOTEBOOKS


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_committed_without_outputs(path):
    # Outputs can hold logs, paths or tokens, and they don't belong in git.
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_code_cells_are_valid_python(path):
    for source in code_cells(path):
        ast.parse(source)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_secret_values(path):
    assert not SECRET_RE.search(path.read_text())


def test_notebook_refers_to_configs_that_exist():
    source = "\n".join(code_cells(ROOT / "notebooks" / "week1_ttft.ipynb"))
    names = set(re.findall(r"\b(t4_\w+|a100_\w+)\b", source))
    assert {"t4_qwen05b", "t4_llama32_3b", "a100_llama31_8b"} <= names
    for name in names:
        assert (ROOT / "configs" / "servers" / f"{name}.yaml").exists()
        assert (ROOT / "configs" / "sweeps" / f"week1_{name}.yaml").exists()


def test_notebook_shows_command_output():
    # A real Colab run printed nothing for commands that ran without a pipe.
    source = "\n".join(code_cells(ROOT / "notebooks" / "week1_ttft.ipynb"))
    assert "stdout=subprocess.PIPE" in source


def test_notebook_stops_the_server_last():
    assert 'server", "stop"' in code_cells(ROOT / "notebooks" / "week1_ttft.ipynb")[-1]
