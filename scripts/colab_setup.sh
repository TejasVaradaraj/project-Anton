#!/usr/bin/env bash
# Set up a fresh Colab GPU runtime for the final runs: the verified versions, the model and prompts.
# Needs a Hugging Face token in /root/.cache/huggingface/token. Run from the repo root.
set -uo pipefail

pip install -q uv==0.12.3 huggingface_hub 2>&1 | tail -1
uv venv -q --python 3.12 --allow-existing /tmp/old
uv pip install -q --python /tmp/old/bin/python 'lmcache==0.5.1' 'vllm==0.25.1' 2>&1 | tail -2
UV_PYTHON=3.12 uv sync --frozen --no-dev 2>&1 | tail -1
python3 - <<'P'
from huggingface_hub import hf_hub_download as d, snapshot_download as s
d("nebius/SWE-agent-trajectories", "default/train/0000.parquet", repo_type="dataset",
  revision="refs/convert/parquet", local_dir="/tmp/swe")
d("meta-llama/Llama-3.1-8B-Instruct", "tokenizer.json", local_dir="/tmp/tok")
s("meta-llama/Llama-3.1-8B-Instruct", ignore_patterns=["original/*", "*.pth"])
P
PATH=/tmp/old/bin:$PATH vllm --version 2>/dev/null | tail -1
echo SETUP_DONE
