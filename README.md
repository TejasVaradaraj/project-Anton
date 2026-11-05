# project-Anton

A benchmark for **KV-cache offloading and prefetching for tool-calling AI agents**.

An AI agent pauses while its tools run. During that pause the inference server can evict the
agent's KV cache (the model's saved work for each token it has read), so the next step has to
recompute a long context. This project measures how well moving that cache to cheaper storage and
prefetching it back works, and how fast the storage must be for that to pay off.

## Status

**Week 1 of 8, in progress. There are no GPU measurements in this repo yet.** The benchmark has
only been tested against a fake server, which checks that the pipeline works but says nothing
about real performance. Real results will appear under `results/` with their config, git SHA and
environment. See [SETUP.md](SETUP.md) for what has been verified and the known limitations.

## What gets measured

| Metric | Meaning |
|---|---|
| TTFT | Time to first token: from sending the request to the first generated token |
| TPOT | Time per output token: average gap between tokens after the first |
| Prefill rate | Tokens per second the GPU can read, from the slope of cold TTFT vs prompt length |
| Break-even bandwidth | Storage speed at which loading a KV cache takes as long as recomputing it |

Week 1 sends each prompt twice: once uncached (cold), then again right away (warm). Prompts are
exact-length random token IDs, and pairs run in a seeded shuffle. A request only counts as valid
if the server confirms the expected token counts and cache behavior.

## Run it

Tests and lint, which need no GPU:

```bash
uv run pytest
```

```bash
uv run ruff check .
```

For real measurements, open [notebooks/week1_ttft.ipynb](notebooks/week1_ttft.ipynb) on Kaggle or
Colab. On any machine with vLLM installed, the same steps are:

```bash
uv run anton server start --config configs/servers/t4_qwen05b.yaml
```

```bash
uv run anton ttft run --server configs/servers/t4_qwen05b.yaml --sweep configs/sweeps/week1_t4_qwen05b.yaml --run-id my-run
```

```bash
uv run anton ttft analyze results/my-run
```

```bash
uv run anton server stop --config configs/servers/t4_qwen05b.yaml
```

Rerunning with the same `--run-id` continues a stopped run.

## Layout

| Path | Contents |
|---|---|
| `src/project_anton/harness/` | Streaming client, sweep runner, server control, run logging |
| `src/project_anton/analysis/` | KV size math, summaries, prefill fit, plots |
| `configs/` | Server and sweep settings |
| `notebooks/` | Kaggle and Colab entry points |
| `tests/` | Unit tests, plus a fake vLLM server on a real socket |
| `results/<run_id>/` | Summaries in git; `raw/` stays out of git |

## Prior work

This builds on existing ideas and doesn't claim to have invented them: CachedAttention (USENIX ATC
2024), Continuum (2025), vLLM PagedAttention, and LMCache.
