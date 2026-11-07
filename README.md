# project-Anton

A benchmark for **KV-cache offloading and prefetching for tool-calling AI agents**.

An AI agent pauses while its tools run. During that pause the inference server can evict the
agent's KV cache (the model's saved work for each token it has read), so the next step has to
recompute a long context. This project measures how well moving that cache to cheaper storage and
prefetching it back works, and how fast the storage must be for that to pay off.

## Status

**Week 1 of 8: first measurements done.** Everything else on the roadmap is still ahead, including
all of the offloading itself. See [SETUP.md](SETUP.md) for what has been verified and the known
limitations.

## Week 1 result: what the prefix cache saves

Llama-3.1-8B (bf16) on one A100 80GB, vLLM 0.33.0, requests sent one at a time, 32 output tokens.
Run [`20261105T074136Z_a100_llama31_8b`](results/20261105T074136Z_a100_llama31_8b), 5 repeats per
cell. All 62 requests passed the validity checks.

![TTFT vs prompt length](results/20261105T074136Z_a100_llama31_8b/ttft_vs_prompt.png)

| Prompt tokens | Cold TTFT (ms) | Warm TTFT (ms) | Cold / warm | Avg prefill (tokens/s) | KV size (GB) | Break-even storage (GB/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 77.7 | 22.0 | 3.5x | 12,866 | 0.13 | 2.35 |
| 2,000 | 140.8 | 25.3 | 5.6x | 14,202 | 0.26 | 2.27 |
| 4,000 | 285.8 | 30.5 | 9.4x | 13,996 | 0.52 | 2.05 |
| 8,000 | 607.4 | 41.5 | 14.6x | 13,170 | 1.05 | 1.85 |
| 16,000 | 1,374.9 | 61.1 | 22.5x | 11,637 | 2.10 | 1.60 |
| 32,000 | 3,475.0 | 105.7 | 32.9x | 9,209 | 4.19 | 1.24 |

**What was measured**
- Cached prompts reach the first token up to 33x faster, and the gap widens with length.
- Prefill cost per token rises with length, so no single prefill rate describes this GPU. A straight
  line fits with r-squared 0.988 but implies a negative startup time, so it's wrong in spite of that.
- TPOT also rises with context, from 10.8 ms at 1k to 12.2 ms at 32k.
- Cross-check at 2,000 tokens against `vllm bench serve` (20 requests): TTFT 140.8 vs 148.1 ms,
  and TPOT 10.87 vs 10.93 ms.

**What this doesn't show**
- Break-even is calculated from these measurements: KV size / (cold TTFT - warm TTFT). No KV cache
  has been moved to storage yet. It ignores storage latency and transfer overhead, so real storage
  must be faster than these numbers.
- Why the cost per token rises hasn't been tested. One possible cause is that each new token attends
  to a longer context, but that is only a hypothesis so far.
- This is one session on one shared GPU with 5 repeats per cell, one request at a time, using random
  token IDs. It says nothing about load, and it's a smaller model than Llama-3.3-70B.

## What gets measured

| Metric | Meaning |
|---|---|
| TTFT | Time to first token: from sending the request to the first generated token |
| TPOT | Time per output token: average gap between tokens after the first |
| Prefill rate | Tokens per second the GPU can read: prompt length / cold TTFT, at each length |
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
