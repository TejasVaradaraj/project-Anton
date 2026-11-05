# project-Anton report

## Summary

This project set out to measure whether offloading an AI agent's KV cache to cheaper storage, then
prefetching it back, is worth doing, and how fast that storage must be.

**The project stopped early, on purpose.** With vLLM 0.33.0 and LMCache 0.5.6 on this setup, **restored KV
caches changed the model's output, and nothing reported an error**. Measuring the speed of offload and
prefetch further wouldn't mean much until restores are correct, so Weeks 6-7 weren't run on a GPU.

What did come out of it:

- A real cliff in GPU-only serving between 64 and 96 agent sessions.
- Evidence that the cache-hit path isn't a safe proxy for a correct restore.
- Measured restore speeds, a storage requirements table, and agent session sizes from real runs.
- A reproducible harness: client, trace replayer, correctness check and system monitor, with 240 tests.

Everything ran on one A100 80GB with Llama-3.1-8B. Prompts are random token IDs, and each result comes
from one session unless stated.

## Main finding: restores change the output

Each prompt was sent fresh, sent again from the prefix cache, then evicted and sent again, with greedy
decoding. On one server, only the cache state differs, so outputs should match. 12 prompts per setup, from
1,000 to 30,000 tokens. Two separate GPU-only servers matched on all 36 requests with a logprob
difference of exactly 0, so the server itself is deterministic.

| Setup | Path | Outputs changed | First-token logprob change, median (max) |
|---|---|---:|---:|
| GPU only | Evicted, then recomputed | 0 of 12 | 0 (0) |
| GPU only | Ordinary prefix cache | 3 of 12 | 0.016 (0.037) |
| LMCache in-process, CPU RAM | Restored | **11 of 12** | **0.675 (3.385)** |
| LMCache in-process, disk | Restored | **10 of 12** | **0.367 (3.374)** |
| LMCache multi-process, CPU RAM | Restored | **7 of 12** | 0.070 (3.356) |
| LMCache multi-process, 60 s before eviction | Restored | **5 of 12** | 0.066 (2.677) |

How to read this:

- **Ordinary prefix caching is the rounding baseline.** It changes some outputs on these prompts, by small
  amounts. Those prompts appear to have near-tied predictions, so tiny rounding differences can flip a
  token. That's why outputs alone aren't enough, and the size of the first-token change matters.
- **Restores go far beyond that.** Single-prompt changes reach about 90 times the baseline's maximum, and
  the in-process median is about 40 times the baseline's median. Output text is entirely different.
- **Multi-process mode is better but still fails.** Prompts of 1,000 tokens were all correct, while most
  failures came at 16,000 and up.
- **Giving saves 60 seconds didn't fix it**, so a simple save-then-evict timing race isn't the whole cause.

**The cause isn't confirmed.** The leading lead is
[LMCache/LMCache#4887](https://github.com/LMCache/LMCache/pull/4887), an open pull request describing copy
errors between newer vLLM KV layouts and the in-process connector. That wouldn't directly explain the
multi-process failures. Newer LMCache (0.5.7rc4) couldn't be tested in-process, because it removes a
module vLLM 0.33.0 imports. It couldn't be tested in multi-process mode either, because that attempt's
server startup failed on a settings error and I then moved to the stable release.

This should be repeated with real text before calling it a general bug. These are random token IDs, which
are unusually sensitive to small changes, though the ordinary prefix cache acts as a control for that.

Multi-process mode also crashed the vLLM engine under the tier test, with
`assert req_id in self.requests` in `_update_from_kv_xfer_finished`.

## The cliff (Week 3)

Calibrated agents arriving at 1 per second, on the full 56 GiB GPU KV cache.

| Sessions | Setup | Found in cache | Hit TTFT p50 (ms) | Miss TTFT p50 (ms) | TPOT p50 (ms) | Met targets | Goodput (tokens/s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | GPU only | 98.5% | 218 | 1,503 | 46.8 | 99.1% | 251 |
| 96 | GPU only | 67.1% | 280 | 33,324 | 101.0 | 49.9% | 69 |
| 96 | LMCache in-process | 99.4% | 927 | 1 miss | 122.1 | 37.0% | 37 |

Targets are TTFT under 2 s and TPOT under 100 ms. Goodput falls 73% between 64 and 96 sessions, and a turn
that misses the cache waits about 33 s, far more than the 3.5 s a 32k recompute takes alone. TPOT also
passes the target at 96 sessions, so GPU compute is running out at the same point.

LMCache keeps nearly every cache at 96 sessions, but its goodput is lower at every load above 16 sessions.
And Week 4 shows those restores aren't correct, so this isn't a working system to compare with. My
prediction of a cliff near 32 sessions was wrong: sessions finish and leave, so total context overstates
what has to stay live at once.

## Restore speed (Week 4)

Extra time compared with a warm request, on an 8 GiB GPU cache, 5 repeats per length:

| Prompt tokens | Recompute, GPU only (ms) | In-process CPU restore (ms) | In-process disk restore (ms) |
|---:|---:|---:|---:|
| 1,000 | 68 | 133 | 171 |
| 4,000 | 280 | 608 | 873 |
| 16,000 | 1,352 | 2,475 | 3,131 |

- In-process restores delivered about 0.8 GB/s from RAM, and 0.6-0.8 GB/s from disk, which is slower than
  recomputing. fio read that disk at 3.9 GB/s, so storage wasn't the limit.
- Saving is costly in-process: a cold 16k request took 1.6x as long with CPU saves and 5.3x with disk saves.
- **Multi-process mode looked far faster, on very thin evidence.** One 16k restore took 302 ms, about 6.9
  GB/s, and its cold request took 1,497 ms against 1,394 ms with no offload. The engine crashed soon after,
  so that's a single sample.
- 32,000-token prompts were never restored in-process, though vLLM's counter reported a hit. The cause
  isn't known.

## Other results

- **Week 1:** cached prompts reach the first token up to 33x faster, and cost per token rises with prompt
  length, so no single prefill rate describes this GPU. `vllm bench serve` agreed within 5%.
- **Agent traces:** calibrated from 20,008 real SWE-agent runs. Tool outputs are far smaller than the plan
  assumed, with a median final context of 11,788 tokens. Tool pause times stay assumed: no dataset I could
  check records them.
- **Storage requirements:** [`results/requirements`](results/requirements) gives bandwidth, chunk IOPS and
  latency per session size. For example, a median agent session needs about 1.7 GB/s to break even, and
  about 17 GB/s to spend only 10% of the time saved on the restore. These are calculated from Week 1
  timings, assume restores work, and ignore load. A 70B table over assumed prefill rates is a projection.
- **NVMe-oF:** it ran with real kernel drivers over TCP and Soft-RoCE in a VM on a Mac, inside one VM. So it
  shows the protocols work, not what a network costs.

## Roadmap status

| Week | Status |
|---|---|
| 1. First vLLM run | Done on A100. Kaggle T4 run dropped by choice |
| 2. Harness | Done. Datadog push not built |
| 3. Cliff and CPU offload | Done, but offload results are affected by the correctness failure |
| 4. Disk, correctness, profiling | Done. This is where the failure was found |
| 5. Network storage | Partial: protocols run in one local VM, with no real network and no KV I/O replay |
| 6. Prefetch proxy | Built and tested on a fake server. **Not run on a GPU**, on purpose |
| 7. Final sweep, requirements | Requirements table built. **Final sweep and TP=2 not run** |
| 8. Ship | This report and the README. **Dockerfile never built, and no talk or resume update** |

## Limitations

- One GPU type, one small model, mostly one session, and prompts of random token IDs.
- Week 3 runs had a client bug that ended some sessions early. Most points lost 0-2 requests, but LMCache at
  64 sessions sent 13% less load. It's fixed, and the planned re-run didn't happen.
- Colab is a shared, virtualized machine, and the benchmark client ran on the same machine as the server.
- Whether disk tier reads reached the disk, instead of the page cache, wasn't confirmed against disk logs.

## What to do next

1. **Repeat the correctness check with real text.** This is cheap, and it decides how strong the claim is.
2. **Try a vLLM and LMCache version pair known to work together**, starting from what LMCache tests against.
3. **If it still fails, report it upstream** with `results/diag` and `results/week4/correctness`.
4. **Only after restores pass, remove the gate** (`ANTON_ALLOW_UNVERIFIED_OFFLOAD` in the scripts) and run
   `scripts/week6_scenarios.py` and `scripts/week7_final.py`.

## Reproduce

Tests need no GPU:

```bash
uv run pytest
```

Correctness check on a GPU machine, with vLLM and LMCache installed:

```bash
uv run anton server start --config configs/servers/a100_small_cpu.yaml
```

```bash
uv run anton correctness record --server configs/servers/a100_small_cpu.yaml --label cpu --out results/check
```

Raw per-request data isn't in git. Each run's `config.json` and `env.json` record its settings and commit.
