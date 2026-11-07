# project-Anton report

## Summary

This project measures whether offloading an AI agent's KV cache to cheaper storage while its tools run,
then restoring or prefetching it, is worth doing, and how fast the storage must be.

**Answer, on one A100 80GB with Llama-3.1-8B:**

- **Offload works, once restores are correct.** With LMCache's multi-process mode on vLLM 0.25.1, offload
  doubled goodput on a cache-starved agent workload and cut post-tool p95 TTFT by about 5x. At 96
  concurrent agents on the same GPU cache, it kept 3.3x the goodput of GPU-only serving.
- **Prefetch didn't help.** Warm-ups sent just before a tool finished gave 7-12% lower goodput than plain
  offload, in four runs with three lead times.
- **Correctness was the hard part.** On newer versions (vLLM 0.33.0 with LMCache 0.5.6 or 0.5.7), restored
  caches silently changed the model's output. In-process LMCache on the older pair also fails, but only on
  prompts that are exact multiples of its 256-token chunk. Nothing reported an error in either case.
- **Storage needs:** a restore must deliver about 1.2-1.9 GB/s just to beat recomputing. Multi-process
  delivered 3.3-12.2 GB/s from CPU RAM. NVMe-oF over a real 2-VM AWS network delivered about 0.96 GB/s,
  which would lose to recomputing.

Everything ran on Colab A100s. Prompts were random token IDs except where real text is noted, and most
results come from one run.

## 1. Correctness: restores must give the same answer

Each prompt is sent fresh, then again from cache, then evicted and sent again, with greedy decoding. One
server, so only the cache state differs. A split control sends the prompt with only its whole 256-token
chunks cached, which is exactly what a correct restore looks like, so it shows normal rounding.

Real text from SWE-agent runs, 20 prompts of 1,000-30,000 tokens:

| Versions | Path | Outputs changed | First-token logprob change, median (max) | Verdict |
|---|---|---:|---:|---|
| vLLM 0.33.0 | GPU only, split control | 2 | 0 (0.062) | Rounding bar |
| vLLM 0.33.0 + LMCache 0.5.6 | In-process CPU restore | **19** | 0.779 (6.431) | Fails |
| vLLM 0.33.0 + LMCache 0.5.6 | Multi-process restore | **12** | 0.029 (1.267) | Fails |
| vLLM 0.33.0 + LMCache 0.5.7 | Multi-process restore | **14** | 0.012 (1.392) | Fails |
| vLLM 0.25.1 | GPU only, prefix cache | 2 | 0 (0.022) | Baseline |
| vLLM 0.25.1 + LMCache 0.5.1 | In-process CPU restore | 2 | 0 (0.035) | Passes at these lengths |
| vLLM 0.25.1 + LMCache 0.5.1 | In-process disk restore | 2 | 0 (0.035) | Passes at these lengths |
| vLLM 0.25.1 + LMCache 0.5.1 | Multi-process restore | 1 | 0 (0.066) | Passes |

Prompts that are exact multiples of 256 tokens (15,872, 31,744 and 32,000), 15 prompts:

| Versions | Path | Outputs changed | First-token logprob change, median (max) | Verdict |
|---|---|---:|---:|---|
| vLLM 0.25.1 | GPU only, split control | 0 | 0 (0.001) | Baseline |
| vLLM 0.25.1 + LMCache 0.5.1 | Multi-process restore | 1 | 0 (0.027) | **Passes** |
| vLLM 0.25.1 + LMCache 0.5.1 | In-process CPU restore | **15** | 6.375 (6.987) | **Fails** |

- **The in-process failure at exact multiples** reports the whole prompt as cached (31,999 of 32,000
  tokens) and answers in about 83 ms, as fast as a GPU cache hit. So it seems to skip loading the cache, and
  even its plain cache-hit request is wrong. That explains the impossible 83 ms restores in the timings.
- **Random token IDs** gave the same verdicts on the older pair: in-process CPU changed 2 of 12, the same as
  its prefix cache.
- **Why the newer versions fail isn't confirmed.** An open LMCache pull request,
  [LMCache/LMCache#4887](https://github.com/LMCache/LMCache/pull/4887), describes copy errors with newer vLLM
  KV layouts in-process. It wouldn't directly explain the multi-process failures. LMCache 0.5.7 can't run
  in-process with vLLM 0.33.0 at all, because it removes a module vLLM imports.

The offload scripts now refuse to run on any vLLM version other than 0.25.1 unless explicitly overridden,
and all later results use multi-process mode.

## 2. Restore speed

On an 8 GiB GPU cache, vLLM 0.25.1, 5 repeats per length. Restore time is the evicted request's extra time
over a GPU cache hit.

| Prompt tokens | Recompute (ms) | Multi-process restore (ms) | Delivered (GB/s) | Needed to break even (GB/s) |
|---:|---:|---:|---:|---:|
| 1,000 | 55 | 30 | 3.3 | 1.8 |
| 4,000 | 260 | 62 | 8.1 | 1.9 |
| 16,000 | 1,350 | 192 | 10.8 | 1.5 |
| 32,000 | 3,467 | 343 | 12.2 | 1.2 |

- **Saving barely slows cold requests in multi-process mode:** 3,571 ms against 3,503 ms at 32,000 tokens.
  In-process saves cost more (3,925 ms), and disk saves far more (12,580 ms).
- **In-process disk restores barely beat recomputing,** and lose at 4,000 tokens (559 against 313 ms).
- **On the newer versions, in-process restores ran at about 0.8 GB/s,** slower than recomputing.

Full tables: [`results/final/tiers`](results/final/tiers) and
[`results/requirements/multi_process_restores.csv`](results/requirements/multi_process_restores.csv).

## 3. Does offload and prefetch help agents?

24 calibrated agents on an 8 GiB GPU cache, so caches get evicted during tool pauses. Multi-process offload.
Goodput counts requests with TTFT under 2 s and TPOT under 100 ms.

| Scenario | Goodput (tokens/s) | Met targets | Post-tool TTFT p50 / p95 (ms) | Median agent task (s) | Previous prompt cached |
|---|---:|---:|---:|---:|---:|
| A: GPU only | 62 | 69.0% | 271 / 9,247 | 217 | 62% |
| B: Offload | **130** | **96.5%** | 153 / 1,689 | 161 | 99% |
| B: Offload, repeat | 130 | 95.6% | 156 / 1,883 | 161 | 99% |
| C: Offload + prefetch, 0.5 s lead | 121 | 92.3% | 156 / 2,347 | 162 | 99% |
| C: repeat | 118 | 92.1% | 161 / 2,787 | 162 | 99% |
| C: 2 s lead | 115 | 89.7% | 159 / 3,101 | 162 | 99% |
| C: 5 s lead | 117 | 91.5% | 159 / 2,511 | 161 | 99% |
| D: Ideal, large GPU cache | 143 | 100% | 108 / 239 | 159 | 99.8% |

- **Offload gets most of the way to ideal.** It doubles goodput and brings task time within 2% of ideal.
- **Prefetch is consistently a bit worse.** The two B runs agree within 0.3%, and every C run is 7-12% below
  them. Restores already take only 0.2-0.4 s, so there's little latency left for prefetch to hide.
- **Why prefetch costs goodput isn't confirmed.** Likely causes are the extra warm-up requests competing for
  the GPU, and warm-ups pulling caches onto a small GPU cache, pushing out other agents'.
- **A proxy bug crashed the first prefetch run.** When an agent's real request arrived during a warm-up,
  the proxy cancelled the warm-up while its cache was loading, and vLLM crashed with
  `assert req_id in self.requests`. It no longer cancels warm-ups once they're sent. The same assertion
  crashed an earlier multi-process run, which this may explain.
- C and D ran on a later A100 session than A and the first B. The repeats of B and C share one session.

Tables: [`results/final/scenarios/all_scenarios.csv`](results/final/scenarios/all_scenarios.csv).

## 4. The cliff, with and without offload

Calibrated agents arriving at 1 per second. Both setups get the same 44 GiB GPU cache, leaving room for the
multi-process LMCache server's own GPU memory.

| Sessions | Setup | Previous prompt cached | Post-tool TTFT p50 / p95 (ms) | TPOT p50 (ms) | Met targets | Goodput (tokens/s) |
|---:|---|---:|---:|---:|---:|---:|
| 32 | GPU only | 99.8% | 119 / 263 | 18 | 100% | 147 |
| 32 | Offload | 99.8% | 123 / 282 | 18 | 100% | 147 |
| 48 | GPU only | 98.5% | 167 / 473 | 29 | 99.4% | 214 |
| 48 | Offload | 99.8% | 164 / 394 | 29 | 100% | 217 |
| 64 | GPU only | 89.0% | 255 / 8,152 | 50 | 81.0% | 160 |
| 64 | Offload | 99.7% | 217 / 540 | 43 | 100% | **260** |
| 80 | GPU only | 58.7% | 874 / 35,140 | 101 | 49.7% | 56 |
| 80 | Offload | 98.0% | 317 / 3,298 | 59 | 87.5% | **211** |
| 96 | GPU only | 51.2% | 16,348 / 40,760 | 222 | 36.9% | 40 |
| 96 | Offload | 94.8% | 548 / 7,489 | 85 | 58.3% | **132** |

- **GPU-only hits a cliff past 48 sessions.** Agents' caches get evicted, misses wait up to 32 s, and
  recomputing them also slows every other request (TPOT passes 100 ms).
- **Offload moves the cliff.** It keeps 95-99% of caches and stays fully within targets at 64 sessions. At 96
  it starts to degrade too, as compute becomes the limit.

Plot: [`results/final/sweep/comparison/cliff.png`](results/final/sweep/comparison/cliff.png).

## 5. Storage requirements

From the final runs: the bandwidth a restore needs so it costs only part of the time it saves. Calculated
from measured cold and warm TTFT, assuming sequential whole-chunk reads with no queueing.

| Workload | Context (tokens) | KV size (GB) | Time saved (ms) | Break even (GB/s) | Use half the saving (GB/s) | Use 10% of the saving (GB/s) |
|---|---:|---:|---:|---:|---:|---:|
| Agent, median session | 11,788 | 1.55 | 950 | 1.6 | 3.3 | 16.3 |
| Agent, p75 session | 28,335 | 3.71 | 2,927 | 1.3 | 2.5 | 12.7 |
| Agent, p90 session | 30,337 | 3.98 | 3,187 | 1.2 | 2.5 | 12.5 |

At LMCache's 32 MiB chunks, the median session needs 48-485 chunk reads per second. A 70B table over
assumed prefill rates is in [`results/requirements`](results/requirements), and it's a projection.

**Against measured storage paths:**

| Path | Read, 32 MiB chunks (GB/s) | Median session restore | Beats recomputing? |
|---|---:|---:|---|
| Multi-process LMCache, CPU RAM, same machine | 10.8 (at 16k tokens) | About 0.15 s | Yes, by about 6x |
| NVMe-oF over TCP, 2 AWS VMs, RAM disk target | 0.96 | About 1.6 s | No |
| NVMe-oF over Soft-RoCE, same VMs | 0.97 | About 1.7 s | No |

The AWS VMs were the only type this account's free plan allowed (`m7i-flex.large`). One network flow peaked
at 5 Gbit/s and four reached 12.4 Gbit/s, so the network, not NVMe-oF, set the limit. Soft-RoCE is
software emulation, so it says nothing about RDMA hardware. No KV traffic was replayed over the network;
fio read at LMCache's chunk size instead.

## Roadmap status

| Week | Status |
|---|---|
| 1. First vLLM run | Done |
| 2. Harness and traces | Done. Datadog push not built (optional) |
| 3. Cliff and offload | Done, redone on the verified versions |
| 4. Disk, correctness, profiling | Done, including real text and exact chunk lengths |
| 5. Network storage | Done on 2 AWS VMs, with no KV traffic replay |
| 6. Prefetch proxy | Done. Prefetch didn't help |
| 7. Final sweep and requirements | Done. TP=2 dropped |
| 8. Ship | Report, README and Dockerfile done. Talk and resume drafts in [`docs`](docs) |

## Limitations

- One GPU type, one small model (8B, not 70B), one run per setting except the prefetch repeats.
- Most prompts are random token IDs. Real text was used for the correctness checks only.
- Tool pause times are assumptions. The calibration dataset records no timing.
- The benchmark client ran on the same shared Colab machine as the server.
- Restores are verified one request at a time, not under load.
- Disk-tier reads weren't confirmed against disk logs, so some may have come from the page cache.
- The verified versions are older than current releases. Newer versions fail the check here.

## Reproduce

Tests need no GPU:

```bash
uv run pytest
```

On a fresh Colab A100, from the repo root, with a Hugging Face token in
`/root/.cache/huggingface/token`:

```bash
bash scripts/colab_setup.sh
```

```bash
PATH=/tmp/old/bin:$PATH HF_HUB_OFFLINE=1 bash scripts/final_runs.sh results/final
```

```bash
PATH=/tmp/old/bin:$PATH HF_HUB_OFFLINE=1 bash scripts/finish_runs.sh results/final
```

The first runs restore timings, the prefetch comparison and the sweep. The second adds the exact-length
correctness check and skips anything already finished.

Raw per-request data isn't in git. Each run's `config.json` and `env.json` record its settings and commit.
