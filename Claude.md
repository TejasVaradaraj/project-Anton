# project-Anton — Project Context

## Who I am
- Tejasvi Varadaraj, BS Computer Science at UT Dallas (expected May 2028). GitHub: TejasVaradaraj
- Background: application/integration engineering, not systems. Strong at: Python, measured before/after
  results, eval harnesses, CI quality gates, fault-tolerant pipelines.
  - VeriFin (AI Society at UTD): RAG over SEC filings, FinanceBench 31% -> 78%, hallucinations 22% -> 3%,
    pytest eval harness in GitHub Actions, Redis job queue, AWS ECS/RDS/S3, 4.2s p95 latency.
  - KiCad MCP Server: 57 -> 25 tools, lazy S-expression indexer ~40x faster, 466 unit tests.
  - Hackermatch: PERN/TypeScript, Postgres concurrency (unique indexes, triggers, row locks), Socket.io.
- NEW to: GPU inference, vLLM, KV cache, storage (NVMe/NVMe-oF), RDMA, Linux kernel stuff.
  **Explain things in simple vocabulary, step by step.** Assume no systems background.

## The goal
Raise my resume's match with an internship JD from ~40% to ~90% by building a real project.
I can only apply to 10 jobs per 6 months at this company, so I plan to apply after Week 3 (once I have real numbers).

### The JD (SoC Architect team, memory/storage company, San Jose, intern)
- Deploy vLLM serving Llama-3.3-70B on 4x H200.
- Build an automated benchmark harness replaying long-context, multi-turn, and agent-style request traces;
  record TTFT, TPOT, throughput, goodput.
- Configure KV cache offloading (LMCache, NVIDIA Dynamo/NIXL) to network-attached storage over NVMe-oF and RDMA
  (BlueField-3 storage server, then their SSD prototype).
- Daily experiment sweeps (request rate, context length, cache size), reproducible in git.
- Python/pandas analysis of bottlenecks across GPU, network, SSD.
- Validate KV cache correctness with FPGA/SoC/firmware engineers.
- Summarize measured storage requirements (bandwidth, IOPS, latency) for future SSD controllers.
- Wants: vLLM/SGLang/TGI, PyTorch on NVIDIA GPUs under Linux, KV cache, prefix caching, batching,
  tensor/pipeline parallelism, TTFT/TPOT/throughput, profiling, pandas/plotting,
  LMCache/Dynamo/NIXL, RDMA/RoCE, NVMe/NVMe-oF, Docker/Kubernetes.

## The project I chose: project-Anton — KV-cache offloading for tool-calling AI agents
**Real problem:** AI coding agents pause seconds to minutes while tools run (tests, grep, file reads).
During the pause the server evicts the agent's KV cache ("the model's scratch notes"), so the next step
re-computes the whole 15k–40k token context -> slow TTFT, wasted GPU.

**Fix:** offload the agent's KV cache to cheaper tiers during the pause (CPU RAM -> local disk -> remote NVMe over
the network), and an agent-aware proxy that PREFETCHES it back onto the GPU just before the tool finishes
(warm-up request with max_tokens=1 at predicted tool end). Measure how fast storage must be for this to pay off.

**Prior work to cite (not claiming invention):** CachedAttention (USENIX ATC 2024), "Continuum" (2025, KV cache
time-to-live for agents), vLLM PagedAttention, LMCache docs.

**KV math:** bytes/token = 2 x layers x KV heads x head_dim x 2 bytes
- Llama-3.1-8B: 2x32x8x128x2 = 128 KiB/token
- Llama-3.2-3B: 2x28x8x128x2 = 112 KiB/token
- Llama-3.3-70B: 2x80x8x128x2 = 320 KiB/token
- Break-even: loading beats recompute when storage bandwidth > (KV bytes/token) x (GPU prefill tokens/s)

## My resources (decided)
| Resource | Use |
|---|---|
| Kaggle (30 GPU hrs/week, 2x T4 16GB, ~29GB RAM, 12h sessions, background "Save & Run All") | All development/debugging, long sweeps, TP=2 experiment |
| Google Colab (200 compute units/month; A100 40GB ~11-13 units/hr => ~15-17 A100 hrs/month; L4 fallback) | Headline Llama-8B results only |
| Azure ($100 credits) | NVMe-oF + Soft-RoCE experiments (needs full VMs to load kernel modules; Colab/Kaggle can't) |
| Datadog (GitHub Student Pack perk — NOT YET SET UP, optional until Week 2) | Live dashboards; git Parquet stays source of truth |

Models: A100 -> Llama-3.1-8B-Instruct bf16. 1x T4 -> Llama-3.2-3B-Instruct with `--dtype half` (T4 has no bf16).
2x T4 -> Llama-3.1-8B fp16 TP=2. Colab L4 24GB -> 8B bf16 fallback.

## 8-week plan (summary)
- **Week 0 — Setup** (IN PROGRESS, see below).
- **Week 1 — First vLLM run.** Kaggle T4 + 3B: async streaming client measuring TTFT/TPOT/E2E; plot TTFT vs prompt
  length (1k–16k); cold vs warm prefix cache. Sanity-check vs `vllm bench serve`. ~1 A100 hr to repeat with 8B.
- **Week 2 — Harness.** Seeded trace generators: long-context, multi-turn, agent (shared ~3k system/tool prefix,
  10–40 steps, +500–3k tokens per tool output, log-normal tool pauses per tool type, Poisson arrivals; calibrate
  from public SWE-agent trajectories on HF). Send TOKEN IDS to /v1/completions; append the trace's own response
  tokens (max_tokens + ignore_eos) for determinism. Async replayer -> requests.jsonl. sysmon: nvidia-smi dmon,
  iostat, sar, vLLM /metrics. Every run -> results/<run_id>/ with config, git SHA, env, raw data. Resumable sweeps
  (skip finished run_ids) because Colab/Kaggle disconnect. Optional Datadog push via HTTP API (low-cardinality tags).
- **Week 3 — Show the problem + CPU offload (MVP, then APPLY).** Sweep concurrent agents 1..32 on GPU-only ->
  find the "cliff" (hit rate drops, TTFT after tool call jumps). Add LMCache CPU offload
  (`--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'` + LMCACHE_* env vars; check
  docs for pinned version). Goodput with SLO (e.g. TTFT<2s, TPOT<100ms).
- **Week 4 — Disk tier + correctness + profiling.** fio --direct=1 baseline of the Colab/Kaggle disk first.
  Page cache trap: can't drop_caches without root -> use posix_fadvise(DONTNEED) or vmtouch -e; confirm with iostat.
  Correctness: greedy outputs + logprobs with vs without offload, with a no-offload-vs-no-offload noise baseline.
  Stacked bar TTFT breakdown per tier.
- **Week 5 — Network storage on Azure.** Plan A (if T4 GPU quota granted): NC4as_T4_v3 + storage VM in same VNet,
  accelerated networking, proximity placement group; end-to-end NVMe-oF/TCP then Soft-RoCE (rdma_rxe; fallback siw).
  Plan B (likely, student credits): record KV I/O trace on GPU runs, replay it on 2 CPU VMs over NVMe-oF with
  O_DIRECT + fio, then combine in pandas with the measured load-time->TTFT relationship (validated on local tiers).
  Also export a RAM disk (null_blk/brd) to isolate network vs SSD latency.
- **Week 6 — Agent-aware prefetch proxy** (FastAPI). X-Agent-Id header, POST /tool_start, per-tool duration history,
  warm-up request before predicted end. Compare: A baseline GPU-only, B offload, C offload+prefetch, D ideal.
  Measure post-tool TTFT, agent task time, wasted prefetches vs lead time.
- **Week 7 — Final sweep + storage requirements.** Break-even theory vs measured; requirements table
  (bandwidth/IOPS/latency per workload); TP=2 on Kaggle 2xT4 vs TP=1 on A100/L4; 70B on 4xH200 projection
  (labeled as projection).
- **Week 8 — Ship.** README (problem, diagram, 4 key plots, requirements table, one-command reproduce, honest
  limitations), REPORT.md, Kaggle/Colab notebooks per experiment, Dockerfile, 5-min talk, resume update.

Repo layout: configs/{servers,traces,sweeps}, src/project_anton/{traces,harness,proxy,sysmon,correctness,analysis},
scripts/, notebooks/, tests/, results/ (summaries in git, raw in Drive/Blob). SETUP.md logs every version.

## WHERE I LEFT OFF
Done:
- Step 1 (approvals): HF account + Llama access requests + HF token; Kaggle phone verification; Azure checks/quota.
- Step 2 (keys list) — I only have HF_TOKEN. No Datadog account/key yet.
- Step 3 (GitHub repo `project-Anton`, uv, skeleton folders, .gitignore, SETUP.md).
- Step 4a (Kaggle notebook created: Accelerator GPU T4 x2, Internet On, Persistence Files only).

**NEXT: Step 4b** — then 4c, 5, 6, 7:
- 4b: Kaggle Add-ons -> Secrets -> add HF_TOKEN (attach to notebook). (DD_API_KEY later.)
- 4c: Kaggle smoke test cells: nvidia-smi (2x T4) -> load secret via
  `from kaggle_secrets import UserSecretsClient` -> `pip install vllm` (record version, then pin) ->
  start `vllm serve` with subprocess.Popen in background (Qwen/Qwen2.5-0.5B-Instruct, --dtype half,
  --max-model-len 4096, --enable-prefix-caching), poll /health -> one /v1/completions request ->
  grep KV cache size from log -> switch to Llama-3.2-3B (--max-model-len 16384) -> `pip install lmcache`
  matching vLLM version and record whether it works on T4. Stop session when done.
- 5: Colab: Secrets (key icon) HF_TOKEN via `google.colab.userdata`; A100 + High-RAM; note units/hr;
  mount Drive for logs; run Llama-3.1-8B bf16; record KV cache size; "Disconnect and delete runtime".
- 6: Datadog (optional until Week 2): claim via GitHub Student Developer Pack, find DD_SITE from URL,
  create API key, send a test metric to https://api.<DD_SITE>/api/v2/series (expect 202).
- 7: Azure: $100 budget with 25/50/75% alerts; SSH key; Standard_B2s Ubuntu 24.04 test VM with auto-shutdown;
  test `modprobe nvmet nvmet-tcp nvme-tcp nvmet-rdma nvme-rdma rdma_rxe null_blk` (if fail:
  `apt install linux-modules-extra-$(uname -r)`; fallback `siw`); record results; Stop until
  "Stopped (deallocated)", then delete the VM.

After setup: write the Week 1 streaming TTFT/TPOT benchmark client + Kaggle notebook.

## Rules for helping me
- Simple vocabulary; define new terms. Step-by-step with exact commands.
- Never put secrets (HF_TOKEN, DD_API_KEY) in code, notebooks, or git — use Kaggle/Colab Secrets.
- Pin versions; record everything in SETUP.md.
- Be honest about limitations (smaller model, emulated RDMA, synthetic tool timings, projections vs measurements).
