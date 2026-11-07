# Draft resume lines

For Tejasvi to edit. Each line is backed by REPORT.md.

**project-Anton: KV-cache offloading for AI agents** (Python, vLLM, LMCache, PyTorch, NVMe-oF, AWS, Docker)

- Built a benchmark harness that replays calibrated AI-agent workloads against vLLM on an A100, measuring
  TTFT, TPOT and goodput under load, with 256 automated tests.
- Found that KV-cache restores in vLLM 0.33 with LMCache 0.5.6-0.5.7 silently changed model output (up to
  19 of 20 prompts) by comparing greedy outputs against controls, then identified a version pair that
  passes.
- Showed that correct KV-cache offload doubled agent goodput (62 to 130 tokens/s) and kept 3.3x GPU-only
  goodput at 96 concurrent agents.
- Derived storage requirements per agent workload (1.2-16 GB/s, 48-485 IOPS at 32 MiB) and measured
  NVMe-oF over TCP and Soft-RoCE between AWS VMs at about 0.96 GB/s.
- Found and fixed a prefetch-proxy bug that crashed the vLLM engine by aborting requests mid-restore.
