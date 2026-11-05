# SETUP — environment and version log

Record every version, platform, and setup result here so runs are reproducible.
Never put secrets (HF_TOKEN, DD_API_KEY) in this file.

## Local
| Item | Value |
|---|---|
| OS | macOS (Darwin 25.6.0, arm64) |
| uv | 0.12.3 |
| Python (local) | 3.14 (`.python-version`) |
| requires-python | >=3.12 (numpy 2.5 needs 3.12; also matches Kaggle/Colab) |

Unless noted, the values below were run and reported by Tejasvi by hand (recorded 2026-11-05).
A blank means "not recorded yet", not "failed". From now on, each benchmark run also saves its
own `env.json`, which is the authoritative record for that run.

## Status at a glance
| Piece | Status |
|---|---|
| Kaggle T4 + Qwen2.5-0.5B via vLLM | Works |
| Kaggle T4 + Llama-3.2-3B | Not run. Dropped from Week 1: Kaggle stays a Qwen test bed, and the A100 does the Llama runs |
| Colab A100 + Llama-3.1-8B bf16, 32,768 context | Works, Week 1 sweep measured (see README) |
| LMCache on any platform | Not run yet (Week 3) |
| Hugging Face gated model access | Approved and tested |
| Datadog | Account and API key configured |
| AWS EC2 SSH | Works |
| NVMe-oF target + Soft-RoCE on AWS | **Blocked on the tested image** (see AWS) |
| Azure | **Blocked**: subscription restrictions stop VM creation |

**iCloud problem on this Mac.** The repo is on an iCloud-synced Desktop, and iCloud marks files in
`.venv` as hidden (it also leaves duplicates like `project_anton 2.pth`). Python 3.13+ skips hidden
`.pth` files, so `import project_anton` and `uv run anton` fail on and off with
`ModuleNotFoundError`. `uv run pytest` still works because `pyproject.toml` sets `pythonpath`.
The lasting fix is to keep the virtualenv out of iCloud by adding this to `~/.zshrc`:
`export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/project-Anton"`. Kaggle, Colab and CI aren't affected.

## Kaggle (2x T4)
| Item | Value |
|---|---|
| Python | |
| CUDA / driver | |
| vLLM | 0.33.0 |
| lmcache | |
| lmcache works on T4? | |
| KV cache size (Qwen2.5-0.5B, 4k ctx) | |
| KV cache size (Llama-3.2-3B, 16k ctx) | |

## Colab (A100 80GB)
| Item | Value |
|---|---|
| Python | 3.13.16 (Colab), 3.12.3 (harness virtualenv) |
| CUDA / driver | 13.0 / 580.82.07 |
| vLLM | 0.33.0 |
| Compute units/hr | About 8.88 (A100 + High-RAM, shown by Colab on 2026-11-05) |
| Model | meta-llama/Llama-3.1-8B-Instruct, bf16 |
| Max context | 32,768 tokens |
| GPU KV cache size | 56.3 GiB, 461,216 tokens |
| Max concurrency at 32K | 14.08x (a theoretical limit from vLLM's startup log, not a measured load) |

Cross-check (arithmetic only): 461,216 tokens x 128 KiB/token = 56.30 GiB, and
461,216 / 32,768 = 14.08. Both match the formula in `analysis/kv_math.py`, and
`tests/test_kv_math.py` pins them.

**Colab can silently hand you a different GPU.** On 2026-11-05 a new runtime asked for an A100 and
got an L4, with only a small popup saying so. An existing A100 session was still available in
another notebook. Check `nvidia-smi` or `env.json` before trusting a run, and delete any fallback
runtime, since it uses units too.

The earlier plan assumed a 40 GB A100. This one has 80 GB, so 8B fits far more KV cache than
planned. Week 3's "cliff" (where agents start evicting each other) will need more concurrent
agents or a smaller `--gpu-memory-utilization` to appear. Decide that when Week 3 starts.

## AWS ($100 credits, no GPU use planned)
| Item | Value |
|---|---|
| EC2 SSH | Works |
| Image tested | Amazon Linux |
| Kernel | |
| Instance type | |
| `nvme-tcp` | Available |
| `nvmet`, `nvmet-tcp` | **Not available** |
| `rdma_rxe` | **Not available** |
| `nvmet-rdma`, `nvme-rdma`, `null_blk`, `siw` | |

What this means, in plain words:
- NVMe-oF has two sides. The **initiator** (`nvme-tcp`) is the machine that uses the remote disk.
  The **target** (`nvmet` + `nvmet-tcp`) is the machine that serves it. This image can only be
  the initiator, so Week 5 can't run on it alone.
- `rdma_rxe` is Soft-RoCE, which is RDMA emulated in software. Without it, no RDMA experiment
  runs on this image.
- These results only cover "the tested installation". Other images or an extra kernel-modules
  package may behave differently. **Not tested yet:** an Ubuntu image with
  `linux-modules-extra-$(uname -r)`. Testing that is the next AWS step, and it needs approval
  because the VM costs money while running.

## Azure
Blocked: Azure for Students stops VM creation because of subscription restrictions. Week 5 moves
to AWS if the module test above works. Any Soft-RoCE result stays labeled as emulated RDMA.

## Datadog (optional until Week 2)
| Item | Value |
|---|---|
| Account + API key | Configured (key kept out of git) |
| DD_SITE | |
| Test metric returned 202? | |

## Software test results (2026-11-05)
| Check | Result |
|---|---|
| `pytest`, Python 3.14 (macOS) | 146 passed |
| `pytest`, Python 3.12 (macOS) | 146 passed |
| `ruff check` | Passed |
| Fresh clone, notebook's install steps, Python 3.12 | `anton --help` and imports work |
| Fake server set to 33,333 tokens/s | Pipeline measured 33,099 tokens/s (0.7% off) |
| Notebook code on Colab A100 | Server, sweep, analysis, cross-check and stop cells all ran (found 2 bugs, both fixed) |
| Notebook's Google Drive and archive cells | **Not run** (results were saved locally and downloaded) |
| Notebook on Kaggle | **Not run** |
| GitHub Actions CI | Passed on the Week 1 pull request; Week 2 branch not pushed yet |
| Week 2 replay on a real server | **Not run** |

These check the software only. The fake server's numbers are not performance measurements.

## Known limitations
- **GPU differs from the JD target.** Measurements use 8B on one A100 and small models on T4, not
  70B on 4x H200. Anything about 70B is a projection and must be labeled as one.
- **No real RDMA or BlueField hardware.** The best available option is emulated Soft-RoCE, and
  that isn't confirmed to work yet. Its latency says nothing about hardware RoCE.
- **Kaggle/Colab can't load kernel modules.** Storage-protocol work has to run on a VM.
- **Shared machines.** Kaggle and Colab are shared and virtualized, and the benchmark client runs
  on the same machine as the server, so expect noise. Report spread, not just averages.
- **Uncalibrated traces.** Week 2 token counts and pause times are assumptions from the plan, not
  fitted to real agent logs. With them, sessions hit the 32k limit after 11-19 turns, not 10-40.
- **Replay recomputes each response.** Prompts include the trace's response tokens so runs repeat
  exactly, but the model generates different tokens, so its cached response can't match. Each turn
  recomputes one response that a real agent wouldn't, which makes post-tool TTFT slightly pessimistic.
- **Sweep points share a server.** A later point can start with the cache still holding entries from
  an earlier one. They can't match its prompts, but they occupy space until evicted.
- **Synthetic prompts.** Week 1 prompts are random token IDs. Their prefill cost should be
  representative, but they aren't real text, and the generated output is meaningless.
