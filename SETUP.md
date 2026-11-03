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

## Kaggle (2x T4)
| Item | Value |
|---|---|
| Python | |
| CUDA / driver | |
| vllm | |
| lmcache | |
| lmcache works on T4? | |
| KV cache size (Qwen2.5-0.5B, 4k ctx) | |
| KV cache size (Llama-3.2-3B, 16k ctx) | |

## Colab (A100 40GB)
| Item | Value |
|---|---|
| Python | |
| CUDA / driver | |
| vllm | |
| Compute units/hr | |
| KV cache size (Llama-3.1-8B bf16) | |

## Azure
| Item | Value |
|---|---|
| Budget + alerts | |
| VM size / image | |
| Kernel | |
| modprobe nvmet nvmet-tcp nvme-tcp nvmet-rdma nvme-rdma rdma_rxe null_blk | |
| Fallbacks needed (linux-modules-extra, siw) | |

## Datadog (optional until Week 2)
| Item | Value |
|---|---|
| DD_SITE | |
| Test metric returned 202? | |
