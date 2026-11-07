# Draft LMCache issue: not posted

Tejasvi decides whether to post this. It would go to https://github.com/LMCache/LMCache/issues.

**Title:** In-process connector returns wrong output for prompts that are exact multiples of the chunk size
(vLLM 0.25.1, LMCache 0.5.1)

**What happens.** With `LMCacheConnectorV1` (in-process, CPU backend, chunk size 256), a prompt whose length
is an exact multiple of 256 is reported as a full cache hit (for example 31,999 of 32,000 tokens) on the
second request, and the first token arrives in about 80 ms, as fast as a GPU prefix-cache hit. The output
differs from the same prompt computed fresh on all 15 test prompts (15,872, 31,744 and 32,000 tokens), with
first-token logprob changes of 6.4 to 7.0. No error is logged.

**Expected.** Greedy output matching the fresh computation, as it does for lengths that aren't multiples of
256, and as it does in multi-process mode (`lmcache server`) for the same prompts.

**Setup.** One A100 80GB, Llama-3.1-8B-Instruct bf16, `--kv-cache-memory` 8 GiB, prefix caching on,
`LMCACHE_CHUNK_SIZE=256`, `LMCACHE_LOCAL_CPU=True`. Each prompt is sent fresh, sent again, then evicted with
filler prompts and sent again, all with `temperature=0`.

**Possibly related.** With vLLM 0.33.0, LMCache 0.5.6 in-process and multi-process, and 0.5.7
multi-process, also changed outputs on most restored prompts of ordinary lengths. #4887 describes a layout
copy bug that may explain part of that.

**Reproduction.** `scripts/correctness_matrix.sh` and `scripts/make_text_prompts.py` in
https://github.com/TejasVaradaraj/project-Anton. Results are in `results/final/correctness_exact` and
`results/week4_text`.
