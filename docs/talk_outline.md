# Talk outline: 5 minutes

Each slide gets about 40 seconds. Numbers come from REPORT.md.

1. **The problem.** AI coding agents pause while tools run. During the pause, the server can evict the
   agent's KV cache, and the next step recomputes 10k-30k tokens. Opening line: even a waiting agent can
   hold GPU memory, or lose its work.
2. **The idea.** Move the cache to cheaper storage during the pause, bring it back after, or prefetch it
   just before the tool ends. Question: does it help, and how fast must storage be?
3. **The setup.** Llama-3.1-8B on one A100, vLLM and LMCache, agents calibrated from 20,008 real
   SWE-agent runs, and a harness that measures TTFT, TPOT and goodput under load.
4. **The cliff.** GPU-only serving falls apart past 48 agents: goodput drops from 214 to 40 tokens/s, and
   a cache miss waits up to 32 s. Show `results/final/sweep/comparison/cliff.png`.
5. **The twist: wrong answers.** On the newest versions, restored caches changed the model's output on most
   prompts, with no error and a "cache hit" in the metrics. Show the correctness table. A check that
   compares outputs caught it; a dashboard wouldn't have.
6. **The fix.** An older version pair passes in multi-process mode. A second, narrower bug remains
   in-process: prompts that are exact multiples of 256 tokens.
7. **The payoff.** With correct restores, offload doubles goodput (62 to 130 tokens/s) and gets within 2%
   of ideal task time. At 96 agents, it keeps 3.3x GPU-only goodput.
8. **Prefetch didn't help.** Restores take only 0.2-0.4 s, so there was little to hide, and warm-ups cost
   7-12% goodput. Repeated four times before concluding.
9. **Storage requirements.** A restore must deliver 1.2-1.9 GB/s to break even, 12-16 GB/s to spend only a
   tenth of the time it saves. NVMe-oF across two cloud VMs gave 0.96 GB/s: the network is the bottleneck.
10. **Takeaways.** Validate correctness before speed; offload helps agents; storage bandwidth per GPU is the
    number SSD and network designs must hit.
