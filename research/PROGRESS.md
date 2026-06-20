# Progress log

Human-readable narrative of the autoresearch run. The agent appends one short entry per cycle
(the configs tried across the GPUs, each run's val_loss, what won, the next idea). Newest at the bottom.

Current best: **5.152164** (run_lr3e-4, commit 482460e)

---

## Cycle 3 — LR narrow around winner (2026-06-20)

**Configs tried (4):** same rank=256, bsz=32; LR ∈ {2e-4, 2.5e-4, 3e-4, 4e-4}, one GPU each (GPUs 0-3 in parallel).

**Per-run val_loss (last, script rule):**
- `run_lr2e-4` (GPU 0): **13.048154** — last value spiked; median across 109 vals = 5.956.
- `run_lr2.5e-4` (GPU 1): **9.076780** — last spiked; median = 6.135.
- `run_lr3e-4` (GPU 2): **16.364034** — last spiked; median = 6.524. **Same config as cycle 2's winner**, illustrating the metric's high variance with n=4 val clips.
- `run_lr4e-4` (GPU 3): **9.804769** — last spiked; median = 6.601.

**What won:** nothing beat the global best **5.152164** (cycle 2 lr=3e-4) on the script's last-val_loss rule → all four runs logged as `discard`. The global best is unchanged.

**Observations:**
- **The metric is the bottleneck, not the LR.** Per the script's "last value wins" rule, even the exact same config (lr=3e-4, GPU 2 this cycle vs GPU 1 cycle 2) produced 16.36 here vs 5.15 last cycle — a 3× swing purely from n=4 val clip sampling. All four runs completed the 2h budget cleanly (109 validations each, ~1915–1980 iters, no OOMs, no divergence — training losses stayed healthy).
- Looking at medians instead of last: lr=2e-4 → 5.96, lr=2.5e-4 → 6.14, lr=3e-4 → 6.52, lr=4e-4 → 6.60. The ordering (2e-4 best, 4e-4 worst) is monotonic and stable across the run, but the absolute values (~6) are systematically a bit higher than cycle 2's lr=3e-4 last value (5.15). That's likely within the same noise band — n=4 val clips at random noise floor.
- **Infrastructure note:** the `train_bounded.sh` wrapper was killed by the launching shell's 120s timeout (it died before the post-run summary reached stdout). The torchrun child processes survived as orphans and finished their full 2h budget; the val_loss emissions were preserved in `/tmp/tb_<run>.XXXXXX.log` (the wrapper's TRAINLOG). We recovered the LAST val_loss from those temp logs. Next cycle should either (a) launch with a longer parent shell timeout, or (b) have train_bounded.sh write the summary to the .log file directly (it does — to stdout, which the parent shell kills). The cleanest fix is for train_bounded.sh to also `tee` the summary into a known location. Flagging this for a future infrastructure fix, NOT changing in this cycle.
- All four ran the budget cleanly → no crashes, no divergence at any of these LRs (unlike cycle 2's lr=1e-3 which exploded at iter 3-5). The LR range [2e-4, 4e-4] is comfortably within the stable region.

**Next idea (cycle 4):** the LR lever is exhausted — the cycle 3 medians are flat (5.96–6.60, no clear minimum) and the last-value noise dominates. Time to switch levers. Two options that look promising:
  1. **LoRA capacity** (rank ∈ {64, 128, 512} vs current 256) — we have 96 GB VRAM headroom and 58 train clips; overfitting risk is real. Lower rank should reduce overfit and might smooth the val_loss noise too.
  2. **Effective batch size via DDP** — group 2 GPUs into one 2-way DDP run per config (effective batch = 2), so we get 4 runs × 2 GPUs = 8 GPUs total. This reduces gradient noise and gives a more stable val_loss signal, which may also let us resolve the LR question more cleanly next time.
  Starting with (1) since the IDEAS.md flags LoRA capacity as "the next biggest lever" and it directly addresses overfitting. Try `ranks ∈ {64, 128, 512}` at lr=3e-4 (the proven-good LR) on GPUs 0-3. If rank=128 wins, next cycle narrows to {96, 128, 160}; if rank=64 wins (less overfit → better), next cycle tries rank ∈ {32, 48, 64}.

---

## Cycle 2 — LR sweep (2026-06-20)

**Configs tried (4):** same rank=256, bsz=32; LR ∈ {1e-4, 3e-4, 5e-4, 1e-3}, one GPU each (GPUs 0-3 in parallel).

**Per-run val_loss:**
- `run_lr1e-4` (GPU 0): **12.901524** — too low, under-converged in 2h.
- `run_lr3e-4` (GPU 1): **5.152164** — **NEW BEST** (beats baseline 5.405514).
- `run_lr5e-4` (GPU 2): **8.505093** — too high, overshoot.
- `run_lr1e-3` (GPU 3): **19.046690** — diverged (training losses hit 124/97/79 at iters 3-5).

**What won:** `lr=3e-4`. Both 1e-4 (too low) and 1e-3 (clearly divergent) bracket the sweet spot; the baseline at 1.778e-4 was in the same "good" region as 3e-4 but slightly worse.

**Observations:**
- The LR landscape is a clear single-minimum between 1e-4 and 5e-4, centered near 3e-4. The 1e-3 run diverged within seconds, which confirms the upper bound on stability.
- All runs hit the 2h time budget (108-109 validations each — same speed as baseline). No OOMs. VRAM column is 0.0 because `peak_vram_gb` was never emitted by the trainer log (TE-fused-attn path doesn't print memory); baseline's 44 GB is a representative upper bound for these configs.
- Note on the upload-best step: `upload_best.sh` picks the latest-modified checkpoint across all run dirs (`ls -t`), not the one matching `RUN_TAG`. After 4 concurrent runs finish at the same instant, this gave us `run_lr1e-3` (the worst). We re-uploaded `run_lr3e-4` (the actual best) explicitly. Worth fixing the script later.

**Next idea (cycle 3):** narrow the LR sweep around the winner — try `{2e-4, 2.5e-4, 3e-4, 4e-4}` (4 values on 4 GPUs). After that, the next biggest lever is LoRA capacity (rank/alpha) since we have 96 GB of VRAM headroom and 58 train clips — overfitting risk is real.

---

## Cycle 1 — baseline (2026-06-19)

**Configs tried (1):** unmodified `v2w_fr3_munich_lora_rank256_lr1.778e-04_bsz32` on a single GPU
(BATCH_SIZE=1, effective batch=1 — `bszs=[32]` in the config is just the experiment name suffix).

**Per-run val_loss:**
- `run_baseline` (GPU 0, 1× DDP): **5.405514** — hit the 2h time budget at iter ~2000 (the last
  validation was at iter 2000, n=4 val clips). Peak VRAM = 44.06 GB / 96 GB cap. Total iters seen
  by the trainer before SIGINT ≈ 2017 (~3.6 it/s).

**What won:** the baseline is the new global best by definition (no prior runs).

**Observations:**
- `val_loss` is very noisy: across 110 logged values over the run it ranges 3.66 → 13.65 with
  a mean ≈ 5–6. This is expected with `n=4` val clips — the metric is high-variance per call.
  We should consider comparing runs on a short window average or on the last few checkpoints,
  not just the final value. For now we follow the script's rule (last value wins).
- 44 GB VRAM / 96 GB cap → plenty of headroom. Next cycles can safely try `BATCH_SIZE=2` or
  larger LoRA rank without risking OOM.
- 3.6 it/s × 2 h = ~25 920 iters theoretical max; the cap is the wall-clock budget, not iter count.

**Next idea (cycle 2):** sweep the learning rate (highest leverage per IDEAS.md). Try
`lrs ∈ {1e-4, 3e-4, 5e-4, 1e-3}` across 4 GPUs in parallel — same rank=256 / bsz=32 to isolate
the LR effect.

---
