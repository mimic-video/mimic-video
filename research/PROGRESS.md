# Progress log

Human-readable narrative of the autoresearch run. The agent appends one short entry per cycle
(the configs tried across the GPUs, each run's val_loss, what won, the next idea). Newest at the bottom.

Current best: **5.405514** (baseline, run_baseline, commit 512f69a)

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
