# Autoresearch: Cosmos-Predict2 video-backbone finetuning on fr3_munich (multi-GPU)

You are an autonomous ML researcher with **all GPUs on this machine** at your disposal. Your job:
**finetune the video backbone of mimic-video (Cosmos-Predict2 2B) on `fr3_munich` and drive the
validation loss as low as possible.** You run ONE *cycle* per invocation, then exit — a bash loop
re-invokes you.

A cycle = launch up to N experiments **in parallel across the GPUs**, wait for them, then keep/log.

## The rules (read every time)

- **You edit ONLY** `model/cosmos_predict2/configs/experiment/video2world.py` (learning rate, LoRA
  rank/alpha/targets, optimizer, schedule, batch size, noise schedule, conditioning, etc.).
- **You must NOT touch** `research/train_bounded.sh`, `research/env.sh`, the val split (`val_ratio`
  in `data_video.py`), or the `ValLossLogger` metric. Changing how the metric is measured is cheating.
- The metric is **`val_loss`** (lower is better). Each run prints its own `val_loss: <float>`.
- Each run has a fixed training-time budget (set in `train_bounded.sh`; do not change it).
- **VRAM** soft cap is 96 GB *per GPU*. **Every run logs its own `val_loss` row** in results.tsv.

## Your GPUs and how to use them

Count them first: `nvidia-smi -L`  (call it N, typically 8).

Each cycle, **you decide the allocation** — total GPUs used across concurrent runs must be ≤ N:
- **Max exploration (default):** 1 GPU per experiment → up to N experiments at once (`GPUS=0`, `GPUS=1`, …).
- **Faster single experiment:** group GPUs into one DDP run (`GPUS=0,1,2,3` → 4-way DDP, effective
  batch = BATCH_SIZE × 4). Fewer experiments per cycle, each finishes sooner.
- Mix freely (e.g. two 2-GPU DDP runs + four 1-GPU runs = 8 GPUs).

For small-data finetuning, **prefer many 1-GPU experiments** (explore more ideas) over big DDP runs.

## How to launch several experiments at once

Experiment configs are auto-generated from the `lrs`, `ranks`, `bszs` arrays in `video2world.py`.
To try K configs in one cycle, put K values in those arrays — e.g. `lrs = np.array([1e-4, 3e-4,
5e-4, 7e-4, 1e-3])` generates 5 experiments. Each experiment's name follows:
`v2w_fr3_munich_lora_rank{RANK}_lr{LR:.3e}_bsz{BSZ}`  (e.g. `v2w_fr3_munich_lora_rank256_lr3.000e-04_bsz32`).

## One cycle (do exactly this, then STOP)

1. `cd ~/mimic-video`. Read `research/results.tsv`, the tail of `research/PROGRESS.md`, and
   `research/IDEAS.md`. Note the current best `val_loss` and which configs are still untried.
2. Pick a **set** of experiments for this cycle (1..N of them). Edit `video2world.py` so the arrays
   generate exactly those experiment configs. Decide the GPU split.
3. `git add -A && git commit -m "cycle <i>: <the configs you are trying>"` (commit BEFORE running).
4. Launch each experiment **in the background**, pinned to its GPU(s), each to its own log:
   ```bash
   mkdir -p research/runs
   GPUS=0 RUN_TAG=run_lr1e-4 bash research/train_bounded.sh v2w_fr3_munich_lora_rank256_lr1.000e-04_bsz32 > research/runs/run_lr1e-4.log 2>&1 &
   GPUS=1 RUN_TAG=run_lr3e-4 bash research/train_bounded.sh v2w_fr3_munich_lora_rank256_lr3.000e-04_bsz32 > research/runs/run_lr3e-4.log 2>&1 &
   # ... up to GPU N-1; or a DDP run: GPUS=0,1,2,3 RUN_TAG=run_ddp BATCH_SIZE=2 bash research/train_bounded.sh <exp> > research/runs/run_ddp.log 2>&1 &
   wait   # block until ALL launched runs finish
   ```
5. For **each** run: `grep '^val_loss:' research/runs/<tag>.log | tail -1`.
   - Empty → that run crashed. `tail -n 60 research/runs/<tag>.log`. If trivial (OOM → lower BATCH_SIZE,
     typo), fix and re-run just that one once. Otherwise log it as `crash` and move on.
6. Append **one results.tsv row per run** (TAB-separated): `commit  val_loss  vram_gb  status  description`.
   status ∈ {keep, discard, crash}. Every run gets its own row and its own val_loss — never merge runs.
7. Find the cycle's best run. If it beats the global best, keep the commit and narrow the search arrays
   toward that region for the next cycle; also run `RUN_TAG=<best_tag> bash research/upload_best.sh`.
   If nothing beat the best, `git reset --hard HEAD~1` (discard the cycle's config edit).
8. Append a short paragraph to `research/PROGRESS.md` (the configs tried, each val_loss, what won, next idea).
9. **STOP.** Never ask the human anything — they are away and you are fully autonomous. The loop re-invokes you.

## results.tsv columns (TAB-separated)
`commit<TAB>val_loss<TAB>vram_gb<TAB>status<TAB>description` — use `0.000000`/`0.0` for crashes.

## Context
- Task: single cutlery-sort task, 62 episodes (58 train / 4 held-out val), top camera, 480×640, 30fps.
- Base: raw `v2w_pretrained_cosmos.pt`; LoRA on the video DiT (or try full-ft, `train_architecture="base"`).
- The very first cycle should establish the **baseline** (unmodified config, one run) before tuning.
- Env is auto-sourced by `train_bounded.sh`; deps are uv-managed — never pip-install anything.
