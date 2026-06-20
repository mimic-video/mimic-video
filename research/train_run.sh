#!/usr/bin/env bash
# Launch a full (un-bounded) training run on a chosen GPU set and record a reproducible manifest
# (exact command + env + git SHA + world size) next to the trainer's auto-dumped config.yaml.
# This is the counterpart to the frozen, sweep-oriented train_bounded.sh — use this for real runs.
#
# Verify params first (resolves + prints the full config, writes config.yaml, does NOT train):
#   DRYRUN=1 RUN_NAME=v2w_fr3_munich_8gpu_lr3e-4 \
#     bash research/train_run.sh v2w_fr3_munich_lora_rank256_lr3.000e-04_bsz32
#
# Then launch for real (same args, drop DRYRUN):
#   RUN_NAME=v2w_fr3_munich_8gpu_lr3e-4 \
#     bash research/train_run.sh v2w_fr3_munich_lora_rank256_lr3.000e-04_bsz32
#
# Env knobs (all optional): GPUS (default all 8), RUN_NAME (default=experiment), WANDB (1),
#   MAX_ITER (4000), SAVE_ITER (500), VAL_ITER (200), LOG_ITER (50), MASTER_PORT, WANDB_PROJECT.
# Extra positional args after the experiment are passed verbatim as Hydra overrides.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/research/env.sh"
cd "$MODEL_DIR"

EXP="${1:?usage: train_run.sh <EXPERIMENT> [extra hydra overrides...]}"; shift || true
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RUN_NAME="${RUN_NAME:-$EXP}"
NPROC="$(awk -F, '{print NF}' <<<"$GPUS")"
FIRST_GPU="${GPUS%%,*}"
MASTER_PORT="${MASTER_PORT:-$((29500 + FIRST_GPU))}"
MAX_ITER="${MAX_ITER:-4000}"
SAVE_ITER="${SAVE_ITER:-1000}"
VAL_ITER="${VAL_ITER:-100}"
LOG_ITER="${LOG_ITER:-50}"
# Per-GPU local batch. The 2B v2w model + activation checkpointing uses ~28GB fixed + ~15GB/sample,
# so batch>=4 OOMs a 97GB card. 2 (~59GB) is the safe default. Effective batch = BATCH * world_size
# * GRAD_ACCUM; raise GRAD_ACCUM (not BATCH) to grow effective batch without more VRAM.
BATCH="${BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
# PROFILE=1 enables the profiler callback: it dumps a compute-vs-NCCL-comm breakdown at ~iter 13 and
# an interconnect (latency/bandwidth) sweep at train end. It's OVERLAID on a normal run — wandb,
# validation and checkpointing stay on — because the profiler only samples a short early window of
# pure training steps, so those don't interfere. For a quick throwaway profile, pass MAX_ITER=20.
RUN_VALIDATION=true
if [ -n "${PROFILE:-}" ]; then
  export MIMIC_PROFILE=1
fi
export MIMIC_WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-mimic-video}"
# Reduce allocator fragmentation (the OOM message recommends this); harmless if already set.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OVERRIDES=(
  experiment="$EXP"
  job.name="$RUN_NAME"
  trainer.max_iter="$MAX_ITER"
  trainer.logging_iter="$LOG_ITER"
  trainer.run_validation="$RUN_VALIDATION"
  trainer.validation_iter="$VAL_ITER"
  checkpoint.save_iter="$SAVE_ITER"
  dataloader_train.batch_size="$BATCH"
  trainer.grad_accum_iter="$GRAD_ACCUM"
  # /dev/shm is only 1GB and is shared by ALL ranks on the node; native-res uint8 video clips are
  # ~56MB each, so any num_workers>0 across 8 ranks overflows it ("shm: No space left on device").
  # Load in-process (no IPC, no /dev/shm); decode is ~0.33s/batch, dwarfed by the 2B model step.
  dataloader_train.num_workers=0
  dataloader_train.prefetch_factor=null
  dataloader_train.persistent_workers=false
  # The val sampler (ResumableDistributedSampler) forces drop_last=True, so a small val set
  # (~6 clips here) floors to len(val)//world_size = 0 per rank under multi-GPU DDP and the
  # ProgressBarCallback asserts "Invalid number of validation iterations: 0". Drop the sampler so
  # every rank evaluates the full val set; the trainer all-reduces val loss, so the mean is correct.
  dataloader_val.sampler=null
  "$@"
)

OUTDIR="checkpoints/posttraining/video2world/$RUN_NAME"

# --- dry run: resolve + print the config and write config.yaml, then stop. No GPUs touched. ---
if [ -n "${DRYRUN:-}" ]; then
  echo "[train_run] DRYRUN — resolving config for '$RUN_NAME' (no training)"
  "$PY" -m scripts.train --config=cosmos_predict2/configs/config.py --dryrun -- "${OVERRIDES[@]}"
  echo "[train_run] wrote $OUTDIR/config.yaml — inspect it to confirm every param."
  exit $?
fi

mkdir -p "$OUTDIR"
# Reproducibility manifest: everything the config.yaml can't capture (cmd, env, GPUs, code version).
MANIFEST="$OUTDIR/run_manifest.txt"
{
  echo "run_name:   $RUN_NAME"
  echo "experiment: $EXP"
  echo "utc:        $(date -u +%FT%TZ)"
  echo "host:       $(hostname)"
  echo "git_sha:    $(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
  echo "git_dirty:  $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted file(s)"
  echo "gpus:       $GPUS  (world_size=$NPROC, master_port=$MASTER_PORT)"
  echo "wandb:      MIMIC_WANDB=$MIMIC_WANDB  project=$WANDB_PROJECT"
  echo "command:"
  echo "  CUDA_VISIBLE_DEVICES=$GPUS MIMIC_WANDB=$MIMIC_WANDB WANDB_PROJECT=$WANDB_PROJECT \\"
  echo "  $TORCHRUN --nproc_per_node=$NPROC --master_port=$MASTER_PORT \\"
  echo "    -m scripts.train --config=cosmos_predict2/configs/config.py -- ${OVERRIDES[*]}"
} | tee "$MANIFEST"
echo "[train_run] manifest -> $MANIFEST ; config.yaml will be written by the trainer at startup."

export CUDA_VISIBLE_DEVICES="$GPUS"
exec "$TORCHRUN" --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  -m scripts.train --config=cosmos_predict2/configs/config.py -- "${OVERRIDES[@]}"
