#!/usr/bin/env bash
# FROZEN. Run ONE bounded finetuning experiment on an assigned set of GPUs and emit a single
# comparable `val_loss: <float>` line. Safe to run many copies concurrently (one per GPU group).
# Do NOT edit this as part of experiments (see program.md).
#
# Usage:    GPUS=0 RUN_TAG=myrun bash research/train_bounded.sh <EXPERIMENT_NAME> > research/runs/myrun.log 2>&1
# GPUS:     comma-separated GPU ids. 1 id => single-GPU run; N ids => N-way DDP for ONE experiment.
#           e.g. GPUS=0  |  GPUS=0,1,2,3  (effective batch = BATCH_SIZE * N)
# RUN_TAG:  unique label -> own checkpoint dir + job name (default: the experiment name).
# Other env: TRAIN_BUDGET_S (900), VALIDATION_ITER (25), BATCH_SIZE (per-GPU local batch, default 1).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/research/env.sh"
cd "$MODEL_DIR"

EXP="${1:-v2w_fr3_munich_lora_rank256_lr1.778e-04_bsz32}"
GPUS="${GPUS:-0}"
RUN_TAG="${RUN_TAG:-$EXP}"
TRAIN_BUDGET_S="${TRAIN_BUDGET_S:-900}"
LOAD_GRACE_S="${LOAD_GRACE_S:-180}"
VALIDATION_ITER="${VALIDATION_ITER:-25}"
BATCH_SIZE="${BATCH_SIZE:-1}"

export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC="$(awk -F, '{print NF}' <<<"$GPUS")"          # GPUs in this run => DDP world size
FIRST_GPU="${GPUS%%,*}"
MASTER_PORT="$(( 29400 + FIRST_GPU ))"              # distinct per concurrent run (GPU sets don't overlap)
TOTAL_S=$(( TRAIN_BUDGET_S + LOAD_GRACE_S ))
OUTDIR="checkpoints/posttraining/video2world/$RUN_TAG"

TRAINLOG="$(mktemp /tmp/tb_${RUN_TAG//\//_}.XXXXXX.log)"
echo "[train_bounded] run=$RUN_TAG exp=$EXP gpus=$GPUS (ddp x$NPROC) budget=${TRAIN_BUDGET_S}s val_every=${VALIDATION_ITER} bsz=${BATCH_SIZE}/gpu"
echo "[train_bounded] started $(date -u +%FT%TZ)"

# num_workers=0 is REQUIRED (/dev/shm is small here); job.name=$RUN_TAG isolates each run's outputs.
timeout --signal=SIGINT --kill-after=120s "${TOTAL_S}s" \
  "$TORCHRUN" --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment="$EXP" job.name="$RUN_TAG" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  dataloader_train.num_workers=0 dataloader_train.prefetch_factor=null dataloader_train.persistent_workers=false \
  trainer.max_iter=100000000 \
  trainer.run_validation=true trainer.validation_iter="$VALIDATION_ITER" \
  checkpoint.save_iter=100000000 \
  > "$TRAINLOG" 2>&1
rc=$?
[ "$rc" -eq 124 ] && echo "[train_bounded] hit time budget (expected) — using latest validation."

echo "[train_bounded] last training losses:"; grep -aE "Iteration [0-9]+: .*Loss:" "$TRAINLOG" | tail -3
LAST_VL="$(grep -aE '^val_loss:' "$TRAINLOG" | tail -1 | awk '{print $2}')"
N_VAL="$(grep -acE '^val_loss:' "$TRAINLOG")"
PEAK_VRAM_GB="$(grep -aoE '([0-9]+\.[0-9]+) ?GB' "$TRAINLOG" | grep -oE '[0-9]+\.[0-9]+' | sort -gr | head -1)"

echo "----- train_bounded summary ($RUN_TAG) -----"
if [ -n "$LAST_VL" ]; then
  mkdir -p "$OUTDIR" 2>/dev/null || true
  echo "$LAST_VL" > "$OUTDIR/val_loss.txt" 2>/dev/null || true
  echo "run_tag: $RUN_TAG"
  echo "val_loss: $LAST_VL"
  echo "num_validations: $N_VAL"
  [ -n "${PEAK_VRAM_GB:-}" ] && echo "peak_vram_gb: $PEAK_VRAM_GB"
  echo "[train_bounded] OK $(date -u +%FT%TZ)"
  rm -f "$TRAINLOG"; exit 0
else
  echo "run_tag: $RUN_TAG"
  echo "[train_bounded] NO val_loss — crashed before first validation. Tail:"
  tail -n 40 "$TRAINLOG"
  rm -f "$TRAINLOG"; exit 1
fi
