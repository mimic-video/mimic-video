#!/usr/bin/env bash
# FROZEN. The autoresearch harness: run one bounded finetuning experiment and emit a single
# comparable `val_loss: <float>` line. Do NOT edit this as part of experiments (see program.md).
#
# Usage:   bash research/train_bounded.sh [EXPERIMENT_NAME]
# Knobs (env): TRAIN_BUDGET_S (default 900 = 15 min), VALIDATION_ITER (default 25), BATCH_SIZE (1).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/research/env.sh"
cd "$MODEL_DIR"

EXP="${1:-v2w_fr3_munich_lora_rank256_lr1.778e-04_bsz32}"
TRAIN_BUDGET_S="${TRAIN_BUDGET_S:-900}"      # training wall-clock budget (excl. ~load); bump for longer runs
LOAD_GRACE_S="${LOAD_GRACE_S:-180}"          # headroom for model load + an in-flight validation
VALIDATION_ITER="${VALIDATION_ITER:-25}"     # validate (=> print val_loss) every N steps
BATCH_SIZE="${BATCH_SIZE:-1}"
TOTAL_S=$(( TRAIN_BUDGET_S + LOAD_GRACE_S ))

TRAINLOG="$(mktemp /tmp/tb_train.XXXXXX.log)"
echo "[train_bounded] exp=$EXP  budget=${TRAIN_BUDGET_S}s (total cap ${TOTAL_S}s)  val_every=${VALIDATION_ITER}  bsz=${BATCH_SIZE}"
echo "[train_bounded] started $(date -u +%FT%TZ)"

# num_workers=0 is REQUIRED: /dev/shm is 64MB here, worker shared-memory transfer OOMs otherwise.
timeout --signal=SIGINT --kill-after=120s "${TOTAL_S}s" "$TORCHRUN" --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment="$EXP" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  dataloader_train.num_workers=0 dataloader_train.prefetch_factor=null dataloader_train.persistent_workers=false \
  trainer.max_iter=100000000 \
  trainer.run_validation=true trainer.validation_iter="$VALIDATION_ITER" \
  checkpoint.save_iter=100000000 \
  > "$TRAINLOG" 2>&1
rc=$?
[ "$rc" -eq 124 ] && echo "[train_bounded] training hit time budget (expected) — using latest validation."

# Surface a little context, then the canonical metric line.
echo "[train_bounded] last training losses:"; grep -aE "Iteration [0-9]+: .*Loss:" "$TRAINLOG" | tail -3
LAST_VL="$(grep -aE '^val_loss:' "$TRAINLOG" | tail -1 | awk '{print $2}')"
N_VAL="$(grep -acE '^val_loss:' "$TRAINLOG")"
PEAK_VRAM_GB="$(grep -aoE 'Peak memory.*?([0-9]+\.[0-9]+) ?GB|max_memory_allocated[^0-9]*([0-9.]+)' "$TRAINLOG" | grep -oE '[0-9]+\.[0-9]+' | sort -gr | head -1)"

echo "----- train_bounded summary -----"
if [ -n "$LAST_VL" ]; then
  echo "val_loss: $LAST_VL"
  echo "num_validations: $N_VAL"
  [ -n "${PEAK_VRAM_GB:-}" ] && echo "peak_vram_gb: $PEAK_VRAM_GB"
  echo "[train_bounded] OK $(date -u +%FT%TZ)"
  rm -f "$TRAINLOG"; exit 0
else
  echo "[train_bounded] NO val_loss produced — run crashed before first validation. Tail of training log:"
  tail -n 40 "$TRAINLOG"
  rm -f "$TRAINLOG"; exit 1
fi
