#!/usr/bin/env bash
# Upload the current-best finetuned video-backbone checkpoint to HF so the human can inspect it.
# Called by the agent only when a NEW global-best val_loss is achieved (see program.md).
# Repo: microagi-luca/mimic-video-fr3-munich-backbone (private). Uses ~/.cache/huggingface/token.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/research/env.sh"
HF_REPO="${HF_BEST_REPO:-microagi-luca/mimic-video-fr3-munich-backbone}"

# Latest saved model checkpoint across experiments (the kept/best one the agent just produced).
CKPT="$(ls -t "$MODEL_DIR"/checkpoints/posttraining/video2world/*/checkpoints/model/iter_*.pt 2>/dev/null | head -1 || true)"
if [ -z "$CKPT" ]; then echo "[upload_best] no checkpoint found — did training save one? (checkpoint.save_iter)"; exit 1; fi
EXP_DIR="$(dirname "$(dirname "$(dirname "$CKPT")")")"
echo "[upload_best] uploading $CKPT  ->  $HF_REPO"

"$PY" - "$HF_REPO" "$CKPT" "$EXP_DIR" "$REPO/research/results.tsv" <<'PY'
import os, sys
from huggingface_hub import HfApi
hf_repo, ckpt, exp_dir, results = sys.argv[1:5]
api = HfApi()
api.create_repo(hf_repo, private=True, exist_ok=True, repo_type="model")
api.upload_file(path_or_fileobj=ckpt, path_in_repo="video_backbone/"+os.path.basename(ckpt), repo_id=hf_repo)
cfg = os.path.join(exp_dir, "config.yaml")
if os.path.exists(cfg):
    api.upload_file(path_or_fileobj=cfg, path_in_repo="config.yaml", repo_id=hf_repo)
if os.path.exists(results):
    api.upload_file(path_or_fileobj=results, path_in_repo="results.tsv", repo_id=hf_repo)
print("[upload_best] done:", hf_repo)
PY
