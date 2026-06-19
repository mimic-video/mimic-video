#!/usr/bin/env bash
# Runtime environment for the Cosmos-Predict2 cu130 training env on Blackwell (sm_120).
# Source before any torchrun/python that touches the model:  source research/env.sh
#
# Venv resolution (first hit wins):
#   1. $COSMOS_VENV   (explicit override)
#   2. /opt/venv      (if you baked a Docker image; see research/Dockerfile)
#   3. <repo>/model/.venv   (built with `uv sync --extra cu130`, or the mounted one on a new VM)
# Does NOT pin CUDA_VISIBLE_DEVICES — train_bounded.sh sets it per run so many runs share the GPUs.

_RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export MIMIC_ROOT="$(cd "$_RESEARCH_DIR/.." && pwd)"
export MODEL_DIR="$MIMIC_ROOT/model"

if [ -n "${COSMOS_VENV:-}" ] && [ -d "$COSMOS_VENV" ]; then VENV="$COSMOS_VENV"
elif [ -d /opt/venv ]; then VENV=/opt/venv
else VENV="$MODEL_DIR/.venv"; fi
export VENV
_SP="$VENV/lib/python3.12/site-packages"

# TE (cu13) + torch need the cu13 nvidia libs and torch's libs on the loader path.
# nvidia-cublas==13.5 (forced via uv override) provides the grouped-GEMM symbols TE 2.16 needs.
export LD_LIBRARY_PATH="$_SP/nvidia/cu13/lib:$_SP/nvidia/cudnn/lib:$_SP/nvidia/cusparselt/lib:$_SP/nvidia/nccl/lib:$_SP/nvidia/nvshmem/lib:$_SP/torch/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="${CUDA_HOME:-$( [ -d /usr/local/cuda-13.0 ] && echo /usr/local/cuda-13.0 || echo /usr/local/cuda )}"

# Blackwell has no flash-attn build; use TE fused attention (+ the repo's SDPA fallback).
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export TOKENIZERS_PARALLELISM=false

export PY="$VENV/bin/python"
export TORCHRUN="$VENV/bin/torchrun"
