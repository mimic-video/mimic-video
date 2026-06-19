#!/usr/bin/env bash
# Runtime environment for the Cosmos-Predict2 cu130 training env on this Blackwell (sm_120) box.
# Source this before any torchrun/python that touches the model:  source research/env.sh
# The env itself is built/reproduced with:  (cd model && uv sync --extra cu130)

# Resolve repo root (this file lives in <repo>/research/) and the model venv.
_RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export MIMIC_ROOT="$(cd "$_RESEARCH_DIR/.." && pwd)"
export MODEL_DIR="$MIMIC_ROOT/model"
_SP="$MODEL_DIR/.venv/lib/python3.12/site-packages"

# TE (cu13) + torch need the cu13 nvidia libs and torch's libs on the loader path.
# nvidia-cublas==13.5 (forced via uv override) provides the grouped-GEMM symbols TE 2.16 needs.
export LD_LIBRARY_PATH="$_SP/nvidia/cu13/lib:$_SP/nvidia/cudnn/lib:$_SP/nvidia/cusparselt/lib:$_SP/nvidia/nccl/lib:$_SP/nvidia/nvshmem/lib:$_SP/torch/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda-13.0

# Blackwell has no flash-attn build; use TE fused attention (the repo's SDPA fallback covers the rest).
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

export PY="$MODEL_DIR/.venv/bin/python"
export TORCHRUN="$MODEL_DIR/.venv/bin/torchrun"
