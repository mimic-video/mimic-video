# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description:
# Single point of entry for all generic attention ops (self and cross attention), that tries to
# deliver the best performance possible given any use case (GPU and environment).
#
# On Hopper GPUs (i.e. H100, H20, H200), Flash Attention 3 is the best-performing choice, but it
# needs to be installed. When it is not available, the second best choice is cuDNN attention, which
# we get using PyTorch's SDPA API.
#
# For all other use cases, we will just use PyTorch's SDPA, but we need to specify backends and
# priorities.
# Flash Attention 2, which is one of the backends, is the best choice for Ampere GPUs (both RTX and
# datacenter-class).
#
# For anything pre-Ampere, the only choice is "memory-efficient" (xformers) FMHA.
#
# For Ada and Blackwell RTX, it is unclear at the moment, so we defer to Flash Attention 2, and
# fallbacks are cuDNN and xformers.
#
# For Blackwell datacenter-class (B200, GB200), cuDNN is the best choice.
#
#
# Dispatching to the desired backends/paths are done by checking the compute capability (really SM
# number, which is just compute capability * 10) of the GPU device the input tensors are on.
#
# Here's a breakdown of relevant compute capabilities:
#
# | GPU / category | Arch  |
# |================|=======|
# | A100           | SM80  |
# | A40            | SM80  |
# | Ampere RTX     | SM86  |
# |----------------|-------|
# | Ada Lovelace   | SM89  |
# |----------------|-------|
# | H20            | SM90  |
# | H100           | SM90  |
# | H200           | SM90  |
# |----------------|-------|
# | B200           | SM100 |
# | Blackwell RTX  | SM103 |
# |----------------|-------|
#

from functools import partial

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from flash_attn.flash_attn_interface import flash_attn_func

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False


def _chunked_attention(
    q_B_H_S_D: torch.Tensor,
    k_B_H_S_D: torch.Tensor,
    v_B_H_S_D: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    max_score_elems: int = 2**27,
) -> torch.Tensor:
    """Memory-bounded scaled dot-product attention.

    Slices the query dimension so the ``[B, H, chunk, S]`` score matrix stays within
    ``max_score_elems`` elements. Mathematically identical to
    ``scaled_dot_product_attention``, but never materializes the full ``S x S`` matrix --
    which is what the CPU/XPU MATH fallback would otherwise do for fp32.

    Returns a tensor in the ``[B, S, H, D]`` layout (the caller's expectation).
    """
    batch, heads, q_len, head_dim = q_B_H_S_D.shape
    k_len = k_B_H_S_D.shape[2]
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    max_score_elems = max(1, max_score_elems)
    chunk = max(1, min(q_len, max_score_elems // max(1, batch * heads * k_len)))

    outputs = []
    for start in range(0, q_len, chunk):
        q_chunk = q_B_H_S_D[:, :, start : start + chunk]
        scores = torch.matmul(q_chunk, k_B_H_S_D.transpose(-2, -1)) * scale
        if causal:
            q_idx = torch.arange(start, start + q_chunk.shape[2], device=q_B_H_S_D.device).view(1, 1, -1, 1)
            k_idx = torch.arange(k_len, device=q_B_H_S_D.device).view(1, 1, 1, -1)
            scores = scores.masked_fill(k_idx > q_idx, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            probs = torch.nn.functional.dropout(probs, p=dropout_p, training=True)
        outputs.append(torch.matmul(probs, v_B_H_S_D))
    out = torch.cat(outputs, dim=2)
    return out.transpose(1, 2).contiguous()


def get_device_cc(device) -> int:
    """
    Returns the compute capability of a given torch device if it's a CUDA device, otherwise returns 0.

    Args:
        device: torch device.

    Returns:
        device_cc (int): compute capability in the SmXXX format (i.e. 90 for Hopper).
    """
    if torch.cuda.is_available() and torch.version.cuda and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor
    return 0


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    deterministic=False,
    dtype=None,
):
    if dtype is None:
        # Preserve the historical bfloat16 default on CUDA so existing behaviour and
        # numerics are untouched; off CUDA (XPU/CPU) follow the model dtype instead of
        # forcing bf16, which lets the pipeline run in float32.
        dtype = torch.bfloat16 if q.device.type == "cuda" else q.dtype
    supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    is_half = dtype in [torch.bfloat16, torch.float16]
    compute_cap = get_device_cc(q.device)

    if dtype not in supported_dtypes:
        raise NotImplementedError(f"{dtype=} is not supported.")

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    if q.device.type != "cuda":
        # Intel XPU / CPU: the SDPA backends selected below (flash-attn, cuDNN, memory-efficient)
        # are CUDA-only. Rely on PyTorch's default backend selection for the device instead.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if dtype == torch.float32:
            # On XPU the only fp32 SDPA backend is MATH, which materializes the full S x S
            # score matrix (tens of GB for the video DiT). Use a memory-bounded, chunked
            # attention instead; the algorithm is identical on CPU and XPU.
            return _chunked_attention(q, k, v, causal=causal, dropout_p=dropout_p, softmax_scale=softmax_scale)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )
        return out.transpose(1, 2).contiguous()

    # If Flash Attention 3 is installed, and the user's running on a Hopper GPU (compute capability
    # 9.0, or SM90), use Flash Attention 3.
    if compute_cap == 90 and FLASH_ATTN_3_AVAILABLE and is_half:
        return flash_attn_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )  # [0] before idk why
    else:
        # If Blackwell or Hopper (SM100 or SM90), cuDNN has native FMHA kernels. The Hopper one is
        # not always as fast as Flash Attention 3, but when Flash Attention is unavailable, it's
        # still a far better choice than Flash Attention 2 (Ampere).
        if compute_cap in [90, 100] and is_half:
            SDPA_BACKENDS = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
            BEST_SDPA_BACKEND = SDPBackend.CUDNN_ATTENTION
        elif is_half:
            SDPA_BACKENDS = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
            BEST_SDPA_BACKEND = SDPBackend.FLASH_ATTENTION if compute_cap >= 80 else SDPBackend.EFFICIENT_ATTENTION
        else:
            assert dtype == torch.float32, f"Unrecognized {dtype=}."
            SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION]
            BEST_SDPA_BACKEND = SDPBackend.EFFICIENT_ATTENTION

        if deterministic:
            raise NotImplementedError(
                "Deterministic mode in attention is only supported when Flash Attention 3 is available."
            )

        # Torch 2.6 and later allows priorities for backends, but for older versions
        # we can only run with a specific backend. As long as we pick ones we're certain
        # will work on that device, it should be fine.
        try:
            sdpa_kernel(backends=SDPA_BACKENDS, set_priority_order=True)
            sdpa_kernel_ = partial(sdpa_kernel, set_priority_order=True)
        except TypeError:
            sdpa_kernel_ = sdpa_kernel
            SDPA_BACKENDS = [BEST_SDPA_BACKEND]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        with sdpa_kernel_(backends=SDPA_BACKENDS):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=causal,
                dropout_p=dropout_p,
                scale=softmax_scale,
            )

        out = out.transpose(1, 2).contiguous()
        return out
