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

"""Pure-PyTorch stand-ins for the Transformer Engine ops used by the DiTs.

The Cosmos/mimic-video DiTs are written against three Transformer Engine (TE)
primitives: ``RMSNorm``, ``DotProductAttention`` and ``apply_rotary_pos_emb``.
TE ships CUDA kernels only, so on Intel XPU (and CPU) we substitute the
mathematically equivalent PyTorch implementations below. On CUDA machines the
real TE ops are still used -- nothing here is imported unless TE is missing.

The implementations intentionally mirror TE's reference (non-fused) math:

* ``RMSNorm``: normalization is accumulated in fp32 and the learnable weight is
  applied afterwards, matching ``transformer_engine.pytorch.RMSNorm``.
* ``apply_rotary_pos_emb``: ``freqs`` carries the *angles*; cos/sin are computed
  in fp32 and cast back, then ``t * cos + rotate_half(t) * sin`` is applied to
  the first ``rot_dim`` channels (the TE convention for ``bshd``/``sbhd``).
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Drop-in replacement for ``transformer_engine.pytorch.RMSNorm``.

    Mirrors TE's fp32 computation: the normalization and the gamma multiply both
    happen in fp32 and the result is cast back to the input dtype exactly once.
    Supports TE's ``zero_centered_gamma`` (``(1 + gamma)`` instead of ``gamma``).
    """

    def __init__(self, dim: int, eps: float = 1e-5, zero_centered_gamma: bool = False, **kwargs) -> None:
        del kwargs  # TE-specific flags (sequence_parallel, params_dtype, ...) are irrelevant here
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.weight = nn.Parameter(torch.ones(dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.zero_centered_gamma:
            nn.init.zeros_(self.weight)
        else:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        gamma = self.weight.float()
        if self.zero_centered_gamma:
            gamma = gamma + 1.0
        # Apply gamma in fp32 and cast once, matching TE's fused kernel.
        return (x * gamma).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Match ``transformer_engine.pytorch.attention._rotate_half`` exactly."""
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Drop-in replacement for ``transformer_engine.pytorch.attention.apply_rotary_pos_emb``.

    ``fused`` and any context-parallel kwargs are accepted (and ignored) so this can
    be swapped in transparently; the result is identical to TE's non-fused path.
    """
    del fused, kwargs
    if tensor_format not in ("sbhd", "bshd"):
        raise NotImplementedError(
            f"Only formats 'sbhd' and 'bshd' are supported without Transformer Engine, got {tensor_format!r}."
        )

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]
    if cur_seq_len > max_seq_len:
        raise AssertionError(f"Rotary embeddings only supported up to {max_seq_len} sequence length, got {cur_seq_len}.")

    freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [s, 1, 1, d] -> [1, s, 1, d]

    # cos/sin in fp32 then cast, for better precision (mirrors TE).
    cos_ = torch.cos(freqs.float()).to(t.dtype)
    sin_ = torch.sin(freqs.float()).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)
