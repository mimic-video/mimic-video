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

"""Compatibility layer for the Transformer Engine primitives used by the DiTs.

Transformer Engine (TE) and ``flash-attn`` only ship CUDA kernels, so the DiTs
could not be imported at all on Intel XPU or CPU. This module makes those imports
optional and transparently substitutes pure-PyTorch implementations (see
:mod:`cosmos_predict2.module.torch_ops`) when the CUDA-only packages are missing.

On a CUDA machine with TE/flash-attn installed the behaviour is unchanged: the
real TE ops are re-exported and the requested attention backends are preserved.
"""

from __future__ import annotations

from imaginaire.utils import log

try:  # pragma: no cover - depends on the host
    import transformer_engine as te

    TE_AVAILABLE = True
except ImportError:  # pragma: no cover - XPU / CPU / CUDA without TE
    te = None
    TE_AVAILABLE = False

if TE_AVAILABLE:  # pragma: no cover - CUDA-only path
    from transformer_engine.pytorch.attention import (
        DotProductAttention,
        apply_rotary_pos_emb,
    )

    RMSNorm = te.pytorch.RMSNorm
else:  # pragma: no cover - exercised on XPU / CPU
    from cosmos_predict2.module.torch_ops import RMSNorm, apply_rotary_pos_emb  # noqa: F401

    DotProductAttention = None

try:  # pragma: no cover - depends on the host
    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:  # pragma: no cover
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False

_WARNED_BACKENDS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Log a fallback warning only once per process (the DiTs build many attention layers)."""
    if key not in _WARNED_BACKENDS:
        _WARNED_BACKENDS.add(key)
        log.warning(message)


def resolve_atten_backend(backend: str) -> str:
    """Return an attention backend that is actually usable on this machine.

    Requested CUDA-only backends (``transformer_engine``, ``flash_attn_no_cp``)
    fall back to the portable PyTorch SDPA backend (``torch``) when their
    dependency is missing. Every other backend is passed through untouched.
    """
    if backend == "transformer_engine" and not TE_AVAILABLE:
        _warn_once("transformer_engine", "Transformer Engine not available; falling back to the PyTorch SDPA backend.")
        return "torch"
    if backend == "flash_attn_no_cp" and not FLASH_ATTN_AVAILABLE:
        _warn_once("flash_attn_no_cp", "flash-attn not available; falling back to the PyTorch SDPA backend.")
        return "torch"
    return backend
