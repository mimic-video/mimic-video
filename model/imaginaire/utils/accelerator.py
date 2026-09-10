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

"""Device-agnostic helpers for CUDA / Intel XPU / CPU execution.

The code base was originally written for CUDA accelerators, which led to
``torch.cuda`` being hard-coded in a handful of places. These helpers centralize
the accelerator dispatch so the same inference stack can run on Intel XPU (via
the PyTorch XPU backend) or on CPU for reference / parity checks.

Only the small set of operations that genuinely differ between backends live
here; everything else is expected to be written device-agnostically (``.to(device)``
instead of ``.cuda()``).

Note: this module is named ``accelerator`` because ``imaginaire.utils.device``
already hosts the NVML-based ``Device`` affinity helper.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

SUPPORTED_DEVICE_TYPES: tuple[str, ...] = ("cuda", "xpu", "cpu")


def cuda_available() -> bool:
    """Return whether a usable CUDA backend is present."""
    return torch.cuda.is_available()


def xpu_available() -> bool:
    """Return whether a usable Intel XPU backend is present."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_available(device_type: str) -> bool:
    """Return whether ``device_type`` is usable on this machine."""
    if device_type == "cuda":
        return cuda_available()
    if device_type == "xpu":
        return xpu_available()
    if device_type == "cpu":
        return True
    raise ValueError(f"Unsupported device type {device_type!r}; expected one of {SUPPORTED_DEVICE_TYPES}.")


def default_device() -> str:
    """Pick the best available accelerator, falling back to CPU."""
    if cuda_available():
        return "cuda"
    if xpu_available():
        return "xpu"
    return "cpu"


def resolve_device(device: str | torch.device | None = None) -> str:
    """Validate a user-provided device string and return a canonical representation.

    Accepts the ``--device`` values ``cuda``/``xpu``/``cpu`` (optionally with an
    index such as ``cuda:0``) as well as ``torch.device`` objects. ``None``
    resolves to :func:`default_device`.
    """
    if device is None:
        return default_device()
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"Unsupported device {str(dev)!r}; expected one of {SUPPORTED_DEVICE_TYPES} (optionally with an index)."
        )
    if not is_available(dev.type):
        raise RuntimeError(f"Requested device {str(dev)!r} but the {dev.type} backend is not available on this machine.")
    return str(dev)


def empty_cache(device: str | torch.device | None = None) -> None:
    """Release the accelerator caching allocator (no-op on CPU / unavailable backends)."""
    dev = torch.device(device if device is not None else default_device())
    if dev.type == "cuda" and cuda_available():
        torch.cuda.empty_cache()
    elif dev.type == "xpu" and xpu_available():
        torch.xpu.empty_cache()


def synchronize(device: str | torch.device | None = None) -> None:
    """Wait for all kernels on ``device`` to finish (no-op on CPU)."""
    dev = torch.device(device if device is not None else default_device())
    if dev.type == "cuda" and cuda_available():
        torch.cuda.synchronize(dev)
    elif dev.type == "xpu" and xpu_available():
        torch.xpu.synchronize(dev)


def device_index(device: str | torch.device | None = None) -> int:
    """Return the index of ``device`` (0 when unspecified)."""
    dev = torch.device(device if device is not None else default_device())
    return dev.index if dev.index is not None else 0


def cuda_graphs_supported(device: str | torch.device | None = None) -> bool:
    """Return whether ``torch.cuda.CUDAGraph`` may be used on ``device``.

    CUDA Graphs are a CUDA-only feature; on XPU/CPU the model runs eagerly.
    """
    dev = torch.device(device if device is not None else default_device())
    return dev.type == "cuda" and cuda_available()


def amp_autocast(device: str | torch.device | None, dtype: torch.dtype, enabled: bool = True):
    """Return an autocast context manager appropriate for ``device``.

    Unlike ``torch.amp.autocast("cuda", ...)`` this does not raise on XPU/CPU and
    simply disables autocast when the device backend has no autocast support.
    """
    dev = torch.device(device if device is not None else default_device())
    supported = (dev.type == "cuda" and cuda_available()) or (dev.type == "xpu" and xpu_available())
    if not (enabled and supported):
        return nullcontext()
    return torch.amp.autocast(dev.type, dtype=dtype)


def device_report() -> str:
    """Human-readable summary of the accelerator situation, for logs."""
    lines = [f"torch {torch.__version__}"]
    if cuda_available():
        lines.append(f"cuda: {torch.cuda.device_count()} device(s) [{torch.cuda.get_device_name(0)}]")
    else:
        lines.append("cuda: unavailable")
    if xpu_available():
        lines.append(f"xpu: {torch.xpu.device_count()} device(s) [{torch.xpu.get_device_name(0)}]")
    else:
        lines.append("xpu: unavailable")
    lines.append(f"default: {default_device()}")
    return " | ".join(lines)
