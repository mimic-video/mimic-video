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
"""VLSP: Video-Latent Source Prior for Action Flow.

Existing mimic-video uses the partially-denoised video latent purely as a
*condition* for the action DiT.  VLSP additionally uses that latent to define
the *source* endpoint of the action flow-matching sampler:

    baseline:   source ~ N(0, I)
    VLSP:       source ~ q_phi(s | h_v, o, ...) = N(mu_phi(h_v), diag(sigma_phi(h_v)^2))

with flow matching

    x_t = (1 - t) * a + t * s
    u_t = s - a
    L   = || v_theta(x_t, t, c) - u_t ||^2  +  lambda_KL * KL(q_phi || N(0, I))

This module is intentionally free of transformer-engine / flash-attn imports so
it can be unit-tested on CPU.  The returned source always lives in the *same
normalized action space* as ``x0_B_HA_A`` because the action flow loss is
computed on normalized actions.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from imaginaire.utils import log, misc

if TYPE_CHECKING:
    # Imported for typing only; keeps this nn.Module importable without dragging
    # in the hydra / omegaconf config stack (handy for unit tests).
    from cosmos_predict2.configs.config_world2action import ActionSourcePriorConfig

# Modes that require the learned VideoLatentSourcePrior network.
VIDEO_PRIOR_MODES: frozenset[str] = frozenset(
    {
        "video_prior_sample",
        "video_prior_mean",
        "video_prior_residual",
        "video_prior_blend",
        "video_prior_dropout",
        "shuffled_video_prior",
    }
)

# All recognised source modes.
ALL_SOURCE_MODES: frozenset[str] = VIDEO_PRIOR_MODES | frozenset({"gaussian", "gt_action_noisy_debug"})

# Stable integer ids so modes can be logged as scalars.
SOURCE_MODE_IDS: dict[str, int] = {
    "gaussian": 0,
    "video_prior_sample": 1,
    "video_prior_mean": 2,
    "video_prior_residual": 3,
    "video_prior_blend": 4,
    "video_prior_dropout": 5,
    "shuffled_video_prior": 6,
    "gt_action_noisy_debug": 7,
}
COND_MODE_IDS: dict[str, int] = {
    "normal": 0,
    "zero_video": 1,
    "shuffled_video": 2,
    "dropout_video": 3,
}


# --------------------------------------------------------------------------- #
#  Small building blocks (CPU friendly, no transformer-engine dependency)      #
# --------------------------------------------------------------------------- #
class _SinusoidalEmbedding(nn.Module):
    """Continuous sinusoidal embedding for a scalar (e.g. the video sigma)."""

    def __init__(self, dim: int, min_period: float = 4e-3, max_period: float = 4.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"embedding dim ({dim}) must be even")
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, t_B: torch.Tensor) -> torch.Tensor:
        # t_B: [B] scalar in [0, 1) (already squashed). Returns [B, dim].
        fraction = torch.linspace(0.0, 1.0, self.dim // 2, device=t_B.device, dtype=torch.float32)
        freqs = math.tau / (self.min_period * (self.max_period / self.min_period) ** fraction)
        ang = t_B.float()[:, None] * freqs[None, :]
        return torch.cat((torch.sin(ang), torch.cos(ang)), dim=-1)


class _MLP(nn.Module):
    """Residual-free GELU MLP, ``depth`` hidden layers of width ``dim``."""

    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(dim)]
        for _ in range(max(1, depth)):
            layers += [nn.Linear(dim, dim), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _MeanPool(nn.Module):
    """Ordinary (or masked) mean over the token dimension, then project."""

    def __init__(self, dim_in: int, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, hidden)

    def forward(self, x_B_N_D: torch.Tensor, mask_B_N: torch.Tensor | None = None) -> torch.Tensor:
        if mask_B_N is None:
            pooled = x_B_N_D.mean(dim=1)
        else:
            m = mask_B_N.to(x_B_N_D.dtype)[..., None]
            pooled = (x_B_N_D * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class _AttentionPool(nn.Module):
    """A single learned query attends to the video tokens."""

    def __init__(self, dim_in: int, hidden: int, num_heads: int) -> None:
        super().__init__()
        self.kv_proj = nn.Linear(dim_in, hidden)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)

    def forward(self, x_B_N_D: torch.Tensor, mask_B_N: torch.Tensor | None = None) -> torch.Tensor:
        kv = self.kv_proj(x_B_N_D)
        q = self.query.expand(x_B_N_D.shape[0], -1, -1).to(kv.dtype)
        key_padding_mask = None if mask_B_N is None else ~mask_B_N
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return out[:, 0]


class _PerceiverPool(nn.Module):
    """A small learned latent array cross-attends to the video tokens."""

    def __init__(self, dim_in: int, hidden: int, num_latents: int, num_heads: int) -> None:
        super().__init__()
        self.kv_proj = nn.Linear(dim_in, hidden)
        self.latents = nn.Parameter(torch.zeros(1, num_latents, hidden))
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.ff = _MLP(hidden, depth=1)

    def forward(self, x_B_N_D: torch.Tensor, mask_B_N: torch.Tensor | None = None) -> torch.Tensor:
        kv = self.kv_proj(x_B_N_D)
        lat = self.latents.expand(x_B_N_D.shape[0], -1, -1).to(kv.dtype)
        key_padding_mask = None if mask_B_N is None else ~mask_B_N
        out, _ = self.attn(lat, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        out = lat + out
        out = self.ff(out)
        return out.mean(dim=1)  # pool the latent array


# --------------------------------------------------------------------------- #
#  The learned prior network                                                   #
# --------------------------------------------------------------------------- #
class VideoLatentSourcePrior(nn.Module):
    """Maps (video latent, state, context-timestep[, language]) to a horizon-aware
    diagonal-Gaussian over the flow-matching source.

    Output ``mu`` / ``logstd`` have shape ``[B, HA, action_dim]``.  The network is
    horizon-aware: a learned per-step query is broadcast onto the pooled global
    feature so the prior can differ across action steps rather than blindly
    broadcasting a single global vector.
    """

    def __init__(
        self,
        cfg: ActionSourcePriorConfig,
        *,
        action_dim: int,
        video_emb_dim: int,
        state_dim: int,
        max_horizon: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self.max_horizon = max_horizon
        self.use_state = cfg.use_state
        self.use_context_timestep = cfg.use_context_timestep
        self.use_language = cfg.use_language
        hidden = cfg.hidden_dim

        self.ctx_norm = nn.LayerNorm(video_emb_dim)
        if cfg.pool_type == "mean":
            self.pool: nn.Module = _MeanPool(video_emb_dim, hidden)
        elif cfg.pool_type == "attention":
            self.pool = _AttentionPool(video_emb_dim, hidden, cfg.num_attention_heads)
        elif cfg.pool_type == "perceiver":
            self.pool = _PerceiverPool(video_emb_dim, hidden, cfg.num_perceiver_latents, cfg.num_attention_heads)
        else:
            raise ValueError(f"unknown pool_type: {cfg.pool_type!r}")

        if self.use_state:
            self.state_proj = nn.Linear(state_dim, hidden)
        if self.use_context_timestep:
            self.time_embed = _SinusoidalEmbedding(hidden)
            self.time_proj = nn.Linear(hidden, hidden)
        if self.use_language:
            # language embedding width is unknown at construction in the general
            # case; default off.  A LazyLinear keeps it optional & dim-agnostic.
            self.lang_proj = nn.LazyLinear(hidden)

        # learned horizon queries -> horizon awareness
        self.horizon_queries = nn.Parameter(torch.zeros(1, max_horizon, hidden))
        self.trunk = _MLP(hidden, depth=cfg.mlp_depth)
        self.mu_head = nn.Linear(hidden, action_dim)
        self.logstd_head = nn.Linear(hidden, action_dim)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.horizon_queries, std=0.02)
        # Start close to a standard-ish prior: mean 0, constant logstd = init_logstd.
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logstd_head.weight)
        nn.init.constant_(self.logstd_head.bias, self.cfg.init_logstd)

    @property
    def _pdtype(self) -> torch.dtype:
        return self.mu_head.weight.dtype

    def forward(
        self,
        crossattn_emb: torch.Tensor,
        state_B_HO_O: torch.Tensor | None,
        context_timesteps_B_1: torch.Tensor | None,
        horizon: int,
        language_B_L_D: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Accept either flattened [B, N, D] or structured [B, T, H, W, D] latents.
        x = crossattn_emb
        if x.dim() == 5:
            b, t, h, w, d = x.shape
            x = x.reshape(b, t * h * w, d)
        x = x.to(self._pdtype)

        h = self.pool(self.ctx_norm(x))  # [B, hidden]
        B = h.shape[0]

        if self.use_state:
            if state_B_HO_O is not None and state_B_HO_O.shape[1] > 0:
                h = h + self.state_proj(state_B_HO_O.to(h.dtype).mean(dim=1))
            else:
                # keep state_proj in the autograd graph even when no obs are
                # provided, so DDP (find_unused_parameters=False) is happy.
                zeros = h.new_zeros((B, self.state_proj.in_features))
                h = h + 0.0 * self.state_proj(zeros)

        if self.use_context_timestep:
            if context_timesteps_B_1 is not None:
                sigma = context_timesteps_B_1.reshape(context_timesteps_B_1.shape[0], -1)[:, 0].float()
                squashed = sigma / (1.0 + sigma)  # map [0, inf) -> [0, 1)
                h = h + self.time_proj(self.time_embed(squashed).to(h.dtype))
            else:
                h = h + 0.0 * self.time_proj(h.new_zeros((B, self.time_proj.in_features)))

        if self.use_language and language_B_L_D is not None:
            lang = language_B_L_D
            if lang.dim() == 3:
                lang = lang.mean(dim=1)
            h = h + self.lang_proj(lang.to(h.dtype))

        per_step = self.horizon_queries[:, :horizon, :].to(h.dtype) + h.unsqueeze(1)  # [B, HA, hidden]
        per_step = self.trunk(per_step)
        mu = self.mu_head(per_step)
        logstd = self.logstd_head(per_step).clamp(min=self.cfg.logstd_min, max=self.cfg.logstd_max)
        return mu, logstd


# --------------------------------------------------------------------------- #
#  Top-level dispatcher                                                        #
# --------------------------------------------------------------------------- #
class ActionSourcePrior(nn.Module):
    """Produces the flow-matching source endpoint for the action sampler.

    ``forward`` returns ``(source_B_HA_A, metrics)`` where ``source`` lives in the
    normalized action space (float32).  The learned :class:`VideoLatentSourcePrior`
    is only instantiated when it is actually needed (``enabled`` and a video-prior
    mode), so disabled / gaussian configs carry no extra parameters and old
    checkpoints load cleanly.
    """

    def __init__(
        self,
        cfg: ActionSourcePriorConfig,
        *,
        action_dim: int,
        video_emb_dim: int,
        state_dim: int,
        max_horizon: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self.max_horizon = max_horizon

        self._uses_net = bool(cfg.enabled) and cfg.mode in VIDEO_PRIOR_MODES
        if self._uses_net:
            self.net: VideoLatentSourcePrior | None = VideoLatentSourcePrior(
                cfg,
                action_dim=action_dim,
                video_emb_dim=video_emb_dim,
                state_dim=state_dim,
                max_horizon=max_horizon,
            )
        else:
            self.net = None

    # ------------------------------------------------------------------ #
    @property
    def effective_mode(self) -> str:
        """The mode actually used; forced to ``gaussian`` when disabled."""
        return self.cfg.mode if self.cfg.enabled else "gaussian"

    @property
    def has_trainable_params(self) -> bool:
        return self.net is not None

    # ------------------------------------------------------------------ #
    def _gaussian_baseline(
        self,
        shape: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
        training: bool,
        seed: int | None,
    ) -> torch.Tensor:
        """Exactly reproduces the original source draw.

        Training uses ``torch.randn`` (matching ``draw_training_t_and_epsilon``);
        inference with a seed uses ``misc.arch_invariant_rand`` (matching the old
        ``World2ActionPipeline.__call__``).
        """
        if seed is not None and not training:
            return misc.arch_invariant_rand(shape, dtype=dtype, device=device, seed=seed)
        return torch.randn(shape, device=device, dtype=dtype)

    def _make_dropout_mask(
        self,
        shape: tuple[int, int, int],
        keep_prob: float,
        rand,
    ) -> torch.Tensor:
        """Bernoulli(keep_prob) keep-mask broadcastable over ``shape`` = (B, HA, A).

        ``dropout_granularity`` controls the broadcast shape:
          * ``sample`` / ``trajectory``: one decision per trajectory -> (B, 1, 1)
          * ``element``: an independent decision per scalar -> (B, HA, A)
        """
        b, ha, a = shape
        granularity = self.cfg.dropout_granularity
        if granularity in ("sample", "trajectory"):
            mask_shape: tuple[int, int, int] = (b, 1, 1)
        elif granularity == "element":
            mask_shape = (b, ha, a)
        else:
            raise ValueError(f"unknown dropout_granularity: {granularity!r}")
        return rand(mask_shape).lt(keep_prob).float()

    # ------------------------------------------------------------------ #
    def forward(
        self,
        shape: tuple[int, int, int],
        *,
        crossattn_emb: torch.Tensor | None,
        state_B_HO_O: torch.Tensor | None,
        context_timesteps_B_1: torch.Tensor | None,
        x0_B_HA_A: torch.Tensor | None = None,
        language_B_L_D: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        training: bool = False,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, HA, A = shape
        mode = self.effective_mode

        # Resolve device; source lives in float32 (same as the original epsilon/x0).
        if x0_B_HA_A is not None:
            device = x0_B_HA_A.device
        elif crossattn_emb is not None:
            device = crossattn_emb.device
        elif state_B_HO_O is not None:
            device = state_B_HO_O.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        # Generator for deterministic, *distinct* gaussian draws inside video modes.
        gen = generator
        if gen is None and seed is not None and not training:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))

        def randn(shp: tuple[int, ...]) -> torch.Tensor:
            return torch.randn(shp, device=device, dtype=dtype, generator=gen)

        def rand(shp: tuple[int, ...]) -> torch.Tensor:
            return torch.rand(shp, device=device, dtype=dtype, generator=gen)

        metrics: dict[str, torch.Tensor] = {}

        # ---- baseline N(0, I) -------------------------------------------------
        if mode == "gaussian":
            source = self._gaussian_baseline(shape, device, dtype, training, seed)
            return self._finalize(source, shape, x0_B_HA_A, metrics, mode, randn)

        # ---- debug: noisy ground-truth actions -------------------------------
        if mode == "gt_action_noisy_debug":
            if x0_B_HA_A is None or not training:
                log.warning(
                    "gt_action_noisy_debug requested without x0 / outside training; "
                    "falling back to gaussian source."
                )
                source = self._gaussian_baseline(shape, device, dtype, training, seed)
                return self._finalize(source, shape, x0_B_HA_A, metrics, "gaussian", randn)
            source = x0_B_HA_A.float() + self.cfg.debug_noise_std * randn(shape)
            return self._finalize(source, shape, x0_B_HA_A, metrics, mode, randn)

        # ---- video-latent prior modes ----------------------------------------
        assert self.net is not None, "video prior modes require an instantiated prior network"
        if crossattn_emb is None:
            raise ValueError(f"source_mode={mode!r} requires video crossattn_emb")

        cond = crossattn_emb
        shuffle_enabled = 0.0
        if mode == "shuffled_video_prior":
            perm = torch.randperm(B, device=device, generator=gen)
            cond = cond[perm]
            shuffle_enabled = 1.0

        if self.cfg.detach_video_latents:
            cond = cond.detach()

        mu, logstd = self.net(cond, state_B_HO_O, context_timesteps_B_1, HA, language_B_L_D)
        mu = mu.float()
        logstd = logstd.float()
        std = logstd.exp()
        temperature = float(self.cfg.sampling_temperature)

        # reparameterized "video" sample: s = mu + T * sigma * eps
        eps = randn(shape)
        source_video = mu + temperature * std * eps

        dropout_rate_actual = torch.zeros((), device=device)
        if mode == "video_prior_sample":
            source = source_video
        elif mode == "video_prior_mean":
            source = mu  # deterministic
        elif mode == "video_prior_residual":
            # source = gaussian noise + residual * mu  (mu is a learned offset)
            source = eps + float(self.cfg.residual_scale) * mu
        elif mode == "video_prior_blend":
            alpha = float(self.cfg.blend_alpha)
            eps_g = randn(shape)
            source = alpha * source_video + math.sqrt(max(1.0 - alpha * alpha, 0.0)) * eps_g
        elif mode == "video_prior_dropout":
            eps_g = randn(shape)
            keep = self._make_dropout_mask(shape, 1.0 - float(self.cfg.source_dropout_prob), rand)
            source = keep * source_video + (1.0 - keep) * eps_g
            dropout_rate_actual = (1.0 - keep).mean()
        elif mode == "shuffled_video_prior":
            source = source_video
        else:  # pragma: no cover - guarded by ALL_SOURCE_MODES
            raise ValueError(f"unknown source_mode: {mode!r}")

        # Keep BOTH heads in the autograd graph for *every* mode so DDP (with the
        # default find_unused_parameters=False) never sees an unused parameter,
        # even when a mode is deterministic (mean / residual) or all samples drop
        # out.  The added term is numerically zero.
        source = source + 0.0 * (mu.mean() + logstd.mean())

        # tensors needed for regularization
        metrics["mu"] = mu
        metrics["logstd"] = logstd
        metrics["source/shuffle_enabled"] = torch.as_tensor(shuffle_enabled, device=device)
        metrics["source/dropout_rate_actual"] = dropout_rate_actual.detach()
        with torch.no_grad():
            metrics["source/mu_mean"] = mu.mean()
            metrics["source/mu_std"] = mu.std()
            metrics["source/logstd_mean"] = logstd.mean()
            metrics["source/std_mean"] = std.mean()
        return self._finalize(source, shape, x0_B_HA_A, metrics, mode, randn)

    # ------------------------------------------------------------------ #
    def _finalize(
        self,
        source: torch.Tensor,
        shape: tuple[int, int, int],
        x0_B_HA_A: torch.Tensor | None,
        metrics: dict[str, torch.Tensor],
        mode: str,
        randn,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source = source.reshape(shape).float()

        # When VLSP is disabled the source is the plain Gaussian baseline.  We
        # deliberately avoid *any* extra work (no GPU sync, no extra RNG draw) so
        # the training/inference RNG stream stays bit-identical to the original.
        if not self.cfg.enabled:
            return source, metrics

        if not torch.isfinite(source).all():
            raise FloatingPointError(f"non-finite source produced by mode={mode!r}")
        with torch.no_grad():
            device = source.device
            metrics.setdefault("source/mu_mean", torch.zeros((), device=device))
            metrics.setdefault("source/mu_std", torch.zeros((), device=device))
            metrics.setdefault("source/logstd_mean", torch.zeros((), device=device))
            metrics.setdefault("source/std_mean", torch.ones((), device=device))
            metrics.setdefault("source/shuffle_enabled", torch.zeros((), device=device))
            metrics.setdefault("source/dropout_rate_actual", torch.zeros((), device=device))
            metrics["source/source_mean"] = source.mean()
            metrics["source/source_std"] = source.std()
            metrics["source/source_mode_id"] = torch.as_tensor(
                float(SOURCE_MODE_IDS.get(mode, -1)), device=device
            )
            if x0_B_HA_A is not None:
                metrics["source/source_vs_x0_mse"] = F.mse_loss(source, x0_B_HA_A.float())
            metrics["source/source_vs_gaussian_mse"] = F.mse_loss(source, randn(shape))
        return source, metrics


# --------------------------------------------------------------------------- #
#  Action conditioning (independent of the source prior input)                 #
# --------------------------------------------------------------------------- #
def apply_action_conditioning(
    crossattn_emb: torch.Tensor,
    *,
    mode: str,
    dropout_prob: float = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    training: bool = False,
) -> torch.Tensor:
    """Transform the video latent before it is fed to the action DiT cross-attn.

    This is *separate* from the source-prior input so e.g. the source can use the
    real video latent while the decoder receives a zeroed/shuffled condition.

      * ``normal``        : pass-through (baseline)
      * ``zero_video``    : replace with zeros (source-only experiments)
      * ``shuffled_video``: shuffle across the batch (negative control)
      * ``dropout_video`` : per-sample random zeroing with prob ``dropout_prob``
    """
    if mode == "normal":
        return crossattn_emb

    B = crossattn_emb.shape[0]
    device = crossattn_emb.device
    gen = generator
    if gen is None and seed is not None and not training:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 1234)

    if mode == "zero_video":
        return torch.zeros_like(crossattn_emb)
    if mode == "shuffled_video":
        perm = torch.randperm(B, device=device, generator=gen)
        return crossattn_emb[perm]
    if mode == "dropout_video":
        drop_B = torch.rand(B, device=device, generator=gen) < float(dropout_prob)
        out = crossattn_emb.clone()
        out[drop_B] = 0.0
        return out
    raise ValueError(f"unknown action_conditioning.mode: {mode!r}")


# --------------------------------------------------------------------------- #
#  Regularization                                                              #
# --------------------------------------------------------------------------- #
def compute_prior_regularization(
    metrics: dict[str, torch.Tensor],
    cfg: ActionSourcePriorConfig,
) -> tuple[torch.Tensor | float, dict[str, torch.Tensor]]:
    """Optional regularizers on the diagonal-Gaussian prior.

    Returns ``(loss, log_dict)``.  ``loss`` is ``0.0`` (a python float, a no-op in
    ``loss_flow + loss_prior``) when the prior is not a learned diagonal Gaussian
    or all weights are zero.
    """
    mu = metrics.get("mu")
    logstd = metrics.get("logstd")
    logs: dict[str, torch.Tensor] = {}
    if mu is None or logstd is None:
        return 0.0, logs

    total: torch.Tensor | float = 0.0

    if cfg.kl_weight != 0.0:
        # KL(N(mu, sigma^2) || N(0, I)) = 0.5 * (mu^2 + sigma^2 - 1 - 2*logstd)
        var = (2.0 * logstd).exp()
        kl = 0.5 * (mu.pow(2) + var - 1.0 - 2.0 * logstd).mean()
        total = total + cfg.kl_weight * kl
        logs["loss/source_kl"] = kl.detach()

    if cfg.mean_l2_weight != 0.0:
        mean_l2 = mu.pow(2).mean()
        total = total + cfg.mean_l2_weight * mean_l2
        logs["loss/source_mean_l2"] = mean_l2.detach()

    if cfg.std_reg_weight != 0.0:
        # Penalize collapse toward sigma << 1 (logstd < 0).
        std_reg = F.relu(-logstd).mean()
        total = total + cfg.std_reg_weight * std_reg
        logs["loss/source_std_reg"] = std_reg.detach()

    return total, logs
