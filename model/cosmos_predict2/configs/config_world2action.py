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

import attrs
from torch import nn

from cosmos_predict2.configs.defaults.ema import EMAConfig
from imaginaire.config import make_freezable
from imaginaire.lazy_config import LazyDict


@make_freezable
@attrs.define(slots=False)
class SchedulerConfig:
    alpha: float
    beta: float
    num_denoising_steps: int


@make_freezable
@attrs.define(slots=False)
class ActionSourcePriorConfig:
    """VLSP (Video-Latent Source Prior) configuration.

    Controls how the *source* endpoint of the action flow-matching sampler is
    produced.  The baseline flow matching uses ``source ~ N(0, I)``; VLSP can
    instead derive the source from the partially-denoised video latent, i.e.
    ``source ~ q_phi(s | video_latent, obs, ...)``.

    When ``enabled`` is False the module is a no-op and reproduces the original
    Gaussian behaviour exactly, so old checkpoints / configs are unaffected.
    """

    # master switch.  enabled=False => exact baseline Gaussian source.
    enabled: bool = False
    # one of the source modes implemented in action_source_prior.py:
    #   gaussian | video_prior_sample | video_prior_mean | video_prior_residual
    #   video_prior_blend | video_prior_dropout | shuffled_video_prior
    #   gt_action_noisy_debug
    mode: str = "gaussian"

    # ---- prior network architecture ----
    pool_type: str = "mean"  # mean | attention | perceiver
    hidden_dim: int = 1024
    num_perceiver_latents: int = 8
    num_attention_heads: int = 8
    mlp_depth: int = 2

    # ---- diagonal Gaussian parameterization ----
    logstd_min: float = -5.0
    logstd_max: float = 1.0
    init_logstd: float = -1.0

    # ---- mode-specific knobs ----
    residual_scale: float = 1.0  # video_prior_residual
    blend_alpha: float = 1.0  # video_prior_blend
    source_dropout_prob: float = 0.0  # video_prior_dropout
    dropout_granularity: str = "sample"  # sample | trajectory | element
    sampling_temperature: float = 1.0  # scales the stochastic component

    # ---- inputs to the prior ----
    detach_video_latents: bool = True
    use_state: bool = True
    use_context_timestep: bool = True
    use_language: bool = False

    # ---- regularization weights (all default off) ----
    kl_weight: float = 0.0
    mean_l2_weight: float = 0.0
    std_reg_weight: float = 0.0

    # ---- debug ----
    debug_noise_std: float = 0.05


@make_freezable
@attrs.define(slots=False)
class ActionConditioningConfig:
    """How the video latent is fed to the action DiT cross-attention.

    This is *independent* of the source prior input so that e.g. the source can
    be derived from the real video latent while the action decoder receives a
    zeroed (source-only) or shuffled video condition.
    """

    # normal | zero_video | shuffled_video | dropout_video
    mode: str = "normal"
    dropout_prob: float = 0.0


@make_freezable
@attrs.define(slots=False)
class World2ActionPipelineConfig:
    precision: str
    scheduler: SchedulerConfig
    net: LazyDict[nn.Module]
    ema: EMAConfig
    xattn_layer_idx: int
    # VLSP: optional video-latent source prior + action conditioning.
    # Defaults preserve the original behaviour exactly (disabled / gaussian / normal).
    action_source_prior: ActionSourcePriorConfig = attrs.field(factory=ActionSourcePriorConfig)
    action_conditioning: ActionConditioningConfig = attrs.field(factory=ActionConditioningConfig)
