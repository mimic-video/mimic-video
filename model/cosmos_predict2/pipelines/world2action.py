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

from contextlib import contextmanager

import numpy as np
import torch
from megatron.core import parallel_state
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard

from cosmos_predict2.configs.config_world2action import (
    World2ActionPipelineConfig,
)
from cosmos_predict2.models.action_source_prior import ActionSourcePrior, apply_action_conditioning
from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict
from cosmos_predict2.module.normalizer import StaticBatchNormalizer
from cosmos_predict2.pipelines.base import BasePipeline
from cosmos_predict2.schedulers.beta_scheduler import BetaScheduler
from cosmos_predict2.utils.dtensor_helper import (
    DTensorFastEmaModelUpdater,
    broadcast_dtensor_model_states,
)
from imaginaire.lazy_config import instantiate
from imaginaire.utils import log
from imaginaire.utils.ema import FastEmaModelUpdater


class World2ActionPipeline(BasePipeline):
    def __init__(
        self,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.dit: torch.nn.Module
        self.config: World2ActionPipelineConfig
        self.ema_dit: torch.nn.Module
        self.scheduler: BetaScheduler
        self.model_names = ["dit"]
        self.use_unified_sequence_parallel = False
        self.normalizer = StaticBatchNormalizer()
        # VLSP: video-latent source prior (set in from_config). When VLSP is
        # disabled this is an ActionSourcePrior with no parameters that simply
        # reproduces the Gaussian source, so nothing downstream changes.
        self.source_prior: ActionSourcePrior
        self.source_prior_ema: ActionSourcePrior | None = None

    @staticmethod
    def from_config(
        config: World2ActionPipelineConfig,
        dit_path: str = "",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "World2ActionPipeline":
        # Create a pipe
        pipe = World2ActionPipeline(device=device, torch_dtype=dtype)
        pipe.config = config
        pipe.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
        log.warning(f"precision {pipe.precision}")

        # 2. setup up denoising scheduler
        pipe.scheduler = BetaScheduler(
            config.scheduler.alpha,
            config.scheduler.beta,
            config.scheduler.num_denoising_steps,
        )

        # 3. Set up DiT
        with init_weights_on_device():
            dit_config = config.net
            pipe.dit = instantiate(dit_config).eval()

        # VLSP source-prior weights are stashed here (if present in the checkpoint)
        # and loaded after the prior is constructed in step 4b below. This makes
        # inference/eval (which load via from_config, not the training Model) pick
        # up trained source-prior weights instead of leaving them random.
        sp_state_dict: dict = {}
        sp_ema_state_dict: dict = {}
        if dit_path:
            log.info(f"Loading DiT from {dit_path}")
            state_dict = load_state_dict(dit_path)
            state_dict_dit_compatible = dict()
            for k, v in state_dict.items():
                if k.startswith("net."):
                    state_dict_dit_compatible[k[4:]] = v
                elif k.startswith("source_prior_ema."):
                    sp_ema_state_dict[k[len("source_prior_ema.") :]] = v
                elif k.startswith("source_prior."):
                    sp_state_dict[k[len("source_prior.") :]] = v
                else:
                    state_dict_dit_compatible[k] = v
            pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
            del state_dict, state_dict_dit_compatible
            log.success(f"Successfully loaded DiT from {dit_path}")

        else:
            pipe.dit.to_empty(device="cuda")
            pipe.dit.init_weights()
            log.warning("dit_path not provided, initializing DiT with random weights")

        # 4. Handle EMA
        if config.ema.enabled:
            pipe.dit_ema = instantiate(dit_config).eval()
            pipe.dit_ema.requires_grad_(False)

            # default when not using FSDP, otherwise updated when enabling fsdp
            pipe.dit_ema_worker = FastEmaModelUpdater()

            s = config.ema.rate
            pipe.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
            # copying is only necessary when starting the training at iteration 0.
            # Actual state_dict should be loaded after the pipe is created.
            pipe.dit_ema_worker.copy_to(src_model=pipe.dit, tgt_model=pipe.dit_ema)

        # 4b. VLSP: build the action source prior. Dims are read from the action
        # DiT config so the prior produces (B, HA, action_dim) in the same
        # normalized action space the flow loss is computed on.
        net_cfg = config.net
        pipe.source_prior = ActionSourcePrior(
            config.action_source_prior,
            action_dim=int(net_cfg.out_channels),
            video_emb_dim=int(net_cfg.crossattn_emb_channels),
            state_dim=int(net_cfg.in_channels),
            max_horizon=int(net_cfg.max_horizon),
        ).to(device=device, dtype=dtype)

        # Load trained source-prior weights from the checkpoint, if present
        # (non-strict: old/baseline checkpoints simply have none).
        if pipe.source_prior.has_trainable_params and len(sp_state_dict) > 0:
            sp_res = pipe.source_prior.load_state_dict(sp_state_dict, strict=False)
            log.success(
                f"Loaded source_prior from checkpoint "
                f"(missing={len(sp_res.missing_keys)}, unexpected={len(sp_res.unexpected_keys)})"
            )
        elif pipe.source_prior.has_trainable_params and dit_path:
            log.warning("No source_prior.* weights found in checkpoint; source prior is randomly initialized.")

        if config.ema.enabled and pipe.source_prior.has_trainable_params:
            pipe.source_prior_ema = ActionSourcePrior(
                config.action_source_prior,
                action_dim=int(net_cfg.out_channels),
                video_emb_dim=int(net_cfg.crossattn_emb_channels),
                state_dim=int(net_cfg.in_channels),
                max_horizon=int(net_cfg.max_horizon),
            ).to(device=device, dtype=dtype)
            if len(sp_ema_state_dict) > 0:
                pipe.source_prior_ema.load_state_dict(sp_ema_state_dict, strict=False)
            else:
                pipe.source_prior_ema.load_state_dict(pipe.source_prior.state_dict())
            pipe.source_prior_ema.requires_grad_(False)
        else:
            pipe.source_prior_ema = None

        pipe.dit = pipe.dit.to(device=device, dtype=dtype)
        torch.cuda.empty_cache()

        # 5. training states
        if parallel_state.is_initialized():
            pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            pipe.data_parallel_size = 1

        return pipe

    def apply_fsdp(self, dp_mesh: DeviceMesh) -> None:
        self.dit.fully_shard(mesh=dp_mesh)
        self.dit = fully_shard(self.dit, mesh=dp_mesh, reshard_after_forward=True)
        broadcast_dtensor_model_states(self.dit, dp_mesh)
        if self.dit_ema:
            self.dit_ema.fully_shard(mesh=dp_mesh)
            self.dit_ema = fully_shard(self.dit_ema, mesh=dp_mesh, reshard_after_forward=True)
            broadcast_dtensor_model_states(self.dit_ema, dp_mesh)
            self.dit_ema_worker = DTensorFastEmaModelUpdater()
            # No need to copy weights to EMA when applying FSDP, it is already copied before applying FSDP.

    def apply_cp(self) -> None:
        raise NotImplementedError

    def denoising_model(self) -> torch.nn.Module:
        return self.dit

    # ------------------------------------------------------------------ #
    #  VLSP: source prior + action conditioning                          #
    # ------------------------------------------------------------------ #
    @property
    def source_prior_enabled(self) -> bool:
        return bool(self.config.action_source_prior.enabled)

    @property
    def source_prior_has_params(self) -> bool:
        return self.source_prior is not None and self.source_prior.has_trainable_params

    def sample_action_source(
        self,
        *,
        x0_shape: tuple[int, int, int] | torch.Size,
        crossattn_emb: torch.Tensor | None,
        state_B_HO_O: torch.Tensor | None,
        context_timesteps_B_1: torch.Tensor | None,
        x0_B_HA_A: torch.Tensor | None = None,
        language_B_L_D: torch.Tensor | None = None,
        training: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample the flow-matching source endpoint (VLSP entry point).

        ``source_mode="gaussian"`` reproduces the original baseline exactly. Other
        modes derive the source from the video latent (see action_source_prior.py).
        The returned source lives in the normalized action space.
        """
        return self.source_prior(
            tuple(x0_shape),
            crossattn_emb=crossattn_emb,
            state_B_HO_O=state_B_HO_O,
            context_timesteps_B_1=context_timesteps_B_1,
            x0_B_HA_A=x0_B_HA_A,
            language_B_L_D=language_B_L_D,
            generator=generator,
            training=training,
            seed=seed,
        )

    def prepare_action_condition(
        self,
        crossattn_emb: torch.Tensor,
        *,
        mode: str | None = None,
        training: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Transform the video latent before it is fed to the action DiT.

        This is *separate* from the source prior input so that e.g. the source can
        use the real video latent while the decoder receives a zeroed or shuffled
        condition. When VLSP is disabled the condition is always passed through
        unchanged (exact baseline behaviour).
        """
        if not self.source_prior_enabled:
            return crossattn_emb
        if mode is None:
            mode = self.config.action_conditioning.mode
        return apply_action_conditioning(
            crossattn_emb,
            mode=mode,
            dropout_prob=float(self.config.action_conditioning.dropout_prob),
            seed=seed,
            generator=generator,
            training=training,
        )

    def denoise(
        self,
        xt_B_HA_A: torch.Tensor,
        timesteps_B_HA_1: torch.Tensor,
        state_B_HO_O: torch.Tensor,
        crossattn_emb: torch.Tensor,
        context_timesteps_B_1: torch.Tensor,
        *,
        obs_dropout: float,
        use_cuda_graphs: bool = False,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Performs denoising on the input noise data, and condition

        Args:
            xt_B_T_A (torch.Tensor): The input noise data.
            state_B_H_A (torch.Tensor): conditional information, here robot state
            use_cuda_graphs (bool, optional): Whether to use CUDA Graphs for inference. Defaults to False.

        Returns:
            torch.Tensor: The denoising field v_t = epsilon - x0.
        """
        # scale to have unit variance. don't know if this helps.
        xt_B_HA_A = xt_B_HA_A / ((1 - timesteps_B_HA_1) ** 2 + timesteps_B_HA_1**2).sqrt()
        B, HO, _O = state_B_HO_O.shape

        timesteps_cond_B_HO_1 = torch.empty(
            B, HO, 1, dtype=timesteps_B_HA_1.dtype, device=timesteps_B_HA_1.device
        ).fill_(1e-3)
        timesteps_B_T_1 = torch.cat((timesteps_cond_B_HO_1, timesteps_B_HA_1), dim=1)

        vt_pred_B_T_A = self.dit(
            state_B_HO_O=state_B_HO_O.to(**self.tensor_kwargs),
            xt_B_HA_A=xt_B_HA_A.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T_1.squeeze(2),
            context_timesteps_B_1=context_timesteps_B_1,
            crossattn_emb=crossattn_emb.to(**self.tensor_kwargs),
            obs_dropout=obs_dropout,
            use_cuda_graphs=use_cuda_graphs,
            return_hidden_states=return_hidden_states,
        )  # vt_pred_B_HA_A[, hidden_states]
        if return_hidden_states:
            vt_pred_B_T_A, hidden_states = vt_pred_B_T_A

        vt_pred_B_HA_A = vt_pred_B_T_A[:, HO:, :]

        if return_hidden_states:
            return vt_pred_B_HA_A, hidden_states

        return vt_pred_B_HA_A

    @torch.no_grad()
    def __call__(
        self,
        state_B_HO_O: torch.Tensor,
        crossattn_emb: torch.Tensor,
        context_timesteps_B_1: torch.Tensor,
        seed: int = 0,
        use_cuda_graphs: bool = False,
    ) -> torch.Tensor:
        B, HO, _O = state_B_HO_O.shape
        T = self.config.net.max_horizon
        HA = T - HO

        if HO > 0:
            state_B_HO_O = self.normalizer.norms["obs/lowdim_concat"](state_B_HO_O)

        # VLSP: the source endpoint of the action flow. With source_mode="gaussian"
        # this is identical to the original misc.arch_invariant_rand(seed=...) draw.
        sample_B_HA_A, _source_metrics = self.sample_action_source(
            x0_shape=(B, HA, self.dit.out_channels),
            crossattn_emb=crossattn_emb,
            state_B_HO_O=state_B_HO_O,
            context_timesteps_B_1=context_timesteps_B_1,
            x0_B_HA_A=None,
            training=False,
            seed=seed,
        )
        sample_B_HA_A = sample_B_HA_A.to(**self.tensor_kwargs)

        # VLSP: optionally zero/shuffle the video condition fed to the action DiT
        # (no-op when action_conditioning.mode == "normal" or VLSP disabled).
        crossattn_for_action = self.prepare_action_condition(
            crossattn_emb,
            training=False,
            seed=seed,
        )

        timestep_B_HA_1 = torch.ones(
            (B, HA, 1),
            dtype=torch.float32,
            device=self.tensor_kwargs["device"],
        )

        for _ in range(self.scheduler.num_denoising_steps):
            vt_pred_B_HA_A = self.denoise(
                sample_B_HA_A,
                timestep_B_HA_1,
                state_B_HO_O,
                crossattn_for_action,
                context_timesteps_B_1,
                obs_dropout=0.0,
                use_cuda_graphs=use_cuda_graphs,
            )
            sample_B_HA_A, timestep_B_HA_1 = self.scheduler.step(
                vt_pred_B_HA_A,
                sample_B_HA_A,
                timestep_B_HA_1,
            )

        return self.normalizer.norms["action/lowdim_concat"].unnormalize(sample_B_HA_A)

    @contextmanager
    def ema_scope(self, context: None, is_cpu: bool = False):
        if self.config.ema.enabled:
            # https://github.com/pytorch/pytorch/issues/144289
            for module in self.dit.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.dit_ema_worker.cache(self.dit.parameters(), is_cpu=is_cpu)
            self.dit_ema_worker.copy_to(src_model=self.dit_ema, tgt_model=self.dit)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.dit.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.dit_ema_worker.restore(self.dit.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")
