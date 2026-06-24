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
import collections
import gc
import math
from collections.abc import Mapping
from typing import Any

import attrs
import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from omegaconf import DictConfig
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as F
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.configs.config_video2world import EMAConfig
from cosmos_predict2.configs.config_world2action import World2ActionPipelineConfig
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.models.action_source_prior import COND_MODE_IDS, compute_prior_regularization
from cosmos_predict2.pipelines.video2world import (
    Video2WorldPipeline,
    Video2WorldPipelineConfig,
)
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from cosmos_predict2.utils.checkpointer import non_strict_load_model
from cosmos_predict2.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2.utils.torch_future import clip_grad_norm_
from imaginaire.lazy_config import LazyDict, instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log


@attrs.define(slots=False)
class World2ActionModelConfig:
    train_architecture: str  # base or lora
    lora_rank: int
    lora_alpha: int
    lora_target_modules: str
    init_lora_weights: bool

    precision: str
    loss_reduce: str
    loss_scale: float
    ema: EMAConfig

    # This is used for the original way to load models
    action_dit_path: str
    video_dit_path: str
    pipe_config: World2ActionPipelineConfig
    video_pipe_config: Video2WorldPipelineConfig

    fsdp_shard_size: int  # 0 means not using fsdp, -1 means set to world size
    data_config: DictConfig


def _dp_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        group = parallel_state.get_data_parallel_group()
        world = parallel_state.get_data_parallel_world_size()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        x /= world
    return x


def _dp_mean_dict(d: dict[str, object], device: torch.device) -> dict[str, float]:
    keys = list(d.keys())
    t = torch.stack([torch.as_tensor(d[k], device=device, dtype=torch.float32) for k in keys], dim=0)
    t = _dp_mean(t)
    return {k: t[i].item() for i, k in enumerate(keys)}


class World2ActionModel(ImaginaireModel):
    def __init__(self, config: World2ActionModelConfig):
        super().__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        # 1. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")

        self.pipe: World2ActionPipeline = World2ActionPipeline.from_config(
            config.pipe_config,
            dit_path=config.action_dit_path,
            **self.tensor_kwargs,
        )

        self.video2world_pipe: Video2WorldPipeline = Video2WorldPipeline.from_config(
            config.video_pipe_config,
            dit_path=config.video_dit_path,
            use_text_encoder=False,
        )
        self.video2world_pipe.requires_grad_(False)
        if config.video_pipe_config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(config.video_pipe_config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        self.freeze_parameters()
        if config.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.dit,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_target_modules=config.lora_target_modules,
                init_lora_weights=config.init_lora_weights,
            )
            if self.pipe.dit_ema:
                self.add_lora_to_model(
                    self.pipe.dit_ema,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                    init_lora_weights=config.init_lora_weights,
                )
        else:
            self.pipe.denoising_model().requires_grad_(True)

        # VLSP: the video-latent source prior is always fully trainable (even in
        # LoRA mode); the video model stays frozen.  Match the trainable dtype of
        # the action DiT so the optimizer param group is homogeneous: LoRA upcasts
        # its trainable params to fp32, base training keeps them in self.precision.
        if self.pipe.source_prior_has_params:
            self.pipe.source_prior.requires_grad_(True)
            self.pipe.source_prior.train()
            if config.train_architecture == "lora":
                for p in self.pipe.source_prior.parameters():
                    p.data = p.data.to(torch.float32)

        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

        if config.fsdp_shard_size != 0 and torch.distributed.is_initialized():
            if config.fsdp_shard_size == -1:
                fsdp_shard_size = torch.distributed.get_world_size()
                replica_group_size = 1
            else:
                fsdp_shard_size = min(config.fsdp_shard_size, torch.distributed.get_world_size())
                replica_group_size = torch.distributed.get_world_size() // fsdp_shard_size
            dp_mesh = init_device_mesh(
                "cuda",
                (replica_group_size, fsdp_shard_size),
                mesh_dim_names=("replicate", "shard"),
            )
            log.info(f"Using FSDP with shard size {fsdp_shard_size} | device mesh: {dp_mesh}")
            self.pipe.apply_fsdp(dp_mesh)
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")

    # New function, added for i4 adaption
    @property
    def net(self) -> torch.nn.Module:
        return self.pipe.dit

    # New function, added for i4 adaption
    @property
    def net_ema(self) -> torch.nn.Module:
        return self.pipe.dit_ema

    def is_image_batch(self, batch: dict) -> bool:
        return False

    # New function, added for i4 adaption
    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            config_model (ModelConfig): The config object for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """
        # Include the VLSP source prior's parameters in the optimizer. The DiT
        # keeps its base/LoRA trainability; the source prior is fully trainable.
        if self.pipe.source_prior_has_params:
            optim_module: nn.Module = nn.ModuleList([self.net, self.pipe.source_prior])
        else:
            optim_module = self.net
        optimizer: torch.optim.Optimizer = instantiate(optimizer_config, model=optim_module)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int,
    ) -> None:
        """
        update the net_ema
        """
        del scheduler, optimizer

        if self.config.pipe_config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(iteration)
            self.pipe.dit_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)
            if self.pipe.source_prior_ema is not None:
                self._update_source_prior_ema(ema_beta)

    @torch.no_grad()
    def _update_source_prior_ema(self, beta: float) -> None:
        """Param-wise EMA for the VLSP source prior (weights = beta*ema + (1-beta)*new).

        Implemented for the standard DDP / non-FSDP path. FSDP+EMA for the source
        prior is not specially handled (the prior is small and left replicated);
        see VLSP.md for the limitation.
        """
        src_params = dict(self.pipe.source_prior.named_parameters())
        for name, p_ema in self.pipe.source_prior_ema.named_parameters():
            p_ema.mul_(beta).add_(src_params[name].detach(), alpha=1.0 - beta)
        src_buffers = dict(self.pipe.source_prior.named_buffers())
        for name, b_ema in self.pipe.source_prior_ema.named_buffers():
            if name in src_buffers:
                b_ema.copy_(src_buffers[name])

    # New function, added for i4 adaption
    def on_train_start(self, memory_format: torch.memory_format, dataset_stats: dict, stats_id: str) -> None:
        if self.config.pipe_config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        self.net.to(memory_format=memory_format, **self.tensor_kwargs)

        self.stats_id = stats_id
        self.pipe.normalizer.build_from_stats(
            dataset_stats,
            normalization_types=extract_normalization_types(self.config.data_config.policy_io.policy_io),
            concat_groups=self.config.data_config.policy_io.concat_groups,
            **self.tensor_kwargs,
        )
        self.pipe.normalizer.requires_grad_(False)

    def freeze_parameters(self) -> None:
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def add_lora_to_model(
        self,
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,
    ):
        from peft import LoraConfig, inject_adapter_in_model

        # Add LoRA to UNet
        self.lora_alpha = lora_alpha

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    def draw_training_t_and_epsilon(
        self,
        x0_size: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = torch.randn(x0_size, dtype=torch.float32, device=self.tensor_kwargs["device"])
        t_B = self.pipe.scheduler.sample_t(x0_size[0])

        return t_B.unsqueeze(1).repeat(1, x0_size[1]).unsqueeze(2), epsilon

    def draw_training_t(self, x0_size: torch.Size) -> torch.Tensor:
        """Draw the flow-matching timestep only.

        The source endpoint is drawn separately via ``pipe.sample_action_source``;
        with ``source_mode="gaussian"`` that draw matches the old epsilon exactly,
        and drawing it *before* the timestep preserves the original RNG order.
        """
        t_B = self.pipe.scheduler.sample_t(x0_size[0])
        return t_B.unsqueeze(1).repeat(1, x0_size[1]).unsqueeze(2)

    def compute_loss(
        self,
        x0_B_HA_A: torch.Tensor,
        source_B_HA_A: torch.Tensor,
        t_B_HA_1: torch.Tensor,
        crossattn_emb: torch.Tensor,
        video_sigma_B_1: torch.Tensor,
        state_B_HO_O: torch.Tensor,
        source_metrics: dict[str, torch.Tensor],
    ) -> tuple[dict, torch.Tensor]:
        """Flow-matching loss with a configurable source endpoint (VLSP).

        Interpolate between the (normalized) action ``x0`` and the source ``s``,
        predict the flow field ``u_t = s - x0`` and regress it, optionally adding
        a regularizer on the learned source prior:

            x_t = (1 - t) * x0 + t * s
            u_t = s - x0
            L   = || v_theta(x_t, t, c) - u_t ||^2  +  L_prior

        With ``source_mode="gaussian"`` (the default) ``s`` is N(0, I) and this is
        identical to the original action flow loss.
        """
        # scale to have unit variance. don't know if this helps.
        xt_B_HA_A = (1 - t_B_HA_1) * x0_B_HA_A + t_B_HA_1 * source_B_HA_A
        ut_B_HA_A = source_B_HA_A - x0_B_HA_A

        vt_B_HA_A = self.pipe.denoise(
            xt_B_HA_A,
            t_B_HA_1,
            state_B_HO_O,
            crossattn_emb,
            video_sigma_B_1,
            obs_dropout=0.2,
            return_hidden_states=False,
        ).float()
        loss_flow = F.mse_loss(vt_B_HA_A, ut_B_HA_A, reduction=self.loss_reduce) * self.loss_scale

        # Optional regularizers on q_phi(s | video) (KL / mean-L2 / std). All
        # weights default to 0.0, so loss_prior is a no-op for the baseline.
        loss_prior, prior_logs = compute_prior_regularization(source_metrics, self.pipe.config.action_source_prior)
        loss = loss_flow + loss_prior

        with torch.no_grad():
            var_inst_x0 = x0_B_HA_A.float().var(dim=(1, 2)).mean()

            if not self.pipe.source_prior_enabled:
                # Exact baseline logging path (same keys / collective as before).
                metrics = _dp_mean(torch.stack([loss.float(), var_inst_x0], dim=0).to(x0_B_HA_A.device))
                if not dist.is_available() or not dist.is_initialized() or parallel_state.get_data_parallel_rank() == 0:
                    output_batch = {"loss": metrics[0].item(), "Var_inst[x_0]": metrics[1].item()}
                else:
                    output_batch = {}
            else:
                scalars: dict[str, torch.Tensor] = {
                    "loss": loss.detach().float(),
                    "loss/flow": loss_flow.detach().float(),
                    "Var_inst[x_0]": var_inst_x0,
                }
                if torch.is_tensor(loss_prior):
                    scalars["loss/source_prior"] = loss_prior.detach().float()
                scalars.update(prior_logs)
                for key, val in source_metrics.items():
                    if key in ("mu", "logstd"):
                        continue  # tensors used for regularization only
                    scalars[key] = val
                cond_mode = self.pipe.config.action_conditioning.mode
                scalars["condition/mode_id"] = torch.as_tensor(
                    float(COND_MODE_IDS.get(cond_mode, -1)), device=loss.device
                )
                scalars["condition/shuffle_enabled"] = torch.as_tensor(
                    1.0 if cond_mode == "shuffled_video" else 0.0, device=loss.device
                )
                reduced = _dp_mean_dict(scalars, device=x0_B_HA_A.device)
                if not dist.is_available() or not dist.is_initialized() or parallel_state.get_data_parallel_rank() == 0:
                    output_batch = reduced
                else:
                    output_batch = {}

        del var_inst_x0
        gc.collect(0)

        return output_batch, loss

    def get_crossattn_emb(
        self,
        data_batch: dict,
        video_sigma_B_1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, video_B_C_T_H_W, condition = self.video2world_pipe.get_mimic_data_and_condition(data_batch)

        video_epsilon_B_C_T_H_W = torch.randn(video_B_C_T_H_W.size(), **self.tensor_kwargs)

        if video_sigma_B_1 is None:
            video_sigma_B_1 = self.draw_video_sigma(video_B_C_T_H_W.size(), condition)

        world_pred = self.video2world_pipe.denoise(
            video_B_C_T_H_W + video_epsilon_B_C_T_H_W * rearrange(video_sigma_B_1, "b t -> b 1 t 1 1"),
            video_sigma_B_1,
            condition,
            use_cuda_graphs=False,
            return_only_hidden_states_up_to=self.pipe.config.xattn_layer_idx,
            return_decoded_video=False,
        )

        crossattn_emb = world_pred.hidden_states[self.pipe.config.xattn_layer_idx]

        del world_pred
        gc.collect(0)

        B, T, H, W, D = crossattn_emb.shape
        crossattn_emb = crossattn_emb.reshape(B, T * H * W, D)

        return crossattn_emb, video_sigma_B_1

    def predict(self, data_batch: dict, video_sigma_B_1: torch.Tensor) -> torch.Tensor:
        crossattn_emb, video_sigma_B_1 = self.get_crossattn_emb(data_batch, video_sigma_B_1)
        state_B_HO_O = data_batch["obs/lowdim_concat"]

        return self.pipe(state_B_HO_O, crossattn_emb, video_sigma_B_1)

    def draw_video_sigma(self, x0_size: torch.Size, condition: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x0_size[0]

        sigma_B = self.video2world_pipe.scheduler.sample_sigma(batch_size)
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO

        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        if is_video_batch:
            # Implement the high sigma strategy LOGUNIFORM200_100000
            LOG_200 = math.log(200)
            LOG_100000 = math.log(100000)
            mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < 0.05
            log_new_sigma = (
                torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (LOG_100000 - LOG_200)
                + LOG_200
            )
            sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
        return sigma_B_1

    def training_step(self, data_batch: dict, iteration: int) -> tuple[dict, torch.Tensor]:
        data_batch["obs/language_embedding"] = data_batch["obs/language_embedding"].squeeze(1)
        B, _HA, A = data_batch["action/lowdim_concat"].shape
        if "obs/lowdim_concat" not in data_batch:
            data_batch["obs/lowdim_concat"] = torch.empty((B, 0, A), **self.tensor_kwargs)

        crossattn_emb, video_sigma_B_1 = self.get_crossattn_emb(data_batch)

        normalised_data_batch: dict = self.pipe.normalizer(data_batch, strict=False)

        x0_B_HA_A = normalised_data_batch["action/lowdim_concat"]

        state_B_HO_O = normalised_data_batch["obs/lowdim_concat"]

        language_B_L_D = (
            data_batch.get("obs/language_embedding")
            if self.pipe.config.action_source_prior.use_language
            else None
        )

        # VLSP: draw the flow source first (gaussian == the old epsilon draw),
        # then the timestep -> identical RNG order to the original baseline.
        source_B_HA_A, source_metrics = self.pipe.sample_action_source(
            x0_shape=x0_B_HA_A.size(),
            crossattn_emb=crossattn_emb,
            state_B_HO_O=state_B_HO_O,
            context_timesteps_B_1=video_sigma_B_1,
            x0_B_HA_A=x0_B_HA_A,
            language_B_L_D=language_B_L_D,
            training=True,
        )
        t_B_HA_1 = self.draw_training_t(x0_B_HA_A.size())

        # VLSP: optionally zero / shuffle / drop the video condition fed to the
        # action DiT (independent of the source prior input above).
        crossattn_for_action = self.pipe.prepare_action_condition(crossattn_emb, training=True)

        output_batch, loss = self.compute_loss(
            x0_B_HA_A,
            source_B_HA_A,
            t_B_HA_1,
            crossattn_for_action,
            video_sigma_B_1,
            state_B_HO_O,
            source_metrics,
        )

        return output_batch, loss

    @torch.inference_mode()
    def validation_step(self, data_batch: dict, iteration: int):
        output_batch, loss = self.training_step(data_batch, iteration)
        unnormed_x0_B_HA_A = data_batch["action/lowdim_concat"]

        output_batch["mses"] = collections.defaultdict(list)

        # get mses for gt video + noise
        self.video2world_pipe.scheduler.set_timesteps(35, device=self.tensor_kwargs["device"])
        for video_sigma in self.video2world_pipe.scheduler.sigmas:
            video_sigma_B_1 = video_sigma.repeat(unnormed_x0_B_HA_A.shape[0]).unsqueeze(1)
            unnormed_x0_pred_B_HA_A = self.predict(data_batch, video_sigma_B_1).float()

            mses_gtvid = {
                "gtvid/full": F.mse_loss(unnormed_x0_pred_B_HA_A, unnormed_x0_B_HA_A.float()),
            }
            mses_gtvid = _dp_mean_dict(mses_gtvid, device=unnormed_x0_pred_B_HA_A.device)

            if dist.is_available() and dist.is_initialized() and parallel_state.get_data_parallel_rank() != 0:
                continue

            for name, mse in mses_gtvid.items():
                output_batch["mses"][name].append((video_sigma.item(), mse))

        del (
            video_sigma,
            video_sigma_B_1,
            unnormed_x0_pred_B_HA_A,
        )
        gc.collect()

        # get mses for generated video
        input_vid = data_batch["obs/workspace_rgb"]
        B, C, T, H, W = input_vid.shape
        assert T in (1, 5)
        vid_input = torch.zeros((B, C, 61, H, W), device=input_vid.device, dtype=input_vid.dtype)
        vid_input[:, :, :T, :, :] = input_vid

        context = self.video2world_pipe.generate_video(
            vid_input=vid_input,
            num_latent_conditional_frames=1 if T == 1 else 2,
            prompt_embedding=data_batch["obs/language_embedding"],
            guidance=0.0,
            num_sampling_step=35,
            seed=0,
            use_cuda_graphs=False,
            return_all_context=True,
            hidden_state_layer_idx=self.pipe.config.xattn_layer_idx,
        )
        for video_sigma, crossattn_emb in context:
            video_sigma_B_1 = video_sigma.repeat(unnormed_x0_B_HA_A.shape[0]).unsqueeze(1)

            hidden_state_shape = crossattn_emb.shape
            crossattn_emb = crossattn_emb.reshape(hidden_state_shape[0], -1, hidden_state_shape[-1])

            genvid_unnormed_x0_pred_B_HA_A = self.pipe(
                state_B_HO_O=data_batch["obs/lowdim_concat"],
                crossattn_emb=crossattn_emb,
                context_timesteps_B_1=video_sigma_B_1,
                seed=0,
                use_cuda_graphs=False,
            )

            mses_genvid = {
                "genvid/full": F.mse_loss(genvid_unnormed_x0_pred_B_HA_A, unnormed_x0_B_HA_A.float()),
            }
            mses_genvid = _dp_mean_dict(mses_genvid, device=genvid_unnormed_x0_pred_B_HA_A.device)

            if dist.is_available() and dist.is_initialized() and parallel_state.get_data_parallel_rank() != 0:
                continue

            for name, mse in mses_genvid.items():
                output_batch["mses"][name].append((video_sigma.item(), mse))

        return output_batch, loss

    # ------------------ Checkpointing ------------------

    def state_dict(self) -> dict[str, Any]:
        # the checkpoint format should be compatible with traditional imaginaire4
        # pipeline contains both net and net_ema
        # checkpoint should be saved/loaded from Model
        # checkpoint should be loadable from pipeline as well - We don't use Model for inference only jobs.

        net_state_dict = self.pipe.dit.state_dict(prefix="net.")
        if self.config.pipe_config.ema.enabled:
            ema_state_dict = self.pipe.dit_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)

        # VLSP: persist the source prior (and its EMA) alongside the DiT weights.
        # Keys are prefixed so they round-trip through load_state_dict below and
        # do not collide with the action/ema DiT keys.
        if self.pipe.source_prior_has_params:
            net_state_dict.update(self.pipe.source_prior.state_dict(prefix="source_prior."))
            if self.config.pipe_config.ema.enabled and self.pipe.source_prior_ema is not None:
                net_state_dict.update(self.pipe.source_prior_ema.state_dict(prefix="source_prior_ema."))

        # convert DTensor to Tensor
        for key, val in net_state_dict.items():
            if isinstance(val, DTensor):
                # Convert to full tensor
                net_state_dict[key] = val.full_tensor().detach().cpu()
            else:
                net_state_dict[key] = val.detach().cpu()

        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:e
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                    those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                    matching keys one-by-one. This is typically used when loading parts of state dicts
                                    or using customized loading procedures. Defaults to False.
        """
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        _sp_state_dict = collections.OrderedDict()
        _sp_ema_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v
            elif k.startswith("source_prior_ema."):
                _sp_ema_state_dict[k[len("source_prior_ema.") :]] = v
            elif k.startswith("source_prior."):
                _sp_state_dict[k[len("source_prior.") :]] = v

        # VLSP: load the source prior non-strictly so that (a) old checkpoints with
        # no source-prior keys load fine (the freshly-initialized prior is kept),
        # and (b) an old action-decoder checkpoint can seed a brand-new source
        # prior under mode != "gaussian".
        if self.pipe.source_prior_has_params:
            if len(_sp_state_dict) > 0:
                sp_res = self.pipe.source_prior.load_state_dict(_sp_state_dict, strict=False)
                log.info(
                    f"Loaded source_prior: missing={len(sp_res.missing_keys)}, "
                    f"unexpected={len(sp_res.unexpected_keys)}"
                )
            else:
                log.warning("No source_prior.* weights in checkpoint; using freshly initialized source prior.")
            if (
                self.config.pipe_config.ema.enabled
                and self.pipe.source_prior_ema is not None
                and len(_sp_ema_state_dict) > 0
            ):
                self.pipe.source_prior_ema.load_state_dict(_sp_ema_state_dict, strict=False)

        state_dict = _reg_state_dict

        if strict:
            reg_results: _IncompatibleKeys = self.pipe.dit.load_state_dict(
                _reg_state_dict, strict=strict, assign=assign
            )

            if self.config.pipe_config.ema.enabled:
                ema_results: _IncompatibleKeys = self.pipe.dit_ema.load_state_dict(
                    _ema_state_dict, strict=strict, assign=assign
                )

            return _IncompatibleKeys(
                missing_keys=reg_results.missing_keys
                + (ema_results.missing_keys if self.config.pipe_config.ema.enabled else []),
                unexpected_keys=reg_results.unexpected_keys
                + (ema_results.unexpected_keys if self.config.pipe_config.ema.enabled else []),
            )
        else:
            log.critical("load model in non-strict mode")
            log.critical(non_strict_load_model(self.pipe.dit, _reg_state_dict), rank0_only=False)
            if self.config.pipe_config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(
                    non_strict_load_model(self.pipe.dit_ema, _ema_state_dict),
                    rank0_only=False,
                )

    # ------------------ public methods ------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.pipe_config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.pipe.ema_exp_coefficient + 1)

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
    ) -> torch.Tensor:
        params = list(self.net.parameters())
        if self.pipe.source_prior_has_params:
            params = params + list(self.pipe.source_prior.parameters())
        return clip_grad_norm_(
            params,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )
