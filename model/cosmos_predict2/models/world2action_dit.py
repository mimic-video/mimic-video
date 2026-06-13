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

import itertools as it
import math
from collections.abc import Callable

import einops
import torch
import transformer_engine as te
from einops import rearrange
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from transformer_engine.pytorch.attention import DotProductAttention

try:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb
except ImportError:  # TE >= 2.x moved it to the rope submodule
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

from cosmos_predict2.networks.selective_activation_checkpoint import (
    CheckpointMode,
    SACConfig,
)
from imaginaire.utils import log
from imaginaire.utils.graph import create_cuda_graph


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------- Feed Forward Network -----------------------
class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)

        x = self.activation(x)
        x = self.layer2(x)
        return x


class ActionEmbedder(nn.Module):
    """
    Expand `in_dim` to `out_dim` with an MLP whose per-layer growth is
    *roughly* `target_mult` (default 4.0).  The number of hops is chosen
    automatically to keep every hop near that multiplier.

    Example:
        in_dim=20, out_dim=2048, target_mult=4
        total_ratio = 102.4
        hops = round(log(total_ratio) / log(4)) = 3
        base ≈ 102.4 ** (1/3) ≈ 4.68
        widths: 20 → 94 → 440 → 2048   (all hops ≈4.68x)
    """

    def __init__(self, in_dim: int, out_dim: int, target_mult: float = 4.0):
        super().__init__()
        if not (in_dim > 0 < out_dim and out_dim > in_dim):
            raise ValueError("dims must be positive and out_dim > in_dim")
        if target_mult < 2.0:
            raise ValueError("target_mult should be at least 2")

        total_ratio = out_dim / in_dim
        hops = max(1, round(math.log(total_ratio, target_mult)))
        base = total_ratio ** (1.0 / hops)

        dims = [in_dim]
        for _ in range(hops - 1):
            next_dim = round(dims[-1] * base)
            dims.append(next_dim)
        dims.append(out_dim)

        self.layers = nn.Sequential()
        for d_in, d_out in it.pairwise(dims[:-1]):
            self.layers.append(nn.Linear(d_in, d_out))
            self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(dims[-2], dims[-1]))

        self.scaling_factor = 1.0 / math.sqrt(out_dim)
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.layers[:-1]:
            if isinstance(m, nn.Linear):
                # relu-gelu mismatch leads to slightly smaller variance in the end. fine.
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        nn.init.kaiming_uniform_(self.layers[-1].weight, nonlinearity="linear")
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x_B_T_A: torch.Tensor) -> torch.Tensor:
        x_B_T_D = self.layers(x_B_T_A)
        x_B_T_D = x_B_T_D * self.scaling_factor

        return x_B_T_D


def torch_attention_op(q_B_S_H_D: torch.Tensor, k_B_S_H_D: torch.Tensor, v_B_S_H_D: torch.Tensor) -> torch.Tensor:
    """Computes multi-head attention using PyTorch's native implementation.

    This function provides a PyTorch backend alternative to Transformer Engine's attention operation.
    It rearranges the input tensors to match PyTorch's expected format, computes scaled dot-product
    attention, and rearranges the output back to the original format.

    The input tensor names use the following dimension conventions:

    - B: batch size
    - S: sequence length
    - H: number of attention heads
    - D: head dimension

    Args:
        q_B_S_H_D: Query tensor with shape (batch, seq_len, n_heads, head_dim)
        k_B_S_H_D: Key tensor with shape (batch, seq_len, n_heads, head_dim)
        v_B_S_H_D: Value tensor with shape (batch, seq_len, n_heads, head_dim)

    Returns:
        Attention output tensor with shape (batch, seq_len, n_heads * head_dim)
    """
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    result_B_S_HD = rearrange(
        torch.nn.functional.scaled_dot_product_attention(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D),
        "b h ... l -> b ... (h l)",
    )

    return result_B_S_HD


class Attention(nn.Module):
    """
    A flexible attention module supporting both self-attention and cross-attention mechanisms.

    This module implements a multi-head attention layer that can operate in either self-attention
    or cross-attention mode. The mode is determined by whether a context dimension is provided.
    The implementation uses scaled dot-product attention and supports optional bias terms and
    dropout regularization.

    Args:
        query_dim (int): The dimensionality of the query vectors.
        context_dim (int, optional): The dimensionality of the context (key/value) vectors.
            If None, the module operates in self-attention mode using query_dim. Default: None
        n_heads (int, optional): Number of attention heads for multi-head attention. Default: 8
        head_dim (int, optional): The dimension of each attention head. Default: 64
        dropout (float, optional): Dropout probability applied to the output. Default: 0.0
        qkv_format (str, optional): Format specification for QKV tensors. Default: "bshd"
        backend (str, optional): Backend to use for the attention operation. Default: "transformer_engine"

    Examples:
        >>> # Self-attention with 512 dimensions and 8 heads
        >>> self_attn = Attention(query_dim=512)
        >>> x = torch.randn(32, 16, 512)  # (batch_size, seq_len, dim)
        >>> out = self_attn(x)  # (32, 16, 512)

        >>> # Cross-attention
        >>> cross_attn = Attention(query_dim=512, context_dim=256)
        >>> query = torch.randn(32, 16, 512)
        >>> context = torch.randn(32, 8, 256)
        >>> out = cross_attn(query, context)  # (32, 16, 512)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
        backend: str = "transformer_engine",
    ) -> None:
        super().__init__()
        log.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{n_heads} heads with a dimension of {head_dim}."
        )
        self.is_selfattn = context_dim is None  # self attention

        self.backend = backend

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        if self.backend == "transformer_engine":
            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
            )
        elif self.backend == "torch":
            self.attn_op = torch_attention_op
        elif self.backend == "flash_attn_no_cp":
            try:
                from flash_attn.flash_attn_interface import flash_attn_func

                self.attn_op = flash_attn_func
            except ImportError:  # flash-attn unavailable (e.g. Blackwell): torch SDPA, same [B,S,H,D] I/O
                import torch.nn.functional as _F

                def _flash_attn_func_fallback(q, k, v, *args, **kwargs):
                    out = _F.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                    )
                    return out.transpose(1, 2)

                self.attn_op = _flash_attn_func_fallback
        else:
            raise NotImplementedError(self.backend)

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            rope_emb: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            return q, k, v

        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

        return q, k, v

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        result = self.attn_op(q, k, v)  # [B, S, H, D]
        result = rearrange(result, "b s h d -> b s (h d)")
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
        """
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v)

    def set_context_parallel_group(
        self, process_group: ProcessGroup, ranks: list[int], stream: torch.cuda.Stream
    ) -> None:
        self.attn_op.set_context_parallel_group(process_group, ranks, stream)


class ContinuousSinusoidalPosEmb(nn.Module):
    def __init__(self, num_channels: int, min_period: int, max_period: int):
        super().__init__()

        if num_channels % 2 != 0:
            raise ValueError(f"embedding_dim ({num_channels}) must be even.")

        self._num_channels = num_channels
        self._min_period = min_period
        self._max_period = max_period

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        fraction = torch.linspace(0.0, 1.0, self._num_channels // 2, device=timesteps_B_T.device, dtype=torch.float64)
        freqs = math.tau / (self._min_period * (self._max_period / self._min_period) ** fraction)
        sinusoid_input_B_T_D = einops.einsum(timesteps_B_T, freqs, "b i, j -> b i j")
        return torch.cat((torch.sin(sinusoid_input_B_T_D), torch.cos(sinusoid_input_B_T_D)), dim=-1).bfloat16()


class PairTimestepEmbedding(nn.Module):
    """
    DxD Linear -> SiLU -> rank-R bilinear (CP) -> {3D or D}.
    If use_adaln_lora=True: returns (emb_B_T_D, adaln_lora_B_T_3D) with pass-through emb.
    If use_adaln_lora=False: returns (emb_B_T_D, None) with emb = output of the module.

    Args:
        in_features:  D (timestep feature width from ContinuousSinusoidalPosEmb)
        out_features: D (model width x_dim). Typically == in_features.
        rank:         R for CP bilinear (small, e.g. 64..256)
        use_adaln_lora: if True, output 3*D for AdaLN (added in blocks) and pass-through emb.
                        if False, output D and return it as emb (backcompat).
    """

    def __init__(self, in_features: int, out_features: int, rank: int, use_adaln_lora: bool = True):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.rank = rank
        self.use_adaln_lora = use_adaln_lora

        self.proj_x = nn.Linear(in_features, out_features, bias=True)
        self.proj_c = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.SiLU()

        self.B = nn.Linear(out_features, rank, bias=True)  # bias True to model marginals
        self.C = nn.Linear(out_features, rank, bias=True)  # subtract bias-bias term later to avoid global bias
        self.A = nn.Linear(rank, (3 if use_adaln_lora else 1) * out_features, bias=False)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        std_in = 1.0 / math.sqrt(self.in_dim)
        nn.init.trunc_normal_(self.proj_x.weight, std=std_in, a=-3 * std_in, b=3 * std_in)
        nn.init.trunc_normal_(self.proj_c.weight, std=std_in, a=-3 * std_in, b=3 * std_in)
        nn.init.zeros_(self.proj_x.bias)
        nn.init.zeros_(self.proj_c.bias)

        std_mid = 1.0 / math.sqrt(self.out_dim)
        eps = 0.382050174  # std(silu(N(0, 0.5))), supposed to include all the matmuls given the other inits
        nn.init.trunc_normal_(self.B.weight, std=std_mid, a=-3 * std_mid, b=3 * std_mid)
        nn.init.constant_(self.B.bias, eps)
        nn.init.trunc_normal_(self.C.weight, std=std_mid, a=-3 * std_mid, b=3 * std_mid)
        nn.init.constant_(self.C.bias, eps)

        std_r = 1.0 / math.sqrt(self.rank)
        nn.init.trunc_normal_(self.A.weight, std=std_r, a=-3 * std_r, b=3 * std_r)

    def forward(
        self,
        action_timestep_feats_B_T_D: torch.Tensor,
        video_timestep_feats_B_1_D: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        video_timestep_feats_B_T_D = video_timestep_feats_B_1_D.expand_as(action_timestep_feats_B_T_D)

        emb_act_B_T_D = self.act(self.proj_x(action_timestep_feats_B_T_D))
        emb_vid_B_T_D = self.act(self.proj_c(video_timestep_feats_B_T_D))

        # rank-R bilinear (CP): z = (B phi_x + b) ⊙ (C phi_c + c) - b ⊙ c
        z_act_B_T_R = self.B(emb_act_B_T_D)
        z_vid_B_T_R = self.C(emb_vid_B_T_D)
        z_B_T_R = z_act_B_T_R * z_vid_B_T_R - self.B.bias * self.C.bias

        out_B_T_Dor3D = self.A(z_B_T_R)

        if self.use_adaln_lora:
            emb_B_T_DorR = z_B_T_R
            adaln_lora_B_T_3D = out_B_T_Dor3D
        else:
            emb_B_T_DorR = out_B_T_Dor3D
            adaln_lora_B_T_3D = None

        return emb_B_T_DorR, adaln_lora_B_T_3D


class PairTimeEmbedder(nn.Module):
    """
    TimContinuousSinusoidalPosEmbesteps (shared) -> PairTimestepEmbedding.
    Returns:
      if use_adaln_lora: (emb_B_T_R, adaln_lora_B_T_3D)
      else:              (emb_B_T_D, None)      # back-compat single output
    """

    def __init__(self, model_channels: int, rank: int, use_adaln_lora: bool = True):
        super().__init__()
        self.use_adaln_lora = use_adaln_lora
        self.t_embed_action = ContinuousSinusoidalPosEmb(model_channels, min_period=4e-3, max_period=4.0)
        # self.t_embed_video = ContinuousSinusoidalPosEmb(model_channels, min_period=math.tau, max_period=1e4 * math.tau)
        self.pair = PairTimestepEmbedding(
            in_features=model_channels,
            out_features=model_channels,
            rank=rank,
            use_adaln_lora=use_adaln_lora,
        )

    def forward(
        self,
        t_action_B_T: torch.Tensor,
        t_video_B_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        emb_action_B_T_D = self.t_embed_action(t_action_B_T)
        emb_video_B_T_D = self.t_embed_action(t_video_B_1 / (1.0 + t_video_B_1))
        return self.pair(emb_action_B_T_D, emb_video_B_T_D)


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            out_channels,
            bias=False,
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False),
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        x_B_T_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        def _fn(
            _x_B_T_D: torch.Tensor,
            _norm_layer: nn.Module,
            _scale_B_T_1_1_D: torch.Tensor,
            _shift_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            return _norm_layer(_x_B_T_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        x_B_T_D = _fn(x_B_T_D, self.layer_norm, scale_B_T_D, shift_B_T_D)
        x_B_T_O = self.linear(x_B_T_D)
        return x_B_T_O


class Block(nn.Module):
    """
    A transformer block that combines cross-attention, self-attention, and MLP layers with AdaLN modulation.
    Each component (cross-attention, self-attention, MLP) has its own layer normalization and AdaLN modulation.

    Parameters:
        x_dim (int): Dimension of input features
        pair_timestep_feature_rank (int): Dimension of pair timestep embedding
        context_dim (int): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Multiplier for MLP hidden dimension. Default: 4.0
        use_adaln_lora (bool): Whether to use AdaLN-LoRA modulation. Default: False
        adaln_lora_dim (int): Hidden dimension for AdaLN-LoRA layers. Default: 256

    The block applies the following sequence:
    1. Self-attention with AdaLN modulation
    2. Cross-attention with AdaLN modulation
    3. MLP with AdaLN modulation

    Each component uses skip connections and layer normalization.
    """

    def __init__(
        self,
        x_dim: int,
        pair_timestep_feature_rank: int | None,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float,
        use_adaln_lora: bool,
        adaln_lora_dim: int,
        backend: str,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.pair_timestep_feature_rank = pair_timestep_feature_rank

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            backend=backend,
        )

        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            backend=backend,
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(pair_timestep_feature_rank, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(pair_timestep_feature_rank, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(pair_timestep_feature_rank, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.pair_timestep_feature_rank)
            torch.nn.init.trunc_normal_(
                self.adaln_modulation_self_attn[1].weight,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            torch.nn.init.trunc_normal_(
                self.adaln_modulation_cross_attn[1].weight,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_D: torch.Tensor,
        emb_B_T_DorR: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_D: torch.Tensor | None = None,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
        extra_per_block_pos_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_D = x_B_T_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_DorR) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_DorR) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_DorR) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_DorR
            ).chunk(3, dim=-1)
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_DorR
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_DorR).chunk(3, dim=-1)

        def _fn(_x_B_T_D, _norm_layer, _scale_B_T_D, _shift_B_T_D):
            return _norm_layer(_x_B_T_D) * (1 + _scale_B_T_D) + _shift_B_T_D

        def _x_fn(
            _x_B_T_D: torch.Tensor,
            layer_norm_cross_attn: Callable,
            _scale_cross_attn_B_T_D: torch.Tensor,
            _shift_cross_attn_B_T_D: torch.Tensor,
        ) -> torch.Tensor:
            _normalized_x_B_T_D = _fn(
                _x_B_T_D,
                layer_norm_cross_attn,
                _scale_cross_attn_B_T_D,
                _shift_cross_attn_B_T_D,
            )
            _result_B_T_D = self.cross_attn(
                _normalized_x_B_T_D,
                crossattn_emb,
                rope_emb=rope_emb_L_D,
            )
            return _result_B_T_D

        result_B_T_D = _x_fn(
            x_B_T_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_D,
            shift_cross_attn_B_T_D,
        )
        x_B_T_D = x_B_T_D + result_B_T_D * gate_cross_attn_B_T_D

        normalized_x_B_T_D = _fn(
            x_B_T_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_D,
            shift_self_attn_B_T_D,
        )
        result_B_T_D = self.self_attn(
            normalized_x_B_T_D,
            None,
            rope_emb=rope_emb_L_D,
        )
        x_B_T_D = x_B_T_D + result_B_T_D * gate_self_attn_B_T_D

        normalized_x_B_T_D = _fn(
            x_B_T_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_D,
            shift_mlp_B_T_D,
        )
        result_B_T_D = self.mlp(normalized_x_B_T_D)
        x_B_T_D = x_B_T_D + gate_mlp_B_T_D * result_B_T_D
        return x_B_T_D


class World2ActionDIT(nn.Module):
    """
    A clean impl of DIT adapted for robot actions.

    Args:
        max_horizon (int): Maximum number of action+obs timesteps in sequence.
        in_channels (int): Number of input channels (robot action dim).
        out_channels (int): Number of output channels (robot action dim).
        model_channels (int): Base number of channels used throughout the model.
        num_blocks (int): Number of transformer blocks.
        num_heads (int): Number of heads in the multi-head attention layers.
        mlp_ratio (float): Expansion ratio for MLP blocks.
        crossattn_emb_channels (int): Number of embedding channels for cross-attention.
        use_adaln_lora (bool): Whether to use AdaLN-LoRA.
        adaln_lora_dim (int): Dimension for AdaLN-LoRA.
    """

    def __init__(
        self,
        max_horizon: int,
        in_channels: int,
        out_channels: int,
        # attention settings
        model_channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        atten_backend: str,
        # cross attention settings
        crossattn_emb_channels: int,
        use_adaln_lora: bool,
        adaln_lora_dim: int,
        pair_timestep_feature_rank: int,
        sac_config: SACConfig,
    ) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.atten_backend = atten_backend
        self.cuda_graphs = {}

        self.ctx_norm = nn.LayerNorm(crossattn_emb_channels, eps=1e-6)

        self.obs_mask_token = nn.Parameter(torch.empty((1, 1, self.model_channels), dtype=torch.float32))
        self.obs_embedder = ActionEmbedder(in_channels, model_channels)
        self.action_embedder = ActionEmbedder(out_channels, model_channels)

        self.pos_embedding = nn.Parameter(torch.empty((1, self.max_horizon, self.model_channels), dtype=torch.float32))

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = PairTimeEmbedder(model_channels, pair_timestep_feature_rank, use_adaln_lora=use_adaln_lora)

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    pair_timestep_feature_rank=pair_timestep_feature_rank,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=atten_backend,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

        self.t_embedding_norm = te.pytorch.RMSNorm(
            pair_timestep_feature_rank if use_adaln_lora else model_channels, eps=1e-6
        )
        self.init_weights()
        self.enable_selective_checkpoint(sac_config)

    def init_weights(self) -> None:
        self.obs_embedder.init_weights()
        self.action_embedder.init_weights()

        std = 1.0 / math.sqrt(self.model_channels)
        torch.nn.init.trunc_normal_(self.obs_mask_token, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_embedding, std=std, a=-3 * std, b=3 * std)

        self.t_embedder.pair.init_weights()

        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

    def prepare_embedded_sequence(
        self,
        state_B_HO_O: torch.Tensor,
        xt_B_HA_A: torch.Tensor,
        *,
        obs_dropout: float,
    ) -> torch.Tensor:
        obs_emb = self.obs_embedder(state_B_HO_O)
        action_emb = self.action_embedder(xt_B_HA_A)

        obs_dropout_mask = torch.bernoulli(
            torch.full((obs_emb.shape[0], 1, 1), obs_dropout, device=obs_emb.device, dtype=obs_emb.dtype)
        )
        obs_emb = (1.0 - obs_dropout_mask) * obs_emb + obs_dropout_mask * self.obs_mask_token

        return torch.cat((obs_emb, action_emb), dim=1) + self.pos_embedding

    def forward(
        self,
        state_B_HO_O: torch.Tensor,
        xt_B_HA_A: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        context_timesteps_B_1: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        obs_dropout: float,
        use_cuda_graphs: bool = False,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
        """
        assert not (self.training and use_cuda_graphs), "CUDA Graphs are supported only for inference"
        x_B_T_D = self.prepare_embedded_sequence(state_B_HO_O, xt_B_HA_A, obs_dropout=obs_dropout)
        crossattn_emb = self.ctx_norm(crossattn_emb)

        t_embedding_B_T_DorR, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T, context_timesteps_B_1)
        t_embedding_B_T_DorR = self.t_embedding_norm(t_embedding_B_T_DorR)

        if use_cuda_graphs:
            shapes_key = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                x_B_T_D,
                t_embedding_B_T_DorR,
                crossattn_emb,
                None,
                adaln_lora_B_T_3D,
                None,
            )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks

        block_kwargs = {
            "rope_emb_L_D": None,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": None,
        }
        hidden_states = []
        for block in blocks:
            x_B_T_D = block(
                x_B_T_D,
                t_embedding_B_T_DorR,
                crossattn_emb,
                **block_kwargs,
            )
            if return_hidden_states:
                hidden_states.append(x_B_T_D.detach().clone())

        x_B_T_O = self.final_layer(x_B_T_D, t_embedding_B_T_DorR, adaln_lora_B_T_3D=adaln_lora_B_T_3D)

        if return_hidden_states:
            return x_B_T_O, hidden_states

        return x_B_T_O

    def enable_selective_checkpoint(self, sac_config: SACConfig):
        if sac_config.mode == CheckpointMode.NONE:
            pass
        else:
            log.debug(
                f"Enable selective checkpoint with {sac_config.mode}, for every {sac_config.every_n_blocks} blocks. Total blocks: {len(self.blocks)}"
            )
            _context_fn = sac_config.get_context_fn()
            for block_id, block in self.blocks.named_children():
                if int(block_id) % sac_config.every_n_blocks == 0:
                    log.debug(f"Enable selective checkpoint for block {block_id}")
                    block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=_context_fn,
                        preserve_rng_state=False,
                    )
                    self.blocks.register_module(block_id, block)
            self.register_module(
                "final_layer",
                ptd_checkpoint_wrapper(
                    self.final_layer,
                    context_fn=_context_fn,
                    preserve_rng_state=False,
                ),
            )

        return self

    def fully_shard(self, mesh: DeviceMesh) -> None:
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)

        fully_shard(self.action_embedder, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=True)

    def enable_context_parallel(self, process_group: ProcessGroup | None = None) -> None:
        raise NotImplementedError

    @property
    def is_context_parallel_enabled(self) -> bool:
        return False
