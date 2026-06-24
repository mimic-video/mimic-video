# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Lightweight smoke test for VLSP (Video-Latent Source Prior).

Exercises the source-prior module + action-conditioning helpers with fake
tensors -- no GPU / transformer-engine / dataset required.  Run with:

    python -m scripts.debug_vlsp

It checks:
  1. the prior instantiates and returns sources of shape [B, HA, A];
  2. ``gaussian`` works without any video latent;
  3. ``video_prior_sample`` works with a fake crossattn_emb;
  4. ``zero_video`` / ``shuffled_video`` conditioning preserve shape;
  5. ``video_prior_mean`` is deterministic for equal inputs;
  6. source-prior checkpoints round-trip and old (empty) checkpoints load when
     the prior is disabled.
"""

from __future__ import annotations

import torch

from cosmos_predict2.models.action_source_prior import (
    ActionSourcePrior,
    apply_action_conditioning,
    compute_prior_regularization,
)

# The real attrs config requires the hydra/omegaconf stack; fall back to a tiny
# duck-typed stand-in so this smoke test runs in a minimal (CPU-only) env too.
try:  # pragma: no cover - depends on environment
    from cosmos_predict2.configs.config_world2action import ActionSourcePriorConfig

    def make_cfg(**kw):
        return ActionSourcePriorConfig(**kw)

except Exception:  # noqa: BLE001
    from dataclasses import dataclass, fields

    @dataclass
    class _Cfg:
        enabled: bool = False
        mode: str = "gaussian"
        pool_type: str = "mean"
        hidden_dim: int = 256
        num_perceiver_latents: int = 8
        num_attention_heads: int = 8
        mlp_depth: int = 2
        logstd_min: float = -5.0
        logstd_max: float = 1.0
        init_logstd: float = -1.0
        residual_scale: float = 1.0
        blend_alpha: float = 1.0
        source_dropout_prob: float = 0.0
        dropout_granularity: str = "sample"
        sampling_temperature: float = 1.0
        detach_video_latents: bool = True
        use_state: bool = True
        use_context_timestep: bool = True
        use_language: bool = False
        kl_weight: float = 0.0
        mean_l2_weight: float = 0.0
        std_reg_weight: float = 0.0
        debug_noise_std: float = 0.05

    def make_cfg(**kw):
        valid = {f.name for f in fields(_Cfg)}
        return _Cfg(**{k: v for k, v in kw.items() if k in valid})


B, HO, HA, A, D = 2, 1, 15, 10, 2048
SHAPE = (B, HA, A)
MAX_HORIZON = 61


def _build(**kw) -> ActionSourcePrior:
    cfg = make_cfg(hidden_dim=128, num_attention_heads=4, **kw)
    return ActionSourcePrior(cfg, action_dim=A, video_emb_dim=D, state_dim=A, max_horizon=MAX_HORIZON)


def _fake_inputs():
    g = torch.Generator().manual_seed(0)
    crossattn = torch.randn(B, 7 * 5 * 5, D, generator=g)
    state = torch.randn(B, HO, A, generator=g)
    ctx_t = torch.rand(B, 1, generator=g) * 100.0
    x0 = torch.randn(B, HA, A, generator=g)
    return crossattn, state, ctx_t, x0


def main() -> None:
    torch.manual_seed(0)
    crossattn, state, ctx_t, x0 = _fake_inputs()

    # 1 + 3. gaussian works without any video latent.
    prior = _build(enabled=False, mode="gaussian")
    source, metrics = prior(
        SHAPE, crossattn_emb=None, state_B_HO_O=None, context_timesteps_B_1=None, x0_B_HA_A=None, training=True
    )
    assert source.shape == SHAPE == x0.shape, source.shape
    assert torch.isfinite(source).all()
    assert sum(p.numel() for p in prior.parameters()) == 0, "disabled prior must carry no params"
    print("[1/7] gaussian (disabled) -> shape", tuple(source.shape), "OK")

    # gaussian determinism via seed (inference path uses arch_invariant_rand).
    s1, _ = prior(SHAPE, crossattn_emb=None, state_B_HO_O=None, context_timesteps_B_1=None, training=False, seed=3)
    s2, _ = prior(SHAPE, crossattn_emb=None, state_B_HO_O=None, context_timesteps_B_1=None, training=False, seed=3)
    assert torch.equal(s1, s2)
    print("[2/7] gaussian seeded determinism OK")

    # 4. video_prior_sample works with a fake crossattn_emb.
    prior = _build(enabled=True, mode="video_prior_sample", pool_type="attention")
    source, metrics = prior(
        SHAPE,
        crossattn_emb=crossattn,
        state_B_HO_O=state,
        context_timesteps_B_1=ctx_t,
        x0_B_HA_A=x0,
        training=True,
    )
    assert source.shape == SHAPE and torch.isfinite(source).all()
    loss_prior, logs = compute_prior_regularization(metrics, prior.cfg)
    print("[3/7] video_prior_sample -> shape", tuple(source.shape), "metrics:", sorted(metrics))

    # pred shape sanity: u_t = source - x0 has the same shape as x0.
    pred_like = source - x0
    assert pred_like.shape == x0.shape
    print("[4/7] u_t = source - x0 shape", tuple(pred_like.shape), "OK")

    # 5. zero_video / shuffled_video conditioning preserve shape.
    zero_c = apply_action_conditioning(crossattn, mode="zero_video")
    shuf_c = apply_action_conditioning(crossattn, mode="shuffled_video", seed=1)
    norm_c = apply_action_conditioning(crossattn, mode="normal")
    assert zero_c.shape == shuf_c.shape == norm_c.shape == crossattn.shape
    assert torch.count_nonzero(zero_c) == 0
    assert torch.equal(norm_c, crossattn)
    print("[5/7] zero_video / shuffled_video conditioning preserve shape OK")

    # 6. video_prior_mean is deterministic for equal inputs.
    prior = _build(enabled=True, mode="video_prior_mean", pool_type="mean")
    m1, _ = prior(SHAPE, crossattn_emb=crossattn, state_B_HO_O=state, context_timesteps_B_1=ctx_t, training=False, seed=7)
    m2, _ = prior(
        SHAPE, crossattn_emb=crossattn, state_B_HO_O=state, context_timesteps_B_1=ctx_t, training=False, seed=123
    )
    assert torch.equal(m1, m2), "video_prior_mean must be deterministic regardless of seed"
    print("[6/7] video_prior_mean determinism OK")

    # 7. checkpoint round-trip + old/empty checkpoint into a disabled prior.
    enabled = _build(enabled=True, mode="video_prior_sample")
    sd = enabled.state_dict()
    reloaded = _build(enabled=True, mode="video_prior_sample")
    reloaded.load_state_dict(sd)
    disabled = _build(enabled=False, mode="gaussian")
    res = disabled.load_state_dict({}, strict=False)
    assert len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0
    print(f"[7/7] ckpt round-trip ({len(sd)} keys) + empty-load-into-disabled OK")

    # extra: every video mode x pool is finite and correctly shaped.
    for mode in [
        "video_prior_sample",
        "video_prior_mean",
        "video_prior_residual",
        "video_prior_blend",
        "video_prior_dropout",
        "shuffled_video_prior",
    ]:
        for pool in ["mean", "attention", "perceiver"]:
            p = _build(
                enabled=True,
                mode=mode,
                pool_type=pool,
                blend_alpha=0.5,
                source_dropout_prob=0.3,
                dropout_granularity="element",
                kl_weight=0.1,
                mean_l2_weight=0.1,
                std_reg_weight=0.1,
            )
            s, met = p(
                SHAPE,
                crossattn_emb=crossattn,
                state_B_HO_O=state,
                context_timesteps_B_1=ctx_t,
                x0_B_HA_A=x0,
                training=True,
            )
            assert s.shape == SHAPE and torch.isfinite(s).all(), (mode, pool)
            reg, _ = compute_prior_regularization(met, p.cfg)
            assert torch.is_tensor(reg) and torch.isfinite(reg).all(), (mode, pool)
    print("[extra] all 6 video modes x 3 pools + regularizers finite OK")

    print("\nALL VLSP SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
