from hydra.core.config_store import ConfigStore

from cosmos_predict2.configs.config_world2action import (
    ActionConditioningConfig,
    ActionSourcePriorConfig,
    SchedulerConfig,
    World2ActionPipelineConfig,
)
from cosmos_predict2.configs.defaults.ema import EMAConfig
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.world2action_dit import World2ActionDIT as VarNoiseWorld2ActionDIT
from imaginaire.lazy_config import LazyCall as L

ACTION_DECODER_NETS = {
    "libero": L(VarNoiseWorld2ActionDIT)(
        max_horizon=61,
        in_channels=10,
        out_channels=10,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
    "bridge": L(VarNoiseWorld2ActionDIT)(
        max_horizon=16,
        in_channels=10,
        out_channels=10,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
}


def register_pipe() -> None:
    cs = ConfigStore.instance()

    for name, net in ACTION_DECODER_NETS.items():
        cs.store(
            group="world2action_pipe",
            package="world2action_pipe",
            name=name,
            node=L(World2ActionPipelineConfig)(
                precision="bfloat16",
                scheduler=SchedulerConfig(alpha=1.0, beta=1.0, num_denoising_steps=10),
                net=net,
                ema=EMAConfig(enabled=False),
                # VLSP defaults; all scalar sub-fields are overridable from the
                # experiment / CLI exactly like scheduler.num_denoising_steps.
                action_source_prior=ActionSourcePriorConfig(),
                action_conditioning=ActionConditioningConfig(),
            ),
        )
