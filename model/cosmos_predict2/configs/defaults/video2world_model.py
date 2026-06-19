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

from hydra.core.config_store import ConfigStore

from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.models.video2world_model import (
    Predict2ModelManagerConfig,
    Predict2Video2WorldModel,
    Predict2Video2WorldModelConfig,
)
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.lazy_config import LazyCall as L

_PREDICT2_VIDEO2WORLD_DDP_2B_480P_10FPS = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10),
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path=get_cosmos_predict2_video2world_checkpoint(model_size="2B", resolution="480", fps=10),
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=0,
            high_sigma_ratio=0.05,
            loss_scale=100.0,
        ),
        _recursive_=False,
    ),
)


def register_model() -> None:
    cs = ConfigStore.instance()

    cs.store(
        group="model",
        package="_global_",
        name="v2w_pretrained_cosmos",
        node=_PREDICT2_VIDEO2WORLD_DDP_2B_480P_10FPS,
    )
    # Distinct name so the video-backbone trainer (Predict2Video2WorldModel) is not shadowed by
    # world2action_model.py's same-named "v2w_pretrained_cosmos" (the action-decoder model).
    cs.store(
        group="model",
        package="_global_",
        name="v2w_video2world_finetune",
        node=_PREDICT2_VIDEO2WORLD_DDP_2B_480P_10FPS,
    )
