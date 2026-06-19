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
import copy

import numpy as np
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from omegaconf import MISSING

from cosmos_predict2.callbacks.video_eval import VideoEvalCallback
from cosmos_predict2.configs.defaults.data_video import train_datasets
from imaginaire.lazy_config import LazyCall as L

BASE: dict = dict(
    defaults=[
        {"override /model": "v2w_video2world_finetune"},  # Predict2Video2WorldModel: trains the VIDEO dit (not the action-decoder model that shadows v2w_pretrained_cosmos)
        {"override /video_dataset_train": MISSING},
        {"override /video_dataset_val": MISSING},
        {"override /dataloader_val": "vanilla"},
        {"override /dataloader_train": "vanilla"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "constant"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        cpu_offloading_activations=False,
        cpu_offloading_weights=False,
    ),
    trainer=dict(
        distributed_parallelism="ddp",
        grad_accum_iter=1,
        max_iter=1_000_000,
        logging_iter=1_000,
        run_validation=False,
        callbacks=dict(
            video_eval=L(VideoEvalCallback)(fuse_lora=MISSING),
        ),
    ),
    optimizer=dict(
        lr=MISSING,
    ),
)

lrs = np.logspace(-5, -3, 9)[[5]]
bszs = [32]
ranks = [256]


def get_local_batch_size(global_bsz: int) -> int:
    res = global_bsz / parallel_state.get_data_parallel_world_size()

    if not res.is_integer():
        msg = "That batch size doesn't work with the number of gpus you have."
        raise ValueError(msg)

    return int(res)


cs = ConfigStore.instance()

for rank in ranks:
    for dataset in train_datasets:
        for lr in lrs:
            for bsz in bszs:
                train_type = f"lora_rank{rank}" if rank is not None else "fullft"
                name = f"v2w_{dataset}_{train_type}_lr{lr:.3e}_bsz{bsz}"

                cfg = copy.deepcopy(BASE)

                cfg["defaults"][1]["override /video_dataset_train"] = dataset
                cfg["defaults"][2]["override /video_dataset_val"] = dataset

                cfg["optimizer"]["lr"] = lr.item()
                cfg["job"]["name"] = name
                cfg["dataloader_train"] = {"batch_size": L(get_local_batch_size)(global_bsz=bsz)}

                if rank is not None:
                    cfg["model"]["config"].update(
                        dict(
                            train_architecture="lora",
                            lora_rank=rank,
                            lora_alpha=32,
                            init_lora_weights=True,
                            lora_target_modules="q_proj,k_proj,v_proj,output_proj,x_embedder.proj.1,linear_1,linear_2,mlp.layer1,mlp.layer2",
                        )
                    )
                cfg["trainer"]["callbacks"]["video_eval"]["fuse_lora"] = rank is not None

                cs.store(
                    group="experiment",
                    package="_global_",
                    name=name,
                    node=cfg,
                )
