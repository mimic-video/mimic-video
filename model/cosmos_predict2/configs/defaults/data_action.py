import pathlib

import hydra
import numpy as np
import omegaconf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.utils.data import DataLoader

from cosmos_predict2.data.action.types import LieRepr, NormalizationType, ObsType
from cosmos_predict2.data.resumable_sampler import ResumableDistributedSampler
from cosmos_predict2.module.normalizer import array_to_stats
from imaginaire.lazy_config import LazyCall as L
from imaginaire.utils import distributed


class MockBridgeDataset:
    def __getitem__(self, idx) -> dict:
        return {
            "action/workspace_rgb": np.zeros((3, 56, 480, 640), dtype=np.float32),
            "action/lowdim_concat": np.zeros((15, 10), dtype=np.float32),
            "obs/workspace_rgb": np.zeros((3, 5, 480, 640), dtype=np.float32),
            "obs/language_embedding": np.zeros((1, 512, 1024), dtype=np.float32),
            "obs/lowdim_concat": np.zeros((1, 10), dtype=np.float32),
        }

    def __len__(self) -> int:
        return 4096

    def get_statistics(self) -> dict:
        return {
            "action/eef_rot_lowdim": array_to_stats(np.random.randn(10, 15, 6).astype(dtype=np.float32)),
            "action/eef_pos_lowdim": array_to_stats(np.random.randn(10, 15, 3).astype(dtype=np.float32)),
            "action/gripper_action_lowdim": array_to_stats(np.random.randn(10, 15, 1).astype(dtype=np.float32)),
            "obs/eef_rot_lowdim": array_to_stats(np.random.randn(10, 15, 6).astype(dtype=np.float32)),
            "obs/eef_pos_lowdim": array_to_stats(np.random.randn(10, 15, 3).astype(dtype=np.float32)),
            "obs/gripper_state_lowdim": array_to_stats(np.random.randn(10, 15, 1).astype(dtype=np.float32)),
        }

    @property
    def stats_id(self) -> str:
        return "mock_bridge"


def _worker_init_filesystem_sharing(worker_id: int) -> None:
    # Runs inside each DataLoader worker before it shares any tensor back to the main process.
    # This pod's /dev/shm is only 64MB and samples are large (full-res video clips), so share tensors
    # via the filesystem (/tmp, ~1.4TB) instead of /dev/shm to avoid "No space left on device".
    import torch

    torch.multiprocessing.set_sharing_strategy("file_system")


def get_data_config(config_name: str):
    omegaconf.OmegaConf.register_new_resolver("eval", eval, replace=True)
    omegaconf.OmegaConf.register_new_resolver("ObsType", ObsType.__getitem__, replace=True)
    omegaconf.OmegaConf.register_new_resolver("LieRepr", LieRepr.__getitem__, replace=True)
    omegaconf.OmegaConf.register_new_resolver("NormalizationType", NormalizationType.__getitem__, replace=True)

    with hydra.initialize(
        version_base=None,
        config_path="../dataloading/",
    ):
        cfg = hydra.compose(config_name=config_name)
    omegaconf.OmegaConf.resolve(cfg)

    return cfg


@distributed.in_rank_order
def get_dataset(data_config: omegaconf.DictConfig, is_train: bool):
    return hydra.utils.instantiate(
        data_config.dataset.dataset,
        train=is_train,
        verbose=distributed.is_rank0(),
    )


video_action_dataset_train = L(get_dataset)(data_config="${data_config}", is_train=True)
video_action_dataset_val = L(get_dataset)(data_config="${data_config}", is_train=False)

mock_video_action_dataset = L(MockBridgeDataset)()

DATA_CONFIGS = {
    f.stem: L(get_data_config)(config_name=f.stem)
    for f in (pathlib.Path(__file__).parents[1] / "dataloading").iterdir()
    if f.is_file()
}


def register_training_and_val_action_data():
    cs = ConfigStore()
    for name, cfg in DATA_CONFIGS.items():
        cs.store(
            group="data_config",
            package="data_config",
            name=name,
            node=cfg,
        )

    from megatron.core import parallel_state

    mimic_dataloader_train = L(DataLoader)(
        dataset=video_action_dataset_train,
        sampler=L(ResumableDistributedSampler)(
            dataset=video_action_dataset_train,
            num_replicas=L(parallel_state.get_data_parallel_world_size)(),
            rank=L(parallel_state.get_data_parallel_rank)(),
            shuffle=True,
            seed=0,
        ),
        batch_size=MISSING,
        prefetch_factor=8,
        drop_last=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        in_order=False,
        worker_init_fn=_worker_init_filesystem_sharing,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="mimic",
        node=mimic_dataloader_train,
    )

    mimic_dataloader_val = L(DataLoader)(
        dataset=video_action_dataset_val,
        sampler=L(ResumableDistributedSampler)(
            dataset=video_action_dataset_val,
            num_replicas=L(parallel_state.get_data_parallel_world_size)(),
            rank=L(parallel_state.get_data_parallel_rank)(),
            shuffle=False,
            seed=0,
        ),
        batch_size=1,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="mimic",
        node=mimic_dataloader_val,
    )

    mock_dataloader_train = L(DataLoader)(
        dataset=mock_video_action_dataset,
        sampler=L(ResumableDistributedSampler)(
            dataset=mock_video_action_dataset,
            num_replicas=L(parallel_state.get_data_parallel_world_size)(),
            rank=L(parallel_state.get_data_parallel_rank)(),
            shuffle=True,
            seed=0,
        ),
        batch_size=MISSING,
        prefetch_factor=None,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=None,
        in_order=None,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="mock",
        node=mock_dataloader_train,
    )

    mock_dataloader_val = L(DataLoader)(
        dataset=mock_video_action_dataset,
        sampler=L(ResumableDistributedSampler)(
            dataset=mock_video_action_dataset,
            num_replicas=L(parallel_state.get_data_parallel_world_size)(),
            rank=L(parallel_state.get_data_parallel_rank)(),
            shuffle=True,
            seed=0,
        ),
        batch_size=1,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="mock",
        node=mock_dataloader_train,
    )
