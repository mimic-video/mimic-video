import copy

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from omegaconf import MISSING
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as Dataset_

from cosmos_predict2.data.dataset_video import Dataset, MultiDataset
from cosmos_predict2.data.resumable_sampler import ResumableDistributedSampler
from imaginaire.lazy_config import LazyCall as L


def get_sampler(dataset) -> ResumableDistributedSampler:
    return ResumableDistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

train_datasets: dict[str, Dataset_] = {
    "fr3": L(Dataset)(
        # bimanual_fr3 (LeRobot) -> top camera, re-encoded to per-episode H.264 by
        # data_preprocessing/lerobot/convert_video.py. Single camera => is_multi_img=False.
        dataset_dir="../datasets/bimanual_fr3_mimic/video",  # relative to model/ (the training cwd)
        num_frames=61,
        video_size=[480, 640],
        data_fps=30.0,  # native fps of the converted mp4s
        is_val=False,
        obs_history=5,
    ),
    "bridge": L(Dataset)(
        dataset_dir=...,
        num_frames=61,
        video_size=[480, 640],
        is_val=False,
        obs_history=5,
        is_multi_img=True,
    ),
    "libero_spatial_agentview": L(Dataset)(
        dataset_dir=...,
        num_frames=61,
        video_size=[480, 640],
        data_fps=20.0,
        is_val=False,
        include_only_with_substrings=["libero_spatial", "agentview"],
        obs_history=5,
    ),
    "libero_goal_agentview": L(Dataset)(
        dataset_dir=...,
        num_frames=61,
        video_size=[480, 640],
        data_fps=20.0,
        is_val=False,
        include_only_with_substrings=["libero_goal", "agentview"],
        obs_history=5,
    ),
    "libero_object_agentview": L(Dataset)(
        dataset_dir=...,
        num_frames=61,
        video_size=[480, 640],
        data_fps=20.0,
        is_val=False,
        include_only_with_substrings=["libero_object", "agentview"],
        obs_history=5,
    ),
}

val_datasets: dict[str, Dataset_] = {}
for k, v in train_datasets.items():
    ds = copy.deepcopy(v)
    ds.is_val = True  # ty:ignore[unresolved-attribute]
    val_datasets[k] = ds


dataset_mixes = {}

for name, mix in dataset_mixes.items():
    train_datasets[name] = L(MultiDataset)(**{k: train_datasets[k] for k in mix.split(",")})
    val_datasets[name] = L(MultiDataset)(**{k: val_datasets[k] for k in mix.split(",")})

dataloader_video_train = L(DataLoader)(
    dataset="${video_dataset_train}",
    sampler=L(get_sampler)(dataset="${video_dataset_train}"),
    batch_size=MISSING,
    prefetch_factor=8,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    in_order=False,
)
dataloader_video_val = L(DataLoader)(
    dataset="${video_dataset_val}",
    sampler=L(get_sampler)(dataset="${video_dataset_val}"),
    batch_size=1,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    in_order=False,
)


def register_training_and_val_video_data() -> None:
    cs = ConfigStore.instance()

    for name, ds in train_datasets.items():
        cs.store(
            group="video_dataset_train",
            package="video_dataset_train",
            name=name,
            node=ds,
        )

    for name, ds in val_datasets.items():
        cs.store(
            group="video_dataset_val",
            package="video_dataset_val",
            name=name,
            node=ds,
        )

    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="vanilla",
        node=dataloader_video_train,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="vanilla",
        node=dataloader_video_val,
    )
