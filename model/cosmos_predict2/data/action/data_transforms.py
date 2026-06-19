import abc
import re
import typing
from collections.abc import Iterable, Iterator
from functools import partial
from operator import itemgetter, methodcaller

import einops
import numpy as np
import PIL.Image
from torchvision import transforms

from cosmos_predict2.data.action import convert_pose_repr
from cosmos_predict2.data.action.types import SCIPY_ROTATION_CONVERSIONS, LieRepr, ObsMeta, ObsType


class DataTransform(abc.ABC):
    def __init__(
        self, targets: list[str], metas: dict[str, ObsMeta], additional_data: dict[str, str] | None = None
    ) -> None:
        self._targets = targets
        self._metas = metas
        self._additional_data = additional_data or {}

    @abc.abstractmethod
    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        raise NotImplementedError

    @property
    def targets(self) -> list[str]:
        return self._targets

    @property
    def additional_data(self) -> dict[str, str]:
        return self._additional_data

    @property
    def new_components(self) -> dict[str, dict]:
        return {}

    @property
    @abc.abstractmethod
    def remove_original(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ignore_for_normalization(self) -> bool:
        raise NotImplementedError


class DeconstructPoseMat(DataTransform):
    def __init__(self, out_key_prefix: str, **kwargs):
        super().__init__(**kwargs)

        self._out_key_prefix = out_key_prefix

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        """Assumes input is a single pose matrix."""
        mat = targets[0][1]

        rot = mat[:, :3, :3]
        yield f"{self._out_key_prefix}_rot_lowdim", rot

        trans = mat[:, :3, 3]
        yield f"{self._out_key_prefix}_pos_lowdim", trans

    @property
    def new_components(self) -> dict[str, dict]:
        pose_mat_meta = next(
            (item for key, item in self._metas.items() if re.search(self._targets[0], key) is not None)
        )
        return {
            f"{self._out_key_prefix}_rot_lowdim": {"obs_type": ObsType.ROTATION_MATRIX, "repr": pose_mat_meta["repr"]},
            f"{self._out_key_prefix}_pos_lowdim": {"obs_type": ObsType.CARTESIAN_POS, "repr": pose_mat_meta["repr"]},
        }

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return False


class Concat(DataTransform):
    def __init__(self, out_key: str, **kwargs):
        super().__init__(**kwargs)

        self._out_key = out_key

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        """Assumes all actions are flattened and have the same horizon."""
        _, targets = zip(*targets, strict=False)
        yield self._out_key, np.concatenate(targets, axis=-1)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True


class Flatten(DataTransform):
    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        for key, value in targets:
            T, *_ = value.shape
            yield key, value.reshape(T, -1).astype(np.float32)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return False


class ConvertLowdimRepr(DataTransform):
    def __init__(self, relative_base_value_idx: int | None = None, *, is_action: bool | None = None, **kwargs):
        super().__init__(**kwargs)

        self._relative_base_value_idx = relative_base_value_idx
        self._is_action = is_action

    def __call__(
        self, targets: list[tuple[str, np.ndarray]], relative_base_value: np.ndarray | None = None
    ) -> Iterator[tuple[str, np.ndarray]]:
        kwargs = (
            {"relative_base_value": relative_base_value[self._relative_base_value_idx]}
            if relative_base_value is not None
            else {"is_action": self._is_action}
        )
        for key, value in targets:
            res = convert_pose_repr.convert_to_repr(
                value,
                self._metas[key]["obs_type"],
                typing.cast(LieRepr, self._metas[key]["repr"]),
                self._metas[key]["target_repr"],
                **kwargs,  # ty:ignore[invalid-argument-type]
            )
            yield key, res

    @property
    def new_components(self) -> dict[str, dict]:
        return {
            key: {**meta, "repr": meta["target_repr"]}
            for key, meta in self._metas.items()
            if any(re.search(pattern, key) is not None for pattern in self._targets)
        }

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return False


class RotationMatrixTo6D(DataTransform):
    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        for key, rot_mat in targets:
            yield key, rot_mat[:, :2].reshape(-1, 6)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return False


class ToRotationMatrix(DataTransform):
    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        for key, rot in targets:
            rot = SCIPY_ROTATION_CONVERSIONS[self._metas[key]["obs_type"]][0](rot)
            yield key, rot.as_matrix()

    @property
    def new_components(self) -> dict[str, dict]:
        return {
            key: {**meta, "obs_type": ObsType.ROTATION_MATRIX}
            for key, meta in self._metas.items()
            if any(re.search(pattern, key) is not None for pattern in self._targets)
        }

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return False


class CosmosProcessImage(DataTransform):
    def __init__(
        self,
        resize_sizes: dict[str, tuple[int, int]] | tuple[int, int],
        obs_types: dict[str, ObsType] | None = None,
        defer_resize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._resize_sizes = resize_sizes
        self._obs_types = obs_types or {}
        self._defer_resize = defer_resize

    def _get_resize_sizes(self, key: str) -> tuple[int, int]:
        return self._resize_sizes[key] if isinstance(self._resize_sizes, dict) else self._resize_sizes

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        for key, imgs in targets:
            if self._defer_resize:
                # Ship native-resolution uint8 across the DataLoader/shared-memory boundary (~21x
                # smaller than upscaled float32). The resize to model resolution and the [-1, 1]
                # normalization are done on the GPU in the pipeline (Video2WorldPipeline), which is
                # both faster and keeps /dev/shm tiny.
                yield key, einops.rearrange(imgs, "t h w c -> c t h w")  # (C, T, H, W) uint8, native res
                continue
            imgs = np.stack(
                [transforms.Resize(self._get_resize_sizes(key))(PIL.Image.fromarray(img, "RGB")) for img in imgs],
                axis=0,
            )
            imgs = einops.rearrange(imgs, "t h w c -> c t h w")
            yield key, 2.0 * (imgs.astype(np.float32) / 255.0 - 0.5)

    @property
    def new_components(self) -> dict[str, dict]:
        return {
            key: {**meta, "obs_type": ObsType.RGB}
            for key, meta in self._metas.items()
            if any(re.search(pattern, key) is not None for pattern in self._targets)
        }

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True


def make_data_transforms(
    data_transform_specs: list[dict], data_components: dict[str, ObsMeta]
) -> Iterator[DataTransform]:
    new_components = {}
    for data_transform_spec in data_transform_specs:
        data_transform_cls = globals()[data_transform_spec.pop("name")]
        data_transform = data_transform_cls(**data_transform_spec, metas=data_components | new_components)
        new_components |= data_transform.new_components
        yield data_transform


def apply_data_transforms(data: dict, ops: Iterable[DataTransform], *, should_ignore_transforms_for_norm: bool) -> dict:
    for op in ops:
        if should_ignore_transforms_for_norm and op.ignore_for_normalization:
            continue

        get_fn = partial(methodcaller, "pop") if op.remove_original else itemgetter
        targets = [
            (key, get_fn(key)(data))
            for pattern in op.targets
            for key in sorted(data)
            if re.search(pattern, key) is not None
        ]

        if targets:
            additional_data = {key: data[val] for key, val in op.additional_data.items()}
            data.update(op(targets, **additional_data))

    return data
