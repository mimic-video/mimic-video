import bisect
import logging
import math
import multiprocessing
import typing
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import tqdm
import zarr

from cosmos_predict2.data.action.interpolate import get_closest_indices, get_previous_indices, interpolate_lowdim
from cosmos_predict2.data.action.types import S_TO_NS, LieRepr, ObsMeta, ObsType
from cosmos_predict2.data.action.utils import (
    linear_search_with_initial_guess_left,
    linear_search_with_initial_guess_right,
)


class ChunkReader:
    def __init__(
        self,
        episode_paths: list[Path],
        *,
        data_components: dict[str, ObsMeta],
        timestep_anchor: str,
        should_include_padded_tails: bool = True,
        episode_mask: np.ndarray | None = None,
        stats_id: str | None = None,
        data_dir: Path | None = None,
        logger: logging.Logger | None = None,
        verbose: bool = False,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)

        episode_paths: np.ndarray = np.array(
            [path for i, path in enumerate(episode_paths) if episode_mask is None or episode_mask[i]]
        )

        if len(episode_paths) == 0:
            msg = "ChunkReader on no episodes doesn't make sense."
            raise ValueError(msg)

        self._data_components = data_components
        self._timestep_anchor = timestep_anchor
        self._should_include_padded_tails = should_include_padded_tails

        non_persistent_action_components = {
            component: meta
            for component, meta in self._data_components.items()
            if component.startswith("action/") and meta["obs_type"] not in ObsType.PERSISTENT
        }

        max_pred_duration = max(
            (ac["horizon"] - 1) / ac["target_frequency"] for ac in non_persistent_action_components.values()
        )
        max_shift_duration = max(ac["shift_right_by"] for ac in non_persistent_action_components.values())

        if len(episode_paths) > 10 * multiprocessing.cpu_count():
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                timesteps = pool.map(
                    partial(
                        self._get_timesteps,
                        non_persistent_action_components=non_persistent_action_components,
                        max_pred_duration=max_pred_duration,
                        max_shift_duration=max_shift_duration,
                        verbose=verbose,
                    ),
                    tqdm.tqdm(
                        episode_paths,
                        desc="Iterating dataset to get episode lengths.",
                        total=len(episode_paths),
                        disable=not verbose,
                    ),
                )
        else:
            timesteps = [
                self._get_timesteps(
                    episode_path,
                    non_persistent_action_components=non_persistent_action_components,
                    max_pred_duration=max_pred_duration,
                    max_shift_duration=max_shift_duration,
                    verbose=verbose,
                )
                for episode_path in tqdm.tqdm(
                    episode_paths,
                    desc="Iterating dataset to get episode lengths.",
                    total=len(episode_paths),
                    disable=not verbose,
                )
            ]

        data = [(p, t, n) for p, (t, n) in zip(episode_paths, timesteps, strict=True) if n > 0]
        if not data:
            msg = "ChunkReader on only empty episodes doesn't make sense."
            raise ValueError(msg)

        self._episode_paths, self._timesteps, self._num_timesteps = zip(*data, strict=True)

        self._episode_paths = np.array(self._episode_paths)
        self._num_timesteps = np.array(self._num_timesteps)

        self._cumulative_num_timesteps = np.cumsum(self._num_timesteps)
        self._cumulative_num_timesteps = np.insert(self._cumulative_num_timesteps, 0, 0)

        self._restrict_keys = None

        self._stats_id = stats_id
        self._data_dir = data_dir
        self._verbose = verbose

    def _get_timesteps(
        self,
        episode_path: Path,
        *,
        non_persistent_action_components: dict,
        max_pred_duration: float,
        max_shift_duration: float,
        verbose: bool,
    ) -> tuple[np.ndarray | None, int]:
        with zarr.open(str(episode_path), "r") as root:
            try:
                min_num_timestamps = min(
                    len(root[f"{component.split('/')[1]}_timestamps"])
                    for component, meta in self._data_components.items()
                    if meta["obs_type"] not in ObsType.PERSISTENT
                )
            except KeyError as e:
                self._logger.warning(
                    f"Episode {episode_path} is lacking data.\n Tried to access {e}.\n"
                    f"Available components: {root.tree()}"
                )
                return None, 0

            if min_num_timestamps <= 2:
                if verbose:
                    self._logger.warning(f"Episode {episode_path} is lacking data.")
                return None, 0

            latest_first_timestamp = max(
                root[f"{component.split('/')[1]}_timestamps"][0] for component in non_persistent_action_components
            )
            earliest_last_timestamp = min(
                root[f"{component.split('/')[1]}_timestamps"][-1] for component in non_persistent_action_components
            )
            end_timestep = (
                earliest_last_timestamp
                if self._should_include_padded_tails
                else earliest_last_timestamp - (max_pred_duration + max_shift_duration) * S_TO_NS
            )

            if self._timestep_anchor.startswith("uniform@"):
                step = S_TO_NS / float(self._timestep_anchor.removeprefix("uniform@"))
                timesteps = np.arange(latest_first_timestamp, end_timestep, step)
                return timesteps, len(timesteps)

            timesteps = root[f"{self._timestep_anchor}_timestamps"][...]

            start_idx = linear_search_with_initial_guess_right(timesteps, latest_first_timestamp, 0)
            timesteps = timesteps[start_idx:]

            end_idx = linear_search_with_initial_guess_left(timesteps, end_timestep, len(timesteps) - 1)
            timesteps = timesteps[: end_idx + 1]

            if len(timesteps) == 0:
                if verbose:
                    self._logger.warning(
                        f"Episode {episode_path} is lacking the data to contain any"
                        " valid chunks in the current configuration."
                    )
                return None, 0

            return timesteps, len(timesteps)

    def __len__(self) -> int:
        return self._cumulative_num_timesteps[-1]

    def restrict_keys(self, keys: set[str] | None) -> None:
        self._restrict_keys = keys

    def _read_chunk(
        self, root, key: str, meta: ObsMeta, step_timestamp: int, progress: float, *, is_action: bool
    ) -> np.ndarray | None:
        if self._restrict_keys is not None and key not in self._restrict_keys:
            return None

        shift_timesteps = meta["shift_right_by"] * S_TO_NS
        # int() avoids uint64 underflow when the backward (obs) window goes before t=0:
        # uint64(0) - pred_duration would wrap to ~2**64 instead of a negative timestamp.
        this_step_timestamp = int(step_timestamp) + shift_timesteps

        actual_timestamps = root[f"{key}_timestamps"]
        n_timestamps = len(actual_timestamps)

        pred_duration = (meta["horizon"] - 1) / meta["target_frequency"] if meta["horizon"] > 1 else 0
        pred_duration_ns: int = math.ceil(pred_duration * S_TO_NS)

        if is_action:
            chunk_start_timestamp = this_step_timestamp
            chunk_end_timestamp = chunk_start_timestamp + pred_duration_ns
        else:
            chunk_start_timestamp = this_step_timestamp - pred_duration_ns
            chunk_end_timestamp = this_step_timestamp

        start_idx = min(
            max(0, n_timestamps - 2),
            linear_search_with_initial_guess_left(
                actual_timestamps, chunk_start_timestamp, round(progress * n_timestamps)
            ),
        )
        end_idx = max(
            1,
            linear_search_with_initial_guess_right(
                actual_timestamps,
                chunk_end_timestamp,
                round(
                    (progress + pred_duration_ns / max(1, actual_timestamps[-1] - actual_timestamps[0])) * n_timestamps
                ),
            ),
        )

        actual_timestamps = actual_timestamps[start_idx : end_idx + 1]
        if len(actual_timestamps) == 0:
            # Degenerate window: the requested chunk lies entirely outside this component's
            # available timestamps. Pad by repeating the nearest frame `horizon` times.
            self._logger.warning(
                f"empty chunk window key={key} horizon={meta['horizon']} is_action={is_action} "
                f"start_idx={start_idx} end_idx={end_idx} n={n_timestamps} "
                f"cs={float(chunk_start_timestamp):.0f} ce={float(chunk_end_timestamp):.0f}"
            )
            idx = int(min(max(start_idx, 0), n_timestamps - 1))
            nearest = root[key][idx]
            return np.repeat(nearest[None], meta["horizon"], axis=0)
        requested_timestamps = np.linspace(
            chunk_start_timestamp, chunk_end_timestamp, meta["horizon"], dtype=np.float64
        )

        # horizon is 1 and there is an exact match for the right timestamp
        if start_idx == end_idx:
            values: np.ndarray = np.expand_dims(root[key][start_idx], 0)

        elif (
            meta["obs_type"] in ObsType.INTERPOLABLE
            and typing.cast(LieRepr, meta["repr"]) in LieRepr.INDEPENDENTLY_INTERPOLABLE
        ):
            values = root[key][start_idx : end_idx + 1]
            if values.ndim == 1:
                values = values[:, None]

            values = interpolate_lowdim(values, actual_timestamps, requested_timestamps, meta, is_action=is_action)

        else:
            if meta["obs_type"] in ObsType.PERSISTENT:
                indices = get_previous_indices(actual_timestamps, requested_timestamps)
            else:
                indices = get_closest_indices(actual_timestamps, requested_timestamps)
            start_offset = indices.min()
            values = root[key][start_idx + start_offset : start_idx + indices.max() + 1][indices - start_offset]

        if values.ndim == 1 and values.dtype != np.object_:
            values = values[:, None]

        if values.dtype == np.object_ and meta["obs_type"] in ObsType.COLOR_VISUAL:
            values = np.array(
                [cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_UNCHANGED) for value in values]
            )

        return values

    def read_chunk(self, idx: int) -> dict:
        """Load a chunk of data from storage and convert it to a unified format.

        This function is only responsible for returning any data that might end up in the batch. This includes
        interpolation to select the correct data at the requested timestamps.
        Any postprocessing of items or subsampling of keys, etc. is done in the dataset.
        """
        episode_idx = bisect.bisect_right(self._cumulative_num_timesteps, idx) - 1
        step_idx = idx - self._cumulative_num_timesteps[episode_idx]
        step_timestamp = self._timesteps[episode_idx][step_idx]

        progress = step_idx / self._num_timesteps[episode_idx]

        with zarr.open(self._episode_paths[episode_idx], "r") as root:
            return {
                key: vals
                for key, meta in self._data_components.items()
                if (
                    vals := self._read_chunk(
                        root, key.split("/")[1], meta, step_timestamp, progress, is_action=key.startswith("action/")
                    )
                )
                is not None
            }
