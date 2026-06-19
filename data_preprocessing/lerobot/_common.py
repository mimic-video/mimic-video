"""Shared helpers for converting a LeRobot v3.0 dataset into the mimic-video formats.

This is written for the `bimanual_fr3` dataset but is deliberately generic over:
  - which camera key becomes the single `workspace_rgb` view (``--camera``), and
  - which arm slice of the 20-dim state/action vector is used (``--arm``).

For the feasibility PoC we keep ONLY the left arm (dims [0:10] = pos(3) + rot6d(6) + gripper(1))
so the released 10-dim Bridge action decoder can be reused with no model-code changes.

Nothing here imports the mimic/`imaginaire` package, so these converters run in a light
environment (numpy + pyarrow + imageio). The T5 steps (get_t5_embeddings.py / precompute_t5.py)
are the only parts that need the full training env + GPU.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

S_TO_NS = 1_000_000_000

# Layout of one arm inside the 20-dim LeRobot state/action vector.
ARM_SLICES = {"left": slice(0, 10), "right": slice(10, 20)}
# Within a 10-dim arm block: pos xyz, rot6d (first two rows of R, pytorch3d convention), gripper.
POS = slice(0, 3)
ROT6D = slice(3, 9)
GRIPPER = 9


@dataclass
class Episode:
    index: int
    frames: np.ndarray  # global frame indices into the (concatenated) camera mp4, in order
    timestamps_ns: np.ndarray  # (T,) uint64, per-episode clock starting at 0
    pos: np.ndarray  # (T, 3) float32
    rot6d: np.ndarray  # (T, 6) float32  (from observation.state)
    gripper_state: np.ndarray  # (T,) float32  (proprio, from observation.state)
    gripper_action: np.ndarray  # (T,) float32  (command, from action)
    task: str


def read_info(dataset_dir: pathlib.Path) -> dict:
    return json.loads((dataset_dir / "meta" / "info.json").read_text())


def _read_table(path: pathlib.Path):
    return pq.read_table(str(path))


def task_map(dataset_dir: pathlib.Path) -> dict[int, str]:
    """Map task_index -> instruction string, tolerant of LeRobot's parquet quirks.

    In bimanual_fr3 the task text lives in the column ``__index_level_0__`` (the string was the
    DataFrame index when written); normal LeRobot exports use a ``task`` column.
    """
    t = _read_table(dataset_dir / "meta" / "tasks.parquet")
    cols = t.column_names
    text_col = "task" if "task" in cols else ("__index_level_0__" if "__index_level_0__" in cols else cols[-1])
    idx = t.column("task_index").to_pylist() if "task_index" in cols else list(range(t.num_rows))
    text = t.column(text_col).to_pylist()
    return {int(i): str(s) for i, s in zip(idx, text)}


def load_episodes(dataset_dir: pathlib.Path, *, arm: str = "left") -> list[Episode]:
    """Parse the main data parquet into per-episode arrays (sorted by global frame order)."""
    arm_slice = ARM_SLICES[arm]
    t = _read_table(dataset_dir / "data" / "chunk-000" / "file-000.parquet")

    index = np.asarray(t.column("index").to_pylist(), dtype=np.int64)
    order = np.argsort(index, kind="stable")  # global frame order == mp4 frame order

    episode_index = np.asarray(t.column("episode_index").to_pylist(), dtype=np.int64)[order]
    frame_index = np.asarray(t.column("frame_index").to_pylist(), dtype=np.int64)[order]
    timestamp = np.asarray(t.column("timestamp").to_pylist(), dtype=np.float64).reshape(-1)[order]
    state = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)[order]
    action = np.asarray(t.column("action").to_pylist(), dtype=np.float32)[order]
    task_index = np.asarray(t.column("task_index").to_pylist(), dtype=np.int64)[order]
    global_frame = np.arange(len(order), dtype=np.int64)  # position in the concatenated mp4

    tasks = task_map(dataset_dir)
    arm_state = state[:, arm_slice]
    arm_action = action[:, arm_slice]

    episodes: list[Episode] = []
    for ep in sorted(set(episode_index.tolist())):
        m = episode_index == ep
        # keep this episode's frames in their own frame_index order
        local_order = np.argsort(frame_index[m], kind="stable")
        sel = np.flatnonzero(m)[local_order]
        ts = timestamp[sel]
        ts = ts - ts[0]  # episodes already reset to ~0, but be safe
        episodes.append(
            Episode(
                index=int(ep),
                frames=global_frame[sel],
                timestamps_ns=(ts * S_TO_NS).round().astype(np.uint64),
                pos=arm_state[sel][:, POS].astype(np.float32),
                rot6d=arm_state[sel][:, ROT6D].astype(np.float32),
                gripper_state=arm_state[sel][:, GRIPPER].astype(np.float32),
                gripper_action=arm_action[sel][:, GRIPPER].astype(np.float32),
                task=tasks.get(int(task_index[sel][0]), ""),
            )
        )
    return episodes


def camera_mp4s(dataset_dir: pathlib.Path, camera: str) -> list[pathlib.Path]:
    cam_dir = dataset_dir / "videos" / camera
    files = sorted(p for p in cam_dir.rglob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No mp4 found for camera {camera!r} under {cam_dir}")
    return files


def decode_frames_by_episode(
    dataset_dir: pathlib.Path, camera: str, episodes: list[Episode]
) -> dict[int, np.ndarray]:
    """Decode the (possibly multi-file) camera video once and bucket frames per episode.

    Uses imageio's bundled ffmpeg, which decodes AV1. Returns {episode_index: (T,H,W,3) uint8}.
    """
    import imageio.v2 as iio

    # global frame index -> episode index
    owner: dict[int, int] = {}
    for ep in episodes:
        for f in ep.frames.tolist():
            owner[f] = ep.index
    buckets: dict[int, list[np.ndarray]] = {ep.index: [] for ep in episodes}

    g = 0
    for mp4 in camera_mp4s(dataset_dir, camera):
        reader = iio.get_reader(str(mp4))
        try:
            for frame in reader:
                ep = owner.get(g)
                if ep is not None:
                    buckets[ep].append(np.asarray(frame, dtype=np.uint8))
                g += 1
        finally:
            reader.close()

    out: dict[int, np.ndarray] = {}
    for ep in episodes:
        frames = buckets[ep.index]
        if len(frames) != len(ep.frames):
            raise RuntimeError(
                f"Episode {ep.index}: decoded {len(frames)} frames but expected {len(ep.frames)}. "
                "Video/parquet frame counts disagree."
            )
        out[ep.index] = np.stack(frames, axis=0)
    return out


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(T,6) -> (T,3,3). pytorch3d/mimic convention: the 6D are the first two ROWS of R.

    Mirrors mimic's RotationMatrixTo6D (``rot_mat[:, :2].reshape(-1, 6)``) so the value round-trips:
    storing R here and letting the Bridge transforms re-extract 6D yields the same representation.
    NOTE: this assumes the dataset's rot6d also follows the row convention. It is self-consistent for
    training regardless; only real-robot deployment would require confirming the convention.
    """
    a1 = d6[:, 0:3]
    a2 = d6[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)  # rows b1,b2,b3


def pose_matrices(pos: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
    """(T,3),(T,6) -> (T,4,4) float32 homogeneous pose."""
    t = pos.shape[0]
    out = np.zeros((t, 4, 4), dtype=np.float32)
    out[:, :3, :3] = rotation_6d_to_matrix(rot6d)
    out[:, :3, 3] = pos
    out[:, 3, 3] = 1.0
    return out
