"""LeRobot v3.0 -> mimic-video ACTION (zarr) format.

Produces one `.zarr` group per episode whose structure is IDENTICAL to what
`data_preprocessing/action/process_bridge.py` emits, so the entire Bridge dataloading stack
(dataset/bridge_fr3.yaml + policy_io/bridge.yaml + transform/bridge_to_bridge.yaml + the 10-dim
Bridge action net) is reused unchanged.

Per-episode zarr arrays (each paired with a `<name>_timestamps` uint64-ns array):
    workspace_rgb            (T, H, W, 3) uint8     - chosen camera (top), native res; resized at load
    eef_pose_lowdim          (T, 4, 4)    float32   - left-arm pose, reconstructed from pos + rot6d
    gripper_state_lowdim     (T,)         float32   - proprio gripper (observation.state)
    gripper_action_lowdim    (T,)         float32   - command gripper (action)
    language_instruction     (1,)         bytes
    language_embedding       (1,512,1024) float16   - added later by precompute_t5.py

Like Bridge, there is NO separate action pose array: the action end-effector pose is read as
future timesteps of `eef_pose_lowdim` (then made relative to the current pose by the transform).
Since action ~= state in this dataset (mean abs diff ~0.013) this is equivalent to using commands.

Example:
    uv run --no-project --with numpy --with pyarrow --with imageio --with imageio-ffmpeg \
        --with zarr==2.18.2 --with numcodecs \
        python convert_action.py \
        --dataset-dir ../../datasets/bimanual_fr3 \
        --out-dir ../../datasets/bimanual_fr3_mimic/action \
        --camera observation.images.top --arm left
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import zarr
from numcodecs import Blosc

from _common import decode_frames_by_episode, load_episodes, pose_matrices, read_info

COMPRESSOR = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)


def _ds(root: zarr.Group, name: str, data: np.ndarray, chunks: tuple) -> None:
    root.create_dataset(name, shape=data.shape, dtype=data.dtype, chunks=chunks, compressor=COMPRESSOR)
    root[name][...] = data


def _ts(root: zarr.Group, name: str, ts: np.ndarray) -> None:
    root.create_dataset(name, shape=(len(ts),), dtype="uint64", chunks=(len(ts),), compressor=COMPRESSOR)
    root[name][...] = ts.astype(np.uint64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path, help="Directory to write per-episode .zarr groups.")
    ap.add_argument("--camera", default="observation.images.top")
    ap.add_argument("--arm", default="left", choices=["left", "right"])
    args = ap.parse_args()

    read_info(args.dataset_dir)  # validate it's a LeRobot dataset
    episodes = load_episodes(args.dataset_dir, arm=args.arm)
    print(f"{len(episodes)} episodes; arm={args.arm}; camera={args.camera!r}")
    frames_by_ep = decode_frames_by_episode(args.dataset_dir, args.camera, episodes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        images = frames_by_ep[ep.index]  # (T,H,W,3) uint8
        ts = ep.timestamps_ns
        t = len(ts)
        if not (0 < images.shape[0] == ep.pos.shape[0] == t):
            raise RuntimeError(f"Episode {ep.index}: length mismatch images={images.shape[0]} pose={ep.pos.shape[0]} ts={t}")

        pose = pose_matrices(ep.pos, ep.rot6d)  # (T,4,4)
        t_img = min(65, t)
        t_ld = min(1024, t)

        out_path = (args.out_dir / f"ep{ep.index:03d}.zarr").resolve()
        root: zarr.Group
        with zarr.open(str(out_path), "w") as root:
            _ds(root, "workspace_rgb", images, (t_img, *images.shape[1:]))
            _ts(root, "workspace_rgb_timestamps", ts)

            _ds(root, "eef_pose_lowdim", pose, (t_ld, 4, 4))
            _ts(root, "eef_pose_lowdim_timestamps", ts)

            _ds(root, "gripper_state_lowdim", ep.gripper_state.astype(np.float32), (t_ld,))
            _ts(root, "gripper_state_lowdim_timestamps", ts)

            _ds(root, "gripper_action_lowdim", ep.gripper_action.astype(np.float32), (t_ld,))
            _ts(root, "gripper_action_lowdim_timestamps", ts)

            root.create_dataset("language_instruction", shape=(1,), dtype=bytes, chunks=(1,), compressor=COMPRESSOR)
            root["language_instruction"][...] = np.array([ep.task.encode()])
            _ts(root, "language_instruction_timestamps", np.array([0], dtype=np.uint64))

        print(f"  ep{ep.index:03d}: T={t} -> {out_path.name}")

    print(f"Done. Next: python ../action/precompute_t5.py --dataset-path {args.out_dir}")


if __name__ == "__main__":
    main()
