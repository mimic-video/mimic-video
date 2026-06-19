"""LeRobot v3.0 -> mimic-video VIDEO finetuning format.

Splits the concatenated per-camera video into one H.264 mp4 per episode (re-encoding from AV1),
and writes a matching metas/<ep>.txt instruction. After this, run:

    cd ../../model && python -m scripts... NO -> run the repo's T5 step:
    cd <repo>/data_preprocessing/video && python get_t5_embeddings.py --dataset_path <out_dir>

Output layout (consumed by cosmos_predict2.data.dataset_video.Dataset):
    <out_dir>/video/ep000.mp4 ...
    <out_dir>/metas/ep000.txt ...
    <out_dir>/t5_xxl/ep000.pickle   (created later by get_t5_embeddings.py)

Example:
    uv run --no-project --with numpy --with pyarrow --with imageio --with imageio-ffmpeg \
        python convert_video.py \
        --dataset-dir ../../datasets/bimanual_fr3 \
        --out-dir ../../datasets/bimanual_fr3_mimic/video \
        --camera observation.images.top --arm left
"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v2 as iio

from _common import decode_frames_by_episode, load_episodes, read_info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=pathlib.Path, help="LeRobot dataset root (has meta/, data/, videos/).")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path, help="Where to write video/ and metas/.")
    ap.add_argument("--camera", default="observation.images.top", help="Camera key to use as the single workspace view.")
    ap.add_argument("--arm", default="left", choices=["left", "right"], help="Unused for video; kept for parity/logging.")
    ap.add_argument("--fps", type=int, default=0, help="Output mp4 fps. 0 => use the dataset's native fps from info.json.")
    ap.add_argument("--quality", type=int, default=8, help="imageio libx264 quality (matches the bridge converter).")
    args = ap.parse_args()

    info = read_info(args.dataset_dir)
    fps = args.fps or int(round(info.get("fps", 30)))

    episodes = load_episodes(args.dataset_dir, arm=args.arm)
    print(f"{len(episodes)} episodes; re-encoding camera {args.camera!r} to H.264 @ {fps}fps")
    frames_by_ep = decode_frames_by_episode(args.dataset_dir, args.camera, episodes)

    video_dir = args.out_dir / "video"
    metas_dir = args.out_dir / "metas"
    video_dir.mkdir(parents=True, exist_ok=True)
    metas_dir.mkdir(parents=True, exist_ok=True)

    for ep in episodes:
        stem = f"ep{ep.index:03d}"
        frames = frames_by_ep[ep.index]  # (T,H,W,3) native resolution; Dataset resizes to 480x640 at load
        out_mp4 = video_dir / f"{stem}.mp4"
        # macro_block_size=1 keeps native HxW (e.g. 180x320) instead of silently padding to a
        # multiple of 16, so the video pipeline and the action zarr see identical frames.
        with iio.get_writer(str(out_mp4), codec="libx264", fps=fps, quality=args.quality, macro_block_size=1) as w:
            for frame in frames:
                w.append_data(frame)
        (metas_dir / f"{stem}.txt").write_text(ep.task, encoding="utf-8")
        print(f"  {stem}: {len(frames)} frames -> {out_mp4.name}  task={ep.task!r}")

    print(f"Done. Next: python get_t5_embeddings.py --dataset_path {args.out_dir}")


if __name__ == "__main__":
    main()
