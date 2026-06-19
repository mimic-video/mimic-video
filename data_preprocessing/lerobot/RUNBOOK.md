# Runbook: finetuning Cosmos-Predict2 + training an action decoder on `bimanual_fr3`

End-to-end recipe for the LeRobot dataset at `datasets/bimanual_fr3` (bimanual Franka FR3, 7 episodes,
1 task, 30 fps, 3 cameras @ 180×320 AV1). This is a **proof-of-concept**: 7 episodes is enough to
validate that the whole stack runs and overfits, **not** to get a generalizing policy.

## Design decisions baked in
- **Left arm only** → 10-dim action/state (`pos(3) + rot6d(6) + gripper(1)`) → reuses the released
  **Bridge** action decoder net unchanged (no model-code edits).
- **`top` camera** → the single `workspace_rgb` view (wrist cameras are dropped).
- Converters emit data **structurally identical to `process_bridge.py`**, so the whole Bridge
  dataloading stack (transform + policy_io + net) is reused; only **2 new YAMLs** were added.

## What was created
| File | Purpose |
|---|---|
| `data_preprocessing/lerobot/_common.py` | shared parsing / AV1 decode / 6D→matrix |
| `data_preprocessing/lerobot/convert_video.py` | LeRobot → `video/*.mp4` + `metas/*.txt` |
| `data_preprocessing/lerobot/convert_action.py` | LeRobot → per-episode `*.zarr` (Bridge schema) |
| `configs/experiment/video2world.py` (line 28) | **bug fix**: `/model` override → `v2w_pretrained_cosmos` |
| `configs/defaults/data_video.py` | registers video dataset **`fr3`** |
| `configs/dataloading/dataset/bridge_fr3.yaml` + `configs/dataloading/bridge_fr3.yaml` | action data_config **`bridge_fr3`** |

---

## Step 1 — Convert data (CPU; runs anywhere)
Both converters are standalone (numpy + pyarrow + imageio + zarr). Run via an ephemeral `uv` env, or
inside the repo venv (which already has zarr/numcodecs — just add `pyarrow imageio imageio-ffmpeg`).
Verified working on all 7 episodes.

```bash
cd data_preprocessing/lerobot

uv run --no-project --with numpy --with pyarrow --with imageio --with imageio-ffmpeg \
  python convert_video.py \
  --dataset-dir ../../datasets/bimanual_fr3 \
  --out-dir ../../datasets/bimanual_fr3_mimic/video \
  --camera observation.images.top --arm left

uv run --no-project --with numpy --with pyarrow --with imageio --with imageio-ffmpeg \
  --with 'zarr<3' --with numcodecs \
  python convert_action.py \
  --dataset-dir ../../datasets/bimanual_fr3 \
  --out-dir ../../datasets/bimanual_fr3_mimic/action \
  --camera observation.images.top --arm left
```
Produces `datasets/bimanual_fr3_mimic/{video/{video,metas}, action/ep0??.zarr}`.

## Step 2 — Precompute T5 embeddings (needs GPU + repo venv + `checkpoints/text_encoder/t5-11b`)
```bash
cd ../video   && python get_t5_embeddings.py --dataset_path ../../datasets/bimanual_fr3_mimic/video
cd ../action  && python precompute_t5.py     --dataset-path ../../datasets/bimanual_fr3_mimic/action
```
The first writes `video/t5_xxl/ep0??.pickle`; the second adds `language_embedding` into each zarr.

## Step 3 — Finetune the video model (LoRA, PoC settings)
```bash
cd ../../model
torchrun --nproc_per_node=1 -m scripts.train --config=cosmos_predict2/configs/config.py -- \
  experiment=v2w_fr3_lora_rank256_lr1.778e-04_bsz32 \
  dataloader_train.batch_size=1 \
  trainer.max_iter=3000 \
  checkpoint.save_iter=500
```
Output: `${IMAGINAIRE_OUTPUT_ROOT:-checkpoints}/posttraining/video2world/v2w_fr3_lora_rank256_lr1.778e-04_bsz32/`.
LoRA is auto-fused to a full checkpoint on save. Watch for overfitting — 7 episodes is tiny.

## Step 4 — Train the action decoder
**Fastest path first:** train on the **stock** backbone to validate the action pipeline before
involving your finetuned video model.
```bash
cd model   # (already here)
torchrun --nproc_per_node=1 -m scripts.train --config=cosmos_predict2/configs/config.py -- \
  experiment=w2a_bridge_fr3_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128 \
  dataloader_train.batch_size=8 \
  trainer.max_iter=20000
```
(`bsz128` is just the registered name; the CLI override sets the real per-GPU batch. `save_iter=1000`
and validation-on are already set for bridge-style configs.)
Output: `${IMAGINAIRE_OUTPUT_ROOT:-checkpoints}/vam/bridge/<experiment>/`.

**To condition on your finetuned video model instead of the stock one:** copy the fused checkpoint to
`checkpoints/video_backbone/<name>.pt`, add `"<name>"` to `VIDEO_MODEL_CKPT_NAMES`
(`configs/defaults/world2action_model.py`), then use `experiment=w2a_bridge_fr3_<name>_lr1.000e-04_layer20_bsz128`
(or override `model.config.video_dit_path=/abs/path.pt`).

---

## Caveats & gotchas
- **Scale:** 7 episodes / 1 task → expect memorization, not generalization.
- **Aspect:** frames are 180×320 (16:9) stored natively and stretched to 480×640 (4:3) at load by both
  pipelines (consistently). If the vertical squish hurts, add letterboxing to **both** converters.
- **Resolution:** the 2B model is fixed at 480×640, so 180×320 is upscaled ~2.7× — inherently soft.
- **rot6d convention:** reconstruction assumes the pytorch3d "first two rows" convention (matches
  mimic). Self-consistent for training; confirm against your robot before any real-robot deployment.
- **action ≈ state** (mean abs diff ≈ 0.013), so, Bridge-style, the action EEF pose is read as future
  timesteps of `eef_pose_lowdim`; only the gripper has a distinct command channel.
- **Re-running `convert_action`:** delete `…/action/paths.pkl` and `…/action/.statistics_cache/` if you
  change the episode set — they are cached by the dataloader.
- **Blackwell/GPU env:** install the CUDA extra matching your port (cu128/cu130), not the README's cu126.
- **Wrist cameras** (`wrist_left`, `wrist_right`) are unused; true multi-view would need model changes.

## Verification done on this machine
- Both converters ran on all 7 episodes (AV1 decoded via imageio).
- zarr matches the Bridge schema: `workspace_rgb (T,180,320,3) uint8`, `eef_pose_lowdim (T,4,4) f32`,
  `gripper_state/action_lowdim (T,) f32`, `language_instruction (1,) bytes`, all with `_timestamps` (ns).
- Reconstructed poses are valid rotations: `max|RRᵀ−I| = 0.0`, `det = 1.0`.
- mp4s are H.264, 320×180, 30 fps. Timestamps are a clean 30 Hz.
- **Not run here** (need GPU + repo venv): the T5 steps and the two trainings.
