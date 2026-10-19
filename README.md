# mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs

<p align="center">
    <a href="https://arxiv.org/abs/2512.15692"><strong>Paper</strong></a>
    &nbsp;·&nbsp;
    <a href="https://mimic-video.github.io"><strong>Website</strong></a>
    &nbsp;·&nbsp;
    <a href="https://huggingface.co/jonpai/mimic-video"><strong>Checkpoints</strong></a>
</p>

## Introduction

<p align="center">
    <img src="assets/figure_one.png" alt="Figure One">
</p>

mimic-video extracts generalist language-conditioned robot policies (Video-Action Models / VAMs) from pretrained video models by conditioning small action decoders on the video backbones' latent representations. By drawing on the video model's knowledge of real-world dynamics and behaviors, performant action decoders can be learned efficiently and without updating the video model. Employing decoupled flow times for video and for actions, efficient inference can be performed with a single video model forward pass per action chunk.

We instantiate our approach with the lightweight 2B Cosmos-Predict2 video model and release trained checkpoints for Bridge and LIBERO.

## Repository Overview

We provide our data ([DATA.md](DATA.md)), modeling ([MODEL.md](MODEL.md)), and evaluation ([Evaluation](#Evaluation)) code. See the respective markdowns for details.

```
mimic-video
├── data_preprocessing  # data preprocessing
├── eval                # evaluation
└── model               # dataloading, model architecture, training, and inference
```

## Environment Setup and Downloading Checkpoints

1. Create uv environment.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd model

# for hopper
uv sync --extra cu126
# for blackwell
uv sync --extra cu129

source .venv/bin/activate
```

2. (Optional) Download trained bridge or libero checkpoints.
```bash
hf auth login

python scripts/download_checkpoints.py
```
If you want to run guardrails (default enabled), additionally request access to and download `nvidia/Cosmos-Guardrail1` and `meta-llama/Llama-Guard-3-8B` to the checkpoints directory.

## Intel XPU Support

Inference can run on Intel XPU (as well as CUDA and CPU). All inference entry points accept a `--device` flag
(`cuda`, `xpu` or `cpu`, optionally with an index such as `xpu:1`) and default to the best available accelerator.

### Setup

Use the PyTorch XPU build (`torch`, `torchvision`, `torchaudio` from the Intel XPU wheel index) and install the
remaining dependencies without the CUDA extras:

```bash
cd model
python -m venv --system-site-packages .venv   # reuse the system torch+xpu
source .venv/bin/activate
pip install attrs loguru einops termcolor pyyaml "hydra-core==1.3.2" omegaconf mediapy "imageio[pyav,ffmpeg]" \
    iopath "megatron-core==0.12.1" "diffusers==0.33.1" ftfy scikit-image pandas opencv-python-headless \
    peft protobuf tyro webdataset boto3 h11 fvcore nltk threadpoolctl better-profanity decord2
```

Transformer Engine, flash-attn and apex are CUDA-only. They are now optional: when they are missing the DiTs
transparently fall back to PyTorch SDPA attention, and fused optimizers raise a clear error only if actually used.

### Running

```bash
# Text/image -> video generation
python scripts/run_video2world.py --device xpu --disable_guardrail \
    --prompt "a robot arm picks up a carrot" --input_path /path/to/frame.png --save_path out.mp4

# VAM policy (LIBERO / SIMPLER). The policy path is device-agnostic; the simulators still require their own setup.
python eval/libero/run.py --device xpu ...
```

The 11B T5 text encoder is ~45 GB in fp32 and does not fit next to the Cosmos models on a single 24 GB card.
Use `--text_encoder_device` to place it elsewhere (e.g. `--text_encoder_device cpu` or `--text_encoder_device xpu:1`),
optionally combined with `--downcast_text_encoder`.

### Simulator-free parity testing

`scripts/vam_inference.py` runs the full `Video2World2ActionPipeline` on a deterministic synthetic observation and
writes the predicted action chunk to `.npy`, which makes it easy to compare devices:

```bash
python scripts/vam_inference.py --device cpu --output /tmp/vam_cpu.npy
python scripts/vam_inference.py --device xpu --output /tmp/vam_xpu.npy
python scripts/compare_outputs.py /tmp/vam_cpu.npy /tmp/vam_xpu.npy --atol 1e-3
```

Notes on precision and parity:

* `bfloat16` (the native training precision) is quantized with 8 mantissa bits: one ULP is already
  `3.9e-3` for values in `[0.5, 1)`, so an absolute tolerance of `1e-3` is not meaningful there.
  Native bf16 CPU-vs-XPU runs agree to within a few ULP (measured max abs diff `6.0e-3`).
* For a strict parity check run both devices with `--dtype float32`. XPU has no fused fp32 SDPA kernel, so the
  non-CUDA attention path uses a memory-bounded chunked attention; measured CPU-vs-XPU agreement is then
  max abs diff `3.5e-6` for the action decoder and `~1e-3` on the generated video tensor.

## Training

You can find an overview over the training repository (which is built on the [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2) repo) in [MODEL.md](./MODEL.md). A quickstart to training your own models is given below.

Multi-node & multi-gpu configuration is handled through [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html).

### Wandb setup

Fill in your wandb entity name in the [wandb callback config](./model/cosmos_predict2/configs/defaults/callbacks.py).

### Video Model Finetuning

This assumes you have downloaded at least the text encoder, video tokenizer, and `v2w_pretrained_cosmos`.

1. Extract videos and language instructions.
   1. Choose a `/path/to/data/dataset/`.
   2. Populate `/path/to/data/dataset/video/` with `ep.mp4` and `/path/to/data/dataset/metas/` with `ep.txt` files. Example scripts for bridge and libero are provided in [data_preprocessing/video](./data_preprocessing/video/). Instructions for downloading these datasets (and fixing libero) are in [Action Decoder Pretraining](#action-decoder-pretraining).
2. Precompute language embeddings in `/path/to/data/dataset/language_embeddings/`.
```bash
cd data_preprocessing/video/
python precompute_t5_embeddings.py --dataset-path /path/to/data/dataset/
```
3. Optionally precompute video embeddings in `/path/to/data/dataset/video_embeddings/`.
```bash
python precompute_video_embeddings.py --dataset-dir /path/to/data/dataset/ --target-fps ? --num-samples-per-second ?
```
4. Create video finetuning config.
   1. Add your dataset to `train_datasets` in [data_video.py](./model/cosmos_predict2/configs/defaults/data_video.py).
   2. Add your experiment hyperparameters to [video2world.py](./model/cosmos_predict2/configs/experiment/video2world.py).
5. Start training with [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html). The experiment name is defined in [video2world.py](./model/cosmos_predict2/configs/experiment/video2world.py) from the step before.
```bash
torchrun -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment=...
```

### Action Decoder Pretraining

This assumes you have downloaded the text encoder, video tokenizer, and the video backbone you would like to train an action decoder for.

#### Bridge

1. Download raw data and unzip.
```bash
aria2c -x 16 -s 16 -c "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip"
7z x demos_8_17.zip -obridge_raw/
rm bridge_raw/raw/traj.tar  # a duplicate episode
rm -r bridge_raw/raw/bridge_data_v1/berkeley/toykitchen1/flip_pot_upright_in_sink_distractors/2021-06-02_16-36-21/raw/traj_group0/traj18/  # only three images
rm -r bridge_raw/raw/bridge_data_v2/datacol2_toykitchen1/pnp_push_sweep/04/2023-07-05_16-29-29/raw/traj_group0/traj49/  # no lowdim data
rm -r bridge_raw/raw/bridge_data_v2/datacol2_toykitchen1/pnp_push_sweep/01/2023-07-03_15-11-24/raw/traj_group0/traj24/  # no lowdim data
```
2. Convert to safetensors.
```bash
cd data_preprocessing/action/
python process_bridge.py --raw-dir ../../bridge_raw/raw --output-dir /path/to/data/bridge/
```
3. Precompute language embeddings.
```bash
# If you already computed them for video model finetuning:
python transfer_t5_bridge.py --dataset-path /path/to/data/bridge/ --language-embeddings-path /path/to/data/bridge_video/language_embeddings/
# Otherwise:
python precompute_t5.py --dataset-path /path/to/data/bridge/
```
4. Optionally transfer video embeddings (if you already computed them for video model finetuning).
```bash
python transfer_video_embeddings_bridge.py --dataset-path /path/to/data/bridge/ --video-embeddings-path /path/to/data/bridge_video/video_embeddings/
```
5. Create training config.
   1. Adapt dataset.data_dir in [bridge.yaml](./model/cosmos_predict2/configs/dataloading/dataset/bridge.yaml) to point to the directory containing the data you want to train on. See [DATA.md](./DATA.md) for details on the data config structure.
   2. Choose training hyperparameters (cross-attention layer, learning rate, batch size) and the video model checkpoint in [experiment/world2action.py](./model/cosmos_predict2/configs/experiment/world2action.py). To use the same hyperparameters as the pretrained checkpoints you can select the correct configuration via the experiment name without changing code.
6. Start training with [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html). The experiment name is defined in [world2action.py](./model/cosmos_predict2/configs/experiment/world2action.py) from the step before.
```bash
cd ../../model
torchrun -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment=...
```

#### LIBERO

1. Follow the [LIBERO dependency installation](#install-dependencies-1) steps.
2. Download the official datasets.
```bash
cd LIBERO
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```
3. Regenerate h5 recordings (filter success, filter no-op, rotate image, re-render at higher resolution).
```bash
cd ../../../data_preprocessing/action/
PYTHONPATH=../../eval/libero/LIBERO/ python regenerate_libero.py --in-dir /path/to/libero/datasets/ --out-dir /path/to/libero/regenerated_datasets/
```
4. Convert to safetensors.
```bash
python process_libero.py --input-dir /path/to/libero/regenerated_datasets/ --output-dir /path/to/data/libero/
```
5. Precompute language embeddings.
```bash
# It takes ~no time
python precompute_t5.py --dataset-path /path/to/data/libero/
```
6. Optionally transfer video embeddings (if you already computed them for video model finetuning).
```bash
python transfer_video_embeddings_libero.py --dataset-path /path/to/data/libero/ --video-embeddings-path /path/to/data/libero_video/video_embeddings/
```
7. Create training config.
   1. Adapt dataset.data_dir in the libero suite you want to train: [libero/](./model/cosmos_predict2/configs/dataloading/) to point to the directory containing the data you want to train on. See [DATA.md](./DATA.md) for details on the data config structure.
   2. Choose training hyperparameters (cross-attention layer, learning rate, batch size) and the video model checkpoint in [experiment/world2action.py](./model/cosmos_predict2/configs/experiment/world2action.py). To use the same hyperparameters as the pretrained checkpoints you can select the correct configuration via the experiment name without changing code.
8. Start training with [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html). The experiment name is defined in [world2action.py](./model/cosmos_predict2/configs/experiment/world2action.py) from the step before.
```bash
cd ../../model
torchrun -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment=...
```

## Evaluation

We have integrated [vanilla SIMPLER-Bridge](./eval/bridge/SimplerEnv/simpler_env/main_inference.py), [human-in-the-loop SIMPLER-Bridge](./eval/bridge/SimplerEnv/simpler_env/main_inference_hil.py) (for ground-truth future video generation), and [vanilla LIBERO](./eval/libero/run.py) evals in this repo. To reproduce the sim results with our checkpoints, follow these quick steps:

### SIMPLER Bridge

#### Install dependencies

```bash
sudo apt install libvulkan1

cd eval/bridge

uv pip install -r SimplerEnv/requirements.txt
uv pip install -e SimplerEnv/ManiSkill2_real2sim
uv pip install -e SimplerEnv
```

#### Run evaluation

This assumes you have the checkpoints from [Environment Setup and Downloading Checkpoints](#environment-setup-and-downloading-checkpoints).

##### Normal policy eval

```bash
# Adapt the `GPUS` list (line 1) to which GPUs to parallelize over (now: 0-7).
# Adapt line 8 for how many evals can run in parallel per GPU (now: 2).
# Fill in `checkpoint_dir` with the path to the checkpoint directory (line 33).
bash eval.sh
```

##### Human-in-the-loop evaluation (oracle study)

For this one, you have to sit down and teleop. The policy will get the ground-truth future video from the teleop, add noise, and then decode actions.

```bash
# Fill in `checkpoint_dir` with the path to the checkpoint directory (line 1).
bash eval_hil.sh
```

### LIBERO

#### Install dependencies

```bash
cd eval/libero

uv pip install -r LIBERO/requirements.txt
uv pip install -e LIBERO
```

#### Run evaluation

This also assumes you have the checkpoints from [Environment Setup and Downloading Checkpoints](#environment-setup-and-downloading-checkpoints).

```bash
# Adapt the `GPUS` list (line 1) to which GPUs to parallelize over (now: 0-7).
# Adapt line 8 for how many evals can run in parallel per GPU (now: 2).
# Fill in `checkpoint_dir` with the path to the checkpoint directory (line 29).
bash eval.sh
```

# License

```
Copyright 2026 mimic-video authors and mimic robotics AG

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this repository except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

# BibTeX

```bibtex
@misc{pai2025mimicvideo,
      title={mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs}, 
      author={Jonas Pai and Liam Achenbach and Victoriano Montesinos and Benedek Forrai and Oier Mees and Elvis Nava},
      year={2025},
      eprint={2512.15692},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.15692}, 
}
```
