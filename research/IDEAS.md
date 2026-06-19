# Experiment ideas (groomed by the agent)

Backlog of things to try to lower `val_loss` on fr3_munich video-backbone finetuning. The agent
picks ONE per iteration, tries it, keeps/discards on val_loss, and edits this list as it learns.
Edit only `model/cosmos_predict2/configs/experiment/video2world.py` to realize these.

## First: establish the baseline
- Run the unmodified config once and record it in results.tsv as `keep  baseline`.

## Learning rate (highest leverage)
- The config sweeps `lrs = np.logspace(-5, -3, 9)[[5]]` (~1.78e-4). Try neighbors: 1e-4, 3e-4, 5e-4, 1e-3.
- Try a warmup + cosine/linear decay instead of the constant scheduler.

## LoRA capacity
- rank ∈ {64, 128, 256, 512}; alpha ∈ {16, 32, 64}. Lower rank = less overfit on 58 train clips.
- Target modules: try adding/removing attention vs MLP targets; try attention-only.
- Try full finetune (`train_architecture="base"`) vs LoRA — compare val_loss and VRAM.

## Batch / throughput (VRAM permitting; /dev/shm forces num_workers=0)
- batch_size 1 → 2 → 4 (watch VRAM, 96GB cap). Larger batch = less gradient noise.
- grad_accum_iter to raise effective batch without VRAM cost.

## Regularization (only 58 train clips — overfitting is the main risk)
- Lower LoRA rank, weight decay, dropout on LoRA, fewer steps (the budget is fixed so this is implicit).
- Data: num_frames / obs_history / conditioning frames count.

## Noise / diffusion schedule
- sigma sampling distribution, min/max conditional frames, rectified-flow loss weighting.

## Notes
- Keep changes minimal and reversible; one variable at a time when possible.
- If two near-misses each helped a little, try combining them.
