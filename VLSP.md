# VLSP: Video-Latent Source Prior for Action Flow

## 1. Method summary

mimic-video generates actions with a flow-matching action decoder. The partially
denoised video-model latent is used as a **cross-attention condition** for the
action DiT. The action flow is integrated from a Gaussian **source** endpoint:

```python
source   = torch.randn_like(action)        # N(0, I)
x_t      = (1 - t) * action + t * source
target_v = source - action
pred_v   = action_dit(x_t, t, cond=video_latents, obs, ...)
```

**VLSP** keeps the video latent as a condition *and additionally* uses it to
parameterize the **source** of the flow:

```python
video_latents      = partial_video_world_model(...)
source, metrics    = source_prior(video_latents, obs, context_timestep, ...)
x_t                = (1 - t) * action + t * source
target_v           = source - action
pred_v             = action_dit(x_t, t, cond=<normal | zero | shuffled>)
```

> **Existing mimic-video uses video latents as a *condition*.
> VLSP uses video latents to *initialize* the action generative process.**

The source-prior input and the action-decoder condition are **independently
configurable**, which is what enables the source-only / shuffled / blended
ablations below.

When `action_source_prior.enabled = False` the code path is **bit-identical** to
the original implementation (verified for both the training epsilon draw and the
inference `arch_invariant_rand(seed=...)` draw).

## 2. Equations

```
h_v        = W_video(o, l, sigma_v)                         # partially denoised video latent
q_phi(s|.) = N( mu_phi(h_v, o, sigma_v), diag(sigma_phi(h_v, ...)^2) )

x_t        = (1 - t) * a + t * s
u_t        = s - a
L_flow     = || v_theta(x_t, t, c) - u_t ||^2
L          = L_flow + lambda_KL * KL( q_phi || N(0, I) )

KL         = 0.5 * ( mu^2 + sigma^2 - 1 - log sigma^2 )     # diagonal Gaussian
```

The source `s` (and therefore `mu`, `sigma`) live in the **normalized action
space**, because the flow loss is computed on normalized actions. The final
prediction is unnormalized by the existing action unnormalizer.

## 3. Source modes

`action_source_prior.mode`:

| mode                    | source `s`                                                    | uses prior net | deterministic |
|-------------------------|--------------------------------------------------------------|:--------------:|:-------------:|
| `gaussian`              | `randn` (baseline; inference uses `arch_invariant_rand`)      | no             | given seed    |
| `video_prior_sample`    | `mu + T * exp(logstd) * eps`                                  | yes            | given seed    |
| `video_prior_mean`      | `mu`                                                          | yes            | yes           |
| `video_prior_residual`  | `eps + residual_scale * mu`                                   | yes            | given seed    |
| `video_prior_blend`     | `alpha * s_video + sqrt(1 - alpha^2) * eps`                   | yes            | given seed    |
| `video_prior_dropout`   | per-sample `mask * s_video + (1 - mask) * eps`               | yes            | given seed    |
| `shuffled_video_prior`  | like `video_prior_sample` with video latents batch-shuffled  | yes            | given seed    |
| `gt_action_noisy_debug` | `x0 + debug_noise_std * randn` (training/debug only)         | no             | no            |

`action_conditioning.mode` (independent of the source):

| mode             | video latent fed to the action DiT                  |
|------------------|-----------------------------------------------------|
| `normal`         | pass-through (baseline)                              |
| `zero_video`     | zeros (source-only experiments)                     |
| `shuffled_video` | shuffled across the batch (negative control)        |
| `dropout_video`  | per-sample random zeroing with prob `dropout_prob`  |

## 4. Config options

`cosmos_predict2/configs/config_world2action.py`:

```python
@attrs.define
class ActionSourcePriorConfig:
    enabled: bool = False               # master switch (False => exact baseline)
    mode: str = "gaussian"

    pool_type: str = "mean"             # mean | attention | perceiver
    hidden_dim: int = 1024
    num_perceiver_latents: int = 8
    num_attention_heads: int = 8
    mlp_depth: int = 2

    logstd_min: float = -5.0
    logstd_max: float = 1.0
    init_logstd: float = -1.0

    residual_scale: float = 1.0         # video_prior_residual
    blend_alpha: float = 1.0            # video_prior_blend
    source_dropout_prob: float = 0.0    # video_prior_dropout
    dropout_granularity: str = "sample" # sample | trajectory | element
    sampling_temperature: float = 1.0   # scales the stochastic term

    detach_video_latents: bool = True
    use_state: bool = True
    use_context_timestep: bool = True
    use_language: bool = False          # off by default (not plumbed to inference)

    kl_weight: float = 0.0
    mean_l2_weight: float = 0.0
    std_reg_weight: float = 0.0

    debug_noise_std: float = 0.05

@attrs.define
class ActionConditioningConfig:
    mode: str = "normal"                # normal | zero_video | shuffled_video | dropout_video
    dropout_prob: float = 0.0
```

Both are fields of `World2ActionPipelineConfig` and are overridable per
experiment / on the CLI at
`model.config.pipe_config.action_source_prior.*` and
`model.config.pipe_config.action_conditioning.*`.

### Pooling architectures (`pool_type`)

- `mean` — masked/ordinary mean over the video token dimension.
- `attention` — a single learned query attends to the video tokens.
- `perceiver` — a small learned latent array cross-attends to the video tokens, then is pooled.

The prior is **horizon-aware**: a learned per-step query is added to the pooled
global feature, so `mu`/`logstd` differ across the `HA` action steps rather than
broadcasting a single global vector.

## 5. Experiment matrix

Registered named experiments (`cosmos_predict2/configs/experiment/world2action.py`):

| experiment                       | source mode             | conditioning      |
|----------------------------------|-------------------------|-------------------|
| `vlsp_baseline_gaussian`         | gaussian (disabled)     | normal            |
| `baseline_gaussian`              | gaussian (disabled)     | normal            |
| `vlsp_source_only_sample`        | video_prior_sample      | zero_video        |
| `vlsp_source_condition_sample`   | video_prior_sample      | normal            |
| `vlsp_source_only_mean`          | video_prior_mean        | zero_video        |
| `vlsp_source_condition_mean`     | video_prior_mean        | normal            |
| `vlsp_blend_alpha_025/050/075`   | video_prior_blend       | normal            |
| `vlsp_shuffled_source`           | shuffled_video_prior    | normal            |
| `vlsp_shuffled_condition`        | video_prior_sample      | shuffled_video    |
| `vlsp_dropout_020`               | video_prior_dropout     | normal            |
| `vlsp_residual`                  | video_prior_residual    | normal            |
| `vlsp_debug_gt_action_noisy`     | gt_action_noisy_debug   | normal            |

These are built on top of a representative base experiment (a libero base when
available). Sweeps over `blend_alpha`, `residual_scale`, `source_dropout_prob`
and `sampling_temperature` can be done by overriding the corresponding field on
the CLI (below).

Full ablation table the code supports without further edits:

```
A. Baseline                source=gaussian               cond=normal
B. Source-only stochastic  source=video_prior_sample     cond=zero_video
C. Source + condition      source=video_prior_sample     cond=normal
D. Source-only determ.     source=video_prior_mean       cond=zero_video
E. Source + cond determ.   source=video_prior_mean       cond=normal
F. Shuffled source ctrl    source=shuffled_video_prior   cond=normal
G. Shuffled cond ctrl      source=video_prior_sample     cond=shuffled_video
H. Blend                   source=video_prior_blend      blend_alpha in {0.25,0.5,0.75,1.0}
I. Residual                source=video_prior_residual   residual_scale in {0.25,0.5,1.0,2.0}
J. Dropout                 source=video_prior_dropout    source_dropout_prob in {0.1,0.2,0.5}
K. Temperature             source=video_prior_sample     sampling_temperature in {0.0,0.5,1.0,1.5}
```

## 6. Example commands

Training uses `scripts.train`. Replace `<base>` with a concrete base experiment
name (e.g. `w2a_libero_goal_agentview_..._bsz1`) when overriding ad-hoc.

**Gaussian baseline (unchanged behaviour):**

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment=vlsp_baseline_gaussian
```

**VLSP source-only (source uses the video latent, decoder gets zeroed video):**

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment=vlsp_source_only_sample
```

**VLSP source + condition:**

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment=vlsp_source_condition_sample
```

**Shuffled-source negative control:**

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment=vlsp_shuffled_source
```

**Blend-alpha sweep (ad-hoc overrides on any base experiment):**

```bash
for ALPHA in 0.25 0.5 0.75 1.0; do
  torchrun --nproc_per_node=4 -m scripts.train \
    --config=cosmos_predict2/configs/config.py -- \
    experiment=<base> \
    model.config.pipe_config.action_source_prior.enabled=true \
    model.config.pipe_config.action_source_prior.mode=video_prior_blend \
    model.config.pipe_config.action_source_prior.blend_alpha=$ALPHA \
    model.config.pipe_config.action_conditioning.mode=normal
done
```

**Temperature sweep / KL regularizer (ad-hoc):**

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment=vlsp_source_condition_sample \
  model.config.pipe_config.action_source_prior.sampling_temperature=0.5 \
  model.config.pipe_config.action_source_prior.kl_weight=1e-4 \
  model.config.pipe_config.action_source_prior.pool_type=perceiver
```

**Evaluation** selects the *same experiment name* so the pipeline is built with
the matching VLSP config; the trained source-prior weights are loaded from the
action-model checkpoint automatically (`World2ActionPipeline.from_config`):

```bash
python -m eval.libero.run \
  --vam-experiment-name vlsp_source_condition_sample \
  --vam-video-model-path <video.pt> \
  --vam-action-model-path <action_vlsp.pt> \
  ...
```

**Smoke test (no GPU / dataset required):**

```bash
cd model && python -m scripts.debug_vlsp
```

## 7. Logging

When the source prior is enabled the following are logged (per training step):

```
loss/flow, loss/source_prior, loss/source_kl, loss/source_mean_l2, loss/source_std_reg
source/mu_mean, source/mu_std, source/logstd_mean, source/std_mean
source/source_mean, source/source_std
source/source_vs_x0_mse, source/source_vs_gaussian_mse
source/dropout_rate_actual, source/source_mode_id, source/shuffle_enabled
condition/mode_id, condition/shuffle_enabled
```

## 8. Checkpoint compatibility

- The source prior is saved under `source_prior.*` (and `source_prior_ema.*`
  when EMA is enabled) in the model checkpoint, alongside the existing `net.*` /
  `net_ema.*` keys.
- Loading is **non-strict** for the source prior, so:
  - **Old checkpoints without source-prior keys load fine** — a freshly
    initialized source prior is kept (and for `mode="gaussian"` there are no
    source-prior parameters at all).
  - **A brand-new source prior can be seeded from an old action-decoder
    checkpoint** (the DiT loads strictly/non-strictly as before; the source prior
    just starts from its initialization).
- Both the training `Model.state_dict()/load_state_dict()` path and the
  inference `World2ActionPipeline.from_config()` path load source-prior weights,
  so eval picks up trained priors instead of leaving them random.

## 9. Optimizer / parallelism notes & limitations

- The source prior is **always fully trainable** (even under
  `train_architecture="lora"`, where the DiT only trains LoRA params). Its
  parameters are added to the optimizer and to `clip_grad_norm_`. In LoRA mode
  the source-prior params are upcast to fp32 to match the LoRA param group dtype.
- **DDP (default):** the whole model is DDP-wrapped, so source-prior gradients
  are synchronized. To keep `find_unused_parameters=False` valid, both prior
  heads (`mu`, `logstd`) are always kept in the autograd graph for every mode
  (the unused term is numerically zero), so deterministic modes (`mean`,
  `residual`) and all-dropout batches do not trip DDP.
- **EMA:** a simple param-wise EMA of the source prior is maintained for the
  standard non-FSDP path (`source_prior_ema`), updated with the DiT EMA beta.
- **FSDP limitation:** the source prior is left **replicated** (not sharded); it
  is small. Under an FSDP-only run (no DDP wrapper), gradients for replicated
  parameters are not automatically all-reduced across data-parallel ranks, and
  FSDP+EMA for the source prior is not specially handled. The default trainer
  (`distributed_parallelism="ddp"`, `fsdp_shard_size=0`) is unaffected. If you
  enable FSDP, add explicit gradient synchronization for the source prior or
  wrap it in its own FSDP unit.
```
