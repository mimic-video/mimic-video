# Streams training & validation metrics to Weights & Biases.
#
# Dormant by default: the run is only created when the env var MIMIC_WANDB is truthy. The frozen
# autoresearch harness (train_bounded.sh) never sets it, so this callback is a no-op there and the
# `val_loss:` stdout contract is untouched. Only rank 0 talks to wandb; the validation loss is
# reduced across ranks first (mirroring ValLossLogger) so the logged value is the global mean.
#
# Enable for a real run:  MIMIC_WANDB=1 (plus `wandb login` or WANDB_API_KEY in the environment).
# Optional overrides: WANDB_PROJECT, WANDB_RUN_GROUP, WANDB_NAME, WANDB_MODE (online|offline).

import os

import torch
import torch.distributed as dist
import wandb

from imaginaire.model import ImaginaireModel
from imaginaire.utils import log
from imaginaire.utils.callback import Callback


class WandbLogger(Callback):
    def __init__(self, every_n: int = 25):
        super().__init__()
        self.every_n = max(1, int(every_n))
        self._active = False  # True only on the rank that owns the wandb run
        self._val_sum = 0.0
        self._val_n = 0

    @staticmethod
    def _enabled() -> bool:
        # Read on every rank; torchrun propagates the parent env to all ranks.
        return os.environ.get("MIMIC_WANDB", "0").strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _is_rank0() -> bool:
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not self._enabled() or not self._is_rank0():
            return
        job = getattr(self, "config", None) and self.config.job
        run_config = {}
        try:
            run_config = {
                "experiment": getattr(job, "name", None),
                "lr": float(self.config.optimizer.lr),
                "max_iter": int(self.config.trainer.max_iter),
            }
        except Exception:  # config shape is best-effort metadata only
            pass
        wandb.init(
            project=os.environ.get("WANDB_PROJECT") or getattr(job, "project", None) or "mimic-video",
            group=os.environ.get("WANDB_RUN_GROUP") or getattr(job, "group", None) or None,
            name=os.environ.get("WANDB_NAME") or getattr(job, "name", None) or None,
            config=run_config,
            resume="allow",
        )
        self._active = wandb.run is not None
        if self._active:
            log.critical(f"[wandb] logging to {wandb.run.url}")

    def on_training_step_end(
        self, model, data_batch, output_batch, loss: torch.Tensor, iteration: int = 0
    ) -> None:
        if not self._active or iteration % self.every_n != 0:
            return
        metrics = {"train/loss": float(loss.detach().item())}
        try:
            metrics["train/lr"] = float(self.config.optimizer.lr)
        except Exception:
            pass
        wandb.log(metrics, step=iteration)

    # ----- validation: accumulate locally, reduce across ranks, log on rank 0 -----
    def on_validation_start(self, model, dataloader, iteration: int = 0) -> None:
        self._val_sum = 0.0
        self._val_n = 0

    def on_validation_step_end(
        self, model, data_batch, output_batch, loss: torch.Tensor, iteration: int = 0
    ) -> None:
        self._val_sum += float(loss.detach().item())
        self._val_n += 1

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not self._enabled():
            return
        total_sum, total_n = self._val_sum, float(self._val_n)
        # All ranks must reach this collective; only rank 0 owns the wandb run.
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([total_sum, total_n], device="cuda", dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_sum, total_n = t[0].item(), t[1].item()
        if self._active and total_n > 0:
            wandb.log({"val/loss": total_sum / total_n}, step=iteration)

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if self._active:
            wandb.finish()
