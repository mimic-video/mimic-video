# Autoresearch metric callback. FROZEN — do not edit as part of experiments.
# Averages the per-batch validation loss the trainer already computes and prints a single,
# parseable `val_loss: <float>` line at the end of each validation pass. This is the ground-truth
# objective for the autoresearch loop (see research/program.md).

import torch
import torch.distributed as dist

from imaginaire.model import ImaginaireModel
from imaginaire.utils import log
from imaginaire.utils.callback import Callback


class ValLossLogger(Callback):
    def __init__(self):
        super().__init__()
        self._sum = 0.0
        self._n = 0

    def on_validation_start(self, model: ImaginaireModel, dataloader, iteration: int = 0) -> None:
        self._sum = 0.0
        self._n = 0

    def on_validation_step_end(
        self, model: ImaginaireModel, data_batch, output_batch, loss: torch.Tensor, iteration: int = 0
    ) -> None:
        self._sum += float(loss.detach().item())
        self._n += 1

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        total_sum = self._sum
        total_n = self._n
        # Reduce across ranks if running distributed, so the printed value is the global mean.
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([total_sum, float(total_n)], device="cuda", dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_sum, total_n = t[0].item(), t[1].item()
        if total_n > 0:
            mean = total_sum / total_n
            # log line (rank-aware) + a clean stdout line the harness greps for.
            log.critical(f"val_loss: {mean:.6f}  (iteration={iteration}, n={int(total_n)})")
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                print(f"val_loss: {mean:.6f}", flush=True)
