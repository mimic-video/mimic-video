# Training profiler — answers "compute vs communication vs data" for this multi-GPU setup, and
# directly benchmarks the interconnect (latency + bandwidth) for the actual DDP gradient payload.
#
# Dormant unless env MIMIC_PROFILE is truthy (the autoresearch harness never sets it). Designed for a
# short run (~20 iters): it profiles a window of steps with torch.profiler, prints a compute/comm
# breakdown + top CUDA kernels on rank 0, exports a Chrome trace, and runs an all-reduce sweep so you
# can see latency-bound (small) vs bandwidth-bound (large) behavior on PCIe.

import os

import torch
import torch.distributed as dist

from imaginaire.model import ImaginaireModel
from imaginaire.utils import log
from imaginaire.utils.callback import Callback


def _is_rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _world() -> int:
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1


class ProfilerCallback(Callback):
    def __init__(self, warmup: int = 5, active: int = 8):
        super().__init__()
        self.warmup = warmup
        self.active = active
        self.enabled = os.environ.get("MIMIC_PROFILE", "0").strip().lower() in ("1", "true", "yes", "on")
        self.prof = None
        self._started = False
        self._done = False
        self._payload_bytes = 0

    # ---- report the DDP all-reduce payload: only trainable (LoRA) params are synced ----
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not self.enabled:
            return
        n_train = n_total = 0
        bytes_train = 0
        for p in model.parameters():
            n_total += p.numel()
            if p.requires_grad:
                n_train += p.numel()
                bytes_train += p.numel() * p.element_size()
        self._payload_bytes = bytes_train
        if _is_rank0():
            log.critical(
                f"[profile] trainable params (all-reduced each step): {n_train/1e6:.2f}M "
                f"of {n_total/1e9:.2f}B total ({100*n_train/max(n_total,1):.3f}%) "
                f"=> grad payload ~{bytes_train/1e6:.1f} MB/step across {_world()} ranks"
            )

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        if not self.enabled or self._done:
            return
        # Start profiling after warmup (skip allocator/cudnn autotune noise); stop after `active` steps.
        if not self._started and iteration >= self.warmup:
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            )
            self.prof.__enter__()
            self._started = True
            self._start_iter = iteration
        if self._started:
            self.prof.step()
            if iteration >= self._start_iter + self.active:
                self.prof.__exit__(None, None, None)
                self._analyze()
                self._done = True

    def _analyze(self) -> None:
        if not _is_rank0():
            return
        ka = self.prof.key_averages()
        # self_device_time_total is in microseconds; sum across the profiled window.
        def dev_us(evt):
            for attr in ("self_device_time_total", "self_cuda_time_total"):
                v = getattr(evt, attr, None)
                if v:
                    return v
            return 0
        total = sum(dev_us(e) for e in ka)
        def is_comm(name):
            n = name.lower()
            return any(k in n for k in ("nccl", "allreduce", "all_reduce", "reduce_scatter", "allgather", "all_gather", "broadcast"))
        comm = sum(dev_us(e) for e in ka if is_comm(e.key))
        compute = total - comm
        steps = max(self.active, 1)
        log.critical("[profile] ===== GPU time breakdown (rank0, %d profiled steps) =====" % steps)
        log.critical(f"[profile] total GPU kernel time : {total/1e3:8.1f} ms  ({total/1e3/steps:6.1f} ms/step)")
        log.critical(f"[profile]   compute             : {compute/1e3:8.1f} ms  ({100*compute/max(total,1):5.1f}%)")
        log.critical(f"[profile]   communication (NCCL): {comm/1e3:8.1f} ms  ({100*comm/max(total,1):5.1f}%)")
        log.critical("[profile] top CUDA kernels by self device time:")
        order = sorted(ka, key=dev_us, reverse=True)[:12]
        for e in order:
            log.critical(f"[profile]   {dev_us(e)/1e3:8.1f} ms  {e.key[:70]}")
        # Chrome trace for detailed inspection (chrome://tracing or perfetto.dev).
        try:
            out = os.path.join(getattr(self.config.job, "path_local", "."), "profile_trace.json")
            self.prof.export_chrome_trace(out)
            log.critical(f"[profile] chrome trace written: {out}")
        except Exception as e:  # noqa: BLE001
            log.critical(f"[profile] could not export trace: {e}")

    # ---- interconnect micro-benchmark: latency (small msgs) vs bandwidth (large msgs) ----
    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not self.enabled or _world() < 2:
            return
        sizes = [1 << 10, 1 << 16, 1 << 20, 4 << 20, 16 << 20, 64 << 20]  # 1KB .. 64MB
        if self._payload_bytes:
            sizes.append(int(self._payload_bytes))
        sizes = sorted(set(sizes))
        iters = 20
        rows = []
        for nbytes in sizes:
            n = max(nbytes // 4, 1)
            t = torch.empty(n, dtype=torch.float32, device="cuda")
            for _ in range(3):  # warmup
                dist.all_reduce(t)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                dist.all_reduce(t)
            e.record()
            torch.cuda.synchronize()
            ms = s.elapsed_time(e) / iters
            world = _world()
            # ring all-reduce moves ~2*(N-1)/N * bytes over the bus ("busbw"); algbw = bytes/time.
            algbw = nbytes / (ms / 1e3) / 1e9  # GB/s
            busbw = algbw * 2 * (world - 1) / world
            rows.append((nbytes, ms, algbw, busbw))
        if _is_rank0():
            log.critical(f"[profile] ===== all-reduce sweep over {_world()} GPUs (PCIe) =====")
            log.critical("[profile]   size        time/op     algbw      busbw")
            for nbytes, ms, algbw, busbw in rows:
                tag = "  <- grad payload" if nbytes == int(self._payload_bytes) else ""
                sz = f"{nbytes/1e6:.2f}MB" if nbytes >= 1e6 else f"{nbytes/1e3:.0f}KB"
                log.critical(f"[profile]   {sz:>9}  {ms:8.3f}ms  {algbw:7.2f}GB/s {busbw:7.2f}GB/s{tag}")
            log.critical("[profile] small-size time ~= interconnect latency floor; large-size busbw ~= sustained PCIe BW.")
