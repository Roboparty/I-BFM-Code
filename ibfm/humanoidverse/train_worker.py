"""Lightweight distributed worker entrypoint.

This module must stay free of MJLab/physics imports.  ``torchrunx`` unpickles the
worker function before it publishes ``LOCAL_RANK`` to the training module.  If
the full training module is imported during that unpickle, every worker can
briefly create a CUDA context on device 0.  That startup-only context is enough
to make a full replay-buffer resume run out of memory on rank 0.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class _TeeStream:
    """Mirror Python stdout/stderr to torchrunx and a persistent rank log."""

    def __init__(self, primary: Any, mirror: Any) -> None:
        self.primary = primary
        self.mirror = mirror

    def write(self, value: str) -> int:
        primary_result = self.primary.write(value)
        self.mirror.write(value)
        return len(value) if primary_result is None else primary_result

    def flush(self) -> None:
        self.primary.flush()
        self.mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.primary.fileno()


def run_train_worker(args: Any, log_dir: Path) -> None:
    """Bind the worker GPU before importing the full training stack."""

    physical_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", str(physical_local_rank)))

    worker_log_dir = Path(log_dir) / "resume_worker_logs"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    worker_log = (worker_log_dir / f"rank_{global_rank}_stdout_stderr.log").open(
        "a", encoding="utf-8", buffering=1
    )
    sys.stdout = _TeeStream(sys.stdout, worker_log)
    sys.stderr = _TeeStream(sys.stderr, worker_log)
    print(
        f"\n[{datetime.now().astimezone().isoformat()}] worker start "
        f"rank={global_rank} physical_local_rank={physical_local_rank}",
        flush=True,
    )
    visible_devices = [
        token.strip()
        for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip()
    ]
    if visible_devices and physical_local_rank >= len(visible_devices):
        raise RuntimeError(
            "Distributed worker cannot isolate its GPU: "
            f"LOCAL_RANK={physical_local_rank}, CUDA_VISIBLE_DEVICES={visible_devices}"
        )

    # Give each worker a one-device CUDA namespace.  Several MJLab/physics
    # dependencies allocate on logical cuda:0 even after set_device(local_rank).
    # Without this isolation, ranks 1..N create startup allocations on physical
    # GPU 0 and a full replay-buffer resume runs out of memory on rank 0.
    selected_device = (
        visible_devices[physical_local_rank]
        if visible_devices
        else str(physical_local_rank)
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_device
    os.environ["UFO_PHYSICAL_LOCAL_RANK"] = str(physical_local_rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

    import torch

    torch.cuda.set_device(0)

    # Import only after selecting the rank-local device.  Importing
    # humanoidverse.train pulls in MJLab and other CUDA-aware dependencies.
    from humanoidverse.train import run_train

    run_train(args, log_dir)
