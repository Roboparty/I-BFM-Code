#!/usr/bin/env python3
"""Run a two-stage open-loop acquire before latching the normal drive flow.

The first stage uses the configured crouching acquire latent.  The second
stage blends to a narrower offline-selected hand-layout latent.  Transition
timing is independent of hand tracking so the same flow can be used when hand
mocap is noisy or unavailable.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "reacquire_sim2sim.py"
SPEC = importlib.util.spec_from_file_location("push166_reacquire_source", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load controller: {SOURCE}")
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)

runtime = source.frozen.module
controller = runtime.DirectPushController
CROUCH_SECONDS = float(
    os.environ.get("UFO_PUSH_TIMED_ACQUIRE_CROUCH_SECONDS", "0.65")
)
SETTLE_SECONDS = float(
    os.environ.get("UFO_PUSH_TIMED_ACQUIRE_SETTLE_SECONDS", "0.25")
)
SETTLE_LATENT_DIR = Path(
    os.environ.get(
        "UFO_PUSH_TIMED_ACQUIRE_SETTLE_LATENT_DIR",
        str(HERE / "latents/selected/acquire_hand_layout_v2"),
    )
)
_SETTLE_LATENT = np.asarray(
    joblib.load(SETTLE_LATENT_DIR / "z_push_direct.pkl"), dtype=np.float32
).reshape(1, -1)
if _SETTLE_LATENT.shape != (1, 256):
    raise ValueError(
        f"expected settle latent [1, 256], got {_SETTLE_LATENT.shape}: "
        f"{SETTLE_LATENT_DIR}"
    )

original_reset = controller.reset
original_latent = controller.latent
original_update = controller.update


def reset_with_timer(self: Any) -> None:
    original_reset(self)
    self._timed_acquire_steps = 0
    self._timed_acquire_complete = False


def latent_with_timer(self: Any, *args: Any, **kwargs: Any):
    observation, latent, mode = original_latent(self, *args, **kwargs)
    if self.phase != "push_direct" or self._v2_drive_seen:
        self._timed_acquire_steps = 0
        return observation, latent, mode
    if not mode.startswith("push_direct_acquire"):
        self._timed_acquire_steps = 0
        return observation, latent, mode

    self._timed_acquire_steps += 1
    crouch_steps = max(1, round(CROUCH_SECONDS / self.dt))
    settle_steps = max(1, round(SETTLE_SECONDS / self.dt))
    if self._timed_acquire_steps > crouch_steps:
        policy = args[0]
        settle = torch.as_tensor(
            _SETTLE_LATENT, dtype=latent.dtype, device=latent.device
        )
        progress = min(
            float(self._timed_acquire_steps - crouch_steps) / float(settle_steps),
            1.0,
        )
        progress = progress * progress * (3.0 - 2.0 * progress)
        latent = policy.project_z((1.0 - progress) * latent + progress * settle)
        mode = f"push_direct_acquire_settle_{progress:.2f}"

    if self._timed_acquire_steps < crouch_steps + settle_steps:
        return observation, latent, mode

    self.contact_drive_latched = True
    self._v2_drive_seen = True
    self._timed_acquire_complete = True
    self.valid_rear_contact_steps = 0
    self.invalid_rear_contact_steps = 0
    self._v2_drive_blend_steps = 0
    self._timed_acquire_steps = 0
    return original_latent(self, *args, **kwargs)


def update_with_timer(
    self: Any,
    metrics: dict[str, float | bool],
    *,
    goal_height: float,
) -> None:
    original_update(self, metrics, goal_height=goal_height)
    if self.phase != "push_direct":
        self._timed_acquire_steps = 0
        self._timed_acquire_complete = False
        return
    if not self._timed_acquire_complete:
        self.contact_drive_latched = False
        self._v2_drive_seen = False
        self.valid_rear_contact_steps = 0
        self.invalid_rear_contact_steps = 0
        self._v2_drive_blend_steps = 0


controller.reset = reset_with_timer
controller.latent = latent_with_timer
controller.update = update_with_timer


if __name__ == "__main__":
    runtime.run(runtime.parse_args())
