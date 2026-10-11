#!/usr/bin/env python3
"""Use a fixed acquire duration before latching the normal drive flow."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "reacquire_sim2sim.py"
SPEC = importlib.util.spec_from_file_location("push166_reacquire_source", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load controller: {SOURCE}")
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)

runtime = source.frozen.module
controller = runtime.DirectPushController
ACQUIRE_SECONDS = float(os.environ.get("UFO_PUSH_TIMED_ACQUIRE_SECONDS", "0.65"))

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
    required_steps = max(1, round(ACQUIRE_SECONDS / self.dt))
    if self._timed_acquire_steps < required_steps:
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
        # The fixed-duration acquire is intentionally independent of hand
        # tracking. Suppress the inherited contact gate until the timer ends.
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
