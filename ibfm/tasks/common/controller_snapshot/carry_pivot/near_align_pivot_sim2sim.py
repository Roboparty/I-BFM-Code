"""Run the isolated near-align controller with Carry-pivot safety exits.

This imports the bundled controller and near-align implementation. Candidate latent directories supply
the reward-inferred pivot bank through their compatibility
``z_transport_turn.pkl`` file.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
NEAR_ALIGN_V1 = HERE.parent / "carry_near_align" / "near_align_sim2sim.py"


def _load_near_align_v1():
    spec = importlib.util.spec_from_file_location("_intact150_near_align_v1", NEAR_ALIGN_V1)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load near-align-v1 wrapper: {NEAR_ALIGN_V1}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _install_pivot_safety_controller(module: Any, near_align: Any) -> None:
    near_align_controller = module.PhaseController

    class PivotSafetyPhaseController(near_align_controller):
        def __init__(self, args, *, dt: float) -> None:
            self.pivot_arrival_speed = _env_float("UFO_PIVOT_ARRIVAL_SPEED", 0.15)
            max_seconds = _env_float("UFO_PIVOT_MAX_SECONDS", 2.0)
            cooldown_seconds = _env_float("UFO_PIVOT_COOLDOWN_SECONDS", 0.50)
            self.pivot_max_steps = max(1, round(max_seconds / dt))
            self.pivot_cooldown_steps = max(0, round(cooldown_seconds / dt))
            self._latest_pivot_metrics: dict[str, float | bool] = {}
            super().__init__(args, dt=dt)

        def reset(self) -> None:
            super().reset()
            self.pivot_active_steps = 0
            self.pivot_cooldown_remaining = 0
            self.pivot_exit_reason = "none"

        def _reset_near_align(self) -> None:
            super()._reset_near_align()
            self.pivot_active_steps = 0
            self.pivot_cooldown_remaining = 0
            self.pivot_exit_reason = "phase_reset"

        def _release_pivot(self, reason: str) -> None:
            self.near_align_active = False
            self.near_align_enter_count = 0
            self.near_align_exit_count = 0
            self.pivot_active_steps = 0
            self.pivot_cooldown_remaining = self.pivot_cooldown_steps
            self.pivot_exit_reason = reason

        def _update_latch(self, distance: float, bearing: float) -> None:
            if self.pivot_cooldown_remaining > 0:
                self.pivot_cooldown_remaining -= 1

            if self.near_align_active:
                self.pivot_active_steps += 1
                object_speed = float(self._latest_pivot_metrics.get("object_speed", np.inf))
                arrived = (
                    distance <= float(self.args.place_align_distance)
                    and object_speed <= self.pivot_arrival_speed
                )
                aligned = abs(bearing) <= self.near_align_exit_bearing_deg
                self.near_align_exit_count = self.near_align_exit_count + 1 if aligned else 0
                if arrived:
                    self._release_pivot("arrived_slow")
                elif self.near_align_exit_count >= self.near_align_exit_steps:
                    self._release_pivot("bearing_aligned")
                elif self.pivot_active_steps >= self.pivot_max_steps:
                    self._release_pivot("timeout")
                return

            if self.pivot_cooldown_remaining > 0 or self.near_align_weight > 1.0e-6:
                self.near_align_enter_count = 0
                return
            enter = distance <= self.near_align_distance and abs(bearing) >= self.near_align_enter_bearing_deg
            self.near_align_enter_count = self.near_align_enter_count + 1 if enter else 0
            if self.near_align_enter_count >= self.near_align_enter_steps:
                self.near_align_active = True
                self.near_align_enter_count = 0
                self.near_align_exit_count = 0
                self.pivot_active_steps = 0
                self.pivot_exit_reason = "active"

        def latent(
            self,
            policy,
            observation,
            metrics,
            *,
            base_rotation,
            box_direction,
            goal_direction,
            z,
        ):
            self._latest_pivot_metrics = metrics
            return super().latent(
                policy,
                observation,
                metrics,
                base_rotation=base_rotation,
                box_direction=box_direction,
                goal_direction=goal_direction,
                z=z,
            )

    module.PhaseController = PivotSafetyPhaseController


def main() -> None:
    near_align = _load_near_align_v1()
    near_align._install_initialise_state_compatibility()
    module = near_align._load_frozen_controller()
    near_align._install_near_align_controller(module)
    _install_pivot_safety_controller(module, near_align)
    args = module.parse_args()
    print(
        "[carry-pivot] isolated v1: reward-inferred pivot bank, "
        "slow-arrival release, 2.0s timeout, 0.5s cooldown"
    )
    module.run(args)


if __name__ == "__main__":
    main()
