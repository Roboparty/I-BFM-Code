"""Run an isolated near-target alignment controller in MuJoCo.

The bundled Carry controller is imported as a runtime dependency.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np

import humanoidverse.carry_goal_sim2sim as carry_goal


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "carry_base"


def _install_initialise_state_compatibility() -> None:
    original = carry_goal._initialise_state
    if "preserve_root_object_relative_pose" in inspect.signature(original).parameters:
        return

    def compatible(*args: Any, preserve_root_object_relative_pose: bool = False, **kwargs: Any):
        if preserve_root_object_relative_pose:
            raise RuntimeError(
                "The old helper cannot preserve root/object pose for place-direct mode"
            )
        return original(*args, **kwargs)

    carry_goal._initialise_state = compatible


def _load_frozen_controller():
    path = BASELINE / "controller_snapshot" / "carry_phased_sim2sim.py"
    spec = importlib.util.spec_from_file_location("_intact150_frozen_sim2sim", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen controller snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _install_near_align_controller(module) -> None:
    base_controller = module.PhaseController

    class NearAlignPhaseController(base_controller):
        def __init__(self, args, *, dt: float) -> None:
            self.near_align_distance = _env_float("UFO_NEAR_ALIGN_DISTANCE", 0.80)
            self.near_align_enter_bearing_deg = _env_float(
                "UFO_NEAR_ALIGN_ENTER_BEARING_DEG", 30.0
            )
            self.near_align_exit_bearing_deg = _env_float(
                "UFO_NEAR_ALIGN_EXIT_BEARING_DEG", 15.0
            )
            self.near_align_enter_steps = _env_int("UFO_NEAR_ALIGN_ENTER_STEPS", 3)
            self.near_align_exit_steps = _env_int("UFO_NEAR_ALIGN_EXIT_STEPS", 5)
            self.near_align_blend_seconds = _env_float(
                "UFO_NEAR_ALIGN_BLEND_SECONDS", 0.20
            )
            if not 0.0 < self.near_align_exit_bearing_deg < self.near_align_enter_bearing_deg <= 180.0:
                raise ValueError("near-align angles must satisfy 0 < exit < enter <= 180")
            if self.near_align_distance <= 0.0 or self.near_align_blend_seconds < 0.0:
                raise ValueError("near-align distance/blend values are invalid")
            super().__init__(args, dt=dt)

        def reset(self) -> None:
            super().reset()
            self.near_align_active = False
            self.near_align_enter_count = 0
            self.near_align_exit_count = 0
            self.near_align_weight = 0.0

        def _reset_near_align(self) -> None:
            self.near_align_active = False
            self.near_align_enter_count = 0
            self.near_align_exit_count = 0
            self.near_align_weight = 0.0

        def _change_phase(self, phase: str) -> None:
            previous = self.phase
            super()._change_phase(phase)
            if phase != previous and phase != "transport":
                self._reset_near_align()

        def _update_latch(self, distance: float, bearing: float) -> None:
            if self.near_align_active:
                aligned = abs(bearing) <= self.near_align_exit_bearing_deg
                self.near_align_exit_count = self.near_align_exit_count + 1 if aligned else 0
                if self.near_align_exit_count >= self.near_align_exit_steps:
                    self.near_align_active = False
                    self.near_align_enter_count = 0
                    self.near_align_exit_count = 0
            else:
                enter = distance <= self.near_align_distance and abs(bearing) >= self.near_align_enter_bearing_deg
                self.near_align_enter_count = self.near_align_enter_count + 1 if enter else 0
                if self.near_align_enter_count >= self.near_align_enter_steps:
                    self.near_align_active = True
                    self.near_align_enter_count = 0
                    self.near_align_exit_count = 0

        def _advance_weight(self) -> None:
            target = 1.0 if self.near_align_active else 0.0
            if self.near_align_blend_seconds <= 0.0:
                self.near_align_weight = target
                return
            delta = self.dt / self.near_align_blend_seconds
            self.near_align_weight += float(
                np.clip(target - self.near_align_weight, -delta, delta)
            )

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
            if self.phase != "transport":
                if self.near_align_active or self.near_align_weight > 0.0:
                    self._reset_near_align()
                return super().latent(
                    policy,
                    observation,
                    metrics,
                    base_rotation=base_rotation,
                    box_direction=box_direction,
                    goal_direction=goal_direction,
                    z=z,
                )

            bearing = module._bearing_deg(base_rotation, goal_direction)
            distance = float(metrics["goal_distance"])
            self._update_latch(distance, bearing)
            self._advance_weight()
            direction = 1 if bearing >= 0.0 else -1
            normal_turn = module._transport_turn_weight(
                goal_distance=distance,
                bearing_deg=bearing,
                turn_boost_distance=self.args.transport_turn_boost_distance,
                place_distance=self.args.place_align_distance,
                heading_blend_full_deg=self.args.transport_heading_blend_full_deg,
                far_max_turn_blend=self.args.transport_max_turn_blend,
                near_max_turn_blend=self.args.transport_near_max_turn_blend,
            )
            turn_latent = policy.project_z(z["transport_turn"][direction])
            normal_latent = policy.project_z(
                (1.0 - normal_turn) * z["transport"] + normal_turn * turn_latent
            )
            latent = policy.project_z(
                (1.0 - self.near_align_weight) * normal_latent
                + self.near_align_weight * turn_latent
            )
            effective_turn = (1.0 - self.near_align_weight) * normal_turn + self.near_align_weight
            self.last_transport_turn_weight = float(effective_turn)
            if self.near_align_active or self.near_align_weight > 1.0e-6:
                mode = "transport_align_left" if direction > 0 else "transport_align_right"
            elif normal_turn < 1.0e-3:
                mode = "transport"
            else:
                mode = "transport_left" if direction > 0 else "transport_right"
            return observation, latent, mode

        def update(self, metrics, *, goal_height: float) -> None:
            previous = self.phase
            if self.phase == "transport" and (
                self.near_align_active
                or self.near_align_weight > 1.0e-6
                or self.near_align_enter_count > 0
            ):
                adjusted = dict(metrics)
                adjusted["goal_distance"] = max(
                    float(metrics["goal_distance"]),
                    float(self.args.place_align_distance) + 1.0e-6,
                )
                super().update(adjusted, goal_height=goal_height)
            else:
                super().update(metrics, goal_height=goal_height)
            if self.phase != previous and self.phase != "transport":
                self._reset_near_align()

    module.PhaseController = NearAlignPhaseController


def main() -> None:
    _install_initialise_state_compatibility()
    module = _load_frozen_controller()
    _install_near_align_controller(module)
    args = module.parse_args()
    print(
        "[near-align] isolated controller: distance<=0.80m, "
        "bearing enter/exit=30/15deg, blend=0.20s"
    )
    module.run(args)


if __name__ == "__main__":
    main()
