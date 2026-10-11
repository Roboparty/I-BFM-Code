"""Near-target Carry alignment with closed-loop pivot candidate switching."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from pivot_metrics import _install_metric_probe
import near_align_pivot_sim2sim as pivot_v1


def _env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _install_closed_loop_controller(module: Any) -> None:
    parent = module.PhaseController

    class ClosedLoopPivotController(parent):
        def __init__(self, args, *, dt: float) -> None:
            bank_path = Path(args.latent_dir) / "z_transport_pivot_candidates.pkl"
            raw_bank = joblib.load(bank_path)
            self.pivot_candidates_np = {
                direction: [
                    np.asarray(value, dtype=np.float32).reshape(1, -1)
                    for value in raw_bank[direction]
                ]
                for direction in (-1, 1)
            }
            original_turn = joblib.load(
                Path(args.latent_dir) / "z_transport_turn_original.pkl"
            )
            self.normal_turn_np = {
                direction: np.asarray(original_turn[direction], dtype=np.float32).reshape(1, -1)
                for direction in (-1, 1)
            }
            metadata_path = Path(args.latent_dir) / "closed_loop_selection.json"
            metadata = json.loads(metadata_path.read_text())
            self.pivot_candidate_names = {
                direction: list(metadata["fallback_order"][str(direction)])
                for direction in (-1, 1)
            }
            self.pivot_arrival_speed = _env_float("UFO_PIVOT_ARRIVAL_SPEED", 0.15)
            self.pivot_response_steps = max(
                1, round(_env_float("UFO_PIVOT_RESPONSE_SECONDS", 0.50) / dt)
            )
            self.pivot_early_wrong_steps = max(
                1, round(_env_float("UFO_PIVOT_EARLY_WRONG_SECONDS", 0.25) / dt)
            )
            self.pivot_min_base_yaw = _env_float("UFO_PIVOT_MIN_BASE_YAW_RAD", 0.04)
            self.pivot_min_box_yaw = _env_float("UFO_PIVOT_MIN_BOX_YAW_RAD", 0.02)
            self.pivot_wrong_yaw = _env_float("UFO_PIVOT_WRONG_YAW_RAD", 0.02)
            self.pivot_max_base_drift = _env_float(
                "UFO_PIVOT_MAX_BASE_DRIFT_M", 0.22
            )
            self.pivot_max_box_drift = _env_float(
                "UFO_PIVOT_MAX_BOX_DRIFT_M", 0.28
            )
            self.pivot_max_steps = max(
                1, round(_env_float("UFO_PIVOT_MAX_SECONDS", 2.0) / dt)
            )
            self.pivot_cooldown_steps = max(
                0, round(_env_float("UFO_PIVOT_COOLDOWN_SECONDS", 0.50) / dt)
            )
            super().__init__(args, dt=dt)

        def reset(self) -> None:
            super().reset()
            self._reset_closed_loop_state(reset_cooldown=True)

        def _reset_near_align(self) -> None:
            super()._reset_near_align()
            self._reset_closed_loop_state(reset_cooldown=True)

        def _reset_closed_loop_state(self, *, reset_cooldown: bool) -> None:
            self.pivot_direction = 0
            self.pivot_candidate_index = 0
            self.pivot_candidate_steps = 0
            self.pivot_total_steps = 0
            self.pivot_base_yaw_integral = 0.0
            self.pivot_box_yaw_integral = 0.0
            self.pivot_candidate_validated = False
            self.pivot_candidate_start_base_xy: np.ndarray | None = None
            self.pivot_candidate_start_box_xy: np.ndarray | None = None
            self.pivot_exit_reason = "none"
            if reset_cooldown:
                self.pivot_cooldown_remaining = 0

        def _start_pivot(self, direction: int) -> None:
            self.near_align_active = True
            self.near_align_enter_count = 0
            self.near_align_exit_count = 0
            self.pivot_direction = int(direction)
            self.pivot_candidate_index = 0
            self.pivot_candidate_steps = 0
            self.pivot_total_steps = 0
            self.pivot_base_yaw_integral = 0.0
            self.pivot_box_yaw_integral = 0.0
            self.pivot_candidate_validated = False
            self.pivot_candidate_start_base_xy = None
            self.pivot_candidate_start_box_xy = None
            self.pivot_exit_reason = "active"
            print(
                "[pivot-feedback] start "
                f"direction={'left' if direction > 0 else 'right'} "
                f"candidate={self.pivot_candidate_names[direction][0]}",
                flush=True,
            )

        def _release_pivot(self, reason: str) -> None:
            self.near_align_active = False
            self.near_align_enter_count = 0
            self.near_align_exit_count = 0
            self.pivot_cooldown_remaining = self.pivot_cooldown_steps
            self.pivot_exit_reason = reason
            print(
                f"[pivot-feedback] exit reason={reason} "
                f"direction={self.pivot_direction:+d} "
                f"candidate={self.pivot_candidate_names[self.pivot_direction][self.pivot_candidate_index]}",
                flush=True,
            )

        def _switch_candidate(self, reason: str) -> None:
            direction = self.pivot_direction
            next_index = self.pivot_candidate_index + 1
            if next_index >= len(self.pivot_candidates_np[direction]):
                self._release_pivot("all_candidates_failed")
                return
            previous = self.pivot_candidate_names[direction][self.pivot_candidate_index]
            self.pivot_candidate_index = next_index
            self.pivot_candidate_steps = 0
            self.pivot_base_yaw_integral = 0.0
            self.pivot_box_yaw_integral = 0.0
            self.pivot_candidate_validated = False
            self.pivot_candidate_start_base_xy = None
            self.pivot_candidate_start_box_xy = None
            current = self.pivot_candidate_names[direction][next_index]
            print(
                f"[pivot-feedback] switch reason={reason} {previous} -> {current}",
                flush=True,
            )

        def _update_feedback(self, metrics: dict[str, float | bool]) -> None:
            if not self.near_align_active or self.pivot_direction == 0:
                return
            direction = self.pivot_direction
            base_xy = np.asarray(
                [metrics["probe_base_x"], metrics["probe_base_y"]], dtype=np.float64
            )
            box_xy = np.asarray(
                [metrics["probe_box_x"], metrics["probe_box_y"]], dtype=np.float64
            )
            if self.pivot_candidate_start_base_xy is None:
                self.pivot_candidate_start_base_xy = base_xy.copy()
                self.pivot_candidate_start_box_xy = box_xy.copy()
            self.pivot_candidate_steps += 1
            self.pivot_total_steps += 1
            self.pivot_base_yaw_integral += (
                direction * float(metrics["probe_base_yaw_rate"]) * self.dt
            )
            self.pivot_box_yaw_integral += (
                direction * float(metrics["probe_box_yaw_rate"]) * self.dt
            )

            early_wrong = (
                self.pivot_candidate_steps >= self.pivot_early_wrong_steps
                and (
                    self.pivot_base_yaw_integral <= -self.pivot_wrong_yaw
                    or self.pivot_box_yaw_integral <= -self.pivot_wrong_yaw
                )
            )
            response_ready = self.pivot_candidate_steps >= self.pivot_response_steps
            base_drift = float(
                np.linalg.norm(base_xy - self.pivot_candidate_start_base_xy)
            )
            box_drift = float(
                np.linalg.norm(box_xy - self.pivot_candidate_start_box_xy)
            )
            response_drifted = response_ready and (
                base_drift > self.pivot_max_base_drift
                or box_drift > self.pivot_max_box_drift
            )
            response_weak = response_ready and (
                self.pivot_base_yaw_integral < self.pivot_min_base_yaw
                or self.pivot_box_yaw_integral < self.pivot_min_box_yaw
            )
            if early_wrong:
                self._switch_candidate("wrong_direction")
            elif response_drifted:
                self._switch_candidate("excessive_drift")
            elif response_weak:
                self._switch_candidate("weak_response")
            elif response_ready and not self.pivot_candidate_validated:
                self.pivot_candidate_validated = True
                print(
                    "[pivot-feedback] validated "
                    f"candidate={self.pivot_candidate_names[direction][self.pivot_candidate_index]} "
                    f"base_yaw={self.pivot_base_yaw_integral:.3f}rad "
                    f"box_yaw={self.pivot_box_yaw_integral:.3f}rad "
                    f"base_drift={base_drift:.3f}m box_drift={box_drift:.3f}m",
                    flush=True,
                )

        def _update_closed_loop_latch(
            self,
            distance: float,
            bearing: float,
            metrics: dict[str, float | bool],
        ) -> None:
            if self.pivot_cooldown_remaining > 0:
                self.pivot_cooldown_remaining -= 1
            if self.near_align_active:
                self._update_feedback(metrics)
                if not self.near_align_active:
                    return
                arrived = (
                    distance <= float(self.args.place_align_distance)
                    and float(metrics["object_speed"]) <= self.pivot_arrival_speed
                )
                aligned = abs(bearing) <= self.near_align_exit_bearing_deg
                self.near_align_exit_count = self.near_align_exit_count + 1 if aligned else 0
                if arrived:
                    self._release_pivot("arrived_slow")
                elif self.near_align_exit_count >= self.near_align_exit_steps:
                    self._release_pivot("bearing_aligned")
                elif self.pivot_total_steps >= self.pivot_max_steps:
                    self._release_pivot("timeout")
                return

            if self.pivot_cooldown_remaining > 0 or self.near_align_weight > 1.0e-6:
                self.near_align_enter_count = 0
                return
            enter = (
                distance <= self.near_align_distance
                and abs(bearing) >= self.near_align_enter_bearing_deg
            )
            self.near_align_enter_count = self.near_align_enter_count + 1 if enter else 0
            if self.near_align_enter_count >= self.near_align_enter_steps:
                self._start_pivot(1 if bearing >= 0.0 else -1)

        def _candidate_tensor(self, direction: int, reference: torch.Tensor) -> torch.Tensor:
            value = self.pivot_candidates_np[direction][self.pivot_candidate_index]
            return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

        def _normal_turn_tensor(self, direction: int, reference: torch.Tensor) -> torch.Tensor:
            value = self.normal_turn_np[direction]
            return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

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
            self._update_closed_loop_latch(distance, bearing, metrics)
            self._advance_weight()
            normal_direction = 1 if bearing >= 0.0 else -1
            active_direction = self.pivot_direction or normal_direction
            normal_turn = module._transport_turn_weight(
                goal_distance=distance,
                bearing_deg=bearing,
                turn_boost_distance=self.args.transport_turn_boost_distance,
                place_distance=self.args.place_align_distance,
                heading_blend_full_deg=self.args.transport_heading_blend_full_deg,
                far_max_turn_blend=self.args.transport_max_turn_blend,
                near_max_turn_blend=self.args.transport_near_max_turn_blend,
            )
            normal_turn_latent = policy.project_z(
                self._normal_turn_tensor(normal_direction, z["transport"])
            )
            normal_latent = policy.project_z(
                (1.0 - normal_turn) * z["transport"] + normal_turn * normal_turn_latent
            )
            pivot_latent = policy.project_z(
                self._candidate_tensor(active_direction, z["transport"])
            )
            latent = policy.project_z(
                (1.0 - self.near_align_weight) * normal_latent
                + self.near_align_weight * pivot_latent
            )
            effective_turn = (
                (1.0 - self.near_align_weight) * normal_turn + self.near_align_weight
            )
            self.last_transport_turn_weight = float(effective_turn)
            if self.near_align_active or self.near_align_weight > 1.0e-6:
                side = "left" if active_direction > 0 else "right"
                mode = f"transport_align_{side}_c{self.pivot_candidate_index}"
            elif normal_turn < 1.0e-3:
                mode = "transport"
            else:
                mode = "transport_left" if normal_direction > 0 else "transport_right"
            return observation, latent, mode

    module.PhaseController = ClosedLoopPivotController


def main() -> None:
    near_align = pivot_v1._load_near_align_v1()
    near_align._install_initialise_state_compatibility()
    module = near_align._load_frozen_controller()
    near_align._install_near_align_controller(module)
    _install_metric_probe(module)
    _install_closed_loop_controller(module)
    args = module.parse_args()
    print(
        "[carry-pivot] closed-loop candidate validation/switching enabled: "
        "0.5s response window, signed base+box yaw checks",
        flush=True,
    )
    module.run(args)


if __name__ == "__main__":
    main()
