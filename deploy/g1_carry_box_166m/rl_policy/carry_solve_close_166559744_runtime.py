"""Isolated real-robot runtime for the 166.560M solve-close bundle.

The module keeps the strong deployment's LuMo, calibration, observation,
actor, command and safety implementations.  It replaces only the phase
controller and consumes a copied latent directory.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
from loguru import logger
import numpy as np
import yaml

from rl_policy.carry_runtime import (
    BestCarry140935168StrongController,
    CarryMocapFrame,
    CarryRuntime,
    STATE_DIM,
    _bearing_deg,
    _transport_turn_weight,
    load_get_up_latent,
    load_phase_latents,
    project_z,
)
from utils.math import matrix_from_quat


ROOT = Path(__file__).resolve().parents[1]
BASE_PROFILE = "intact_logmap150544384_strong"
DEPLOYMENT_VARIANT = "intact_logmap166559744_solve_close"
EXPECTED_LIFT_SHA256 = (
    "34c2f0eb1310418fb7c3341333def67e72f283a5c3a98c69a8c60874933fbf11"
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(config: Mapping[str, Any], name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"solve_close.{name} must be finite")
    return value


def _positive_float(config: Mapping[str, Any], name: str, default: float) -> float:
    value = _finite_float(config, name, default)
    if value <= 0.0:
        raise ValueError(f"solve_close.{name} must be positive")
    return value


def _nonnegative_float(
    config: Mapping[str, Any], name: str, default: float
) -> float:
    value = _finite_float(config, name, default)
    if value < 0.0:
        raise ValueError(f"solve_close.{name} must be non-negative")
    return value


def _positive_int(config: Mapping[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value <= 0:
        raise ValueError(f"solve_close.{name} must be positive")
    return value


def _yaw_from_quaternion(quaternion_wxyz: np.ndarray) -> float:
    rotation = matrix_from_quat(
        np.asarray(quaternion_wxyz, dtype=np.float64).reshape(1, 4)
    )[0]
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def _wrapped_delta(current: float, previous: float) -> float:
    difference = current - previous
    return float(math.atan2(math.sin(difference), math.cos(difference)))


def load_solve_close_assets(latent_dir: Path) -> tuple[
    dict[int, list[np.ndarray]], dict[int, np.ndarray], dict[int, list[str]]
]:
    raw_candidates = joblib.load(latent_dir / "z_transport_pivot_candidates.pkl")
    raw_normal_turn = joblib.load(latent_dir / "z_transport_turn_original.pkl")
    metadata = json.loads(
        (latent_dir / "closed_loop_selection.json").read_text(encoding="utf-8")
    )
    candidates: dict[int, list[np.ndarray]] = {}
    normal_turn: dict[int, np.ndarray] = {}
    names: dict[int, list[str]] = {}
    for direction in (-1, 1):
        values = raw_candidates.get(direction)
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"pivot candidate bank requires a non-empty list for {direction:+d}"
            )
        candidates[direction] = [
            np.asarray(value, dtype=np.float32).reshape(-1) for value in values
        ]
        if any(value.shape != (256,) for value in candidates[direction]):
            raise ValueError("every solve-close pivot candidate must have shape (256,)")
        normal_turn[direction] = np.asarray(
            raw_normal_turn[direction], dtype=np.float32
        ).reshape(-1)
        if normal_turn[direction].shape != (256,):
            raise ValueError("normal transport-turn latent must have shape (256,)")
        names[direction] = list(metadata["fallback_order"][str(direction)])
        if len(names[direction]) != len(candidates[direction]):
            raise ValueError(
                f"pivot candidate names/count mismatch for direction {direction:+d}"
            )
    return candidates, normal_turn, names


class IntactLogmap166559744SolveCloseController(
    BestCarry140935168StrongController
):
    """Strong controller plus topk2000 lift and guarded close-range pivots."""

    def __init__(
        self,
        phase_config: Mapping[str, Any],
        solve_config: Mapping[str, Any],
        latents: dict[str, Any],
        dt: float,
        pivot_candidates: dict[int, list[np.ndarray]],
        normal_turn: dict[int, np.ndarray],
        candidate_names: dict[int, list[str]],
    ) -> None:
        self.pivot_candidates = pivot_candidates
        self.normal_turn = normal_turn
        self.pivot_candidate_names = candidate_names
        self.near_align_distance = _positive_float(
            solve_config, "near_align_distance", 0.80
        )
        self.near_align_enter_bearing_deg = _positive_float(
            solve_config, "near_align_enter_bearing_deg", 30.0
        )
        self.near_align_exit_bearing_deg = _positive_float(
            solve_config, "near_align_exit_bearing_deg", 15.0
        )
        if not (
            0.0
            < self.near_align_exit_bearing_deg
            < self.near_align_enter_bearing_deg
            <= 180.0
        ):
            raise ValueError(
                "solve-close bearing thresholds require 0 < exit < enter <= 180"
            )
        self.near_align_enter_required = _positive_int(
            solve_config, "near_align_enter_steps", 3
        )
        self.near_align_exit_required = _positive_int(
            solve_config, "near_align_exit_steps", 5
        )
        self.near_align_blend_seconds = _nonnegative_float(
            solve_config, "near_align_blend_seconds", 0.20
        )
        self.pivot_arrival_speed = _nonnegative_float(
            solve_config, "pivot_arrival_speed", 0.15
        )
        self.pivot_min_base_yaw = _nonnegative_float(
            solve_config, "pivot_min_base_yaw_rad", 0.04
        )
        self.pivot_min_box_yaw = _nonnegative_float(
            solve_config, "pivot_min_box_yaw_rad", 0.02
        )
        self.pivot_wrong_yaw = _nonnegative_float(
            solve_config, "pivot_wrong_yaw_rad", 0.02
        )
        self.pivot_max_base_drift = _positive_float(
            solve_config, "pivot_max_base_drift", 0.22
        )
        self.pivot_max_box_drift = _positive_float(
            solve_config, "pivot_max_box_drift", 0.28
        )
        self.pivot_max_height_drift = _positive_float(
            solve_config, "pivot_max_height_drift", 0.08
        )
        self.pivot_min_box_upright = _nonnegative_float(
            solve_config, "pivot_min_box_upright", 0.85
        )
        self.pivot_min_recent_yaw = _nonnegative_float(
            solve_config, "pivot_min_recent_yaw_rad", 0.03
        )
        self.pivot_min_bearing_improvement = _nonnegative_float(
            solve_config, "pivot_min_bearing_improvement_deg", 3.0
        )
        super().__init__(phase_config, latents, dt)
        self.pivot_response_steps = max(
            1,
            round(
                _positive_float(solve_config, "pivot_response_seconds", 0.50)
                / self.dt
            ),
        )
        self.pivot_early_wrong_steps = max(
            1,
            round(
                _positive_float(
                    solve_config, "pivot_early_wrong_seconds", 0.25
                )
                / self.dt
            ),
        )
        self.pivot_max_steps = max(
            1,
            round(
                _positive_float(solve_config, "pivot_max_seconds", 5.0)
                / self.dt
            ),
        )
        self.pivot_cooldown_steps = max(
            0,
            round(
                _nonnegative_float(
                    solve_config, "pivot_cooldown_seconds", 0.50
                )
                / self.dt
            ),
        )
        self.pivot_stall_window_steps = max(
            2,
            round(
                _positive_float(
                    solve_config, "pivot_stall_window_seconds", 0.50
                )
                / self.dt
            ),
        )
        # super().__init__ invokes reset before the duration-derived constants
        # above exist. Reset once more so deque sizes use the pinned config.
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.near_align_active = False
        self.near_align_weight = 0.0
        self.near_align_enter_count = 0
        self.near_align_exit_count = 0
        self.pivot_cooldown_remaining = 0
        self.pivot_direction = 0
        self.pivot_candidate_index = 0
        self.pivot_candidate_steps = 0
        self.pivot_total_steps = 0
        self.pivot_base_yaw_integral = 0.0
        self.pivot_box_yaw_integral = 0.0
        self.pivot_candidate_validated = False
        self.pivot_candidate_start_base_xy: np.ndarray | None = None
        self.pivot_candidate_start_box_xy: np.ndarray | None = None
        self.pivot_start_box_height: float | None = None
        self.pivot_exit_reason = "none"
        window = int(getattr(self, "pivot_stall_window_steps", 2))
        self.recent_base_yaw: deque[float] = deque(maxlen=window)
        self.recent_box_yaw: deque[float] = deque(maxlen=window)
        self.recent_abs_bearing: deque[float] = deque(maxlen=window)
        self.previous_base_yaw: float | None = None
        self.previous_box_yaw: float | None = None

    def _clear_recent_guard(self) -> None:
        self.recent_base_yaw.clear()
        self.recent_box_yaw.clear()
        self.recent_abs_bearing.clear()

    def _reset_near_align(self, *, reset_cooldown: bool) -> None:
        self.near_align_active = False
        self.near_align_weight = 0.0
        self.near_align_enter_count = 0
        self.near_align_exit_count = 0
        self.pivot_direction = 0
        self.pivot_candidate_index = 0
        self.pivot_candidate_steps = 0
        self.pivot_total_steps = 0
        self.pivot_base_yaw_integral = 0.0
        self.pivot_box_yaw_integral = 0.0
        self.pivot_candidate_validated = False
        self.pivot_candidate_start_base_xy = None
        self.pivot_candidate_start_box_xy = None
        self.pivot_start_box_height = None
        self.pivot_exit_reason = "none"
        self._clear_recent_guard()
        if reset_cooldown:
            self.pivot_cooldown_remaining = 0

    def _change_phase(self, phase: str) -> None:
        previous = self.phase
        super()._change_phase(phase)
        if phase != previous and phase != "transport":
            self._reset_near_align(reset_cooldown=True)

    def _yaw_deltas(self, frame: CarryMocapFrame) -> tuple[float, float]:
        base_yaw = _yaw_from_quaternion(frame.robot_root_quaternion_wxyz)
        box_yaw = _yaw_from_quaternion(frame.object_quaternion_wxyz)
        base_delta = (
            0.0
            if self.previous_base_yaw is None
            else _wrapped_delta(base_yaw, self.previous_base_yaw)
        )
        box_delta = (
            0.0
            if self.previous_box_yaw is None
            else _wrapped_delta(box_yaw, self.previous_box_yaw)
        )
        self.previous_base_yaw = base_yaw
        self.previous_box_yaw = box_yaw
        return base_delta, box_delta

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
        self.pivot_start_box_height = None
        self.pivot_exit_reason = "active"
        self._clear_recent_guard()
        logger.info(
            "Solve-close pivot start direction={} candidate={}".format(
                "left" if direction > 0 else "right",
                self.pivot_candidate_names[direction][0],
            )
        )

    def _release_pivot(self, reason: str) -> None:
        direction = self.pivot_direction
        candidate = self.pivot_candidate_names[direction][
            self.pivot_candidate_index
        ]
        self.near_align_active = False
        self.near_align_enter_count = 0
        self.near_align_exit_count = 0
        self.pivot_cooldown_remaining = self.pivot_cooldown_steps
        self.pivot_exit_reason = reason
        self.pivot_start_box_height = None
        self._clear_recent_guard()
        logger.info(
            "Solve-close pivot exit reason={} direction={:+d} candidate={}".format(
                reason, direction, candidate
            )
        )

    def _switch_candidate(self, reason: str) -> None:
        direction = self.pivot_direction
        next_index = self.pivot_candidate_index + 1
        if next_index >= len(self.pivot_candidates[direction]):
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
        self._clear_recent_guard()
        logger.warning(
            "Solve-close pivot switch reason={} {} -> {}".format(
                reason,
                previous,
                self.pivot_candidate_names[direction][next_index],
            )
        )

    def _update_feedback(
        self,
        metrics: Mapping[str, Any],
        frame: CarryMocapFrame,
        bearing: float,
        base_yaw_delta: float,
        box_yaw_delta: float,
    ) -> None:
        if not self.near_align_active or self.pivot_direction == 0:
            return
        direction = self.pivot_direction
        base_xy = np.asarray(frame.robot_root_position[:2], dtype=np.float64)
        box_xy = np.asarray(frame.object_position[:2], dtype=np.float64)
        if self.pivot_candidate_start_base_xy is None:
            self.pivot_candidate_start_base_xy = base_xy.copy()
            self.pivot_candidate_start_box_xy = box_xy.copy()
        if self.pivot_start_box_height is None:
            self.pivot_start_box_height = float(metrics["object_height"])

        self.pivot_candidate_steps += 1
        self.pivot_total_steps += 1
        signed_base_delta = direction * base_yaw_delta
        signed_box_delta = direction * box_yaw_delta
        self.pivot_base_yaw_integral += signed_base_delta
        self.pivot_box_yaw_integral += signed_box_delta

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
        previous_index = self.pivot_candidate_index
        if early_wrong:
            self._switch_candidate("wrong_direction")
        elif response_drifted:
            self._switch_candidate("excessive_drift")
        elif response_weak:
            self._switch_candidate("weak_response")
        elif response_ready and not self.pivot_candidate_validated:
            self.pivot_candidate_validated = True
            logger.info(
                "Solve-close pivot validated candidate={} base_yaw={:.3f}rad "
                "box_yaw={:.3f}rad base_drift={:.3f}m box_drift={:.3f}m".format(
                    self.pivot_candidate_names[direction][self.pivot_candidate_index],
                    self.pivot_base_yaw_integral,
                    self.pivot_box_yaw_integral,
                    base_drift,
                    box_drift,
                )
            )
        if (
            not self.near_align_active
            or self.pivot_candidate_index != previous_index
        ):
            return

        self.recent_base_yaw.append(signed_base_delta)
        self.recent_box_yaw.append(signed_box_delta)
        self.recent_abs_bearing.append(abs(float(bearing)))
        if len(self.recent_base_yaw) < self.pivot_stall_window_steps:
            return
        height_drift = abs(
            float(metrics["object_height"]) - self.pivot_start_box_height
        )
        box_upright = float(metrics["box_upright"])
        recent_yaw = min(sum(self.recent_base_yaw), sum(self.recent_box_yaw))
        bearing_improvement = (
            self.recent_abs_bearing[0] - self.recent_abs_bearing[-1]
        )
        if height_drift > self.pivot_max_height_drift:
            self._switch_candidate("height_drift")
        elif box_upright < self.pivot_min_box_upright:
            self._switch_candidate("box_upright")
        elif self.pivot_candidate_validated and recent_yaw < self.pivot_min_recent_yaw:
            self._switch_candidate("recent_yaw_stall")
        elif (
            self.pivot_candidate_validated
            and bearing_improvement < self.pivot_min_bearing_improvement
        ):
            self._switch_candidate("bearing_stall")

    def _update_latch(
        self,
        distance: float,
        bearing: float,
        metrics: Mapping[str, Any],
        frame: CarryMocapFrame,
        base_yaw_delta: float,
        box_yaw_delta: float,
    ) -> None:
        if self.pivot_cooldown_remaining > 0:
            self.pivot_cooldown_remaining -= 1
        if self.near_align_active:
            self._update_feedback(
                metrics, frame, bearing, base_yaw_delta, box_yaw_delta
            )
            if not self.near_align_active:
                return
            arrived = (
                distance <= float(self.cfg.place_align_distance)
                and float(metrics["object_speed"]) <= self.pivot_arrival_speed
            )
            aligned = abs(bearing) <= self.near_align_exit_bearing_deg
            self.near_align_exit_count = (
                self.near_align_exit_count + 1 if aligned else 0
            )
            if arrived:
                self._release_pivot("arrived_slow")
            elif self.near_align_exit_count >= self.near_align_exit_required:
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
        if self.near_align_enter_count >= self.near_align_enter_required:
            self._start_pivot(1 if bearing >= 0.0 else -1)

    def _advance_weight(self) -> None:
        target = 1.0 if self.near_align_active else 0.0
        if self.near_align_blend_seconds <= 0.0:
            self.near_align_weight = target
            return
        max_delta = self.dt / self.near_align_blend_seconds
        self.near_align_weight += float(
            np.clip(target - self.near_align_weight, -max_delta, max_delta)
        )

    def _record_pivot_metrics(
        self, metrics: Mapping[str, Any], bearing: float
    ) -> None:
        if not isinstance(metrics, dict):
            return
        metrics.update(
            {
                "solve_close_bearing_deg": float(bearing),
                "solve_close_align_active": bool(self.near_align_active),
                "solve_close_align_weight": float(self.near_align_weight),
                "solve_close_pivot_direction": int(self.pivot_direction),
                "solve_close_pivot_candidate_index": int(
                    self.pivot_candidate_index
                ),
                "solve_close_pivot_candidate_validated": bool(
                    self.pivot_candidate_validated
                ),
                "solve_close_pivot_base_yaw_integral_rad": float(
                    self.pivot_base_yaw_integral
                ),
                "solve_close_pivot_box_yaw_integral_rad": float(
                    self.pivot_box_yaw_integral
                ),
                "solve_close_pivot_exit_reason": str(self.pivot_exit_reason),
            }
        )

    def select(
        self,
        observation: dict[str, np.ndarray],
        metrics: Mapping[str, Any],
        frame: CarryMocapFrame,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        base_yaw_delta, box_yaw_delta = self._yaw_deltas(frame)
        if self.phase != "transport":
            if (
                self.near_align_active
                or self.near_align_weight > 0.0
                or self.near_align_enter_count > 0
            ):
                self._reset_near_align(reset_cooldown=True)
            return super().select(observation, metrics, frame)

        rotation = matrix_from_quat(
            frame.robot_root_quaternion_wxyz.reshape(1, 4)
        )[0]
        bearing = _bearing_deg(
            rotation,
            frame.target_position_world[:2] - frame.object_position[:2],
        )
        distance = float(metrics["goal_distance"])
        self._update_latch(
            distance,
            bearing,
            metrics,
            frame,
            base_yaw_delta,
            box_yaw_delta,
        )
        self._advance_weight()
        normal_direction = 1 if bearing >= 0.0 else -1
        active_direction = self.pivot_direction or normal_direction
        normal_turn_weight = _transport_turn_weight(
            goal_distance=distance,
            bearing_deg=bearing,
            turn_boost_distance=float(
                getattr(self.cfg, "transport_turn_boost_distance", 1.2)
            ),
            place_distance=float(self.cfg.place_align_distance),
            heading_blend_full_deg=float(
                self.cfg.transport_heading_blend_full_deg
            ),
            far_max_turn_blend=float(self.cfg.transport_max_turn_blend),
            near_max_turn_blend=float(
                getattr(self.cfg, "transport_near_max_turn_blend", 0.8)
            ),
        )
        normal_latent = project_z(
            (1.0 - normal_turn_weight) * self.latents["transport"]
            + normal_turn_weight * self.normal_turn[normal_direction]
        )
        pivot_latent = project_z(
            self.pivot_candidates[active_direction][self.pivot_candidate_index]
        )
        latent = project_z(
            (1.0 - self.near_align_weight) * normal_latent
            + self.near_align_weight * pivot_latent
        )
        actor = np.concatenate(
            (
                observation["state"].copy(),
                observation["last_action"],
                observation["history_actor"],
            )
        )
        if actor.shape != (STATE_DIM + 29 + 372,):
            raise ValueError(f"unexpected solve-close actor shape: {actor.shape}")
        if self.near_align_active or self.near_align_weight > 1.0e-6:
            side = "left" if active_direction > 0 else "right"
            mode = f"transport_align_{side}_c{self.pivot_candidate_index}"
        elif normal_turn_weight < 1.0e-3:
            mode = "transport"
        else:
            mode = (
                "transport_left" if normal_direction > 0 else "transport_right"
            )
        self._record_pivot_metrics(metrics, bearing)
        return actor, latent.astype(np.float32, copy=False), mode

    def update(self, metrics: Mapping[str, Any]) -> None:
        if self.phase == "transport" and (
            self.near_align_active
            or self.near_align_weight > 1.0e-6
            or self.near_align_enter_count > 0
        ):
            # Keep all strong fall/contact-loss paths active while preventing a
            # distance-only transition into place during close-range alignment.
            adjusted = dict(metrics)
            adjusted["goal_distance"] = max(
                float(metrics["goal_distance"]),
                float(self.cfg.place_align_distance) + 1.0e-6,
            )
            super().update(adjusted)
            return
        super().update(metrics)


class SolveCloseCarryRuntime(CarryRuntime):
    """CarryRuntime with only the solve-close phase controller swapped in."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        exp_config = args[0] if args else kwargs.get("exp_config")
        if not isinstance(exp_config, Mapping):
            raise ValueError("solve-close runtime requires an experiment mapping")
        super().__init__(*args, **kwargs)
        if self.controller_profile != BASE_PROFILE:
            raise ValueError(
                f"solve-close requires phase_controller_profile={BASE_PROFILE!r}"
            )
        if str(exp_config.get("deployment_variant", "")).lower() != DEPLOYMENT_VARIANT:
            raise ValueError(
                f"solve-close requires deployment_variant={DEPLOYMENT_VARIANT!r}"
            )
        latent_dir = _resolve(exp_config["latent_dir"])
        actual_lift_hash = _sha256(latent_dir / "z_lift.pkl")
        if actual_lift_hash != EXPECTED_LIFT_SHA256:
            raise ValueError(
                "solve-close z_lift hash mismatch: "
                f"{actual_lift_hash} != {EXPECTED_LIFT_SHA256}"
            )
        candidates, normal_turn, names = load_solve_close_assets(latent_dir)
        self.controller = IntactLogmap166559744SolveCloseController(
            exp_config.get("phase", {}),
            exp_config.get("solve_close", {}),
            self.controller.latents,
            self.controller.dt,
            candidates,
            normal_turn,
            names,
        )
        self.controller_variant = "solve_close_midface_pivot_rolling_guard_v1"
        logger.info(
            "Carry solve-close enabled: topk2000 lift, near-align <= {:.2f}m, "
            "bearing {:.1f}->{:.1f}deg, candidates={}/{}".format(
                self.controller.near_align_distance,
                self.controller.near_align_enter_bearing_deg,
                self.controller.near_align_exit_bearing_deg,
                len(candidates[-1]),
                len(candidates[1]),
            )
        )


def validate_task_config(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"{path} must contain a mapping")
    if str(config.get("phase_controller_profile", "")).lower() != BASE_PROFILE:
        raise RuntimeError(f"{path} does not select {BASE_PROFILE}")
    if str(config.get("deployment_variant", "")).lower() != DEPLOYMENT_VARIANT:
        raise RuntimeError(f"{path} does not select {DEPLOYMENT_VARIANT}")
    latent_dir = _resolve(config["latent_dir"])
    actual_lift_hash = _sha256(latent_dir / "z_lift.pkl")
    if actual_lift_hash != EXPECTED_LIFT_SHA256:
        raise RuntimeError(
            f"solve-close z_lift hash mismatch: {actual_lift_hash}"
        )
    latents = load_phase_latents(latent_dir)
    get_up_path = config.get("get_up_latent")
    if get_up_path is not None:
        latents["get_up"] = load_get_up_latent(_resolve(get_up_path))
    candidates, normal_turn, names = load_solve_close_assets(latent_dir)
    controller = IntactLogmap166559744SolveCloseController(
        config.get("phase", {}),
        config.get("solve_close", {}),
        latents,
        0.02,
        candidates,
        normal_turn,
        names,
    )
    print(
        "PASS solve-close controller: profile={} variant={} distance={:.2f}m "
        "bearing={:.1f}->{:.1f}deg pivot_max={:.1f}s candidates={}/{} "
        "lift_sha256={}".format(
            BASE_PROFILE,
            DEPLOYMENT_VARIANT,
            controller.near_align_distance,
            controller.near_align_enter_bearing_deg,
            controller.near_align_exit_bearing_deg,
            controller.pivot_max_steps * controller.dt,
            len(candidates[-1]),
            len(candidates[1]),
            actual_lift_hash,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the isolated 166.560M solve-close Carry runtime"
    )
    parser.add_argument("--task-config", type=Path, required=True)
    args = parser.parse_args()
    validate_task_config(args.task_config.expanduser().resolve())


if __name__ == "__main__":
    main()
