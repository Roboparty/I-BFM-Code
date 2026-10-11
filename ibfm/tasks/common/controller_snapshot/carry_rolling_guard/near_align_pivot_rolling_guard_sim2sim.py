"""Closed-loop Carry pivot with rolling progress and height/upright guards.

This wrapper extends the bundled Carry pivot controller.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PIVOT_V1 = (
    HERE.parent
    / "carry_pivot"
    / "near_align_pivot_closedloop_sim2sim.py"
)


def _load_pivot_v1():
    pivot_dir = str(PIVOT_V1.parent)
    if pivot_dir not in sys.path:
        sys.path.insert(0, pivot_dir)
    spec = importlib.util.spec_from_file_location(
        "_intact150_pivot_closedloop_v1", PIVOT_V1
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pivot-v1 controller: {PIVOT_V1}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _install_rolling_guard(module: Any) -> None:
    parent = module.PhaseController

    class RollingGuardPivotController(parent):
        def __init__(self, args, *, dt: float) -> None:
            self.pivot_stall_window_steps = max(
                2,
                round(_env_float("UFO_PIVOT_STALL_WINDOW_SECONDS", 0.50) / dt),
            )
            self.pivot_min_recent_yaw = _env_float(
                "UFO_PIVOT_MIN_RECENT_YAW_RAD", 0.03
            )
            self.pivot_min_bearing_improvement = _env_float(
                "UFO_PIVOT_MIN_BEARING_IMPROVEMENT_DEG", 3.0
            )
            self.pivot_max_height_drift = _env_float(
                "UFO_PIVOT_MAX_HEIGHT_DRIFT_M", 0.08
            )
            self.pivot_min_box_upright_guard = _env_float(
                "UFO_PIVOT_MIN_BOX_UPRIGHT", 0.85
            )
            self._guard_bearing_deg = 0.0
            self._pivot_start_box_height: float | None = None
            self._recent_base_yaw: deque[float] = deque(
                maxlen=self.pivot_stall_window_steps
            )
            self._recent_box_yaw: deque[float] = deque(
                maxlen=self.pivot_stall_window_steps
            )
            self._recent_abs_bearing: deque[float] = deque(
                maxlen=self.pivot_stall_window_steps
            )
            super().__init__(args, dt=dt)

        def _clear_recent_guard(self) -> None:
            self._recent_base_yaw.clear()
            self._recent_box_yaw.clear()
            self._recent_abs_bearing.clear()

        def reset(self) -> None:
            super().reset()
            self._pivot_start_box_height = None
            self._clear_recent_guard()

        def _start_pivot(self, direction: int) -> None:
            super()._start_pivot(direction)
            self._pivot_start_box_height = None
            self._clear_recent_guard()

        def _release_pivot(self, reason: str) -> None:
            super()._release_pivot(reason)
            self._pivot_start_box_height = None
            self._clear_recent_guard()

        def _switch_candidate(self, reason: str) -> None:
            super()._switch_candidate(reason)
            self._clear_recent_guard()

        def _update_closed_loop_latch(
            self,
            distance: float,
            bearing: float,
            metrics: dict[str, float | bool],
        ) -> None:
            self._guard_bearing_deg = float(bearing)
            super()._update_closed_loop_latch(distance, bearing, metrics)

        def _update_feedback(self, metrics: dict[str, float | bool]) -> None:
            if not self.near_align_active or self.pivot_direction == 0:
                return
            if self._pivot_start_box_height is None:
                self._pivot_start_box_height = float(metrics["object_height"])

            previous_index = self.pivot_candidate_index
            super()._update_feedback(metrics)
            if (
                not self.near_align_active
                or self.pivot_candidate_index != previous_index
            ):
                return

            direction = self.pivot_direction
            self._recent_base_yaw.append(
                direction * float(metrics["probe_base_yaw_rate"]) * self.dt
            )
            self._recent_box_yaw.append(
                direction * float(metrics["probe_box_yaw_rate"]) * self.dt
            )
            self._recent_abs_bearing.append(abs(self._guard_bearing_deg))

            if len(self._recent_base_yaw) < self.pivot_stall_window_steps:
                return

            height_drift = abs(
                float(metrics["object_height"]) - self._pivot_start_box_height
            )
            box_upright = float(metrics["box_upright"])
            recent_yaw = min(
                sum(self._recent_base_yaw), sum(self._recent_box_yaw)
            )
            bearing_improvement = (
                self._recent_abs_bearing[0] - self._recent_abs_bearing[-1]
            )
            candidate = self.pivot_candidate_names[direction][
                self.pivot_candidate_index
            ]

            if height_drift > self.pivot_max_height_drift:
                print(
                    "[pivot-rolling-guard] reject "
                    f"candidate={candidate} reason=height_drift "
                    f"value={height_drift:.3f}m limit={self.pivot_max_height_drift:.3f}m",
                    flush=True,
                )
                self._switch_candidate("height_drift")
            elif box_upright < self.pivot_min_box_upright_guard:
                print(
                    "[pivot-rolling-guard] reject "
                    f"candidate={candidate} reason=box_upright "
                    f"value={box_upright:.3f} limit={self.pivot_min_box_upright_guard:.3f}",
                    flush=True,
                )
                self._switch_candidate("box_upright")
            elif (
                self.pivot_candidate_validated
                and recent_yaw < self.pivot_min_recent_yaw
            ):
                print(
                    "[pivot-rolling-guard] reject "
                    f"candidate={candidate} reason=recent_yaw_stall "
                    f"value={recent_yaw:.3f}rad limit={self.pivot_min_recent_yaw:.3f}rad",
                    flush=True,
                )
                self._switch_candidate("recent_yaw_stall")
            elif (
                self.pivot_candidate_validated
                and bearing_improvement < self.pivot_min_bearing_improvement
            ):
                print(
                    "[pivot-rolling-guard] reject "
                    f"candidate={candidate} reason=bearing_stall "
                    f"value={bearing_improvement:.2f}deg "
                    f"limit={self.pivot_min_bearing_improvement:.2f}deg",
                    flush=True,
                )
                self._switch_candidate("bearing_stall")

    module.PhaseController = RollingGuardPivotController


def main() -> None:
    closed_loop = _load_pivot_v1()
    near_align = closed_loop.pivot_v1._load_near_align_v1()
    near_align._install_initialise_state_compatibility()
    module = near_align._load_frozen_controller()
    near_align._install_near_align_controller(module)
    closed_loop._install_metric_probe(module)
    closed_loop._install_closed_loop_controller(module)
    _install_rolling_guard(module)
    args = module.parse_args()
    print(
        "[carry-pivot] rolling guard enabled: window=0.5s, "
        "recent yaw>=0.03rad, bearing improvement>=3deg, "
        "height drift<=0.08m, box upright>=0.85",
        flush=True,
    )
    module.run(args)


if __name__ == "__main__":
    main()
