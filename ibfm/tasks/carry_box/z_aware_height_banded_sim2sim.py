#!/usr/bin/env python3
"""Height-banded 3D Carry controller around the frozen 166.560M bundle."""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "common"
ROLLING_GUARD = (
    SOURCE
    / "controller_snapshot/carry_rolling_guard/"
    "near_align_pivot_rolling_guard_sim2sim.py"
)


def _load_rolling_guard() -> Any:
    spec = importlib.util.spec_from_file_location("_zaware_v2_rolling_guard", ROLLING_GUARD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen controller: {ROLLING_GUARD}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _install_platform_collision_filter(module: Any) -> None:
    """Keep box-support contacts while excluding robot-support contacts."""

    original_build_model = module._build_model

    def build_model_with_filtered_supports(*args: Any, **kwargs: Any):
        model = original_build_model(*args, **kwargs)
        box_id = model.geom(module.BOX_GEOM_NAME).id
        model.geom_contype[box_id] = int(model.geom_contype[box_id]) | 2
        model.geom_conaffinity[box_id] = int(model.geom_conaffinity[box_id]) | 2
        support_ids = []
        for support_name in ("initial_box_support", "target_box_support"):
            support_id = model.geom(support_name).id
            model.geom_contype[support_id] = 2
            model.geom_conaffinity[support_id] = 2
            support_ids.append(support_id)

        def collision_enabled(first: int, second: int) -> bool:
            return bool(
                int(model.geom_contype[first]) & int(model.geom_conaffinity[second])
                and int(model.geom_contype[second])
                & int(model.geom_conaffinity[first])
            )

        box_body_id = int(model.geom_bodyid[box_id])
        support_body_ids = {int(model.geom_bodyid[index]) for index in support_ids}
        robot_geom_ids = [
            index
            for index in range(model.ngeom)
            if int(model.geom_bodyid[index]) not in {0, box_body_id, *support_body_ids}
        ]
        if not all(collision_enabled(box_id, support_id) for support_id in support_ids):
            raise RuntimeError("box-support collision filter is not enabled")
        if any(
            collision_enabled(robot_id, support_id)
            for robot_id in robot_geom_ids
            for support_id in support_ids
        ):
            raise RuntimeError("robot-support collision filter is not disabled")
        print(
            "[collision-filter] box-support=on robot-support=off",
            flush=True,
        )
        return model

    module._build_model = build_model_with_filtered_supports


def _load_controller_stack(rolling_guard: Any) -> Any:
    closed_loop = rolling_guard._load_pivot_v1()
    near_align = closed_loop.pivot_v1._load_near_align_v1()
    near_align._install_initialise_state_compatibility()
    module = near_align._load_frozen_controller()
    near_align._install_near_align_controller(module)
    closed_loop._install_metric_probe(module)
    closed_loop._install_closed_loop_controller(module)
    rolling_guard._install_rolling_guard(module)
    return module


def _install_random_height_reset(module: Any) -> None:
    """Turn the existing R reset into a reproducible random-height trial reset."""

    minimum = _env_float("UFO_Z3D_RESET_HEIGHT_MIN", 0.15)
    maximum = _env_float("UFO_Z3D_RESET_HEIGHT_MAX", 0.80)
    if not 0.0 < minimum < maximum:
        raise ValueError("random reset heights must satisfy 0 < min < max")
    seed = int(os.environ.get("UFO_Z3D_RESET_SEED", "166559744"))
    target = np.asarray(
        [
            _env_float("UFO_Z3D_TARGET_X", 2.50),
            _env_float("UFO_Z3D_TARGET_Y", 0.0),
            _env_float("UFO_Z3D_TARGET_BOX_Z", 0.65),
        ],
        dtype=np.float64,
    )
    success_radius = _env_float("UFO_Z3D_RESET_SUCCESS_RADIUS", 0.20)
    output_path = Path(
        os.environ.get(
            "UFO_Z3D_RESET_SUMMARY_PATH",
            str(HERE / "interactive/random_height_trials.json"),
        )
    )
    rng = np.random.default_rng(seed)
    original_initialise = module._initialise_state
    original_set_height = module._set_initial_box_center_height
    state: dict[str, Any] = {
        "active": False,
        "episode": -1,
        "height": None,
        "model": None,
        "data": None,
        "qpos_address": None,
        "had_bilateral_contact": False,
        "max_lift": 0.0,
        "reached_transport": False,
        "results": [],
    }

    def write_summary() -> None:
        results = state["results"]
        pickup_successes = sum(bool(row["pickup_success"]) for row in results)
        placement_successes = sum(bool(row["placement_success"]) for row in results)
        errors = [float(row["final_error_3d_m"]) for row in results]
        payload = {
            "seed": seed,
            "height_range_m": [minimum, maximum],
            "target_xyz_m": target.tolist(),
            "success_definition": f"final object-center 3D error < {success_radius:.3f} m",
            "episodes_completed": len(results),
            "pickup_success_definition": "controller reached transport after bilateral contact and lift",
            "pickup_successes": pickup_successes,
            "pickup_success_rate": pickup_successes / len(results) if results else None,
            "placement_successes": placement_successes,
            "placement_success_rate": (
                placement_successes / len(results) if results else None
            ),
            "mean_final_error_3d_m": float(np.mean(errors)) if errors else None,
            "trials": results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def finish_current(reason: str) -> None:
        if not state["active"]:
            return
        data = state["data"]
        address = int(state["qpos_address"])
        position = np.asarray(data.qpos[address : address + 3], dtype=np.float64).copy()
        error = float(np.linalg.norm(position - target))
        placement_success = error < success_radius
        pickup_success = bool(state["reached_transport"])
        row = {
            "episode": int(state["episode"]),
            "initial_box_z_m": float(state["height"]),
            "final_box_xyz_m": position.tolist(),
            "final_error_3d_m": error,
            "had_bilateral_contact": bool(state["had_bilateral_contact"]),
            "max_lift": float(state["max_lift"]),
            "pickup_success": pickup_success,
            "placement_success": placement_success,
            "finished_by": reason,
        }
        state["results"].append(row)
        state["active"] = False
        write_summary()
        total = len(state["results"])
        pickup_count = sum(bool(item["pickup_success"]) for item in state["results"])
        placement_count = sum(
            bool(item["placement_success"]) for item in state["results"]
        )
        print(
            f"[height-test] result episode={row['episode']} initial_z={row['initial_box_z_m']:.3f}m "
            f"pickup={int(pickup_success)} error={error:.3f}m "
            f"placement={int(placement_success)} "
            f"cumulative_pickup={pickup_count}/{total} "
            f"cumulative_placement={placement_count}/{total}",
            flush=True,
        )

    def initialise_with_result(*args: Any, **kwargs: Any):
        finish_current("reset")
        return original_initialise(*args, **kwargs)

    def set_random_height(
        model: Any,
        data: Any,
        *,
        qpos_address: int,
        qvel_address: int,
        center_height: float | None,
    ) -> None:
        del center_height
        sampled_height = float(rng.uniform(minimum, maximum))
        original_set_height(
            model,
            data,
            qpos_address=qpos_address,
            qvel_address=qvel_address,
            center_height=sampled_height,
        )
        try:
            support_id = model.geom("initial_box_support").id
            box_geom_id = model.geom(module.BOX_GEOM_NAME).id
        except KeyError as exc:
            raise RuntimeError(
                "random-height test scene must contain initial_box_support"
            ) from exc
        rotation = np.asarray(data.geom_xmat[box_geom_id], dtype=np.float64).reshape(3, 3)
        vertical_extent = float(
            np.dot(np.abs(rotation[2]), np.asarray(model.geom_size[box_geom_id]))
        )
        box_bottom = float(data.geom_xpos[box_geom_id, 2] - vertical_extent)
        support_body_id = int(model.geom_bodyid[support_id])
        support_mocap_id = int(model.body_mocapid[support_body_id])
        if support_mocap_id < 0:
            raise RuntimeError("initial_box_support must belong to a mocap body")
        data.mocap_pos[support_mocap_id, 2] = (
            box_bottom - float(model.geom_size[support_id, 2])
        )
        module.mujoco.mj_forward(model, data)
        if _env_enabled("UFO_Z3D_DEBUG_COLLISION"):
            support_top = float(
                data.geom_xpos[support_id, 2] + model.geom_size[support_id, 2]
            )
            print(
                "[collision-debug] "
                f"box_bottom={box_bottom:.6f} support_top={support_top:.6f} "
                f"box_mask={int(model.geom_contype[box_geom_id])}/"
                f"{int(model.geom_conaffinity[box_geom_id])} "
                f"support_mask={int(model.geom_contype[support_id])}/"
                f"{int(model.geom_conaffinity[support_id])}",
                flush=True,
            )
        os.environ["UFO_Z3D_INITIAL_BOX_Z"] = str(sampled_height)
        state.update(
            {
                "active": True,
                "episode": int(state["episode"]) + 1,
                "height": sampled_height,
                "model": model,
                "data": data,
                "qpos_address": qpos_address,
                "had_bilateral_contact": False,
                "max_lift": 0.0,
                "reached_transport": False,
            }
        )
        profile = (
            target[2] >= 0.50
            and (0.30 <= sampled_height < 0.50 or sampled_height >= 0.65)
        )
        print(
            f"[height-test] episode={state['episode']} initial_z={sampled_height:.3f}m "
            f"target={target.tolist()} profile={'z-aware' if profile else 'baseline-fallback'}",
            flush=True,
        )

    def finish_at_exit() -> None:
        try:
            finish_current("viewer_closed")
        except Exception as exc:  # pragma: no cover - last-resort persistence path
            print(f"[height-test] failed to finalize session: {exc}", flush=True)

    module._initialise_state = initialise_with_result
    module._set_initial_box_center_height = set_random_height
    module._z3d_height_trial_state = state
    atexit.register(finish_at_exit)


def _install_height_banded_controller(module: Any) -> None:
    parent = module.PhaseController

    class HeightBandedController(parent):
        def __init__(self, args: Any, *, dt: float) -> None:
            latent_dir = Path(args.latent_dir)
            self.height_latents_np = {
                skill: {
                    band: np.asarray(
                        joblib.load(latent_dir / f"z_{skill}_{band}.pkl"),
                        dtype=np.float32,
                    ).reshape(1, -1)
                    for band in ("low", "mid", "high")
                }
                for skill in ("acquire", "lift", "transport")
            }
            self.place_latents_np = {
                band: np.asarray(
                    joblib.load(latent_dir / f"z_place_{band}.pkl"),
                    dtype=np.float32,
                ).reshape(1, -1)
                for band in ("low", "mid", "high")
            }
            self.release_lower_near_np = None
            if _env_enabled("UFO_Z3D_RELEASE_XY_ADAPTIVE"):
                self.release_lower_near_np = np.asarray(
                    joblib.load(latent_dir / "z_release_lower_near.pkl"),
                    dtype=np.float32,
                ).reshape(1, -1)
                if (
                    self.release_lower_near_np.shape != (1, 256)
                    or not np.isfinite(self.release_lower_near_np).all()
                ):
                    raise ValueError("Invalid release-only lowering latent")
            self.initial_box_z = _env_float("UFO_Z3D_INITIAL_BOX_Z", 0.186)
            self.target_box_z = _env_float("UFO_Z3D_TARGET_BOX_Z", 0.186)
            self.xy_align_distance = _env_float(
                "UFO_Z3D_XY_ALIGN_DISTANCE_OVERRIDE",
                _env_float("UFO_Z3D_XY_ALIGN_DISTANCE", 0.20),
            )
            self.xy_align_exit_distance = _env_float(
                "UFO_Z3D_XY_ALIGN_EXIT_DISTANCE_OVERRIDE",
                _env_float("UFO_Z3D_XY_ALIGN_EXIT_DISTANCE", 0.22),
            )
            if self.xy_align_exit_distance <= self.xy_align_distance:
                raise ValueError(
                    "vertical-lowering exit distance must exceed its entry distance"
                )
            self.place_entry_distance = _env_float("UFO_Z3D_PLACE_ENTRY_DISTANCE", 0.22)
            self.place_entry_speed = _env_float(
                "UFO_Z3D_PLACE_ENTRY_SPEED_OVERRIDE",
                _env_float("UFO_Z3D_PLACE_ENTRY_SPEED", 0.50),
            )
            self.terminal_radius = _env_float("UFO_Z3D_TERMINAL_RADIUS", 0.195)
            self.terminal_exit_radius = _env_float(
                "UFO_Z3D_TERMINAL_EXIT_RADIUS", 0.20
            )
            if self.terminal_exit_radius <= self.terminal_radius:
                raise ValueError(
                    "terminal exit radius must exceed terminal success radius"
                )
            self.terminal_speed = _env_float("UFO_Z3D_TERMINAL_SPEED", 0.15)
            self.terminal_steps_required = max(
                1, int(os.environ.get("UFO_Z3D_TERMINAL_STEPS", "10"))
            )
            self.terminal_exit_steps_required = max(
                1, int(os.environ.get("UFO_Z3D_TERMINAL_EXIT_STEPS", "1"))
            )
            self.early_success_complete_steps = int(
                os.environ.get("UFO_Z3D_EARLY_SUCCESS_COMPLETE_STEPS", "0")
            )
            if self.early_success_complete_steps < 0:
                raise ValueError(
                    "UFO_Z3D_EARLY_SUCCESS_COMPLETE_STEPS must be non-negative"
                )
            self.max_lift_switch_height = _env_float(
                "UFO_Z3D_MAX_LIFT_SWITCH_HEIGHT", 0.78
            )
            self.lift_hysteresis_band = _env_float(
                "UFO_Z3D_LIFT_HYSTERESIS_BAND", 0.50
            )
            self.pickup_lift_height = _env_float(
                "UFO_Z3D_PICKUP_LIFT_HEIGHT", 0.35
            )
            if self.pickup_lift_height <= 0.0:
                raise ValueError("UFO_Z3D_PICKUP_LIFT_HEIGHT must be positive")
            self.min_transport_height = _env_float(
                "UFO_Z3D_MIN_TRANSPORT_HEIGHT", 0.55
            )
            if self.min_transport_height <= 0.0:
                raise ValueError("UFO_Z3D_MIN_TRANSPORT_HEIGHT must be positive")
            self.transport_fixed_band = os.environ.get(
                "UFO_Z3D_TRANSPORT_FIXED_BAND", "original"
            )
            if self.transport_fixed_band not in {"auto", "original", "low", "mid", "high"}:
                raise ValueError(
                    "UFO_Z3D_TRANSPORT_FIXED_BAND must be auto/original/low/mid/high"
                )
            self.dynamic_height_reset = _env_enabled("UFO_Z3D_RANDOM_HEIGHT_RESET")
            self._reacquire_from_current_height = False
            self._strict_thresholds = {
                "place_switch_distance": float(args.place_switch_distance),
                "place_align_distance": float(args.place_align_distance),
                "place_drift_distance": float(args.place_drift_distance),
                "place_abort_distance": float(args.place_abort_distance),
                "place_success_distance": float(args.place_success_distance),
                "place_success_height": float(args.place_success_height),
                "place_release_height": float(args.place_release_height),
                "place_enter_steps": int(args.place_enter_steps),
                "phase_hold_steps": int(args.phase_hold_steps),
            }
            self.use_height_profile = (
                self.target_box_z >= 0.50
                and (
                    0.30 <= self.initial_box_z < 0.50
                    or self.initial_box_z >= 0.65
                )
            )
            self._configure_thresholds(args)
            self._terminal_steps = 0
            self._terminal_latched = False
            self._complete_success_steps = 0
            self._place_lowering_active = False
            self._pickup_reference_height = self.initial_box_z
            self._pickup_reference_tracking = True
            super().__init__(args, dt=dt)

        def reset(self) -> None:
            if getattr(self, "dynamic_height_reset", False):
                self.initial_box_z = _env_float(
                    "UFO_Z3D_INITIAL_BOX_Z", self.initial_box_z
                )
                self.use_height_profile = (
                    self.target_box_z >= 0.50
                    and (
                        0.30 <= self.initial_box_z < 0.50
                        or self.initial_box_z >= 0.65
                    )
                )
                self._configure_thresholds(self.args)
            super().reset()
            self._terminal_steps = 0
            self._terminal_latched = False
            self._complete_success_steps = 0
            self._reacquire_from_current_height = False
            self._place_lowering_active = False
            self._pickup_reference_height = self.initial_box_z
            self._pickup_reference_tracking = True

        def _reset_pickup_reference(self, object_height: float) -> None:
            self._pickup_reference_height = float(object_height)
            self._pickup_reference_tracking = True

        def _fallback_pickup_lift(
            self, metrics: dict[str, float | bool]
        ) -> float:
            object_height = float(metrics["object_height"])
            if self.phase == "lift":
                if bool(metrics["bilateral_contact"]):
                    self._pickup_reference_tracking = False
                elif self._pickup_reference_tracking:
                    self._pickup_reference_height = min(
                        self._pickup_reference_height, object_height
                    )
                else:
                    self._reset_pickup_reference(object_height)
            return float(
                np.clip(
                    (object_height - self._pickup_reference_height)
                    / self.pickup_lift_height,
                    0.0,
                    1.0,
                )
            )

        def _configure_thresholds(self, args: Any) -> None:
            if self.use_height_profile:
                for name, value in self._strict_thresholds.items():
                    setattr(args, name, value)
                return
            baseline = {
                "place_switch_distance": 0.40,
                "place_align_distance": 0.25,
                "place_drift_distance": 0.45,
                "place_abort_distance": 0.80,
                "place_success_distance": min(
                    self._strict_thresholds["place_success_distance"],
                    self.terminal_radius,
                ),
                "place_success_height": 0.05,
                "place_release_height": 0.08,
                "place_enter_steps": 5,
                "phase_hold_steps": 5,
            }
            for name, value in baseline.items():
                setattr(args, name, value)

        def _reject_nonterminal_parent_completion(
            self, previous_phase: str, previous_phase_steps: int
        ) -> None:
            """Only the common strict 3D latch may declare completion."""
            if self.phase != "complete" or self._terminal_latched:
                return
            self.phase = previous_phase
            self.phase_steps = previous_phase_steps + 1
            self.placed_steps = 0
            self.completion_lost_steps = 0

        @staticmethod
        def _centres(skill: str) -> tuple[float, float, float]:
            if skill == "transport":
                return 0.59, 0.75, 0.96
            return 0.325, 0.61, 0.91

        def _tensor(self, values: np.ndarray, reference: torch.Tensor) -> torch.Tensor:
            return torch.as_tensor(values, device=reference.device, dtype=reference.dtype)

        def _banded_latent(
            self,
            policy: Any,
            skill: str,
            height: float,
            reference: torch.Tensor,
        ) -> tuple[torch.Tensor, str]:
            centres = self._centres(skill)
            values = self.height_latents_np[skill]
            if height <= centres[0]:
                return self._tensor(values["low"], reference), "low"
            if height >= centres[2]:
                return self._tensor(values["high"], reference), "high"
            if height <= centres[1]:
                lower_name, upper_name, lower, upper = "low", "mid", centres[0], centres[1]
            else:
                lower_name, upper_name, lower, upper = "mid", "high", centres[1], centres[2]
            weight = float(np.clip((height - lower) / (upper - lower), 0.0, 1.0))
            lower_z = self._tensor(values[lower_name], reference)
            upper_z = self._tensor(values[upper_name], reference)
            return policy.project_z((1.0 - weight) * lower_z + weight * upper_z), f"{lower_name}_{upper_name}"

        def _place_latent(
            self, policy: Any, lift: float, reference: torch.Tensor
        ) -> tuple[torch.Tensor, str]:
            values = self.place_latents_np
            high, mid, low = 0.75, 0.45, 0.15
            if lift >= high:
                return self._tensor(values["high"], reference), "high"
            if lift >= mid:
                weight = (high - lift) / (high - mid)
                first = self._tensor(values["high"], reference)
                second = self._tensor(values["mid"], reference)
                return policy.project_z((1.0 - weight) * first + weight * second), "high_mid"
            if lift >= low:
                weight = (mid - lift) / (mid - low)
                first = self._tensor(values["mid"], reference)
                second = self._tensor(values["low"], reference)
                return policy.project_z((1.0 - weight) * first + weight * second), "mid_low"
            return self._tensor(values["low"], reference), "low"

        def latent(
            self,
            policy: Any,
            observation: dict[str, torch.Tensor],
            metrics: dict[str, float | bool],
            *,
            base_rotation: np.ndarray,
            box_direction: np.ndarray,
            goal_direction: np.ndarray,
            z: dict[str, Any],
        ):
            if not self.use_height_profile:
                return super().latent(
                    policy,
                    observation,
                    metrics,
                    base_rotation=base_rotation,
                    box_direction=box_direction,
                    goal_direction=goal_direction,
                    z=z,
                )
            if self.phase != "place":
                self._place_lowering_active = False
            if self.phase == "lift":
                if not bool(metrics["bilateral_contact"]):
                    acquire_height = (
                        float(metrics["object_height"])
                        if self._reacquire_from_current_height
                        else self.initial_box_z
                    )
                    latent, band = self._banded_latent(
                        policy,
                        "acquire",
                        acquire_height,
                        z["lift"],
                    )
                    return observation, latent, f"acquire_{band}"
                latent, band = self._banded_latent(
                    policy, "lift", float(metrics["object_height"]), z["lift"]
                )
                return observation, latent, f"lift_{band}"

            if self.phase == "transport":
                if self.transport_fixed_band == "original":
                    return super().latent(
                        policy,
                        observation,
                        metrics,
                        base_rotation=base_rotation,
                        box_direction=box_direction,
                        goal_direction=goal_direction,
                        z=z,
                    )
                desired_height = max(0.68, self.target_box_z + 0.06)
                if self.transport_fixed_band == "auto":
                    latent, band = self._banded_latent(
                        policy, "transport", desired_height, z["transport"]
                    )
                else:
                    band = self.transport_fixed_band
                    latent = self._tensor(
                        self.height_latents_np["transport"][band], z["transport"]
                    )
                adjusted_z = dict(z)
                adjusted_z["transport"] = latent
                result = super().latent(
                    policy,
                    observation,
                    metrics,
                    base_rotation=base_rotation,
                    box_direction=box_direction,
                    goal_direction=goal_direction,
                    z=adjusted_z,
                )
                return result[0], result[1], f"{result[2]}_z{band}"

            if self.phase == "place" and self.release_started_step is None:
                place_observation = module._goal_lookahead_observation(
                    observation, self.args.align_goal_lookahead
                )
                goal_distance = float(metrics["goal_distance"])
                if self._place_lowering_active:
                    if goal_distance > self.xy_align_exit_distance:
                        self._place_lowering_active = False
                elif goal_distance <= self.xy_align_distance:
                    self._place_lowering_active = True
                if not self._place_lowering_active:
                    return place_observation, z["align"], "place_xy_align"
                lower, band = self._place_latent(
                    policy, float(metrics["lift"]), z["lower"]
                )
                blend = module._smooth_place_blend(
                    self.phase_steps, self.place_settle_steps, self.place_lower_steps
                )
                latent = policy.project_z((1.0 - blend) * z["align"] + blend * lower)
                return place_observation, latent, f"place_z_{band}_{blend:.2f}"

            if (
                self.phase == "place"
                and self.release_started_step is not None
                and self.release_lower_near_np is not None
                and float(metrics["goal_distance"]) <= self.xy_align_distance
            ):
                adjusted_z = dict(z)
                adjusted_z["lower"] = self._tensor(
                    self.release_lower_near_np, z["lower"]
                )
                return super().latent(
                    policy,
                    observation,
                    metrics,
                    base_rotation=base_rotation,
                    box_direction=box_direction,
                    goal_direction=goal_direction,
                    z=adjusted_z,
                )

            return super().latent(
                policy,
                observation,
                metrics,
                base_rotation=base_rotation,
                box_direction=box_direction,
                goal_direction=goal_direction,
                z=z,
            )

        def update(self, metrics: dict[str, float | bool], *, goal_height: float) -> None:
            trial = getattr(module, "_z3d_height_trial_state", None)
            if trial is not None and trial.get("active"):
                trial["had_bilateral_contact"] = bool(
                    trial["had_bilateral_contact"] or metrics["bilateral_contact"]
                )
                trial["max_lift"] = max(
                    float(trial["max_lift"]), float(metrics["lift"])
                )
                trial["reached_transport"] = bool(
                    trial["reached_transport"] or self.phase == "transport"
                )
            error_3d = float(
                np.hypot(
                    float(metrics["goal_distance"]),
                    float(metrics["object_height"]) - goal_height,
                )
            )
            if self._terminal_latched:
                outside_exit_radius = error_3d > self.terminal_exit_radius
                self.completion_lost_steps = (
                    self.completion_lost_steps + 1 if outside_exit_radius else 0
                )
                if self.completion_lost_steps < self.terminal_exit_steps_required:
                    self._complete_success_steps = (
                        self._complete_success_steps + 1
                        if error_3d < self.terminal_exit_radius
                        else 0
                    )
                    self.phase = "complete"
                    self.phase_steps += 1
                    if (
                        self.early_success_complete_steps > 0
                        and self._complete_success_steps
                        >= self.early_success_complete_steps
                    ):
                        self.args.max_steps = 1
                        print(
                            "[z-aware-v2] early success after "
                            f"{self._complete_success_steps} complete steps "
                            f"error_3d={error_3d:.3f}m",
                            flush=True,
                        )
                    return

                recovery_phase = module._box_recovery_phase(
                    float(metrics["horizontal_distance"]),
                    self.args.approach_switch_distance,
                )
                self._terminal_latched = False
                self._terminal_steps = 0
                self._complete_success_steps = 0
                self.phase = recovery_phase
                self.phase_steps = 0
                self.recovery_count += 1
                self.ready_steps = 0
                self.lost_steps = 0
                self.place_enter_steps = 0
                self.place_drift_steps = 0
                self.placed_steps = 0
                self.failure_steps = 0
                self.completion_lost_steps = 0
                self.release_started_step = None
                self._reset_pickup_reference(float(metrics["object_height"]))
                if hasattr(self, "_reset_near_align"):
                    self._reset_near_align()
                print(
                    "[z-aware-v2] terminal latch released "
                    f"error_3d={error_3d:.3f}m recovery={recovery_phase}",
                    flush=True,
                )
                return

            terminal_ready = (
                error_3d < self.terminal_radius
                and float(metrics["object_speed"]) <= self.terminal_speed
                and float(metrics["box_upright"]) >= 0.90
            )
            self._terminal_steps = self._terminal_steps + 1 if terminal_ready else 0
            if self._terminal_steps >= self.terminal_steps_required:
                self._terminal_latched = True
                self._complete_success_steps = 1
                self.completion_lost_steps = 0
                self.phase = "complete"
                self.phase_steps = 0
                return

            if not self.use_height_profile:
                previous_phase = self.phase
                previous_phase_steps = self.phase_steps
                adjusted = dict(metrics)
                use_pickup_lift = self.target_box_z >= 0.50
                if use_pickup_lift and self.phase in {"lift", "transport"}:
                    adjusted["lift"] = self._fallback_pickup_lift(metrics)
                if use_pickup_lift and self.phase == "lift":
                    required_transport_height = min(
                        max(self.min_transport_height, self.target_box_z),
                        self.max_lift_switch_height,
                    )
                    if float(metrics["object_height"]) < required_transport_height:
                        adjusted["lift"] = min(
                            float(adjusted["lift"]),
                            float(self.args.hold_switch_lift) - 1.0e-6,
                        )
                super().update(adjusted, goal_height=goal_height)
                self._reject_nonterminal_parent_completion(
                    previous_phase, previous_phase_steps
                )
                if (
                    use_pickup_lift
                    and self.phase == "lift"
                    and previous_phase != "lift"
                ):
                    self._reset_pickup_reference(float(metrics["object_height"]))
                return

            adjusted = dict(metrics)
            required_height = min(
                goal_height + float(self.args.hold_switch_lift) * 0.35,
                self.max_lift_switch_height,
            )
            adjusted["lift"] = float(
                np.clip(
                    float(self.args.hold_switch_lift)
                    + (float(metrics["object_height"]) - required_height)
                    / self.lift_hysteresis_band,
                    0.0,
                    1.0,
                )
            )
            if self.phase == "transport":
                ready_for_place = (
                    float(metrics["goal_distance"]) <= self.place_entry_distance
                    and float(metrics["object_speed"]) <= self.place_entry_speed
                )
                if ready_for_place and hasattr(self, "_reset_near_align"):
                    self._reset_near_align()
                elif not ready_for_place:
                    adjusted["goal_distance"] = max(
                        float(metrics["goal_distance"]),
                        float(self.args.place_align_distance) + 1.0e-6,
                    )
            was_transport = self.phase == "transport"
            previous_phase = self.phase
            previous_phase_steps = self.phase_steps
            super().update(adjusted, goal_height=goal_height)
            self._reject_nonterminal_parent_completion(
                previous_phase, previous_phase_steps
            )
            if was_transport or self.phase == "transport":
                self._reacquire_from_current_height = True
            if trial is not None and trial.get("active"):
                trial["reached_transport"] = bool(
                    trial["reached_transport"] or self.phase == "transport"
                )

    module.PhaseController = HeightBandedController


def main() -> None:
    rolling_guard = _load_rolling_guard()
    random_height_reset = _env_enabled("UFO_Z3D_RANDOM_HEIGHT_RESET")
    module = _load_controller_stack(rolling_guard)
    _install_platform_collision_filter(module)
    if random_height_reset:
        _install_random_height_reset(module)
    _install_height_banded_controller(module)
    args = module.parse_args()
    print(
        "[z-aware-v2] height-banded acquire/lift/transport; "
        "XY-first place; strict 3D terminal latch on every height route",
        flush=True,
    )
    module.run(args)


if __name__ == "__main__":
    main()
