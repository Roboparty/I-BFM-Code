"""Interactively run phased Carry reward-inference latents in pure MuJoCo.

The policy and phase latents are fixed.  The placement goal is interactive:
W/S move it along world X, A/D along world Y, H restores the configured goal,
R resets the robot and box while preserving the current goal, and P prints the
current controller state.  The green, non-colliding marker is the box goal.
Optional deterministic disturbances can pull the box down during transport and
relocate it after placement, while the MuJoCo renderer records directly to MP4.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import imageio_ffmpeg
import joblib
import mediapy as media
import mujoco
# HEADLESS PATCH(2026-08-30): lazy import in --viewer branch; eager import inits GLFW and dies on display-less servers
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_MujocoRenderer = mujoco.Renderer


def _smoothstep01(value: float) -> float:
    """Return a smooth 0..1 interpolation for a scalar input."""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _annotate_video_frame(frame: np.ndarray, label: str | None) -> np.ndarray:
    if not label:
        return frame
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(18, image.height // 30))
    padding = max(8, image.height // 90)
    left = padding
    top = padding
    bounds = draw.textbbox((left, top), label, font=font, stroke_width=1)
    draw.rectangle(
        (bounds[0] - padding, bounds[1] - padding, bounds[2] + padding, bounds[3] + padding),
        fill=(0, 0, 0, 180),
    )
    draw.text((left, top), label, fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(image)

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.carry_goal_sim2sim import (
    BOX_BODY_NAME,
    BOX_GEOM_NAME,
    BOX_JOINT_NAME,
    CONTROL_DT,
    FLOOR_GEOM_NAME,
    GOAL_GEOM_NAME,
    SIM_DT,
    ActorObservation,
    _actuator_indices,
    _build_model,
    _initialise_state,
    _joint_addresses,
    _load_box_config,
    _load_motion,
    _match_joint_parameter,
    _patch_mode15_joint_physics,
    _physical_hand_contacts,
    _world_body_velocity,
)
from humanoidverse.carry_phased_reward_inference import (
    _box_recovery_phase,
    _goal_lookahead_observation,
    _masked_object_observation,
    _smooth_place_blend,
    _transport_turn_weight,
)
from humanoidverse.envs.carry_observations import (
    compute_hand_capsule_box_surface_distance,
    update_hand_box_contact_hysteresis,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _load_latent(path: Path, name: str, device: torch.device) -> torch.Tensor:
    value = joblib.load(path / f"{name}.pkl")
    if isinstance(value, dict):
        raise TypeError(f"{name}.pkl contains a latent bank; use _load_latent_bank")
    return torch.as_tensor(value, device=device, dtype=torch.float32).reshape(1, -1)


def _load_get_up_latent(path: Path, device: torch.device) -> torch.Tensor:
    value = joblib.load(path)
    if isinstance(value, dict):
        raise TypeError(f"{path} contains a latent bank; expected [Z] or [T, Z]")
    latent = torch.as_tensor(value, device=device, dtype=torch.float32)
    if latent.ndim == 1:
        latent = latent.unsqueeze(0)
    if latent.ndim != 2 or latent.shape[0] == 0:
        raise ValueError(f"{path} must contain non-empty [Z] or [T, Z] latents, got {tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise ValueError(f"{path} contains non-finite values")
    return latent


def _load_latent_bank(path: Path, name: str, device: torch.device) -> dict[int, torch.Tensor]:
    value = joblib.load(path / f"{name}.pkl")
    if not isinstance(value, dict) or not {-1, 1}.issubset(value):
        raise TypeError(f"{name}.pkl must contain left/right latent keys -1 and 1")
    return {
        direction: torch.as_tensor(value[direction], device=device, dtype=torch.float32).reshape(1, -1)
        for direction in (-1, 1)
    }


def _bearing_deg(rotation: np.ndarray, direction_xy: np.ndarray) -> float:
    local = rotation.T @ np.asarray((direction_xy[0], direction_xy[1], 0.0), dtype=np.float64)
    return math.degrees(math.atan2(float(local[1]), float(local[0])))


def _push_turn_weight(
    *,
    goal_distance: float,
    bearing_deg: float,
    turn_boost_distance: float,
    stop_distance: float,
    heading_blend_full_deg: float,
    far_max_turn_blend: float,
    near_max_turn_blend: float,
) -> float:
    """Increase steering authority near the goal without steering when aligned."""

    if turn_boost_distance <= stop_distance:
        raise ValueError("turn_boost_distance must exceed stop_distance")
    if heading_blend_full_deg <= 0.0:
        raise ValueError("heading_blend_full_deg must be positive")
    if not 0.0 <= far_max_turn_blend <= near_max_turn_blend < 1.0:
        raise ValueError("push turn blend limits must satisfy 0 <= far <= near < 1")
    near = float(np.clip(
        (float(turn_boost_distance) - float(goal_distance))
        / (float(turn_boost_distance) - float(stop_distance)),
        0.0,
        1.0,
    ))
    near = near * near * (3.0 - 2.0 * near)
    turn_cap = (1.0 - near) * float(far_max_turn_blend) + near * float(near_max_turn_blend)
    angle_weight = float(np.clip(abs(float(bearing_deg)) / float(heading_blend_full_deg), 0.0, 1.0))
    return angle_weight * turn_cap


def _quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _apply_initial_lie_down(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: str | None,
    root_height: float,
) -> None:
    """Create a deterministic fallen start for validating the recovery controller."""

    if side is None:
        return
    angle = math.pi / 2.0 if side == "left" else -math.pi / 2.0
    roll = np.asarray((math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0))
    data.qpos[2] = float(root_height)
    data.qpos[3:7] = _quat_mul_wxyz(roll, data.qpos[3:7])
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _control_window_active(step: int, *, start_step: int | None, duration_steps: int) -> bool:
    return start_step is not None and start_step <= step < start_step + duration_steps


def _physical_ground_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    floor_geom_id: int,
    body_ids: tuple[int, int],
) -> np.ndarray:
    """Return physical floor contact for the configured left/right bodies."""

    found = np.zeros(2, dtype=np.float32)
    body_to_index = {int(body_id): index for index, body_id in enumerate(body_ids)}
    for contact in data.contact[: data.ncon]:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 == floor_geom_id:
            other = geom2
        elif geom2 == floor_geom_id:
            other = geom1
        else:
            continue
        index = body_to_index.get(int(model.geom_bodyid[other]))
        if index is not None:
            found[index] = 1.0
    return found


def _world_site_linear_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
) -> np.ndarray:
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_SITE,
        int(site_id),
        velocity,
        0,
    )
    return velocity[3:6].copy()


def _relocate_box(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos_address: int,
    qvel_address: int,
    offset_xy: np.ndarray,
    ground_height: float,
) -> None:
    """Teleport the free box by a world-frame offset and leave it stationary."""

    data.qpos[qpos_address : qpos_address + 2] += np.asarray(offset_xy, dtype=np.float64)
    data.qpos[qpos_address + 2] = float(ground_height)
    data.qvel[qvel_address : qvel_address + 6] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def _set_initial_box_center_height(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos_address: int,
    qvel_address: int,
    center_height: float | None,
) -> None:
    """Optionally place the initial box at an explicit height with zero velocity."""

    if center_height is None:
        return
    data.qpos[qpos_address + 2] = float(center_height)
    data.qvel[qvel_address : qvel_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _align_initial_box_yaw_with_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos_address: int,
    qvel_address: int,
    enabled: bool,
) -> None:
    """Make the initial upright box heading match the robot root heading."""

    if not enabled:
        return
    yaw = _yaw_from_wxyz(data.qpos[3:7])
    half_yaw = 0.5 * yaw
    target_geom_quat = np.asarray(
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        dtype=np.float64,
    )
    box_geom_id = model.geom(BOX_GEOM_NAME).id
    fixed_geom_quat = np.asarray(model.geom_quat[box_geom_id], dtype=np.float64)
    fixed_geom_quat /= np.linalg.norm(fixed_geom_quat)
    inverse_geom_quat = fixed_geom_quat * np.asarray((1.0, -1.0, -1.0, -1.0))
    body_quat = _quat_mul_wxyz(target_geom_quat, inverse_geom_quat)
    data.qpos[qpos_address + 3 : qpos_address + 7] = body_quat / np.linalg.norm(body_quat)
    data.qvel[qvel_address : qvel_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def _controller_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_id: int,
    box_body_id: int,
    box_qpos_address: int,
    contact: np.ndarray,
    goal: np.ndarray,
    ground_height: float,
    lift_height: float,
    push_standoff_distance: float = 0.55,
) -> dict[str, float | bool]:
    base_pos = data.xpos[base_body_id]
    box_pos = data.qpos[box_qpos_address : box_qpos_address + 3]
    base_lin_vel, _base_ang_vel = _world_body_velocity(model, data, base_body_id)
    box_lin_vel, _box_ang_vel = _world_body_velocity(model, data, box_body_id)
    base_rotation = data.xmat[base_body_id].reshape(3, 3)
    box_geom_id = model.geom(BOX_GEOM_NAME).id
    box_rotation = data.geom_xmat[box_geom_id].reshape(3, 3)
    robot_box_xy = box_pos[:2] - base_pos[:2]
    robot_box_norm = max(float(np.linalg.norm(robot_box_xy)), 1e-8)
    box_goal_xy = goal[:2] - box_pos[:2]
    box_goal_norm = max(float(np.linalg.norm(box_goal_xy)), 1e-8)
    goal_direction = box_goal_xy / box_goal_norm
    lateral_direction = np.asarray((-goal_direction[1], goal_direction[0]))
    staging_xy = box_pos[:2] - goal_direction * float(push_standoff_distance)
    return {
        "horizontal_distance": float(np.linalg.norm(box_pos[:2] - base_pos[:2])),
        "goal_distance": float(np.linalg.norm(goal[:2] - box_pos[:2])),
        "object_height": float(box_pos[2]),
        "object_speed": float(np.linalg.norm(box_lin_vel)),
        "object_vertical_speed": float(box_lin_vel[2]),
        "ground_error": float(abs(box_pos[2] - ground_height)),
        "lift": float(np.clip((box_pos[2] - ground_height) / lift_height, 0.0, 1.0)),
        "root_height": float(base_pos[2]),
        "pelvis_up_z": float(base_rotation[2, 2]),
        "robot_upright": float(np.clip(base_rotation[2, 2], 0.0, 1.0)),
        "box_upright": float(np.clip(box_rotation[2, 2], 0.0, 1.0)),
        "any_contact": bool(contact[0] > 0.5 or contact[1] > 0.5),
        "bilateral_contact": bool(contact[0] > 0.5 and contact[1] > 0.5),
        "approach_boxward_speed": float(np.dot(base_lin_vel[:2], robot_box_xy / robot_box_norm)),
        "transport_goalward_speed": float(np.dot(box_lin_vel[:2], box_goal_xy / box_goal_norm)),
        "push_staging_distance": float(np.linalg.norm(staging_xy - base_pos[:2])),
        "push_behind_projection": float(np.dot(robot_box_xy, goal_direction)),
        "push_lateral_error": float(abs(np.dot(robot_box_xy, lateral_direction))),
    }


def _summarize_speed_rollout(
    history: list[dict[str, float | bool | str]],
    *,
    final_phase: str,
    placement_ground_height: float | None = None,
) -> dict[str, Any]:
    """Summarize a deterministic phased rollout for latent-bank selection."""

    def rows_for(phase: str) -> list[dict[str, float | bool | str]]:
        return [row for row in history if row["phase"] == phase]

    def mean(rows: list[dict[str, float | bool | str]], key: str) -> float | None:
        return None if not rows else float(np.mean([float(row[key]) for row in rows]))

    def swing_mean(
        rows: list[dict[str, float | bool | str]],
        contact_key: str,
        height_key: str,
    ) -> float | None:
        swing_rows = [row for row in rows if not bool(row[contact_key])]
        return mean(swing_rows, height_key)

    approach_rows = rows_for("approach")
    lift_rows = rows_for("lift")
    transport_rows = rows_for("transport")
    direct_place_rows = rows_for("place_direct")
    push_transport_rows = rows_for("push_transport")
    push_direct_rows = rows_for("push_direct")
    phase_counts = {
        phase: sum(row["phase"] == phase for row in history)
        for phase in (
            "get_up",
            "approach",
            "lift",
            "transport",
            "place",
            "place_direct",
            "push_contact",
            "push_transport",
            "push_stop",
            "push_direct",
            "complete",
        )
    }
    final = history[-1] if history else None
    completed = final_phase == "complete" or phase_counts["complete"] > 0
    airborne_rows: list[dict[str, float | bool | str]] = []
    first_high_contact_loss: dict[str, float | bool | str] | None = None
    unsafe_high_release = False
    if direct_place_rows and placement_ground_height is not None:
        airborne_threshold = float(placement_ground_height) + 0.05
        airborne_rows = [
            row
            for row in direct_place_rows
            if float(row["object_height"]) > airborne_threshold
        ]
        bilateral_seen = False
        for row in direct_place_rows:
            bilateral = bool(row["bilateral_contact"])
            any_contact = bool(row.get("any_contact", bilateral))
            bilateral_seen = bilateral_seen or bilateral
            high = float(row["object_height"]) > airborne_threshold
            if bilateral_seen and high and not any_contact:
                if first_high_contact_loss is None:
                    first_high_contact_loss = row
                if float(row["object_vertical_speed"]) < -0.35:
                    unsafe_high_release = True
    airborne_bilateral_rate = mean(airborne_rows, "bilateral_contact")
    direct_place_min_root_height = (
        None
        if not direct_place_rows
        else float(min(float(row["root_height"]) for row in direct_place_rows))
    )
    direct_place_controlled = (
        None
        if not direct_place_rows or placement_ground_height is None
        else bool(
            completed
            and direct_place_min_root_height is not None
            and direct_place_min_root_height <= 0.50
            and airborne_bilateral_rate is not None
            and airborne_bilateral_rate >= 0.90
            and not unsafe_high_release
        )
    )
    return {
        "step_count": len(history),
        "final_phase": final_phase,
        "completed": completed,
        "phase_counts": phase_counts,
        "approach_boxward_speed_mean": mean(approach_rows, "approach_boxward_speed"),
        "lift_left_foot_ground_contact_rate": mean(lift_rows, "left_foot_ground_contact"),
        "lift_right_foot_ground_contact_rate": mean(lift_rows, "right_foot_ground_contact"),
        "lift_bilateral_foot_ground_contact_rate": mean(lift_rows, "bilateral_foot_ground_contact"),
        "transport_goalward_speed_mean": mean(transport_rows, "transport_goalward_speed"),
        "transport_bilateral_contact_rate": mean(transport_rows, "bilateral_contact"),
        "transport_lift_mean": mean(transport_rows, "lift"),
        "transport_left_foot_ground_contact_rate": mean(transport_rows, "left_foot_ground_contact"),
        "transport_right_foot_ground_contact_rate": mean(transport_rows, "right_foot_ground_contact"),
        "transport_bilateral_foot_ground_contact_rate": mean(
            transport_rows, "bilateral_foot_ground_contact"
        ),
        "transport_left_foot_scuff_rate": mean(transport_rows, "left_foot_scuff"),
        "transport_right_foot_scuff_rate": mean(transport_rows, "right_foot_scuff"),
        "transport_any_foot_scuff_rate": mean(transport_rows, "any_foot_scuff"),
        "transport_left_swing_foot_height_mean": swing_mean(
            transport_rows, "left_foot_ground_contact", "left_foot_height"
        ),
        "transport_right_swing_foot_height_mean": swing_mean(
            transport_rows, "right_foot_ground_contact", "right_foot_height"
        ),
        "direct_place_goalward_speed_mean": mean(
            direct_place_rows, "transport_goalward_speed"
        ),
        "direct_place_bilateral_contact_rate": mean(
            direct_place_rows, "bilateral_contact"
        ),
        "direct_place_airborne_bilateral_contact_rate": airborne_bilateral_rate,
        "direct_place_min_root_height": direct_place_min_root_height,
        "direct_place_first_high_simultaneous_contact_loss_height": (
            None
            if first_high_contact_loss is None
            else float(first_high_contact_loss["object_height"])
        ),
        "direct_place_first_high_simultaneous_contact_loss_vertical_speed": (
            None
            if first_high_contact_loss is None
            else float(first_high_contact_loss["object_vertical_speed"])
        ),
        "direct_place_unsafe_high_release": unsafe_high_release,
        "direct_place_controlled": direct_place_controlled,
        "direct_place_lift_mean": mean(direct_place_rows, "lift"),
        "push_goalward_speed_mean": mean(push_transport_rows, "transport_goalward_speed"),
        "push_contact_rate": mean(push_transport_rows, "any_contact"),
        "push_lift_mean": mean(push_transport_rows, "lift"),
        "push_direct_goalward_speed_mean": mean(
            push_direct_rows, "transport_goalward_speed"
        ),
        "push_direct_contact_rate": mean(push_direct_rows, "any_contact"),
        "push_direct_lift_mean": mean(push_direct_rows, "lift"),
        "max_lift": None if not history else float(max(float(row["lift"]) for row in history)),
        "final_goal_distance": None if final is None else float(final["goal_distance"]),
        "final_horizontal_distance": None if final is None else float(final["horizontal_distance"]),
        "final_object_height": None if final is None else float(final["object_height"]),
    }


class CapsuleBoxContactHysteresis:
    """Online G1 hand-capsule contact labels matching the expert-data rule."""

    def __init__(
        self,
        box_config: dict[str, Any],
        *,
        hand_body_ids: tuple[int, int],
        box_qpos_address: int,
        enter_distance: float,
        exit_distance: float,
    ) -> None:
        self.hand_body_ids = hand_body_ids
        self.box_qpos_address = int(box_qpos_address)
        self.enter_distance = float(enter_distance)
        self.exit_distance = float(exit_distance)
        self.half_size = torch.as_tensor(box_config["half_size"], dtype=torch.float32)
        self.geom_position = torch.as_tensor(
            box_config.get("geom_pos", [0.0, 0.0, 0.0]), dtype=torch.float32
        )
        geom_quaternion_wxyz = np.asarray(
            box_config.get("geom_quat_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32
        )
        self.geom_quaternion_xyzw = torch.as_tensor(
            np.roll(geom_quaternion_wxyz, -1).copy(), dtype=torch.float32
        )
        self.capsule_segments = torch.as_tensor(
            box_config["hand_capsule_segments"], dtype=torch.float32
        )
        self.capsule_radius = float(box_config["hand_contact_radius"])
        self.previous_contact = torch.zeros(1, 2, dtype=torch.float32)
        self.last_surface_distance = np.full(2, np.inf, dtype=np.float32)

    def reset(self) -> None:
        self.previous_contact.zero_()
        self.last_surface_distance.fill(np.inf)

    @torch.no_grad()
    def update(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        hand_positions = torch.as_tensor(
            np.asarray([data.xpos[body_id] for body_id in self.hand_body_ids]),
            dtype=torch.float32,
        ).reshape(1, 2, 3)
        hand_quaternions_xyzw = torch.as_tensor(
            np.roll(
                np.asarray([data.xquat[body_id] for body_id in self.hand_body_ids]),
                -1,
                axis=-1,
            ).copy(),
            dtype=torch.float32,
        ).reshape(1, 2, 4)
        object_position = torch.as_tensor(
            data.qpos[self.box_qpos_address : self.box_qpos_address + 3], dtype=torch.float32
        ).reshape(1, 3)
        object_quaternion_xyzw = torch.as_tensor(
            np.roll(
                data.qpos[self.box_qpos_address + 3 : self.box_qpos_address + 7], -1
            ).copy(),
            dtype=torch.float32,
        ).reshape(1, 4)
        distance = compute_hand_capsule_box_surface_distance(
            hand_positions,
            hand_quaternions_xyzw,
            self.capsule_segments,
            object_position,
            object_quaternion_xyzw,
            self.half_size,
            self.geom_position,
            self.geom_quaternion_xyzw,
            hand_contact_radius=self.capsule_radius,
        )
        contact = update_hand_box_contact_hysteresis(
            distance,
            self.previous_contact,
            enter_distance=self.enter_distance,
            exit_distance=self.exit_distance,
        )
        self.previous_contact.copy_(contact)
        self.last_surface_distance = distance[0].cpu().numpy().copy()
        return contact[0].cpu().numpy().astype(np.float32), self.last_surface_distance.copy()


class PhaseController:
    def __init__(self, args: argparse.Namespace, *, dt: float) -> None:
        self.args = args
        self.dt = float(dt)
        self.place_settle_steps = max(0, round(args.place_settle_seconds / dt))
        self.place_lower_steps = max(1, round(args.place_lower_seconds / dt))
        self.place_release_steps = max(1, round(args.place_release_seconds / dt))
        self.place_retry_steps = max(1, round(args.place_retry_delay_seconds / dt))
        self.place_max_attempt_steps = max(1, round(args.place_max_attempt_seconds / dt))
        self.reset()

    def reset(self) -> None:
        self.phase = "approach"
        self.phase_steps = 0
        self.ready_steps = 0
        self.lost_steps = 0
        self.place_enter_steps = 0
        self.place_drift_steps = 0
        self.placed_steps = 0
        self.failure_steps = 0
        self.completion_lost_steps = 0
        self.release_started_step: int | None = None
        self.place_attempt = 0
        self.recovery_count = 0
        self.fall_event_count = 0
        self.fall_detect_steps = 0
        self.get_up_stable_steps = 0
        self.last_transport_turn_weight = 0.0
        self.approach_turn_active = False
        self.approach_turn_direction = 1
        self.approach_turn_weight = 0.0
        self.place_lower_progress_steps = 0
        self.place_descent_braking = False

    def _change_phase(self, phase: str) -> None:
        if phase == self.phase:
            return
        self.phase = phase
        self.phase_steps = 0
        self.ready_steps = 0
        self.lost_steps = 0
        self.place_enter_steps = 0
        self.place_drift_steps = 0
        self.release_started_step = None
        self.place_lower_progress_steps = 0
        self.place_descent_braking = False
        self.approach_turn_active = False
        self.approach_turn_weight = 0.0

    def _update_fall_detection(self, metrics: dict[str, float | bool]) -> bool:
        root_height = float(metrics["root_height"])
        pelvis_up_z = float(metrics["pelvis_up_z"])
        fallen = root_height < self.args.fall_root_height or (
            pelvis_up_z < self.args.fall_upright_threshold
            and root_height < self.args.fall_tilted_root_height
        )
        self.fall_detect_steps = self.fall_detect_steps + 1 if fallen else 0
        return self.fall_detect_steps >= self.args.fall_detect_steps

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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, str]:
        self.last_transport_turn_weight = 0.0
        legacy_736_flow = bool(getattr(self.args, "legacy_736_flow", False))
        if self.phase == "get_up":
            sequence = z["get_up"]
            sequence_step = min(self.phase_steps, int(sequence.shape[0]) - 1)
            mode = "get_up_reward" if int(sequence.shape[0]) == 1 else f"get_up_{sequence_step:03d}"
            return (
                _masked_object_observation(observation),
                sequence[sequence_step : sequence_step + 1],
                mode,
            )
        if self.phase == "approach":
            bearing = _bearing_deg(base_rotation, box_direction)
            direction = 1 if bearing >= 0.0 else -1
            turn_enter_deg = float(getattr(self.args, "approach_turn_enter_deg", 12.0))
            turn_exit_deg = float(getattr(self.args, "approach_turn_exit_deg", 5.0))
            if turn_enter_deg <= 0.0:
                turn_weight = (
                    min(abs(bearing) / self.args.heading_blend_full_deg, 1.0)
                    * self.args.max_turn_blend
                )
                self.approach_turn_weight = turn_weight
                self.approach_turn_direction = direction
            else:
                if self.approach_turn_active:
                    if abs(bearing) <= turn_exit_deg or direction != self.approach_turn_direction:
                        self.approach_turn_active = False
                elif self.approach_turn_weight <= 1e-6 and abs(bearing) >= turn_enter_deg:
                    self.approach_turn_active = True
                    self.approach_turn_direction = direction

                target_weight = (
                    float(self.args.max_turn_blend) if self.approach_turn_active else 0.0
                )
                blend_seconds = float(
                    getattr(self.args, "approach_turn_blend_seconds", 0.20)
                )
                if blend_seconds > 0.0:
                    max_delta = float(self.args.max_turn_blend) * self.dt / blend_seconds
                    self.approach_turn_weight += float(
                        np.clip(
                            target_weight - self.approach_turn_weight,
                            -max_delta,
                            max_delta,
                        )
                    )
                else:
                    self.approach_turn_weight = target_weight
                turn_weight = self.approach_turn_weight
                direction = self.approach_turn_direction
            latent = policy.project_z((1.0 - turn_weight) * z["approach"] + turn_weight * z["turn"][direction])
            mode = "forward" if turn_weight < 1e-3 else ("forward_left" if direction > 0 else "forward_right")
            policy_observation = (
                _masked_object_observation(observation)
                if self.args.mask_approach_object_context
                else observation
            )
            return policy_observation, latent, mode
        if self.phase == "lift":
            return observation, z["lift"], "lift"
        if self.phase == "transport":
            bearing = _bearing_deg(base_rotation, goal_direction)
            distance = float(metrics["goal_distance"])
            turn_weight = _transport_turn_weight(
                goal_distance=distance,
                bearing_deg=bearing,
                turn_boost_distance=self.args.transport_turn_boost_distance,
                place_distance=self.args.place_align_distance,
                heading_blend_full_deg=self.args.transport_heading_blend_full_deg,
                far_max_turn_blend=self.args.transport_max_turn_blend,
                near_max_turn_blend=self.args.transport_near_max_turn_blend,
            )
            self.last_transport_turn_weight = turn_weight
            direction = 1 if bearing >= 0.0 else -1
            latent = policy.project_z(
                (1.0 - turn_weight) * z["transport"] + turn_weight * z["transport_turn"][direction]
            )
            mode = "transport" if turn_weight < 1e-3 else ("transport_left" if direction > 0 else "transport_right")
            return observation, latent, mode
        if self.phase == "place":
            place_observation = _goal_lookahead_observation(observation, self.args.align_goal_lookahead)
            if legacy_736_flow:
                if self.release_started_step is None:
                    blend = _smooth_place_blend(
                        self.phase_steps,
                        self.place_settle_steps,
                        self.place_lower_steps,
                    )
                    latent = policy.project_z(
                        (1.0 - blend) * z["align"] + blend * z["lower"]
                    )
                    return place_observation, latent, f"place_lower_{blend:.2f}"
                blend = _smooth_place_blend(
                    self.phase_steps - self.release_started_step,
                    0,
                    self.place_release_steps,
                )
                latent = policy.project_z(
                    (1.0 - blend) * z["lower"] + blend * z["place"]
                )
                return place_observation, latent, f"place_release_{blend:.2f}"
            lower_latent = z["lower"]
            low_blend = 0.0
            if "lower_low" in z:
                low_blend = _smoothstep01(
                    (
                        self.args.lower_low_blend_start_lift
                        - float(metrics["lift"])
                    )
                    / (
                        self.args.lower_low_blend_start_lift
                        - self.args.lower_low_blend_end_lift
                    )
                )
                lower_latent = policy.project_z(
                    (1.0 - low_blend) * z["lower"]
                    + low_blend * z["lower_low"]
                )
            if self.release_started_step is None:
                if self.phase_steps <= self.place_settle_steps:
                    # Hold the box stable (z_hold = static carry) during settle.
                    # z_align is a walking skill: on re-place it can move and
                    # drop the box before lowering begins.
                    return place_observation, z["hold"], "place_settle"
                vertical_speed = float(metrics.get("object_vertical_speed", 0.0))
                ground_error = float(metrics.get("ground_error", float("inf")))
                descent_speed_limit = float(
                    getattr(self.args, "place_descent_speed_limit", 0.15)
                )
                descent_resume_speed = float(
                    getattr(self.args, "place_descent_resume_speed", 0.08)
                )
                if self.place_descent_braking:
                    self.place_descent_braking = (
                        vertical_speed < -descent_resume_speed
                        and ground_error > self.args.place_release_height
                    )
                elif (
                    vertical_speed < -descent_speed_limit
                    and ground_error > self.args.place_release_height
                ):
                    self.place_descent_braking = True
                if self.place_descent_braking:
                    return place_observation, z["align"], "place_descent_brake"
                self.place_lower_progress_steps = min(
                    self.place_lower_progress_steps + 1,
                    self.place_lower_steps,
                )
                blend = _smooth_place_blend(
                    self.place_lower_progress_steps,
                    0,
                    self.place_lower_steps,
                )
                latent = policy.project_z((1.0 - blend) * z["align"] + blend * lower_latent)
                return place_observation, latent, f"place_lower_{blend:.2f}_low{low_blend:.2f}"
            blend = _smooth_place_blend(
                self.phase_steps - self.release_started_step,
                0,
                self.place_release_steps,
            )
            latent = policy.project_z((1.0 - blend) * lower_latent + blend * z["place"])
            return place_observation, latent, f"place_release_{blend:.2f}"
        if legacy_736_flow:
            return observation, z["place"], "placed_verify"
        # After placement, move directly from the released-box pose to the
        # empty-handed standing latent.  The get-up latent remains reserved for
        # actual fall recovery and is not a stationary placement skill.
        stand_blend = _smooth_place_blend(
            self.phase_steps, 0, self.place_release_steps
        )
        return (
            observation,
            policy.project_z(
                (1.0 - stand_blend) * z["place"] + stand_blend * z["stand"]
            ),
            "placed_verify",
        )

    def update(self, metrics: dict[str, float | bool], *, goal_height: float) -> None:
        self.phase_steps += 1
        legacy_736_flow = bool(getattr(self.args, "legacy_736_flow", False))
        fall_detected = self._update_fall_detection(metrics)

        if self.phase == "get_up":
            if (
                legacy_736_flow
                and float(metrics["horizontal_distance"]) <= self.args.get_up_distance
            ):
                self._change_phase("lift")
                self.get_up_stable_steps = 0
                return
            recovered = (
                float(metrics["pelvis_up_z"]) >= self.args.get_up_upright_threshold
                and float(metrics["root_height"]) >= self.args.get_up_root_height
            )
            self.get_up_stable_steps = self.get_up_stable_steps + 1 if recovered else 0
            if self.get_up_stable_steps >= self.args.get_up_stable_steps:
                next_phase = "approach" if legacy_736_flow else (
                    "lift"
                    if float(metrics["horizontal_distance"])
                    <= self.args.approach_switch_distance
                    else "approach"
                )
                self._change_phase(next_phase)
                self.fall_detect_steps = 0
                self.get_up_stable_steps = 0
            return

        if fall_detected:
            recovery_phase = (
                "get_up"
                if not legacy_736_flow
                or float(metrics["horizontal_distance"]) > self.args.get_up_distance
                else "lift"
            )
            if recovery_phase != self.phase:
                self.fall_event_count += 1
                self._change_phase(recovery_phase)
                self.get_up_stable_steps = 0
            return

        self.get_up_stable_steps = 0
        if self.phase == "approach":
            if float(metrics["horizontal_distance"]) <= self.args.approach_switch_distance:
                self.phase = "lift"
                self.phase_steps = 0
                self.lost_steps = 0
            return
        if self.phase == "lift":
            ready = (
                bool(metrics["bilateral_contact"])
                and float(metrics["lift"]) >= self.args.hold_switch_lift
                and float(metrics["robot_upright"]) >= self.args.hold_switch_upright
            )
            self.ready_steps = self.ready_steps + 1 if ready else 0
            if self.ready_steps >= self.args.phase_hold_steps:
                self.phase = "transport"
                self.phase_steps = 0
                self.lost_steps = 0
                self.place_enter_steps = 0
                self.place_attempt_steps = 0
                return
            # If the lift skill cannot establish bilateral contact within the
            # grace window, go back to approach instead of looping forever in a
            # miss-positioned crouch (re-place after relocation).
            if not legacy_736_flow:
                lost_contact = not bool(metrics["bilateral_contact"])
                self.lost_steps = self.lost_steps + 1 if lost_contact else 0
                if self.lost_steps >= max(self.args.phase_hold_steps * 4, 40):
                    self.phase = "approach"
                    self.phase_steps = 0
                    self.ready_steps = 0
                    self.lost_steps = 0
                    self.place_enter_steps = 0
            return
        if self.phase == "transport":
            lost = not bool(metrics["bilateral_contact"]) or float(metrics["lift"]) < self.args.hold_release_lift
            self.lost_steps = self.lost_steps + 1 if lost else 0
            inside_place = not lost and float(metrics["goal_distance"]) <= self.args.place_align_distance
            self.place_enter_steps = self.place_enter_steps + 1 if inside_place else 0
            if self.lost_steps >= self.args.phase_hold_steps:
                self.phase = _box_recovery_phase(
                    float(metrics["horizontal_distance"]), self.args.approach_switch_distance
                )
                self.phase_steps = 0
                self.ready_steps = 0
                self.place_enter_steps = 0
            elif self.place_enter_steps >= self.args.place_enter_steps:
                self.phase = "place"
                self.phase_steps = 0
                self.place_attempt += 1
                self.release_started_step = None
                self.place_enter_steps = 0
                self.place_drift_steps = 0
                self.placed_steps = 0
                self.failure_steps = 0
            return
        ground_error = abs(float(metrics["object_height"]) - goal_height)
        if self.phase == "place":
            drifted_while_holding = (
                self.release_started_step is None
                and bool(metrics["bilateral_contact"])
                and float(metrics["goal_distance"]) > self.args.place_drift_distance
            )
            self.place_drift_steps = self.place_drift_steps + 1 if drifted_while_holding else 0
            if self.place_drift_steps >= self.args.place_drift_steps:
                self.phase = "transport"
                self.phase_steps = 0
                self.release_started_step = None
                self.place_drift_steps = 0
                return
            if (
                self.release_started_step is None
                and bool(metrics["bilateral_contact"])
                and ground_error <= self.args.place_release_height
            ):
                self.release_started_step = self.phase_steps
            accepted_release = (
                legacy_736_flow
                or self.release_started_step is not None
                or bool(getattr(self.args, "accept_settled_ground_release", False))
            )
            placed = (
                # Do not count an accidental high release followed by free fall
                # as a successful controlled placement.  A valid completion
                # normally must first reach the low bilateral-contact release
                # gate.  Selected rollouts may also accept a box that remains
                # released, upright, still, and on target for the full success
                # debounce even if contact was lost just above that gate.
                accepted_release
                and float(metrics["goal_distance"]) <= self.args.place_success_distance
                and ground_error <= self.args.place_success_height
                and float(metrics["object_speed"]) <= self.args.place_success_speed
                and float(metrics["box_upright"]) >= self.args.place_success_upright
                and not bool(metrics["bilateral_contact"])
            )
            self.placed_steps = self.placed_steps + 1 if placed else 0
            if self.placed_steps >= self.args.place_success_steps:
                self.phase = "complete"
                self.phase_steps = 0
                self.completion_lost_steps = 0
                return
            released = not bool(metrics["bilateral_contact"])
            failed = (
                (released and ground_error > self.args.place_release_height)
                or (
                    released
                    and ground_error <= self.args.place_success_height
                    and float(metrics["object_speed"]) <= self.args.place_success_speed
                    and float(metrics["goal_distance"]) > self.args.place_success_distance
                )
                or self.phase_steps >= self.place_max_attempt_steps
                or float(metrics["goal_distance"]) > self.args.place_abort_distance
            )
            self.failure_steps = self.failure_steps + 1 if failed else 0
            if self.failure_steps >= self.place_retry_steps:
                self.phase = _box_recovery_phase(
                    float(metrics["horizontal_distance"]), self.args.approach_switch_distance
                )
                self.phase_steps = 0
                self.recovery_count += 1
                self.ready_steps = 0
                self.lost_steps = 0
                self.place_enter_steps = 0
                self.place_drift_steps = 0
                self.placed_steps = 0
                self.failure_steps = 0
                self.release_started_step = None
            return
        still_complete = (
            float(metrics["goal_distance"]) <= self.args.place_success_distance
            and ground_error <= self.args.place_success_height
            and float(metrics["object_speed"]) <= self.args.place_success_speed
            and float(metrics["box_upright"]) >= self.args.place_success_upright
        )
        self.completion_lost_steps = 0 if still_complete else self.completion_lost_steps + 1
        if self.completion_lost_steps >= self.place_retry_steps:
            self.phase = _box_recovery_phase(
                float(metrics["horizontal_distance"]), self.args.approach_switch_distance
            )
            self.phase_steps = 0
            self.recovery_count += 1
            self.ready_steps = 0
            self.lost_steps = 0
            self.place_enter_steps = 0
            self.place_drift_steps = 0
            self.completion_lost_steps = 0


class PushPhaseController(PhaseController):
    """Ground-push controller that never requests lift or release skills."""

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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, str]:
        self.last_transport_turn_weight = 0.0
        goal_norm = max(float(np.linalg.norm(goal_direction)), 1e-8)
        goal_unit = np.asarray(goal_direction, dtype=np.float64) / goal_norm

        if self.phase == "approach":
            # Navigate to a point behind the box, measured along box -> goal.
            staging_direction = np.asarray(box_direction, dtype=np.float64) - (
                goal_unit * float(self.args.push_standoff_distance)
            )
            if float(np.linalg.norm(staging_direction)) < 1e-6:
                staging_direction = np.asarray(box_direction, dtype=np.float64)
            bearing = _bearing_deg(base_rotation, staging_direction)
            turn_weight = (
                min(abs(bearing) / self.args.heading_blend_full_deg, 1.0)
                * self.args.max_turn_blend
            )
            direction = 1 if bearing >= 0.0 else -1
            latent = policy.project_z(
                (1.0 - turn_weight) * z["approach"] + turn_weight * z["turn"][direction]
            )
            mode = "approach" if turn_weight < 1e-3 else (
                "approach_left" if direction > 0 else "approach_right"
            )
            policy_observation = (
                _masked_object_observation(observation)
                if self.args.mask_approach_object_context
                else observation
            )
            return policy_observation, latent, mode

        if self.phase == "push_contact":
            bearing = _bearing_deg(base_rotation, box_direction)
            turn_weight = (
                min(abs(bearing) / self.args.heading_blend_full_deg, 1.0)
                * self.args.push_contact_max_turn_blend
            )
            direction = 1 if bearing >= 0.0 else -1
            latent = policy.project_z(
                (1.0 - turn_weight) * z["push_contact"]
                + turn_weight * z["turn"][direction]
            )
            mode = "push_contact" if turn_weight < 1e-3 else (
                "push_contact_left" if direction > 0 else "push_contact_right"
            )
            return observation, latent, mode

        if self.phase == "push_transport":
            bearing = _bearing_deg(base_rotation, goal_direction)
            turn_weight = _push_turn_weight(
                goal_distance=float(metrics["goal_distance"]),
                bearing_deg=bearing,
                turn_boost_distance=self.args.push_turn_boost_distance,
                stop_distance=self.args.push_stop_enter_distance,
                heading_blend_full_deg=self.args.transport_heading_blend_full_deg,
                far_max_turn_blend=self.args.push_max_turn_blend,
                near_max_turn_blend=self.args.push_near_max_turn_blend,
            )
            self.last_transport_turn_weight = turn_weight
            direction = 1 if bearing >= 0.0 else -1
            latent = policy.project_z(
                (1.0 - turn_weight) * z["push_transport"]
                + turn_weight * z["push_turn"][direction]
            )
            mode = "push_forward" if turn_weight < 1e-3 else (
                "push_left" if direction > 0 else "push_right"
            )
            return observation, latent, mode

        return observation, z["push_stop"], "push_stop" if self.phase == "push_stop" else "complete"

    def update(self, metrics: dict[str, float | bool], *, goal_height: float) -> None:
        del goal_height
        self.phase_steps += 1
        grounded = float(metrics["lift"]) <= self.args.push_max_lift
        upright = (
            float(metrics["robot_upright"]) >= self.args.push_upright_min
            and float(metrics["box_upright"]) >= self.args.push_upright_min
        )
        aligned_behind = (
            float(metrics["push_behind_projection"]) >= self.args.push_behind_min
            and float(metrics["push_lateral_error"]) <= self.args.push_lateral_max
        )
        push_contact = (
            bool(metrics["bilateral_contact"])
            if self.args.push_require_bilateral_contact
            else bool(metrics["any_contact"])
        )

        if self.phase == "approach":
            ready = (
                float(metrics["push_staging_distance"]) <= self.args.push_staging_tolerance
                and aligned_behind
            )
            self.ready_steps = self.ready_steps + 1 if ready else 0
            if self.ready_steps >= self.args.push_phase_hold_steps:
                self._change_phase("push_contact")
            return

        if self.phase == "push_contact":
            too_far = float(metrics["horizontal_distance"]) > self.args.push_reapproach_distance
            if too_far or not aligned_behind:
                self.lost_steps += 1
            else:
                self.lost_steps = 0
            if self.lost_steps >= self.args.push_lost_contact_steps:
                self._change_phase("approach")
                self.recovery_count += 1
                return
            ready = push_contact and grounded and upright
            self.ready_steps = self.ready_steps + 1 if ready else 0
            if self.ready_steps >= self.args.push_contact_steps:
                self._change_phase("push_transport")
            return

        if self.phase == "push_transport":
            lost = (
                not push_contact
                or not grounded
                or float(metrics["box_upright"]) < self.args.push_upright_min
            )
            self.lost_steps = self.lost_steps + 1 if lost else 0
            near_goal = float(metrics["goal_distance"]) <= self.args.push_stop_enter_distance
            self.place_enter_steps = self.place_enter_steps + 1 if near_goal else 0
            if self.lost_steps >= self.args.push_lost_contact_steps:
                recovery = (
                    "approach"
                    if float(metrics["horizontal_distance"]) > self.args.push_reapproach_distance
                    else "push_contact"
                )
                self._change_phase(recovery)
                self.recovery_count += 1
            elif self.place_enter_steps >= self.args.push_stop_enter_steps:
                self._change_phase("push_stop")
                self.placed_steps = 0
            return

        success = (
            float(metrics["goal_distance"]) <= self.args.push_success_distance
            and float(metrics["object_speed"]) <= self.args.push_success_speed
            and grounded
            and float(metrics["box_upright"]) >= self.args.push_upright_min
        )
        if self.phase == "push_stop":
            drifted = float(metrics["goal_distance"]) > self.args.push_stop_exit_distance
            self.place_drift_steps = self.place_drift_steps + 1 if drifted else 0
            self.placed_steps = self.placed_steps + 1 if success else 0
            if self.placed_steps >= self.args.push_success_steps:
                self._change_phase("complete")
            elif self.place_drift_steps >= self.args.push_stop_exit_steps:
                self._change_phase("push_transport")
                self.recovery_count += 1
            return

        # A completed push resumes automatically if the box is displaced.
        self.completion_lost_steps = 0 if success else self.completion_lost_steps + 1
        if self.completion_lost_steps >= self.args.push_stop_exit_steps:
            recovery = (
                "approach"
                if float(metrics["horizontal_distance"]) > self.args.push_reapproach_distance
                else "push_contact"
            )
            self._change_phase(recovery)
            self.recovery_count += 1
            self.completion_lost_steps = 0


class DirectPlaceController(PhaseController):
    """Run a continuous goal-conditioned placement policy without phase switches."""

    def _reset_direct_place_state(self) -> None:
        self.last_direct_place_terminal_weight = 0.0
        self.direct_place_lower_commitment = 0.0
        self.direct_place_release_latched = False
        self.direct_place_release_commitment = 0.0
        self.direct_place_commit_latched = False
        self.direct_place_heading_ready = False
        self.direct_place_turn_direction: np.ndarray | None = None
        self.direct_place_use_transport: bool | None = None
        self.direct_place_direct_heading_gate = False
        self.direct_place_rear_route = False
        self.direct_place_positive_lateral_transport_route = False
        self.direct_place_initial_bearing_abs: float | None = None
        self.direct_place_initial_bearing_deg: float | None = None
        self.direct_place_far_route = False
        self.direct_place_disable_turn = False
        self.direct_place_position_ready = False
        self.direct_place_seen_bilateral = False
        self.direct_place_contact_loss_steps = 0
        self.direct_place_initial_goal_distance: float | None = None

    def reset(self) -> None:
        super().reset()
        self.phase = "place_direct"
        self._reset_direct_place_state()

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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, str]:
        del box_direction
        self.last_transport_turn_weight = 0.0
        if self.phase == "complete" and "direct_place_stand" in z:
            stand_blend = _smooth_place_blend(
                self.phase_steps,
                0,
                self.place_release_steps,
            )
            released = z.get("direct_place_release", z["direct_place"])
            latent = policy.project_z(
                (1.0 - stand_blend) * released
                + stand_blend * z["direct_place_stand"]
            )
            return observation, latent, f"place_direct_complete_stand_{stand_blend:.2f}"

        available_transport_base = z.get("direct_place_transport")
        if self.direct_place_use_transport is None:
            lateral_min = getattr(
                self.args, "direct_place_lateral_direct_min_deg", None
            )
            if available_transport_base is None:
                self.direct_place_use_transport = False
            elif lateral_min is None:
                self.direct_place_use_transport = True
            else:
                initial_bearing_deg = _bearing_deg(base_rotation, goal_direction)
                initial_bearing = abs(initial_bearing_deg)
                self.direct_place_initial_bearing_abs = initial_bearing
                self.direct_place_initial_bearing_deg = initial_bearing_deg
                lateral_max = float(
                    getattr(
                        self.args,
                        "direct_place_lateral_direct_max_deg",
                        180.0,
                    )
                )
                lateral_direct = (
                    float(lateral_min) <= initial_bearing <= lateral_max
                )
                positive_lateral_transport = (
                    initial_bearing_deg > 0.0
                    and float(lateral_min)
                    <= initial_bearing
                    <= float(
                        getattr(
                            self.args,
                            "direct_place_positive_lateral_transport_max_deg",
                            150.0,
                        )
                    )
                    and bool(
                        getattr(
                            self.args,
                            "direct_place_positive_lateral_use_transport",
                            False,
                        )
                    )
                )
                if positive_lateral_transport:
                    lateral_direct = False
                rear_min = getattr(
                    self.args, "direct_place_rear_direct_min_deg", None
                )
                rear_direct = (
                    rear_min is not None
                    and initial_bearing >= float(rear_min)
                    and float(metrics["goal_distance"])
                    <= float(
                        getattr(
                            self.args,
                            "direct_place_rear_direct_max_distance",
                            0.0,
                        )
                    )
                )
                far_direct_distance = getattr(
                    self.args, "direct_place_far_direct_distance", None
                )
                far_direct = (
                    far_direct_distance is not None
                    and float(metrics["goal_distance"])
                    >= float(far_direct_distance)
                )
                self.direct_place_use_transport = not (
                    lateral_direct or rear_direct or far_direct
                )
                self.direct_place_direct_heading_gate = rear_direct
                self.direct_place_rear_route = rear_direct
                self.direct_place_positive_lateral_transport_route = (
                    positive_lateral_transport
                )
                self.direct_place_far_route = far_direct
                self.direct_place_disable_turn = (
                    far_direct
                    and initial_bearing
                    <= float(
                        getattr(
                            self.args,
                            "direct_place_far_direct_max_bearing_deg",
                            60.0,
                        )
                    )
                )
        transport_navigation = bool(self.direct_place_use_transport)
        if self.direct_place_rear_route:
            observation = _goal_lookahead_observation(
                observation,
                float(getattr(self.args, "direct_place_goal_lookahead", 0.0)),
            )
        elif self.direct_place_positive_lateral_transport_route:
            observation = _goal_lookahead_observation(
                observation,
                float(
                    getattr(
                        self.args,
                        "direct_place_transport_goal_lookahead",
                        0.0,
                    )
                ),
            )
        lateral_direct_route = (
            available_transport_base is not None and not transport_navigation
        )
        ungated_lateral_route = (
            lateral_direct_route and not self.direct_place_rear_route
        )
        use_commit_profile = (
            transport_navigation or available_transport_base is None
        )
        transport_base = (
            available_transport_base if transport_navigation else None
        )
        direct_place_base = (
            transport_base if transport_base is not None else z["direct_place"]
        )
        if self.direct_place_initial_goal_distance is None:
            self.direct_place_initial_goal_distance = float(metrics["goal_distance"])

        direct_turn = z.get("direct_place_turn")
        if self.direct_place_disable_turn:
            direct_turn = None
        turn_mode = ""
        turn_heading_ready = True
        if direct_turn is not None:
            distance = float(metrics["goal_distance"])
            if transport_base is not None:
                bearing = _bearing_deg(base_rotation, goal_direction)
                turn_weight = _transport_turn_weight(
                    goal_distance=distance,
                    bearing_deg=bearing,
                    turn_boost_distance=float(
                        self.args.transport_turn_boost_distance
                    ),
                    place_distance=float(
                        self.args.direct_place_profile_commit_distance
                    ),
                    heading_blend_full_deg=float(
                        self.args.transport_heading_blend_full_deg
                    ),
                    far_max_turn_blend=float(self.args.transport_max_turn_blend),
                    near_max_turn_blend=float(
                        self.args.transport_near_max_turn_blend
                    ),
                )
                turn_ready_deg = None
            else:
                fade_end = float(self.args.direct_place_turn_fade_end_distance)
                direction_norm = float(np.linalg.norm(goal_direction))
                if direction_norm > 1e-6 and (
                    self.direct_place_turn_direction is None or distance >= fade_end
                ):
                    self.direct_place_turn_direction = (
                        np.asarray(goal_direction, dtype=np.float64) / direction_norm
                    )
                stable_goal_direction = (
                    self.direct_place_turn_direction
                    if self.direct_place_turn_direction is not None
                    else np.asarray(goal_direction, dtype=np.float64)
                )
                bearing = _bearing_deg(base_rotation, stable_goal_direction)
                turn_ready_deg = getattr(
                    self.args, "direct_place_turn_ready_deg", None
                )
                if (
                    available_transport_base is not None
                    and not self.direct_place_direct_heading_gate
                ):
                    # This lateral route uses distance fading to avoid near-goal chatter.
                    turn_ready_deg = None
                if turn_ready_deg is not None:
                    position_ready = (
                        distance
                        <= float(
                            getattr(
                                self.args,
                                "direct_place_heading_position_ready_distance",
                                0.0,
                            )
                        )
                        and float(metrics.get("object_speed", 0.0))
                        <= float(
                            getattr(
                                self.args,
                                "direct_place_heading_position_ready_speed",
                                0.0,
                            )
                        )
                        and bool(metrics.get("bilateral_contact", False))
                        and float(metrics.get("box_upright", 1.0)) >= 0.85
                    )
                    self.direct_place_position_ready = (
                        self.direct_place_position_ready or position_ready
                    )
                    turn_resume_deg = float(
                        getattr(
                            self.args,
                            "direct_place_turn_resume_deg",
                            turn_ready_deg,
                        )
                    )
                    abs_bearing = abs(bearing)
                    if self.direct_place_position_ready:
                        self.direct_place_heading_ready = True
                    elif self.direct_place_heading_ready:
                        self.direct_place_heading_ready = abs_bearing <= turn_resume_deg
                    else:
                        self.direct_place_heading_ready = abs_bearing <= float(
                            turn_ready_deg
                        )
                    turn_heading_ready = self.direct_place_heading_ready
                fade_start = float(self.args.direct_place_turn_fade_start_distance)
                turn_fade = _smoothstep01(
                    (distance - fade_end) / (fade_start - fade_end)
                )
                if self.direct_place_position_ready:
                    turn_fade = 0.0
                elif not turn_heading_ready:
                    # A close target must not suppress steering before the robot faces it.
                    turn_fade = 1.0
                angle_weight = float(
                    np.clip(
                        abs(bearing)
                        / float(self.args.direct_place_turn_heading_full_deg),
                        0.0,
                        1.0,
                    )
                )
                turn_weight = (
                    angle_weight
                    * float(self.args.direct_place_turn_max_blend)
                    * turn_fade
                )
            self.last_transport_turn_weight = turn_weight
            direction = 1 if bearing >= 0.0 else -1
            direct_place_base = policy.project_z(
                (1.0 - turn_weight) * direct_place_base
                + turn_weight * direct_turn[direction]
            )
            if turn_weight >= 1e-3:
                turn_mode = "_left" if direction > 0 else "_right"
            if turn_ready_deg is not None:
                turn_mode += f"_hr{int(turn_heading_ready)}"

        profile_names = (
            "direct_place_slow",
            "direct_place_lower",
            "direct_place_release",
        )
        if all(name in z for name in profile_names):
            start = float(self.args.direct_place_terminal_blend_start_distance)
            end = float(self.args.direct_place_terminal_blend_end_distance)
            distance = float(metrics["goal_distance"])
            slow_progress = _smoothstep01((start - distance) / (start - end))
            if not turn_heading_ready:
                slow_progress = 0.0

            lower_full_distance = float(self.args.direct_place_profile_lower_full_distance)
            lower_progress = _smoothstep01(
                (end - distance) / (end - lower_full_distance)
            )
            commit_seconds = float(
                getattr(self.args, "direct_place_profile_commit_seconds", 0.0)
            )
            far_commit_seconds = getattr(
                self.args, "direct_place_profile_far_commit_seconds", None
            )
            far_profile = (
                far_commit_seconds is not None
                and self.direct_place_initial_goal_distance
                >= float(
                    getattr(
                        self.args,
                        "direct_place_profile_far_commit_distance",
                        float("inf"),
                    )
                )
            )
            if far_profile:
                commit_seconds = float(far_commit_seconds)
            if commit_seconds > 0.0:
                commit_distance = float(
                    getattr(self.args, "direct_place_profile_commit_distance", end)
                )
                commit_speed = float(
                    getattr(self.args, "direct_place_profile_commit_speed", 0.20)
                )
                if lateral_direct_route:
                    commit_distance = float(
                        getattr(
                            self.args,
                            "direct_place_lateral_commit_distance",
                            commit_distance,
                        )
                    )
                    commit_speed = float(
                        getattr(
                            self.args,
                            "direct_place_lateral_commit_speed",
                            commit_speed,
                        )
                    )
                reset_distance = float(
                    getattr(self.args, "direct_place_profile_commit_reset_distance", start)
                )
                can_start_commit = (
                    turn_heading_ready
                    and distance <= commit_distance
                    and float(metrics.get("object_speed", 0.0)) <= commit_speed
                    and float(metrics.get("robot_upright", 1.0))
                    >= float(
                        0.0
                        if lateral_direct_route
                        else getattr(
                            self.args,
                            "direct_place_profile_commit_min_upright",
                            0.0,
                        )
                    )
                    and float(metrics.get("box_upright", 1.0))
                    >= float(
                        0.0
                        if lateral_direct_route
                        else getattr(
                            self.args,
                            "direct_place_profile_commit_min_upright",
                            0.0,
                        )
                    )
                    and (
                        lateral_direct_route
                        or not getattr(
                            self.args,
                            "direct_place_profile_commit_require_bilateral",
                            False,
                        )
                        or bool(metrics.get("bilateral_contact", False))
                    )
                )
                commit_latch_enabled = bool(
                    not ungated_lateral_route
                    and getattr(
                        self.args,
                        "direct_place_profile_commit_latch",
                        False,
                    )
                )
                if can_start_commit and commit_latch_enabled:
                    self.direct_place_commit_latched = True
                delta = self.dt / commit_seconds
                lower_descent_speed = getattr(
                    self.args,
                    "direct_place_profile_lower_descent_speed",
                    None,
                )
                lower_brake_rate = float(
                    getattr(
                        self.args,
                        "direct_place_profile_lower_brake_rate",
                        2.0,
                    )
                )
                if self.direct_place_rear_route:
                    lower_descent_speed = getattr(
                        self.args,
                        "direct_place_rear_lower_descent_speed",
                        lower_descent_speed,
                    )
                    lower_brake_rate = float(
                        getattr(
                            self.args,
                            "direct_place_rear_lower_brake_rate",
                            lower_brake_rate,
                        )
                    )
                braking_route = self.direct_place_rear_route or (
                    transport_navigation
                    and self.direct_place_initial_bearing_abs is not None
                    and self.direct_place_initial_bearing_abs
                    < float(
                        getattr(
                            self.args,
                            "direct_place_transport_brake_max_bearing_deg",
                            60.0,
                        )
                    )
                ) or (
                    lateral_direct_route
                    and not self.direct_place_disable_turn
                    and self.direct_place_initial_bearing_deg is not None
                    and self.direct_place_initial_bearing_deg > 0.0
                )
                if not braking_route:
                    lower_descent_speed = None
                lower_brake_floor = 0.5 if transport_navigation else 0.0
                braking_lower = (
                    lower_descent_speed is not None
                    and self.direct_place_lower_commitment > lower_brake_floor
                    and float(metrics.get("object_vertical_speed", 0.0))
                    < -float(lower_descent_speed)
                )
                if braking_lower:
                    self.direct_place_lower_commitment = max(
                        lower_brake_floor,
                        self.direct_place_lower_commitment
                        - delta * lower_brake_rate,
                    )
                elif can_start_commit or self.direct_place_commit_latched:
                    self.direct_place_lower_commitment = min(
                        1.0, self.direct_place_lower_commitment + delta
                    )
                elif distance > reset_distance and not commit_latch_enabled:
                    self.direct_place_lower_commitment = max(
                        0.0, self.direct_place_lower_commitment - delta
                    )
                if getattr(self.args, "direct_place_profile_commit_only", False):
                    lower_progress = self.direct_place_lower_commitment
                else:
                    lower_progress = max(
                        lower_progress, self.direct_place_lower_commitment
                    )
                if getattr(
                    self.args,
                    "direct_place_profile_slow_commit_only",
                    False,
                ) and use_commit_profile:
                    slow_progress = (
                        self.direct_place_lower_commitment
                        if turn_heading_ready
                        else 0.0
                    )
                else:
                    slow_progress = max(
                        slow_progress, self.direct_place_lower_commitment
                    )

                if getattr(
                    self.args,
                    "direct_place_profile_sequential_commit",
                    False,
                ) and use_commit_profile:
                    slow_progress = _smoothstep01(
                        2.0 * self.direct_place_lower_commitment
                    )
                    lower_progress = _smoothstep01(
                        2.0 * self.direct_place_lower_commitment - 1.0
                    )

            lift_start = float(self.args.direct_place_release_lift_start)
            lift_end = float(self.args.direct_place_release_lift_end)
            release_progress = _smoothstep01(
                (lift_start - float(metrics["lift"])) / (lift_start - lift_end)
            )
            release_latch_distance = float(
                getattr(self.args, "direct_place_release_latch_distance", 0.35)
            )
            release_latch_min_upright = float(
                getattr(self.args, "direct_place_release_latch_min_upright", 0.90)
            )
            release_latch_max_speed = float(
                getattr(
                    self.args,
                    "direct_place_release_latch_max_speed",
                    float("inf"),
                )
            )
            release_require_gate = bool(
                getattr(self.args, "direct_place_release_require_gate", False)
            )
            if self.direct_place_far_route:
                release_latch_distance = 0.35
                release_latch_min_upright = 0.90
                release_latch_max_speed = float("inf")
                release_require_gate = False
            release_ready = (
                distance <= release_latch_distance
                and float(metrics.get("object_speed", 0.0))
                <= release_latch_max_speed
                and float(metrics.get("box_upright", 1.0))
                >= release_latch_min_upright
            )
            if (
                release_require_gate and not release_ready
            ):
                release_progress = 0.0
            release_latch_lift = float(
                getattr(self.args, "direct_place_release_latch_lift", 0.0)
            )
            if self.direct_place_far_route:
                release_latch_lift = 0.15
            can_commit_release = (
                release_latch_lift > 0.0
                and turn_heading_ready
                and release_ready
                and float(metrics["lift"]) <= release_latch_lift
                and abs(float(metrics.get("object_vertical_speed", 0.0)))
                <= float(
                    getattr(
                        self.args,
                        "direct_place_release_latch_max_vertical_speed",
                        0.20,
                    )
                )
            )
            release_commit_seconds = float(
                getattr(self.args, "direct_place_release_commit_seconds", 0.0)
            )
            if self.direct_place_far_route:
                release_commit_seconds = 0.0
            if can_commit_release and release_commit_seconds > 0.0:
                self.direct_place_release_commitment = min(
                    1.0,
                    self.direct_place_release_commitment
                    + self.dt / release_commit_seconds,
                )
            elif can_commit_release:
                self.direct_place_release_commitment = 1.0
            elif release_commit_seconds > 0.0:
                self.direct_place_release_commitment = max(
                    0.0,
                    self.direct_place_release_commitment
                    - self.dt / release_commit_seconds,
                )
            release_progress = max(
                release_progress, self.direct_place_release_commitment
            )
            if self.direct_place_release_commitment >= 1.0:
                self.direct_place_release_latched = True
            if self.direct_place_release_latched:
                release_progress = 1.0

            lower_or_release = policy.project_z(
                (1.0 - release_progress) * z["direct_place_lower"]
                + release_progress * z["direct_place_release"]
            )
            slow_latent = (
                z.get("direct_place_transport_slow", z["direct_place_slow"])
                if transport_navigation
                else z["direct_place_slow"]
            )
            near_latent = policy.project_z(
                (1.0 - lower_progress) * slow_latent
                + lower_progress * lower_or_release
            )
            latent = policy.project_z(
                (1.0 - slow_progress) * direct_place_base
                + slow_progress * near_latent
            )

            guard = 0.0
            contact_recovery = False
            descent_guard_enabled = bool(
                getattr(self.args, "direct_place_descent_guard", False)
            )
            if descent_guard_enabled and hasattr(
                self.args, "direct_place_descent_guard_start_speed"
            ):
                vertical_speed = float(metrics.get("object_vertical_speed", 0.0))
                guard_start = float(self.args.direct_place_descent_guard_start_speed)
                guard_full = float(self.args.direct_place_descent_guard_full_speed)
                guard = _smoothstep01(
                    (-vertical_speed - guard_start) / (guard_full - guard_start)
                )
                guard *= float(
                    getattr(self.args, "direct_place_descent_guard_max_blend", 1.0)
                )
            bilateral = bool(metrics.get("bilateral_contact", True))
            if bilateral:
                self.direct_place_seen_bilateral = True
                self.direct_place_contact_loss_steps = 0
            elif (
                self.direct_place_seen_bilateral
                and float(metrics["lift"])
                > float(getattr(self.args, "direct_place_contact_guard_lift", 0.10))
            ):
                self.direct_place_contact_loss_steps += 1
            else:
                self.direct_place_contact_loss_steps = 0
            contact_recovery = (
                self.direct_place_contact_loss_steps
                >= int(getattr(self.args, "direct_place_contact_guard_steps", 1))
            )
            if self.direct_place_release_latched:
                latent = z["direct_place_release"]
                guard = 0.0
            elif contact_recovery:
                latent = z.get(
                    "direct_place_recover",
                    z.get("direct_place_guard", z["direct_place_slow"]),
                )
            elif guard > 0.0:
                guard_latent = z.get("direct_place_guard", z["direct_place_slow"])
                latent = policy.project_z(
                    (1.0 - guard) * latent + guard * guard_latent
                )

            self.last_direct_place_terminal_weight = slow_progress
            return (
                observation,
                latent,
                "place_direct_profile_"
                f"s{slow_progress:.2f}_l{lower_progress:.2f}_r{release_progress:.2f}_"
                f"c{self.direct_place_lower_commitment:.2f}_g{guard:.2f}_"
                f"rec{int(contact_recovery)}_rl{int(self.direct_place_release_latched)}{turn_mode}",
            )

        terminal = z.get("direct_place_terminal")
        if terminal is None:
            self.last_direct_place_terminal_weight = 0.0
            return observation, direct_place_base, f"place_direct{turn_mode}"

        distance = float(metrics["goal_distance"])
        blend_start = getattr(
            self.args, "direct_place_terminal_blend_start_distance", None
        )
        blend_end = getattr(
            self.args, "direct_place_terminal_blend_end_distance", 0.30
        )
        blend_center = self.args.direct_place_terminal_blend_center
        if blend_start is not None:
            blend = _smoothstep01(
                (float(blend_start) - distance)
                / (float(blend_start) - float(blend_end))
            )
        elif blend_center is None:
            blend_distance = float(self.args.direct_place_terminal_blend_distance)
            blend = math.exp(-((distance / blend_distance) ** 2))
        else:
            width = float(self.args.direct_place_terminal_blend_width)
            exponent = float(
                np.clip((distance - float(blend_center)) / width, -60.0, 60.0)
            )
            blend = 1.0 / (1.0 + math.exp(exponent))
        speed_sigma = self.args.direct_place_terminal_speed_sigma
        if speed_sigma is not None:
            object_speed = float(metrics["object_speed"])
            blend *= math.exp(-((object_speed / float(speed_sigma)) ** 2))
        blend *= float(self.args.direct_place_terminal_max_blend)
        self.last_direct_place_terminal_weight = blend
        latent = policy.project_z(
            (1.0 - blend) * direct_place_base + blend * terminal
        )
        return observation, latent, f"place_direct_terminal_{blend:.2f}{turn_mode}"

    def update(self, metrics: dict[str, float | bool], *, goal_height: float) -> None:
        self.phase_steps += 1
        placed = (
            float(metrics["goal_distance"])
            <= self.args.direct_place_success_distance
            and abs(float(metrics["object_height"]) - goal_height)
            <= self.args.place_success_height
            and float(metrics["object_speed"]) <= self.args.place_success_speed
            and float(metrics["box_upright"]) >= self.args.place_success_upright
            and not bool(metrics["any_contact"])
        )
        self.placed_steps = self.placed_steps + 1 if placed else 0
        if self.phase == "complete":
            self.completion_lost_steps = (
                0 if placed else self.completion_lost_steps + 1
            )
            if self.completion_lost_steps >= self.place_retry_steps:
                self.phase = "place_direct"
                self.phase_steps = 0
                self.placed_steps = 0
                self.completion_lost_steps = 0
                self.direct_place_release_latched = False
                self.direct_place_release_commitment = 0.0
            return
        if self.placed_steps >= self.args.place_success_steps:
            self.phase = "complete"
            self.phase_steps = 0
            self.completion_lost_steps = 0


class CarryDirectPlaceController(DirectPlaceController):
    """Use phased pickup/transport, then finish with continuous direct placement."""

    def reset(self) -> None:
        PhaseController.reset(self)
        self._reset_direct_place_state()

    def _resume_carry_for_box(self, metrics: dict[str, float | bool]) -> None:
        next_phase = _box_recovery_phase(
            float(metrics["horizontal_distance"]),
            self.args.approach_switch_distance,
        )
        self._change_phase(next_phase)
        self._reset_direct_place_state()
        self.placed_steps = 0
        self.failure_steps = 0
        self.completion_lost_steps = 0
        self.recovery_count += 1

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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, str]:
        if self.phase in ("place_direct", "complete"):
            return DirectPlaceController.latent(
                self,
                policy,
                observation,
                metrics,
                base_rotation=base_rotation,
                box_direction=box_direction,
                goal_direction=goal_direction,
                z=z,
            )
        return PhaseController.latent(
            self,
            policy,
            observation,
            metrics,
            base_rotation=base_rotation,
            box_direction=box_direction,
            goal_direction=goal_direction,
            z=z,
        )

    def update(self, metrics: dict[str, float | bool], *, goal_height: float) -> None:
        if self.phase in ("place_direct", "complete"):
            if self._update_fall_detection(metrics):
                self.fall_event_count += 1
                self._change_phase("get_up")
                self._reset_direct_place_state()
                self.placed_steps = 0
                self.failure_steps = 0
                self.completion_lost_steps = 0
                self.get_up_stable_steps = 0
                return

            previous_phase = self.phase
            DirectPlaceController.update(self, metrics, goal_height=goal_height)

            if previous_phase == "complete" and self.phase == "place_direct":
                self._resume_carry_for_box(metrics)
                return

            if self.phase == "place_direct":
                released_off_target = (
                    self.direct_place_release_latched
                    and not bool(metrics["any_contact"])
                    and abs(float(metrics["object_height"]) - goal_height)
                    <= self.args.place_success_height
                    and float(metrics["object_speed"])
                    <= self.args.place_success_speed
                    and float(metrics["goal_distance"])
                    > self.args.direct_place_success_distance
                )
                self.failure_steps = (
                    self.failure_steps + 1 if released_off_target else 0
                )
                if self.failure_steps >= self.place_retry_steps:
                    self._resume_carry_for_box(metrics)
            return
        PhaseController.update(self, metrics, goal_height=goal_height)
        if self.phase == "place":
            self._change_phase("place_direct")
            self._reset_direct_place_state()


class DirectPushController(PhaseController):
    """Apply one fixed push latent without phase transitions or hidden rewards."""

    def reset(self) -> None:
        super().reset()
        self.phase = "push_direct"

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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, str]:
        del policy, metrics, base_rotation, box_direction, goal_direction
        self.last_transport_turn_weight = 0.0
        return observation, z["push_direct"], "push_direct"

    def update(self, metrics: dict[str, float | bool], *, goal_height: float) -> None:
        del metrics, goal_height
        self.phase_steps += 1


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    robot_training = load_robot_training_spec(args.robot_config)
    joint_names = list(robot_training.robot.control_joint_names)
    box_cfg = _load_box_config(args.box_config)
    if args.box_size is not None:
        box_size = np.asarray(args.box_size, dtype=np.float64)
        if box_size.shape != (3,) or not np.all(np.isfinite(box_size)) or np.any(box_size <= 0.0):
            raise ValueError(f"--box-size must contain three positive finite full dimensions, got {args.box_size}")
        box_cfg = dict(box_cfg, half_size=tuple(float(value) / 2.0 for value in box_size))
    if args.box_mass is not None:
        if not np.isfinite(args.box_mass) or args.box_mass <= 0.0:
            raise ValueError(f"--box-mass must be a positive finite value in kilograms, got {args.box_mass}")
        box_cfg = dict(box_cfg, mass=float(args.box_mass))
    video_label = args.video_label
    if video_label is None:
        label_parts = []
        if args.box_size is not None:
            label_parts.append("Box size: " + " x ".join(f"{float(value):g}" for value in args.box_size) + " m")
        if args.box_mass is not None:
            label_parts.append(f"Mass: {float(args.box_mass):g} kg")
        video_label = " | ".join(label_parts) or None
    if args.box_friction is not None:
        box_cfg = dict(box_cfg, sim2sim_box_friction=tuple(args.box_friction))
    if args.box_condim is not None:
        box_cfg = dict(box_cfg, sim2sim_box_condim=args.box_condim)
    model = _build_model(
        Path(robot_training.robot.xml_path),
        box_cfg,
        render_size=args.render_size,
        show_goal_marker=True,
        goal_marker_radius=(
            args.push_success_distance
            if args.controller_mode in ("push", "push-direct")
            else args.place_success_distance
        ),
    )
    hand_body_ids = tuple(model.body(name).id for name in robot_training.robot.hands)
    foot_body_ids = tuple(model.body(name).id for name in robot_training.robot.feet)
    if len(foot_body_ids) != 2:
        raise ValueError(f"Expected left/right foot bodies, got {robot_training.robot.feet}")
    foot_site_ids = tuple(
        model.site(f"{str(body_name).split('_', 1)[0]}_foot").id
        for body_name in robot_training.robot.feet
    )
    floor_geom_id = model.geom(FLOOR_GEOM_NAME).id
    if args.hand_friction is not None:
        for body_id in hand_body_ids:
            geom_ids = np.flatnonzero(model.geom_bodyid == int(body_id))
            model.geom_friction[geom_ids] = np.asarray(args.hand_friction)
    if str(robot_training.actuator.get("source")) == "yaml":
        _physics_qpos, physics_qvel = _joint_addresses(model, joint_names)
        actuator_cfg = robot_training.actuator["joints"]
        model.dof_armature[physics_qvel] = np.asarray(
            [float(actuator_cfg[name]["armature"]) for name in joint_names]
        )
        model.dof_frictionloss[physics_qvel] = np.asarray(
            [float(actuator_cfg[name]["friction"]) for name in joint_names]
        )
    else:
        _patch_mode15_joint_physics(model, joint_names)
    data = mujoco.MjData(model)
    qpos_addresses, qvel_addresses = _joint_addresses(model, joint_names)
    actuator_indices = _actuator_indices(model, joint_names)
    _motion_key, motion = _load_motion(args.motion_cache, args.motion_id)

    device = torch.device(args.device)
    policy = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint", device=device.type, strict=False
    ).to(device).eval()
    expected_state_dim = 2 * len(joint_names) + 40
    if int(policy.obs_space["state"].shape[0]) != expected_state_dim:
        raise ValueError(
            "The phased interactive controller requires the non-ablated Carry observation: "
            f"got {int(policy.obs_space['state'].shape[0])}D, expected {expected_state_dim}D"
        )
    approach_policy = policy
    if args.approach_model_folder is not None:
        approach_policy = load_model_from_checkpoint_dir(
            args.approach_model_folder / "checkpoint",
            device=device.type,
            strict=False,
        ).to(device).eval()
        approach_state_dim = int(approach_policy.obs_space["state"].shape[0])
        if approach_state_dim != expected_state_dim:
            raise ValueError(
                "The optional approach policy requires the same non-ablated Carry "
                f"observation: got {approach_state_dim}D, expected {expected_state_dim}D"
            )
    latent_dir = args.latent_dir
    if args.controller_mode == "place-direct":
        z = {"direct_place": _load_latent(latent_dir, "z_direct_place", device)}
        if args.direct_place_turn_latent_dir is not None:
            z["direct_place_turn"] = _load_latent_bank(
                args.direct_place_turn_latent_dir,
                "z_transport_turn",
                device,
            )
        if args.direct_place_transport_latent_dir is not None:
            z["direct_place_transport"] = _load_latent(
                args.direct_place_transport_latent_dir,
                "z_transport",
                device,
            )
            transport_slow_path = (
                args.direct_place_transport_latent_dir / "z_transport_slow.pkl"
            )
            if transport_slow_path.is_file():
                z["direct_place_transport_slow"] = _load_latent(
                    args.direct_place_transport_latent_dir,
                    "z_transport_slow",
                    device,
                )
        if args.stand_latent is not None:
            z["direct_place_stand"] = _load_get_up_latent(
                args.stand_latent, device
            )
            if int(z["direct_place_stand"].shape[0]) != 1:
                raise ValueError(
                    f"{args.stand_latent} must contain one standing latent, got "
                    f"{tuple(z['direct_place_stand'].shape)}"
                )
        if args.direct_place_terminal_latent_dir is not None:
            profile_names = (
                "direct_place_slow",
                "direct_place_lower",
                "direct_place_release",
            )
            profile_files = {
                name: args.direct_place_terminal_latent_dir / f"z_{name}.pkl"
                for name in profile_names
            }
            if all(path.is_file() for path in profile_files.values()):
                for name in profile_names:
                    z[name] = _load_latent(
                        args.direct_place_terminal_latent_dir,
                        f"z_{name}",
                        device,
                    )
                for name in ("direct_place_guard", "direct_place_recover"):
                    if name == "direct_place_guard" and not args.direct_place_descent_guard:
                        continue
                    if (args.direct_place_terminal_latent_dir / f"z_{name}.pkl").is_file():
                        z[name] = _load_latent(
                            args.direct_place_terminal_latent_dir,
                            f"z_{name}",
                            device,
                        )
            else:
                z["direct_place_terminal"] = _load_latent(
                    args.direct_place_terminal_latent_dir,
                    "z_direct_place",
                    device,
                )
    elif args.controller_mode == "push-direct":
        z = {"push_direct": _load_latent(latent_dir, "z_push_direct", device)}
    elif args.controller_mode == "push":
        z = {
            name: _load_latent(latent_dir, f"z_{name}", device)
            for name in ("approach", "push_contact", "push_transport", "push_stop")
        }
        z["turn"] = _load_latent_bank(latent_dir, "z_turn", device)
        z["push_turn"] = _load_latent_bank(latent_dir, "z_push_turn", device)
    else:
        z = {
            name: _load_latent(latent_dir, f"z_{name}", device)
            for name in ("approach", "lift", "transport", "align", "lower", "place", "hold")
        }
        # Optional empty-handed standing latent for the placement-complete hold.
        # z_hold remains the "static carry" (holding the box) used during settle.
        if args.stand_latent is not None:
            z["stand"] = _load_get_up_latent(args.stand_latent, device)
            if int(z["stand"].shape[0]) != 1:
                raise ValueError(
                    f"{args.stand_latent} must contain one standing latent, got "
                    f"{tuple(z['stand'].shape)}"
                )
        elif (latent_dir / "z_stand.pkl").is_file():
            z["stand"] = _load_latent(latent_dir, "z_stand", device)
        else:
            z["stand"] = z["hold"]
        if int(z["stand"].shape[1]) != int(z["approach"].shape[1]):
            raise ValueError(
                f"Stand latent dim {int(z['stand'].shape[1])} does not match policy latent dim "
                f"{int(z['approach'].shape[1])}"
            )
        z["get_up"] = _load_get_up_latent(args.get_up_latent, device)
        if int(z["get_up"].shape[1]) != int(z["approach"].shape[1]):
            raise ValueError(
                f"Get-up latent dim {int(z['get_up'].shape[1])} does not match policy latent dim "
                f"{int(z['approach'].shape[1])}"
            )
        z["turn"] = _load_latent_bank(latent_dir, "z_turn", device)
        z["transport_turn"] = _load_latent_bank(latent_dir, "z_transport_turn", device)
        if args.lower_low_latent is not None:
            z["lower_low"] = torch.as_tensor(
                joblib.load(args.lower_low_latent),
                dtype=torch.float32,
                device=device,
            ).reshape(1, -1)
            if int(z["lower_low"].shape[1]) != int(z["lower"].shape[1]):
                raise ValueError(
                    f"Low lower latent dim {int(z['lower_low'].shape[1])} does not match "
                    f"lower latent dim {int(z['lower'].shape[1])}"
                )
        if args.controller_mode == "carry-direct-place":
            z["direct_place"] = _load_latent(latent_dir, "z_direct_place", device)
            z["direct_place_stand"] = z["stand"]
            if args.direct_place_turn_latent_dir is not None:
                z["direct_place_turn"] = _load_latent_bank(
                    args.direct_place_turn_latent_dir,
                    "z_transport_turn",
                    device,
                )
            profile_dir = args.direct_place_terminal_latent_dir or latent_dir
            profile_names = (
                "direct_place_slow",
                "direct_place_lower",
                "direct_place_release",
            )
            missing_profiles = [
                f"z_{name}.pkl"
                for name in profile_names
                if not (profile_dir / f"z_{name}.pkl").is_file()
            ]
            if missing_profiles:
                raise FileNotFoundError(
                    f"Hybrid direct placement requires {missing_profiles} in {profile_dir}"
                )
            for name in profile_names:
                z[name] = _load_latent(profile_dir, f"z_{name}", device)
            for name in ("direct_place_guard", "direct_place_recover"):
                if name == "direct_place_guard" and not args.direct_place_descent_guard:
                    continue
                if (profile_dir / f"z_{name}.pkl").is_file():
                    z[name] = _load_latent(profile_dir, f"z_{name}", device)

    if args.approach_latent_dir is not None:
        if "approach" not in z:
            raise ValueError(
                "--approach-latent-dir is only valid for controllers with an approach phase"
            )
        base_approach_dim = int(z["approach"].shape[1])
        approach_override = _load_latent(
            args.approach_latent_dir,
            "z_approach",
            device,
        )
        if int(approach_override.shape[1]) != base_approach_dim:
            raise ValueError(
                f"Approach override latent dim {int(approach_override.shape[1])} does not "
                f"match policy latent dim {base_approach_dim}"
            )
        z["approach"] = approach_override
    if args.approach_turn_latent_dir is not None:
        if "approach" not in z or "turn" not in z:
            raise ValueError(
                "--approach-turn-latent-dir is only valid for controllers with an "
                "approach turn bank"
            )
        approach_turn_override = _load_latent_bank(
            args.approach_turn_latent_dir,
            "z_turn",
            device,
        )
        base_approach_dim = int(z["approach"].shape[1])
        for direction, latent in approach_turn_override.items():
            if int(latent.shape[1]) != base_approach_dim:
                raise ValueError(
                    f"Approach turn override {direction} latent dim {int(latent.shape[1])} "
                    f"does not match policy latent dim {base_approach_dim}"
                )
        z["turn"] = approach_turn_override

    defaults = np.asarray([robot_training.default_joint_angles[name] for name in joint_names])
    kp = np.asarray([_match_joint_parameter(name, robot_training.stiffness) for name in joint_names])
    kd = np.asarray([_match_joint_parameter(name, robot_training.damping) for name in joint_names])
    effort = np.asarray(robot_training.effort_limits)
    limits = np.asarray([robot_training.robot.joint_ranges[name] for name in joint_names])
    base_scale = np.asarray([
        float(robot_training.action_scale_overrides.get(name, robot_training.action_scale)) for name in joint_names
    ])
    target_scale = base_scale * effort / kp
    base_body_id = model.body(robot_training.robot.base_body).id
    box_body_id = model.body(BOX_BODY_NAME).id
    box_geom_id = model.geom(BOX_GEOM_NAME).id
    box_joint_id = model.joint(BOX_JOINT_NAME).id
    box_qpos_address = int(model.jnt_qposadr[box_joint_id])
    box_qvel_address = int(model.jnt_dofadr[box_joint_id])
    goal_geom_id = model.geom(GOAL_GEOM_NAME).id
    ground_height = float(box_cfg.get("ground_center_height", 0.185))
    contact_hysteresis = CapsuleBoxContactHysteresis(
        box_cfg,
        hand_body_ids=hand_body_ids,
        box_qpos_address=box_qpos_address,
        enter_distance=args.contact_enter_distance,
        exit_distance=args.contact_exit_distance,
    )

    initial_dof_pos = _initialise_state(
        model,
        data,
        motion,
        initial_frame=args.initial_frame,
        initial_distance=args.initial_distance,
        joint_names=joint_names,
        qpos_addresses=qpos_addresses,
        standing_init=args.standing_init,
        standing_root_height=args.standing_root_height,
        default_dof_pos=defaults,
        initial_bearing_deg=args.initial_bearing_deg,
        initial_box_center_height=args.initial_box_center_height,
        preserve_root_object_relative_pose=args.controller_mode == "place-direct",
    )
    _set_initial_box_center_height(
        model,
        data,
        qpos_address=box_qpos_address,
        qvel_address=box_qvel_address,
        center_height=args.initial_box_center_height,
    )
    _align_initial_box_yaw_with_robot(
        model,
        data,
        qpos_address=box_qpos_address,
        qvel_address=box_qvel_address,
        enabled=args.align_initial_box_yaw_with_robot,
    )
    initial_robot_yaw_deg = math.degrees(_yaw_from_wxyz(data.qpos[3:7]))
    initial_box_rotation = data.geom_xmat[box_geom_id].reshape(3, 3)
    initial_box_yaw_deg = math.degrees(
        math.atan2(initial_box_rotation[1, 0], initial_box_rotation[0, 0])
    )
    _apply_initial_lie_down(
        model,
        data,
        side=args.start_fallen_side,
        root_height=args.start_fallen_root_height,
    )
    initial_box_pos = data.qpos[box_qpos_address : box_qpos_address + 3].copy()
    configured_goal = initial_box_pos.copy()
    configured_goal[:2] += np.asarray((args.goal_offset_x, args.goal_offset_y))
    configured_goal[2] = ground_height
    interaction = {
        "goal": configured_goal.copy(),
        "goal_changed": True,
        "reset": False,
        "restore_goal": False,
        "print": False,
    }
    lock = threading.Lock()

    def key_callback(keycode: int) -> None:
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        with lock:
            if key == "W":
                interaction["goal"][0] += args.goal_step
                interaction["goal_changed"] = True
            elif key == "S":
                interaction["goal"][0] -= args.goal_step
                interaction["goal_changed"] = True
            elif key == "A":
                interaction["goal"][1] += args.goal_step
                interaction["goal_changed"] = True
            elif key == "D":
                interaction["goal"][1] -= args.goal_step
                interaction["goal_changed"] = True
            elif key == "H":
                interaction["restore_goal"] = True
            elif key == "R":
                interaction["reset"] = True
            elif key == "P":
                interaction["print"] = True

    observer = ActorObservation(
        device=device,
        default_dof_pos=defaults,
        target_object_pos=configured_goal,
        hand_contact_offsets=np.asarray(box_cfg["hand_contact_offsets"]),
        include_hand_spatial=True,
    )
    if args.controller_mode == "place-direct":
        controller = DirectPlaceController(args, dt=CONTROL_DT)
    elif args.controller_mode == "carry-direct-place":
        controller = CarryDirectPlaceController(args, dt=CONTROL_DT)
    elif args.controller_mode == "push-direct":
        controller = DirectPushController(args, dt=CONTROL_DT)
    elif args.controller_mode == "push":
        controller = PushPhaseController(args, dt=CONTROL_DT)
    else:
        controller = PhaseController(args, dt=CONTROL_DT)
    current_action = np.clip((initial_dof_pos - defaults) / target_scale, -args.action_clip, args.action_clip)
    previous_q_target = initial_dof_pos.copy()
    substeps = round(CONTROL_DT / SIM_DT)
    step = 0
    mode = "starting"
    drop_start_step: int | None = None
    drop_finished = args.box_drop_force_n <= 0.0
    transport_stable_steps = 0
    complete_stable_steps = 0
    box_relocated = not np.any(np.asarray(args.relocate_box_offset, dtype=np.float64))
    event_log: list[dict[str, Any]] = []
    rollout_history: list[dict[str, float | bool | str]] = []

    trace_stream = None
    trace_writer = None
    if args.trace_csv is not None:
        args.trace_csv.parent.mkdir(parents=True, exist_ok=True)
        trace_stream = args.trace_csv.open("w", encoding="utf-8", newline="", buffering=1)
        trace_fields = (
            "step",
            "time_s",
            "phase_before",
            "phase_after",
            "mode",
            "transport_turn_weight",
            "transport_forward_weight",
            "goal_changed",
            "reset_requested",
            "goal_x",
            "goal_y",
            "goal_z",
            "goal_distance_m",
            "robot_box_distance_m",
            "object_height_m",
            "object_speed_mps",
            "place_descent_braking",
            "place_lower_progress_steps",
            "object_lin_vel_x",
            "object_lin_vel_y",
            "object_lin_vel_z",
            "object_ang_vel_x",
            "object_ang_vel_y",
            "object_ang_vel_z",
            "lift",
            "root_height_m",
            "pelvis_up_z",
            "robot_upright",
            "box_upright",
            "left_contact",
            "right_contact",
            "left_physical_contact",
            "right_physical_contact",
            "left_foot_ground_contact",
            "right_foot_ground_contact",
            "bilateral_foot_ground_contact",
            "left_foot_height_m",
            "right_foot_height_m",
            "left_foot_horizontal_speed_mps",
            "right_foot_horizontal_speed_mps",
            "left_foot_scuff",
            "right_foot_scuff",
            "any_foot_scuff",
            "left_gap_m",
            "right_gap_m",
            "box_x",
            "box_y",
            "box_z",
            "root_x",
            "root_y",
            "root_z",
            "box_force_x",
            "box_force_y",
            "box_force_z",
            "box_torque_x",
            "box_torque_y",
            "box_torque_z",
            "force_active",
            "place_attempt",
            "recovery_count",
            "fall_event_count",
            "fall_detect_steps",
            "get_up_stable_steps",
            "phase_steps",
            "ready_steps",
            "lost_steps",
            "place_enter_steps",
            "place_drift_steps",
            "placed_steps",
            "failure_steps",
            "completion_lost_steps",
            "release_started_step",
        )
        trace_writer = csv.DictWriter(trace_stream, fieldnames=trace_fields)
        trace_writer.writeheader()

    renderer = None
    video_writer = None
    camera = None
    capture_accumulator = 1.0
    if args.save_mp4:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        renderer = _MujocoRenderer(model, height=args.render_size, width=args.render_size)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = float(args.camera_distance)
        camera.azimuth = float(args.camera_azimuth)
        camera.elevation = float(args.camera_elevation)
        video_writer = media.VideoWriter(
            args.output_path,
            shape=(args.render_size, args.render_size),
            fps=float(args.video_fps),
        )
        video_writer.__enter__()

    print("[interactive] W/S: goal +/-X, A/D: goal +/-Y, H: restore goal, R: reset, P: status")
    print(f"[interactive] green marker goal={configured_goal.tolist()} step={args.goal_step:.2f}m")
    print(
        f"[contact] source={args.contact_source} "
        f"enter={args.contact_enter_distance * 100.0:.1f}cm "
        f"exit={args.contact_exit_distance * 100.0:.1f}cm",
        flush=True,
    )
    if args.box_drop_force_n > 0.0:
        print(
            f"[disturbance] box force=(0, 0, -{args.box_drop_force_n:.1f})N for "
            f"{args.box_drop_force_seconds:.2f}s after {args.box_drop_transport_delay_seconds:.2f}s of transport",
            flush=True,
        )
    if not box_relocated:
        print(
            f"[disturbance] after placement, relocate box by "
            f"({args.relocate_box_offset[0]:.2f}, {args.relocate_box_offset[1]:.2f})m",
            flush=True,
        )
    viewer_context = (
        __import__("mujoco.viewer", fromlist=["launch_passive"]).launch_passive(model, data, key_callback=key_callback)
        if args.viewer
        else nullcontext(None)
    )
    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = (float(initial_box_pos[0]), float(initial_box_pos[1]), 0.7)
            viewer.cam.distance = args.camera_distance
            viewer.cam.azimuth = args.camera_azimuth
            viewer.cam.elevation = args.camera_elevation
        while (viewer is None or viewer.is_running()) and (args.max_steps <= 0 or step < args.max_steps):
            started = time.monotonic()
            with lock:
                if interaction["restore_goal"]:
                    interaction["goal"][:] = configured_goal
                    interaction["goal_changed"] = True
                    interaction["restore_goal"] = False
                should_reset = bool(interaction["reset"])
                should_print = bool(interaction["print"])
                interaction["reset"] = False
                interaction["print"] = False
                goal = interaction["goal"].copy()
                goal_changed = bool(interaction["goal_changed"])
                interaction["goal_changed"] = False
            if should_reset:
                initial_dof_pos = _initialise_state(
                    model,
                    data,
                    motion,
                    initial_frame=args.initial_frame,
                    initial_distance=args.initial_distance,
                    joint_names=joint_names,
                    qpos_addresses=qpos_addresses,
                    standing_init=args.standing_init,
                    standing_root_height=args.standing_root_height,
                    default_dof_pos=defaults,
                    initial_bearing_deg=args.initial_bearing_deg,
                    initial_box_center_height=args.initial_box_center_height,
                    preserve_root_object_relative_pose=args.controller_mode == "place-direct",
                )
                _set_initial_box_center_height(
                    model,
                    data,
                    qpos_address=box_qpos_address,
                    qvel_address=box_qvel_address,
                    center_height=args.initial_box_center_height,
                )
                _align_initial_box_yaw_with_robot(
                    model,
                    data,
                    qpos_address=box_qpos_address,
                    qvel_address=box_qvel_address,
                    enabled=args.align_initial_box_yaw_with_robot,
                )
                _apply_initial_lie_down(
                    model,
                    data,
                    side=args.start_fallen_side,
                    root_height=args.start_fallen_root_height,
                )
                current_action = np.clip(
                    (initial_dof_pos - defaults) / target_scale, -args.action_clip, args.action_clip
                )
                previous_q_target = initial_dof_pos.copy()
                observer.reset_history()
                controller.reset()
                contact_hysteresis.reset()
                data.xfrc_applied[:] = 0.0
                print("[interactive] simulation reset", flush=True)
            if goal_changed:
                observer.set_target_object_pos(goal)
                model.geom_pos[goal_geom_id] = (goal[0], goal[1], 0.006)
                print(f"[interactive] goal -> x={goal[0]:.2f}, y={goal[1]:.2f}", flush=True)

            dof_pos = data.qpos[qpos_addresses].copy()
            dof_vel = data.qvel[qvel_addresses].copy()
            _base_lin_vel, base_ang_vel = _world_body_velocity(model, data, base_body_id)
            object_lin_vel, object_ang_vel = _world_body_velocity(model, data, box_body_id)
            physical_contact = _physical_hand_contacts(
                model, data, box_geom_id=box_geom_id, hand_body_ids=hand_body_ids
            )
            hysteresis_contact, _surface_distance = contact_hysteresis.update(data)
            contact = physical_contact if args.contact_source == "physical" else hysteresis_contact
            observation = observer.build(
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                base_pos=data.xpos[base_body_id],
                base_quat_wxyz=data.xquat[base_body_id],
                base_ang_vel_world=base_ang_vel,
                object_pos=data.qpos[box_qpos_address : box_qpos_address + 3],
                object_quat_wxyz=data.qpos[box_qpos_address + 3 : box_qpos_address + 7],
                object_lin_vel=object_lin_vel,
                object_ang_vel=object_ang_vel,
                hand_body_pos=np.asarray([data.xpos[body_id] for body_id in hand_body_ids]),
                hand_body_quat_wxyz=np.asarray([data.xquat[body_id] for body_id in hand_body_ids]),
                hand_contact=contact,
                current_action=current_action,
            )
            metrics = _controller_metrics(
                model,
                data,
                base_body_id=base_body_id,
                box_body_id=box_body_id,
                box_qpos_address=box_qpos_address,
                contact=contact,
                goal=goal,
                ground_height=ground_height,
                lift_height=float(box_cfg.get("lift_height", 0.35)),
                push_standoff_distance=args.push_standoff_distance,
            )
            active_phase = controller.phase
            active_policy = approach_policy if active_phase == "approach" else policy
            box_pos = data.qpos[box_qpos_address : box_qpos_address + 3]
            policy_observation, latent, mode = controller.latent(
                active_policy,
                observation,
                metrics,
                base_rotation=data.xmat[base_body_id].reshape(3, 3),
                box_direction=box_pos[:2] - data.xpos[base_body_id, :2],
                goal_direction=goal[:2] - box_pos[:2],
                z=z,
            )
            raw_action = active_policy.act(
                policy_observation,
                latent,
                mean=True,
            )[0].detach().cpu().numpy()
            current_action = np.clip(raw_action * float(robot_training.normalize_action_to), -args.action_clip, args.action_clip)
            q_target = np.clip(defaults + current_action * target_scale, limits[:, 0], limits[:, 1])
            action_slew_rad_s = 0.0
            if active_phase == "place":
                action_slew_rad_s = float(args.place_action_slew_rad_s)
            elif active_phase == "place_direct":
                action_slew_rad_s = float(args.direct_place_action_slew_rad_s)
                initial_goal_distance = controller.direct_place_initial_goal_distance
                if (
                    initial_goal_distance is not None
                    and initial_goal_distance
                    >= float(args.direct_place_profile_far_commit_distance)
                    and args.direct_place_far_action_slew_rad_s is not None
                ):
                    action_slew_rad_s = float(args.direct_place_far_action_slew_rad_s)
            if action_slew_rad_s > 0.0:
                max_delta = action_slew_rad_s * CONTROL_DT
                q_target = np.clip(
                    q_target,
                    previous_q_target - max_delta,
                    previous_q_target + max_delta,
                )
                q_target = np.clip(q_target, limits[:, 0], limits[:, 1])
                current_action = np.clip(
                    (q_target - defaults) / target_scale,
                    -args.action_clip,
                    args.action_clip,
                )
            previous_q_target = q_target.copy()

            if args.box_drop_force_n > 0.0 and drop_start_step is None:
                transport_stable_steps = transport_stable_steps + 1 if controller.phase == "transport" else 0
                if (
                    controller.phase == "transport"
                    and transport_stable_steps
                    >= max(1, round(args.box_drop_transport_delay_seconds / CONTROL_DT))
                ):
                    drop_start_step = step
                    event_log.append(
                        {
                            "event": "box_downward_force_start",
                            "time_s": step * CONTROL_DT,
                            "force_n": float(args.box_drop_force_n),
                            "duration_s": float(args.box_drop_force_seconds),
                        }
                    )
                    print(
                        f"[disturbance] t={step * CONTROL_DT:.2f}s start box downward force "
                        f"{args.box_drop_force_n:.1f}N",
                        flush=True,
                    )
            force_duration_steps = max(1, round(args.box_drop_force_seconds / CONTROL_DT))
            force_active = args.box_drop_force_n > 0.0 and _control_window_active(
                step,
                start_step=drop_start_step,
                duration_steps=force_duration_steps,
            )
            # Passive viewer perturbations are written to xfrc_applied. Preserve
            # them unless this run explicitly owns the box-force channel.
            if args.box_drop_force_n > 0.0:
                data.xfrc_applied[box_body_id] = 0.0
                if force_active:
                    data.xfrc_applied[box_body_id, 2] = -float(args.box_drop_force_n)
            for _ in range(substeps):
                torque = np.clip(
                    kp * (q_target - data.qpos[qpos_addresses]) - kd * data.qvel[qvel_addresses],
                    -effort,
                    effort,
                )
                data.ctrl[actuator_indices] = torque
                mujoco.mj_step(model, data)
            if (
                not drop_finished
                and drop_start_step is not None
                and step + 1 >= drop_start_step + force_duration_steps
            ):
                drop_finished = True
                data.xfrc_applied[box_body_id] = 0.0
                event_log.append({"event": "box_downward_force_end", "time_s": (step + 1) * CONTROL_DT})
                print(f"[disturbance] t={(step + 1) * CONTROL_DT:.2f}s end box downward force", flush=True)
            post_physical_contact = _physical_hand_contacts(
                model, data, box_geom_id=box_geom_id, hand_body_ids=hand_body_ids
            )
            post_foot_contact = _physical_ground_contacts(
                model,
                data,
                floor_geom_id=floor_geom_id,
                body_ids=foot_body_ids,
            )
            post_foot_velocity = tuple(
                _world_site_linear_velocity(model, data, site_id) for site_id in foot_site_ids
            )
            post_foot_height = tuple(float(data.site_xpos[site_id, 2]) for site_id in foot_site_ids)
            post_foot_horizontal_speed = tuple(
                float(np.linalg.norm(velocity[:2])) for velocity in post_foot_velocity
            )
            post_foot_scuff = tuple(
                bool(contact > 0.5 and speed >= float(args.foot_scuff_speed_threshold))
                for contact, speed in zip(post_foot_contact, post_foot_horizontal_speed)
            )
            post_hysteresis_contact, post_surface_distance = contact_hysteresis.update(data)
            post_contact = (
                post_physical_contact
                if args.contact_source == "physical"
                else post_hysteresis_contact
            )
            post_metrics = _controller_metrics(
                model,
                data,
                base_body_id=base_body_id,
                box_body_id=box_body_id,
                box_qpos_address=box_qpos_address,
                contact=post_contact,
                goal=goal,
                ground_height=ground_height,
                lift_height=float(box_cfg.get("lift_height", 0.35)),
                push_standoff_distance=args.push_standoff_distance,
            )
            post_metrics.update(
                {
                    "left_foot_ground_contact": bool(post_foot_contact[0]),
                    "right_foot_ground_contact": bool(post_foot_contact[1]),
                    "bilateral_foot_ground_contact": bool(np.all(post_foot_contact > 0.5)),
                    "left_foot_height": post_foot_height[0],
                    "right_foot_height": post_foot_height[1],
                    "left_foot_horizontal_speed": post_foot_horizontal_speed[0],
                    "right_foot_horizontal_speed": post_foot_horizontal_speed[1],
                    "left_foot_scuff": post_foot_scuff[0],
                    "right_foot_scuff": post_foot_scuff[1],
                    "any_foot_scuff": bool(any(post_foot_scuff)),
                }
            )
            rollout_history.append({"phase": active_phase, **post_metrics})
            if force_active and (step - int(drop_start_step or step)) % 10 == 0:
                print(
                    f"[disturbance] force step={step - int(drop_start_step or step):02d} "
                    f"box_h={float(post_metrics['object_height']):.3f}m "
                    f"contact={int(post_contact[0])}/{int(post_contact[1])} "
                    f"physical={int(post_physical_contact[0])}/{int(post_physical_contact[1])} "
                    f"gap_cm={post_surface_distance[0] * 100.0:.1f}/{post_surface_distance[1] * 100.0:.1f}",
                    flush=True,
                )
            controller.update(post_metrics, goal_height=ground_height)

            if trace_writer is not None:
                post_object_lin_vel, post_object_ang_vel = _world_body_velocity(model, data, box_body_id)
                trace_writer.writerow(
                    {
                        "step": step + 1,
                        "time_s": (step + 1) * CONTROL_DT,
                        "phase_before": active_phase,
                        "phase_after": controller.phase,
                        "mode": mode,
                        "transport_turn_weight": controller.last_transport_turn_weight,
                        "transport_forward_weight": 1.0 - controller.last_transport_turn_weight
                        if active_phase in ("transport", "push_direct", "place_direct")
                        else 0.0,
                        "goal_changed": int(goal_changed),
                        "reset_requested": int(should_reset),
                        "goal_x": goal[0],
                        "goal_y": goal[1],
                        "goal_z": goal[2],
                        "goal_distance_m": post_metrics["goal_distance"],
                        "robot_box_distance_m": post_metrics["horizontal_distance"],
                        "object_height_m": post_metrics["object_height"],
                        "object_speed_mps": post_metrics["object_speed"],
                        "place_descent_braking": int(controller.place_descent_braking),
                        "place_lower_progress_steps": controller.place_lower_progress_steps,
                        "object_lin_vel_x": post_object_lin_vel[0],
                        "object_lin_vel_y": post_object_lin_vel[1],
                        "object_lin_vel_z": post_object_lin_vel[2],
                        "object_ang_vel_x": post_object_ang_vel[0],
                        "object_ang_vel_y": post_object_ang_vel[1],
                        "object_ang_vel_z": post_object_ang_vel[2],
                        "lift": post_metrics["lift"],
                        "root_height_m": post_metrics["root_height"],
                        "pelvis_up_z": post_metrics["pelvis_up_z"],
                        "robot_upright": post_metrics["robot_upright"],
                        "box_upright": post_metrics["box_upright"],
                        "left_contact": int(post_contact[0]),
                        "right_contact": int(post_contact[1]),
                        "left_physical_contact": int(post_physical_contact[0]),
                        "right_physical_contact": int(post_physical_contact[1]),
                        "left_foot_ground_contact": int(post_foot_contact[0]),
                        "right_foot_ground_contact": int(post_foot_contact[1]),
                        "bilateral_foot_ground_contact": int(np.all(post_foot_contact > 0.5)),
                        "left_foot_height_m": post_foot_height[0],
                        "right_foot_height_m": post_foot_height[1],
                        "left_foot_horizontal_speed_mps": post_foot_horizontal_speed[0],
                        "right_foot_horizontal_speed_mps": post_foot_horizontal_speed[1],
                        "left_foot_scuff": int(post_foot_scuff[0]),
                        "right_foot_scuff": int(post_foot_scuff[1]),
                        "any_foot_scuff": int(any(post_foot_scuff)),
                        "left_gap_m": post_surface_distance[0],
                        "right_gap_m": post_surface_distance[1],
                        "box_x": data.qpos[box_qpos_address],
                        "box_y": data.qpos[box_qpos_address + 1],
                        "box_z": data.qpos[box_qpos_address + 2],
                        "root_x": data.xpos[base_body_id, 0],
                        "root_y": data.xpos[base_body_id, 1],
                        "root_z": data.xpos[base_body_id, 2],
                        "box_force_x": data.xfrc_applied[box_body_id, 0],
                        "box_force_y": data.xfrc_applied[box_body_id, 1],
                        "box_force_z": data.xfrc_applied[box_body_id, 2],
                        "box_torque_x": data.xfrc_applied[box_body_id, 3],
                        "box_torque_y": data.xfrc_applied[box_body_id, 4],
                        "box_torque_z": data.xfrc_applied[box_body_id, 5],
                        "force_active": int(force_active),
                        "place_attempt": controller.place_attempt,
                        "recovery_count": controller.recovery_count,
                        "fall_event_count": controller.fall_event_count,
                        "fall_detect_steps": controller.fall_detect_steps,
                        "get_up_stable_steps": controller.get_up_stable_steps,
                        "phase_steps": controller.phase_steps,
                        "ready_steps": controller.ready_steps,
                        "lost_steps": controller.lost_steps,
                        "place_enter_steps": controller.place_enter_steps,
                        "place_drift_steps": controller.place_drift_steps,
                        "placed_steps": controller.placed_steps,
                        "failure_steps": controller.failure_steps,
                        "completion_lost_steps": controller.completion_lost_steps,
                        "release_started_step": ""
                        if controller.release_started_step is None
                        else controller.release_started_step,
                    }
                )

            if drop_finished and not box_relocated:
                complete_stable_steps = complete_stable_steps + 1 if controller.phase == "complete" else 0
                if (
                    controller.phase == "complete"
                    and complete_stable_steps
                    >= max(1, round(args.relocate_after_complete_seconds / CONTROL_DT))
                ):
                    before = data.qpos[box_qpos_address : box_qpos_address + 3].copy()
                    _relocate_box(
                        model,
                        data,
                        qpos_address=box_qpos_address,
                        qvel_address=box_qvel_address,
                        offset_xy=np.asarray(args.relocate_box_offset, dtype=np.float64),
                        ground_height=ground_height,
                    )
                    after = data.qpos[box_qpos_address : box_qpos_address + 3].copy()
                    box_relocated = True
                    event_log.append(
                        {
                            "event": "box_relocated",
                            "time_s": (step + 1) * CONTROL_DT,
                            "offset_xy": [float(value) for value in args.relocate_box_offset],
                            "before": before.tolist(),
                            "after": after.tolist(),
                        }
                    )
                    print(
                        f"[disturbance] t={(step + 1) * CONTROL_DT:.2f}s relocate box "
                        f"by ({args.relocate_box_offset[0]:.2f}, {args.relocate_box_offset[1]:.2f})m",
                        flush=True,
                    )
            step += 1
            if should_print or step % args.log_every == 0:
                print(
                    f"[interactive] step={step} phase={controller.phase} mode={mode} "
                    f"goal_dist={float(post_metrics['goal_distance']):.3f}m "
                    f"box_dist={float(post_metrics['horizontal_distance']):.3f}m "
                    f"box_h={float(post_metrics['object_height']):.3f}m "
                    f"box_speed={float(post_metrics['object_speed']):.3f}m/s "
                    f"pelvis_z={float(post_metrics['root_height']):.3f}m "
                    f"pelvis_up={float(post_metrics['pelvis_up_z']):.3f} "
                    f"box_upright={float(post_metrics['box_upright']):.3f} "
                    f"contact={int(post_contact[0])}/{int(post_contact[1])} "
                    f"physical={int(post_physical_contact[0])}/{int(post_physical_contact[1])} "
                    f"gap_cm={post_surface_distance[0] * 100.0:.1f}/{post_surface_distance[1] * 100.0:.1f} "
                    f"attempt={controller.place_attempt} recovery={controller.recovery_count}",
                    flush=True,
                )
            if viewer is not None:
                viewer.sync()
                remaining = CONTROL_DT - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
            if renderer is not None and video_writer is not None and camera is not None:
                capture_accumulator += float(args.video_fps) * CONTROL_DT
                if capture_accumulator >= 1.0:
                    box_pos = data.qpos[box_qpos_address : box_qpos_address + 3]
                    focus_xy = (data.xpos[base_body_id, :2] + box_pos[:2] + goal[:2]) / 3.0
                    camera.lookat[:] = (float(focus_xy[0]), float(focus_xy[1]), 0.75)
                    renderer.update_scene(data, camera=camera)
                    frame = np.asarray(renderer.render()).copy()
                    video_writer.add_image(_annotate_video_frame(frame, video_label))
                    capture_accumulator -= 1.0

    if video_writer is not None:
        video_writer.close()
        print(f"[recording] saved MuJoCo video to {args.output_path}", flush=True)
    if renderer is not None:
        renderer.close()
    if trace_stream is not None:
        trace_stream.close()
        print(f"[trace] saved per-step CSV to {args.trace_csv}", flush=True)
    if event_log and args.save_mp4:
        event_path = args.output_path.with_suffix(".events.json")
        event_path.write_text(json.dumps(event_log, indent=2), encoding="utf-8")
        print(f"[disturbance] event log saved to {event_path}", flush=True)
    if args.summary_path is not None:
        summary = _summarize_speed_rollout(
            rollout_history,
            final_phase=controller.phase,
            placement_ground_height=ground_height,
        )
        summary.update(
            {
                "model_folder": str(args.model_folder),
                "approach_model_folder": (
                    None
                    if args.approach_model_folder is None
                    else str(args.approach_model_folder)
                ),
                "latent_dir": str(args.latent_dir),
                "approach_latent_dir": (
                    None
                    if args.approach_latent_dir is None
                    else str(args.approach_latent_dir)
                ),
                "approach_turn_latent_dir": (
                    None
                    if args.approach_turn_latent_dir is None
                    else str(args.approach_turn_latent_dir)
                ),
                "stand_latent": None if args.stand_latent is None else str(args.stand_latent),
                "legacy_736_flow": bool(args.legacy_736_flow),
                "accept_settled_ground_release": bool(args.accept_settled_ground_release),
                "initial_distance": float(args.initial_distance),
                "standing_init": bool(args.standing_init),
                "standing_root_height": float(args.standing_root_height),
                "approach_switch_distance": float(args.approach_switch_distance),
                "approach_max_turn_blend": float(args.max_turn_blend),
                "approach_turn_enter_deg": float(args.approach_turn_enter_deg),
                "approach_turn_exit_deg": float(args.approach_turn_exit_deg),
                "approach_turn_blend_seconds": float(
                    args.approach_turn_blend_seconds
                ),
                "direct_place_descent_guard": bool(
                    args.direct_place_descent_guard
                ),
                "initial_bearing_deg": args.initial_bearing_deg,
                "initial_box_center_height": args.initial_box_center_height,
                "align_initial_box_yaw_with_robot": bool(
                    args.align_initial_box_yaw_with_robot
                ),
                "initial_robot_yaw_deg": float(initial_robot_yaw_deg),
                "initial_box_yaw_deg": float(initial_box_yaw_deg),
                "goal_offset": [float(args.goal_offset_x), float(args.goal_offset_y)],
                "fall_event_count": int(controller.fall_event_count),
                "recovery_count": int(controller.recovery_count),
            }
        )
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[metrics] saved rollout summary to {args.summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument(
        "--approach-model-folder",
        type=Path,
        default=None,
        help=(
            "Optional compatible Carry checkpoint used only while phase=approach; the "
            "main --model-folder policy resumes before lift and all later phases."
        ),
    )
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument(
        "--approach-latent-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory whose z_approach.pkl replaces only the approach latent; "
            "all later phase latents remain loaded from --latent-dir."
        ),
    )
    parser.add_argument(
        "--approach-turn-latent-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory whose z_turn.pkl replaces only the approach steering "
            "bank; transport continues to use z_transport_turn.pkl from --latent-dir."
        ),
    )
    parser.add_argument(
        "--stand-latent",
        type=Path,
        default=None,
        help="Optional single empty-handed standing latent used after placement.",
    )
    parser.add_argument(
        "--controller-mode",
        choices=("carry", "carry-direct-place", "place-direct", "push", "push-direct"),
        default="carry",
        help=(
            "Use the original lift/carry/place controller, phased pickup followed by direct "
            "placement, a carried-start direct-place controller, the phased ground-push "
            "controller, or one fixed direct-push latent."
        ),
    )
    parser.add_argument(
        "--legacy-736-flow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the step-73670656 Carry phase transitions and placement latent flow; "
            "disabled by default."
        ),
    )
    parser.add_argument(
        "--direct-place-terminal-latent-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing continuous slow/lower/release latents. "
            "The controller blends them by distance and lift without changing phase."
        ),
    )
    parser.add_argument(
        "--direct-place-goal-lookahead",
        type=float,
        default=0.0,
        help="Extra observed local XY goal distance used to compensate stopping bias.",
    )
    parser.add_argument(
        "--direct-place-turn-latent-dir",
        type=Path,
        default=None,
        help="Optional directory containing z_transport_turn.pkl for far-field steering.",
    )
    parser.add_argument(
        "--direct-place-transport-latent-dir",
        type=Path,
        default=None,
        help="Optional directory containing the matching z_transport.pkl carry baseline.",
    )
    parser.add_argument(
        "--direct-place-lateral-direct-min-deg",
        type=float,
        default=None,
        help="Route bearings at or above this angle through the direct-place baseline.",
    )
    parser.add_argument(
        "--direct-place-lateral-direct-max-deg",
        type=float,
        default=120.0,
        help="Upper bearing bound for the lateral direct-place route.",
    )
    parser.add_argument(
        "--direct-place-positive-lateral-use-transport",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Route positive signed lateral bearings through the transport latent.",
    )
    parser.add_argument(
        "--direct-place-positive-lateral-transport-max-deg",
        type=float,
        default=150.0,
        help="Upper absolute bearing for the positive-lateral transport route.",
    )
    parser.add_argument(
        "--direct-place-rear-direct-min-deg",
        type=float,
        default=None,
        help="Use gated direct-place steering above this bearing for close rear goals.",
    )
    parser.add_argument(
        "--direct-place-rear-direct-max-distance",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--direct-place-far-direct-distance",
        type=float,
        default=None,
        help="Use the direct-place baseline at or beyond this distance.",
    )
    parser.add_argument(
        "--direct-place-far-direct-max-bearing-deg",
        type=float,
        default=60.0,
        help=(
            "Disable steering for a far direct-place route only when the initial "
            "absolute bearing is at or below this angle."
        ),
    )
    parser.add_argument(
        "--direct-place-transport-goal-lookahead",
        type=float,
        default=0.0,
        help="Extra observed goal distance used only by the transport route.",
    )
    parser.add_argument(
        "--direct-place-rear-lower-descent-speed", type=float, default=None
    )
    parser.add_argument(
        "--direct-place-rear-lower-brake-rate", type=float, default=4.0
    )
    parser.add_argument(
        "--direct-place-lateral-commit-distance",
        type=float,
        default=0.32,
    )
    parser.add_argument(
        "--direct-place-lateral-commit-speed",
        type=float,
        default=0.12,
    )
    parser.add_argument("--direct-place-turn-heading-full-deg", type=float, default=45.0)
    parser.add_argument("--direct-place-turn-max-blend", type=float, default=0.45)
    parser.add_argument("--direct-place-turn-fade-start-distance", type=float, default=0.80)
    parser.add_argument("--direct-place-turn-fade-end-distance", type=float, default=0.60)
    parser.add_argument(
        "--direct-place-turn-ready-deg",
        type=float,
        default=None,
        help="Keep steering and block lowering until heading error enters this angle.",
    )
    parser.add_argument(
        "--direct-place-turn-resume-deg",
        type=float,
        default=30.0,
        help="Resume steering when an aligned robot drifts beyond this angle.",
    )
    parser.add_argument(
        "--direct-place-heading-position-ready-distance",
        type=float,
        default=0.0,
        help="Allow a stable box already at the goal to bypass further heading alignment.",
    )
    parser.add_argument(
        "--direct-place-heading-position-ready-speed",
        type=float,
        default=0.0,
    )
    parser.add_argument("--direct-place-terminal-blend-distance", type=float, default=0.35)
    parser.add_argument(
        "--direct-place-terminal-blend-start-distance",
        type=float,
        default=None,
        help="Distance where continuous near-goal deceleration starts.",
    )
    parser.add_argument("--direct-place-terminal-blend-end-distance", type=float, default=0.30)
    parser.add_argument("--direct-place-profile-lower-full-distance", type=float, default=0.25)
    parser.add_argument("--direct-place-profile-release-distance", type=float, default=0.20)
    parser.add_argument("--direct-place-release-lift-start", type=float, default=0.08)
    parser.add_argument("--direct-place-release-lift-end", type=float, default=0.02)
    parser.add_argument(
        "--direct-place-release-latch-lift",
        type=float,
        default=0.0,
        help="Latch the release latent below this normalized lift; zero disables latching.",
    )
    parser.add_argument("--direct-place-release-latch-distance", type=float, default=0.35)
    parser.add_argument(
        "--direct-place-release-latch-max-vertical-speed", type=float, default=0.20
    )
    parser.add_argument(
        "--direct-place-release-latch-max-speed", type=float, default=float("inf")
    )
    parser.add_argument("--direct-place-release-latch-min-upright", type=float, default=0.90)
    parser.add_argument(
        "--direct-place-release-require-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply distance, total-speed, and upright gates to all release blending.",
    )
    parser.add_argument(
        "--direct-place-release-commit-seconds", type=float, default=0.0
    )
    parser.add_argument("--direct-place-profile-commit-seconds", type=float, default=5.0)
    parser.add_argument(
        "--direct-place-profile-far-commit-seconds",
        type=float,
        default=None,
        help="Optional faster commitment used when the initial goal is beyond the far-distance threshold.",
    )
    parser.add_argument("--direct-place-profile-far-commit-distance", type=float, default=1.25)
    parser.add_argument(
        "--direct-place-profile-commit-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--direct-place-profile-slow-commit-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ramp the slow profile with the timed commitment instead of switching by distance.",
    )
    parser.add_argument(
        "--direct-place-profile-sequential-commit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the first half of commitment to decelerate and the second half to lower.",
    )
    parser.add_argument(
        "--direct-place-profile-commit-min-upright",
        type=float,
        default=0.0,
        help="Minimum robot and box upright score required to start crouching.",
    )
    parser.add_argument(
        "--direct-place-profile-commit-require-bilateral",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require both hand contacts before starting the crouch commitment.",
    )
    parser.add_argument(
        "--direct-place-profile-commit-latch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep decelerating/lowering after the first valid near-goal entry.",
    )
    parser.add_argument(
        "--direct-place-profile-lower-descent-speed",
        type=float,
        default=None,
        help="Back off the lower blend when downward box speed exceeds this value.",
    )
    parser.add_argument(
        "--direct-place-profile-lower-brake-rate", type=float, default=2.0
    )
    parser.add_argument(
        "--direct-place-transport-brake-max-bearing-deg",
        type=float,
        default=60.0,
    )
    parser.add_argument("--direct-place-profile-commit-distance", type=float, default=0.32)
    parser.add_argument("--direct-place-profile-commit-speed", type=float, default=0.20)
    parser.add_argument("--direct-place-profile-commit-reset-distance", type=float, default=0.60)
    parser.add_argument("--direct-place-descent-guard-start-speed", type=float, default=0.10)
    parser.add_argument("--direct-place-descent-guard-full-speed", type=float, default=0.25)
    parser.add_argument("--direct-place-descent-guard-max-blend", type=float, default=1.0)
    parser.add_argument(
        "--direct-place-descent-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy opt-in braking latent driven by instantaneous downward box speed.",
    )
    parser.add_argument(
        "--direct-place-action-slew-rad-s",
        type=float,
        default=0.0,
        help="Optional per-joint PD-target slew limit during direct lowering.",
    )
    parser.add_argument(
        "--direct-place-far-action-slew-rad-s",
        type=float,
        default=None,
        help="Optional direct-place PD-target slew override for far initial goals.",
    )
    parser.add_argument(
        "--direct-place-contact-guard-steps",
        type=int,
        default=2,
        help="Consecutive airborne hand-contact misses before using the recovery latent.",
    )
    parser.add_argument("--direct-place-contact-guard-lift", type=float, default=0.10)
    parser.add_argument("--direct-place-terminal-blend-center", type=float, default=None)
    parser.add_argument("--direct-place-terminal-blend-width", type=float, default=0.03)
    parser.add_argument("--direct-place-terminal-speed-sigma", type=float, default=None)
    parser.add_argument("--direct-place-terminal-max-blend", type=float, default=1.0)
    parser.add_argument(
        "--direct-place-success-distance",
        type=float,
        default=0.30,
        help="Stable released placement distance used to enter complete.",
    )
    parser.add_argument(
        "--get-up-latent",
        type=Path,
        default=Path(
            "analysis/g1_get_up_reward_inference_step73670656_8rank_800k/z_get_up_reward.pkl"
        ),
        help="Fixed reward-inference [Z] get-up latent; [T, Z] remains supported for comparison.",
    )
    parser.add_argument("--motion-cache", type=Path, required=True)
    parser.add_argument("--motion-id", type=int, default=67)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--initial-distance", type=float, default=2.0)
    parser.add_argument(
        "--standing-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start upright at the default joint pose instead of copying Carry motion roll/pitch and joints.",
    )
    parser.add_argument("--standing-root-height", type=float, default=0.80)
    parser.add_argument(
        "--initial-bearing-deg",
        type=float,
        default=None,
        help="Place the box at this bearing in the initial robot yaw frame; 0 is directly ahead.",
    )
    parser.add_argument(
        "--initial-box-center-height",
        type=float,
        default=None,
        help="Optionally override the reset box centre height and zero its free-joint velocity.",
    )
    parser.add_argument(
        "--align-initial-box-yaw-with-robot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start the upright box with the same yaw heading as the robot.",
    )
    parser.add_argument(
        "--start-fallen-side",
        choices=("left", "right"),
        default=None,
        help="Optional deterministic fallen start for recovery validation.",
    )
    parser.add_argument("--start-fallen-root-height", type=float, default=0.35)
    parser.add_argument("--goal-offset-x", type=float, default=2.0)
    parser.add_argument("--goal-offset-y", type=float, default=1.0)
    parser.add_argument("--goal-step", type=float, default=0.25)
    parser.add_argument(
        "--target-follows-object",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="During approach/lift set the observed target to the current box "
        "position (null displacement) so grab actions aim at the box itself; "
        "transport/place still use the configured goal.",
    )
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override the box full dimensions in metres; MuJoCo half-sizes are derived automatically.",
    )
    parser.add_argument(
        "--box-mass",
        type=float,
        default=None,
        help="Override the simulated box mass in kilograms; inertia is derived from its box geometry.",
    )
    parser.add_argument("--box-friction", type=float, nargs=3, default=None)
    parser.add_argument("--box-condim", type=int, choices=(1, 3, 4, 6), default=None)
    parser.add_argument("--hand-friction", type=float, nargs=3, default=None)
    parser.add_argument(
        "--contact-source",
        choices=("physical", "capsule-hysteresis"),
        default="physical",
        help="Contact labels supplied to the policy/controller; MuJoCo collision physics is unchanged.",
    )
    parser.add_argument("--contact-enter-distance", type=float, default=0.045)
    parser.add_argument("--contact-exit-distance", type=float, default=0.060)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--mask-approach-object-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask object context during approach for legacy policies without object-aware approach training.",
    )
    parser.add_argument("--action-clip", type=float, default=5.0)
    parser.add_argument("--camera-distance", type=float, default=5.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N control steps; 0 runs until the viewer closes.")
    parser.add_argument("--save-mp4", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("analysis/carry_phased_sim2sim_disturbance.mp4"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary for deterministic latent-candidate comparison.",
    )
    parser.add_argument(
        "--trace-csv",
        type=Path,
        default=None,
        help="Optional per-control-step CSV trace for phase and disturbance analysis.",
    )
    parser.add_argument("--render-size", type=int, default=720)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--foot-scuff-speed-threshold",
        type=float,
        default=0.10,
        help="Ground-contact foot horizontal speed counted as scuffing.",
    )
    parser.add_argument(
        "--video-label",
        default=None,
        help="Optional text shown in the top-left of every recorded frame; box dimensions are labeled automatically.",
    )
    parser.add_argument("--box-drop-force-n", type=float, default=0.0)
    parser.add_argument("--box-drop-force-seconds", type=float, default=1.0)
    parser.add_argument("--box-drop-transport-delay-seconds", type=float, default=3.0)
    parser.add_argument("--relocate-box-offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"))
    parser.add_argument("--relocate-after-complete-seconds", type=float, default=2.0)
    parser.add_argument("--heading-blend-full-deg", type=float, default=60.0)
    parser.add_argument(
        "--max-turn-blend",
        type=float,
        default=0.90,
        help="Maximum approach turn-latent weight; 0.90 always retains at least one-tenth forward latent.",
    )
    parser.add_argument(
        "--approach-turn-enter-deg",
        type=float,
        default=12.0,
        help="Enter the latched Approach turn mode above this absolute bearing.",
    )
    parser.add_argument(
        "--approach-turn-exit-deg",
        type=float,
        default=5.0,
        help="Return to the unblended Approach latent below this absolute bearing.",
    )
    parser.add_argument(
        "--approach-turn-blend-seconds",
        type=float,
        default=0.20,
        help="Slew time between straight Approach and the latched turn latent.",
    )
    parser.add_argument("--transport-heading-blend-full-deg", type=float, default=45.0)
    parser.add_argument("--transport-max-turn-blend", type=float, default=0.45)
    parser.add_argument(
        "--transport-near-max-turn-blend",
        type=float,
        default=0.8,
        help="Maximum near-goal Transport turn weight; 0.8 retains at least one-fifth forward latent.",
    )
    parser.add_argument(
        "--transport-turn-boost-distance",
        "--transport-slowdown-distance",
        dest="transport_turn_boost_distance",
        type=float,
        default=1.2,
        help="Distance where the turn blend cap starts increasing; forward always uses z_transport.",
    )
    parser.add_argument("--approach-switch-distance", type=float, default=0.75)
    parser.add_argument(
        "--get-up-distance",
        type=float,
        default=1.5,
        help="Deprecated compatibility option; recovery now always finishes Get-up before phase selection.",
    )
    parser.add_argument(
        "--fall-upright-threshold",
        type=float,
        default=0.20,
        help="Severe-tilt threshold for pelvis local +Z dotted with world +Z.",
    )
    parser.add_argument(
        "--fall-root-height",
        type=float,
        default=0.30,
        help="Hard fall threshold for pelvis height above the motion-capture floor.",
    )
    parser.add_argument(
        "--fall-tilted-root-height",
        type=float,
        default=0.50,
        help="Pelvis height threshold applied only together with the severe-tilt threshold.",
    )
    parser.add_argument("--fall-detect-steps", type=int, default=10)
    parser.add_argument("--get-up-upright-threshold", type=float, default=0.85)
    parser.add_argument("--get-up-root-height", type=float, default=0.65)
    parser.add_argument("--get-up-stable-steps", type=int, default=25)
    parser.add_argument("--hold-switch-lift", type=float, default=0.5)
    parser.add_argument("--hold-release-lift", type=float, default=0.25)
    parser.add_argument("--hold-switch-upright", type=float, default=0.85)
    parser.add_argument("--phase-hold-steps", type=int, default=5)
    parser.add_argument("--place-align-distance", type=float, default=0.25)
    parser.add_argument(
        "--place-enter-steps",
        type=int,
        default=5,
        help="Consecutive control steps inside place-align-distance before entering place.",
    )
    parser.add_argument("--place-drift-distance", type=float, default=0.35)
    parser.add_argument(
        "--place-drift-steps",
        type=int,
        default=10,
        help="Consecutive pre-release steps outside place-drift-distance before returning to transport.",
    )
    parser.add_argument("--place-switch-distance", type=float, default=0.20)
    parser.add_argument("--place-abort-distance", type=float, default=0.80)
    parser.add_argument("--place-success-distance", type=float, default=0.20)
    parser.add_argument("--place-success-height", type=float, default=0.05)
    parser.add_argument("--place-success-speed", type=float, default=0.15)
    parser.add_argument("--place-success-upright", type=float, default=0.85)
    parser.add_argument("--place-success-steps", type=int, default=25)
    parser.add_argument("--place-release-height", type=float, default=0.08)
    parser.add_argument("--place-settle-seconds", type=float, default=0.2)
    parser.add_argument("--place-lower-seconds", type=float, default=1.0)
    parser.add_argument(
        "--lower-low-latent",
        type=Path,
        default=None,
        help="Optional low-height static bilateral-hold reward latent blended into z_lower.",
    )
    parser.add_argument("--lower-low-blend-start-lift", type=float, default=0.55)
    parser.add_argument("--lower-low-blend-end-lift", type=float, default=0.10)
    parser.add_argument(
        "--place-action-slew-rad-s",
        type=float,
        default=0.0,
        help="Optional per-joint PD-target slew limit used only during place; 0 disables it.",
    )
    parser.add_argument(
        "--place-descent-speed-limit",
        type=float,
        default=0.15,
        help="Pause z_lower and use z_align when downward box speed exceeds this limit.",
    )
    parser.add_argument(
        "--place-descent-resume-speed",
        type=float,
        default=0.08,
        help="Resume z_lower after braking once downward box speed is below this value.",
    )
    parser.add_argument("--place-release-seconds", type=float, default=2.0)
    parser.add_argument(
        "--accept-settled-ground-release",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Accept a released box that stays upright, still, grounded, and on target "
            "for place-success-steps even if the low bilateral release gate was missed."
        ),
    )
    parser.add_argument("--place-retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--place-max-attempt-seconds", type=float, default=10.0)
    parser.add_argument(
        "--align-goal-lookahead",
        type=float,
        default=0.0,
        help="Extra observed XY goal distance during placement; disabled by default.",
    )
    parser.add_argument("--push-standoff-distance", type=float, default=0.55)
    parser.add_argument("--push-staging-tolerance", type=float, default=0.20)
    parser.add_argument("--push-behind-min", type=float, default=0.30)
    parser.add_argument("--push-lateral-max", type=float, default=0.30)
    parser.add_argument("--push-reapproach-distance", type=float, default=0.90)
    parser.add_argument("--push-contact-max-turn-blend", type=float, default=0.40)
    parser.add_argument("--push-contact-steps", type=int, default=5)
    parser.add_argument(
        "--push-require-bilateral-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require both physical hand contacts before and during ground pushing.",
    )
    parser.add_argument("--push-phase-hold-steps", type=int, default=5)
    parser.add_argument("--push-lost-contact-steps", type=int, default=10)
    parser.add_argument("--push-max-lift", type=float, default=0.15)
    parser.add_argument("--push-upright-min", type=float, default=0.80)
    parser.add_argument("--push-turn-boost-distance", type=float, default=0.80)
    parser.add_argument("--push-max-turn-blend", type=float, default=0.45)
    parser.add_argument("--push-near-max-turn-blend", type=float, default=0.75)
    parser.add_argument("--push-stop-enter-distance", type=float, default=0.25)
    parser.add_argument("--push-stop-enter-steps", type=int, default=5)
    parser.add_argument("--push-stop-exit-distance", type=float, default=0.35)
    parser.add_argument("--push-stop-exit-steps", type=int, default=10)
    parser.add_argument("--push-success-distance", type=float, default=0.20)
    parser.add_argument("--push-success-speed", type=float, default=0.08)
    parser.add_argument("--push-success-steps", type=int, default=25)
    args = parser.parse_args()
    if min(
        args.initial_distance,
        args.goal_step,
        args.start_fallen_root_height,
        args.standing_root_height,
    ) <= 0.0:
        parser.error("initial distance, goal step, and root heights must be positive")
    if args.initial_box_center_height is not None and args.initial_box_center_height <= 0.0:
        parser.error("--initial-box-center-height must be positive")
    if args.save_mp4 and not args.viewer and args.max_steps <= 0:
        parser.error("headless --save-mp4 requires a positive --max-steps")
    if args.render_size <= 0 or args.video_fps <= 0.0 or args.foot_scuff_speed_threshold <= 0.0:
        parser.error("--render-size, --video-fps, and --foot-scuff-speed-threshold must be positive")
    if min(
        args.box_drop_force_n,
        args.box_drop_force_seconds,
        args.box_drop_transport_delay_seconds,
        args.relocate_after_complete_seconds,
    ) < 0.0:
        parser.error("disturbance force and timing values must be non-negative")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("--device must be cpu or cuda[:index]")
    if args.direct_place_terminal_blend_distance <= 0.0:
        parser.error("--direct-place-terminal-blend-distance must be positive")
    if args.direct_place_turn_heading_full_deg <= 0.0:
        parser.error("--direct-place-turn-heading-full-deg must be positive")
    if not 0.0 <= args.direct_place_turn_max_blend < 1.0:
        parser.error("--direct-place-turn-max-blend must be in [0, 1)")
    if not (
        args.direct_place_turn_fade_start_distance
        > args.direct_place_turn_fade_end_distance
        > 0.0
    ):
        parser.error("direct-place turn fade distances must satisfy start > end > 0")
    if args.direct_place_turn_ready_deg is not None and not (
        0.0
        < args.direct_place_turn_ready_deg
        <= args.direct_place_turn_resume_deg
        < 180.0
    ):
        parser.error(
            "direct-place turn angles must satisfy 0 < ready <= resume < 180"
        )
    if args.direct_place_lateral_direct_min_deg is not None and not (
        0.0
        <= args.direct_place_lateral_direct_min_deg
        <= args.direct_place_lateral_direct_max_deg
        <= 180.0
    ):
        parser.error(
            "direct-place lateral route angles must satisfy 0 <= min <= max <= 180"
        )
    if not 0.0 <= args.direct_place_positive_lateral_transport_max_deg <= 180.0:
        parser.error(
            "--direct-place-positive-lateral-transport-max-deg must be in [0, 180]"
        )
    if not 0.0 <= args.direct_place_far_direct_max_bearing_deg <= 180.0:
        parser.error("--direct-place-far-direct-max-bearing-deg must be in [0, 180]")
    if min(
        args.direct_place_goal_lookahead,
        args.direct_place_transport_goal_lookahead,
    ) < 0.0:
        parser.error("direct-place goal lookahead distances must be non-negative")
    if not 0.0 <= args.direct_place_profile_commit_min_upright <= 1.0:
        parser.error("--direct-place-profile-commit-min-upright must be in [0, 1]")
    if args.direct_place_terminal_blend_start_distance is not None and not (
        args.direct_place_terminal_blend_start_distance
        > args.direct_place_terminal_blend_end_distance
        > 0.0
    ):
        parser.error("direct-place blend distances must satisfy start > end > 0")
    if not (
        args.direct_place_terminal_blend_end_distance
        > args.direct_place_profile_lower_full_distance
        >= args.direct_place_profile_release_distance
        > 0.0
    ):
        parser.error(
            "direct-place distances must satisfy blend end > lower full >= release > 0"
        )
    if not 0.0 <= args.direct_place_release_lift_end < args.direct_place_release_lift_start:
        parser.error("direct-place release lift values must satisfy 0 <= end < start")
    if not 0.0 <= args.direct_place_release_latch_lift <= 1.0:
        parser.error("--direct-place-release-latch-lift must be in [0, 1]")
    if min(
        args.direct_place_release_latch_distance,
        args.direct_place_release_latch_max_vertical_speed,
        args.direct_place_profile_commit_speed,
        args.direct_place_profile_commit_reset_distance,
        args.direct_place_descent_guard_start_speed,
        args.direct_place_descent_guard_full_speed,
        args.direct_place_success_distance,
    ) <= 0.0:
        parser.error("direct-place distances and speed thresholds must be positive")
    if not 0.0 <= args.direct_place_release_latch_min_upright <= 1.0:
        parser.error("--direct-place-release-latch-min-upright must be in [0, 1]")
    if args.direct_place_profile_commit_seconds < 0.0:
        parser.error("--direct-place-profile-commit-seconds must be non-negative")
    if (
        args.direct_place_profile_far_commit_seconds is not None
        and args.direct_place_profile_far_commit_seconds <= 0.0
    ):
        parser.error("--direct-place-profile-far-commit-seconds must be positive")
    if args.direct_place_profile_far_commit_distance <= 0.0:
        parser.error("--direct-place-profile-far-commit-distance must be positive")
    if not (
        0.0
        < args.direct_place_profile_commit_distance
        < args.direct_place_profile_commit_reset_distance
    ):
        parser.error("direct-place commitment distances must satisfy 0 < enter < reset")
    if not (
        args.direct_place_descent_guard_start_speed
        < args.direct_place_descent_guard_full_speed
    ):
        parser.error("direct-place descent guard speeds must satisfy start < full")
    if not 0.0 <= args.direct_place_descent_guard_max_blend <= 1.0:
        parser.error("--direct-place-descent-guard-max-blend must be in [0, 1]")
    if args.direct_place_contact_guard_steps <= 0:
        parser.error("--direct-place-contact-guard-steps must be positive")
    if not 0.0 <= args.direct_place_contact_guard_lift <= 1.0:
        parser.error("--direct-place-contact-guard-lift must be in [0, 1]")
    if args.direct_place_terminal_blend_center is not None and (
        args.direct_place_terminal_blend_center <= 0.0
        or args.direct_place_terminal_blend_width <= 0.0
    ):
        parser.error("direct-place sigmoid centre and width must be positive")
    if (
        args.direct_place_terminal_speed_sigma is not None
        and args.direct_place_terminal_speed_sigma <= 0.0
    ):
        parser.error("--direct-place-terminal-speed-sigma must be positive")
    if not 0.0 <= args.direct_place_terminal_max_blend <= 1.0:
        parser.error("--direct-place-terminal-max-blend must be in [0, 1]")
    if not 0.0 <= args.contact_enter_distance < args.contact_exit_distance:
        parser.error("contact thresholds must satisfy 0 <= enter < exit")
    if args.transport_heading_blend_full_deg <= 0.0:
        parser.error("--transport-heading-blend-full-deg must be positive")
    if not (
        0.0 <= args.approach_turn_exit_deg < args.approach_turn_enter_deg
    ):
        parser.error("approach turn thresholds must satisfy 0 <= exit < enter")
    if args.approach_turn_blend_seconds < 0.0:
        parser.error("--approach-turn-blend-seconds must be non-negative")
    if not 0.0 <= args.transport_max_turn_blend <= args.transport_near_max_turn_blend <= 1.0:
        parser.error("transport turn blends must satisfy 0 <= far <= near <= 1")
    if args.transport_turn_boost_distance <= args.place_align_distance:
        parser.error("--transport-turn-boost-distance must exceed --place-align-distance")
    if not (
        args.place_success_distance
        <= args.place_switch_distance
        < args.place_align_distance
        < args.place_drift_distance
        < args.place_abort_distance
    ):
        parser.error("place thresholds must satisfy success <= switch < enter < exit < abort")
    if args.place_enter_steps <= 0 or args.place_drift_steps <= 0:
        parser.error("--place-enter-steps and --place-drift-steps must be positive")
    if args.place_action_slew_rad_s < 0.0:
        parser.error("--place-action-slew-rad-s must be non-negative")
    if args.direct_place_action_slew_rad_s < 0.0 or (
        args.direct_place_far_action_slew_rad_s is not None
        and args.direct_place_far_action_slew_rad_s < 0.0
    ):
        parser.error("direct-place action slew rates must be non-negative")
    if not 0.0 <= args.lower_low_blend_end_lift < args.lower_low_blend_start_lift <= 1.0:
        parser.error("lower-low blend thresholds must satisfy 0 <= end < start <= 1")
    if not 0.0 < args.place_descent_resume_speed < args.place_descent_speed_limit:
        parser.error(
            "place descent speeds must satisfy 0 < resume-speed < speed-limit"
        )
    if args.fall_detect_steps <= 0 or args.get_up_stable_steps <= 0:
        parser.error("fall detection and get-up stability steps must be positive")
    if not (-1.0 <= args.fall_upright_threshold < args.get_up_upright_threshold <= 1.0):
        parser.error("pelvis upright thresholds must satisfy -1 <= fall < recovery <= 1")
    if not (0.0 < args.fall_root_height < args.fall_tilted_root_height < args.get_up_root_height):
        parser.error("root height thresholds must satisfy 0 < hard fall < tilted fall < recovery")
    if min(
        args.push_standoff_distance,
        args.push_staging_tolerance,
        args.push_behind_min,
        args.push_lateral_max,
        args.push_reapproach_distance,
        args.push_contact_steps,
        args.push_phase_hold_steps,
        args.push_lost_contact_steps,
        args.push_max_lift,
        args.push_turn_boost_distance,
        args.push_stop_enter_distance,
        args.push_stop_enter_steps,
        args.push_stop_exit_distance,
        args.push_stop_exit_steps,
        args.push_success_distance,
        args.push_success_speed,
        args.push_success_steps,
    ) <= 0:
        parser.error("push distances, thresholds, step counts, and speed must be positive")
    if not 0.0 <= args.push_contact_max_turn_blend < 1.0:
        parser.error("--push-contact-max-turn-blend must be in [0, 1)")
    if not 0.0 <= args.push_max_turn_blend <= args.push_near_max_turn_blend < 1.0:
        parser.error("push turn blends must satisfy 0 <= far <= near < 1")
    if not 0.0 < args.push_upright_min <= 1.0:
        parser.error("--push-upright-min must be in (0, 1]")
    if not (
        args.push_success_distance
        <= args.push_stop_enter_distance
        < args.push_stop_exit_distance
        < args.push_turn_boost_distance
    ):
        parser.error("push thresholds must satisfy success <= enter < exit < turn-boost")
    return args


if __name__ == "__main__":
    run(parse_args())
