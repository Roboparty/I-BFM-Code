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
import ctypes
import ctypes.util
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
import mujoco.viewer  # lazy-imported in the --viewer branch below (headless servers have no GLFW)
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_MujocoRenderer = mujoco.Renderer

_COORDINATE_FRAME_COLORS = (
    np.asarray((0.95, 0.12, 0.12, 1.0), dtype=np.float32),
    np.asarray((0.12, 0.90, 0.20, 1.0), dtype=np.float32),
    np.asarray((0.12, 0.35, 1.00, 1.0), dtype=np.float32),
)


class X11KeyboardState:
    """Poll held keys because MuJoCo's passive-viewer callback has no release event."""

    _KEYSYMS = {
        "left": 0xFF51,
        "up": 0xFF52,
        "right": 0xFF53,
        "down": 0xFF54,
        "space": 0x0020,
        "b": 0x0062,
        "f": 0x0066,
    }

    def __init__(self) -> None:
        library = ctypes.util.find_library("X11") or "libX11.so.6"
        try:
            self._x11 = ctypes.CDLL(library)
        except OSError as exc:
            raise RuntimeError(
                "manual keyboard mode requires libX11 (X11 or XWayland session)"
            ) from exc
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._x11.XCloseDisplay.restype = ctypes.c_int
        self._x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._x11.XKeysymToKeycode.restype = ctypes.c_ubyte
        self._x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
        self._x11.XQueryKeymap.restype = ctypes.c_int
        self._display = self._x11.XOpenDisplay(None)
        if not self._display:
            raise RuntimeError(
                "manual keyboard mode could not open DISPLAY; run it from an X11/XWayland desktop terminal"
            )
        self._keycodes = {
            name: int(self._x11.XKeysymToKeycode(self._display, keysym))
            for name, keysym in self._KEYSYMS.items()
        }

    def poll(self) -> dict[str, bool]:
        bitmap = (ctypes.c_char * 32)()
        self._x11.XQueryKeymap(self._display, bitmap)
        values = bytes(bitmap)
        return {
            name: bool(values[keycode // 8] & (1 << (keycode % 8)))
            for name, keycode in self._keycodes.items()
        }

    def close(self) -> None:
        if self._display:
            self._x11.XCloseDisplay(self._display)
            self._display = None


def _append_coordinate_frame(
    scene: Any,
    *,
    position: np.ndarray,
    rotation: np.ndarray,
    size: float,
    width: float,
) -> None:
    """Append RGB local-frame arrows to an existing MuJoCo visualization scene."""

    origin = np.asarray(position, dtype=np.float64)
    axes = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    for axis_index, color in enumerate(_COORDINATE_FRAME_COLORS):
        if scene.ngeom >= scene.maxgeom:
            return
        axis = axes[:, axis_index]
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(9, dtype=np.float64),
            color,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            float(width),
            origin,
            origin + float(size) * axis,
        )
        scene.ngeom += 1


def _append_robot_box_coordinate_frames(
    scene: Any,
    data: mujoco.MjData,
    *,
    base_body_id: int,
    box_body_id: int,
    size: float,
    width: float,
) -> None:
    """Show the policy's pelvis/base frame and free-box body frame."""

    for body_id in (base_body_id, box_body_id):
        _append_coordinate_frame(
            scene,
            position=data.xpos[body_id],
            rotation=data.xmat[body_id],
            size=size,
            width=width,
        )


def _append_hand_contact_capsules(
    scene: Any,
    data: mujoco.MjData,
    *,
    hand_body_ids: tuple[int, int],
    capsule_segments: np.ndarray,
    capsule_radius: float,
    contact: np.ndarray,
) -> None:
    """Show the exact left/right hand capsules used by contact hysteresis."""

    local_segments = np.asarray(capsule_segments, dtype=np.float64).reshape(2, 2, 3)
    contact_state = np.asarray(contact, dtype=np.float64).reshape(2)
    idle_colors = (
        np.asarray((0.05, 0.75, 1.00, 0.70), dtype=np.float32),
        np.asarray((1.00, 0.45, 0.05, 0.70), dtype=np.float32),
    )
    active_color = np.asarray((0.10, 1.00, 0.20, 0.90), dtype=np.float32)
    for index, body_id in enumerate(hand_body_ids):
        if scene.ngeom >= scene.maxgeom:
            return
        rotation = data.xmat[body_id].reshape(3, 3)
        origin = data.xpos[body_id]
        endpoints = origin + (rotation @ local_segments[index].T).T
        geom = scene.geoms[scene.ngeom]
        color = active_color if contact_state[index] > 0.5 else idle_colors[index]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(9, dtype=np.float64),
            color,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(capsule_radius),
            endpoints[0],
            endpoints[1],
        )
        scene.ngeom += 1


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
from humanoidverse.envs.motion_observations import compute_humanoid_observations_max
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _load_latent(path: Path, name: str, device: torch.device) -> torch.Tensor:
    value = joblib.load(path / f"{name}.pkl")
    if isinstance(value, dict):
        raise TypeError(f"{name}.pkl contains a latent bank; use _load_latent_bank")
    return torch.as_tensor(value, device=device, dtype=torch.float32).reshape(1, -1)


def _build_privileged_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    body_ids: list[int],
    torso_body_id: int,
    object_context: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Reproduce the training-time max-local-self observation in MuJoCo."""

    body_pos = np.asarray([data.xpos[body_id] for body_id in body_ids], dtype=np.float32)
    body_quat = np.asarray(
        [np.roll(data.xquat[body_id], -1) for body_id in body_ids], dtype=np.float32
    )
    velocities = [_world_body_velocity(model, data, body_id) for body_id in body_ids]
    body_vel = np.asarray([item[0] for item in velocities], dtype=np.float32)
    body_ang_vel = np.asarray([item[1] for item in velocities], dtype=np.float32)

    # The training robot adds one virtual head body 0.35 m above torso_link.
    head_offset_local = np.asarray((0.0, 0.0, 0.35), dtype=np.float32)
    head_offset_world = data.xmat[torso_body_id].reshape(3, 3) @ head_offset_local
    torso_index = body_ids.index(torso_body_id)
    head_pos = body_pos[torso_index] + head_offset_world
    head_quat = body_quat[torso_index]
    head_ang_vel = body_ang_vel[torso_index]
    head_vel = body_vel[torso_index] + np.cross(head_ang_vel, head_offset_local)

    body_pos = np.concatenate((body_pos, head_pos[None]), axis=0)
    body_quat = np.concatenate((body_quat, head_quat[None]), axis=0)
    body_vel = np.concatenate((body_vel, head_vel[None]), axis=0)
    body_ang_vel = np.concatenate((body_ang_vel, head_ang_vel[None]), axis=0)
    tensor = lambda value: torch.as_tensor(
        value, device=device, dtype=torch.float32
    ).unsqueeze(0)
    parts = compute_humanoid_observations_max(
        tensor(body_pos),
        tensor(body_quat),
        tensor(body_vel),
        tensor(body_ang_vel),
        local_root_obs=True,
        root_height_obs=True,
    )
    return torch.cat((*parts.values(), object_context), dim=-1)


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
    box_bearing_deg = _bearing_deg(base_rotation, robot_box_xy)
    return {
        "horizontal_distance": float(np.linalg.norm(box_pos[:2] - base_pos[:2])),
        "goal_distance": float(np.linalg.norm(goal[:2] - box_pos[:2])),
        "object_height": float(box_pos[2]),
        "object_speed": float(np.linalg.norm(box_lin_vel)),
        "object_vertical_speed": float(box_lin_vel[2]),
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
        "push_box_bearing_deg": float(box_bearing_deg),
    }


def _summarize_speed_rollout(
    history: list[dict[str, float | bool | str]],
    *,
    final_phase: str,
) -> dict[str, Any]:
    """Summarize a deterministic phased rollout for latent-bank selection."""

    def rows_for(phase: str) -> list[dict[str, float | bool | str]]:
        return [row for row in history if row["phase"] == phase]

    def mean(rows: list[dict[str, float | bool | str]], key: str) -> float | None:
        return None if not rows else float(np.mean([float(row[key]) for row in rows]))

    approach_rows = rows_for("approach")
    transport_rows = rows_for("transport")
    push_transport_rows = rows_for("push_transport")
    push_direct_rows = rows_for("push_direct")
    push_direct_stop_rows = rows_for("push_direct_stop")
    push_direct_reposition_rows = rows_for("push_direct_reposition")
    phase_counts = {
        phase: sum(row["phase"] == phase for row in history)
        for phase in (
            "get_up",
            "approach",
            "lift",
            "transport",
            "place",
            "push_contact",
            "push_transport",
            "push_stop",
            "push_direct",
            "push_direct_stop",
            "push_direct_reposition",
            "place_direct",
            "complete",
        )
    }
    final = history[-1] if history else None
    return {
        "step_count": len(history),
        "final_phase": final_phase,
        "completed": (
            final_phase in ("complete", "push_direct_stop")
            or phase_counts["complete"] > 0
        ),
        "phase_counts": phase_counts,
        "approach_boxward_speed_mean": mean(approach_rows, "approach_boxward_speed"),
        "transport_goalward_speed_mean": mean(transport_rows, "transport_goalward_speed"),
        "transport_bilateral_contact_rate": mean(transport_rows, "bilateral_contact"),
        "transport_lift_mean": mean(transport_rows, "lift"),
        "push_goalward_speed_mean": mean(push_transport_rows, "transport_goalward_speed"),
        "push_contact_rate": mean(push_transport_rows, "any_contact"),
        "push_lift_mean": mean(push_transport_rows, "lift"),
        "push_direct_goalward_speed_mean": mean(
            push_direct_rows, "transport_goalward_speed"
        ),
        "push_direct_contact_rate": mean(push_direct_rows, "any_contact"),
        "push_direct_lift_mean": mean(push_direct_rows, "lift"),
        "push_direct_stop_object_speed_mean": mean(push_direct_stop_rows, "object_speed"),
        "push_direct_stop_robot_upright_mean": mean(
            push_direct_stop_rows, "robot_upright"
        ),
        "push_direct_reposition_boxward_speed_mean": mean(
            push_direct_reposition_rows, "approach_boxward_speed"
        ),
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
            turn_weight = min(abs(bearing) / self.args.heading_blend_full_deg, 1.0) * self.args.max_turn_blend
            direction = 1 if bearing >= 0.0 else -1
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
            if self.release_started_step is None:
                blend = _smooth_place_blend(self.phase_steps, self.place_settle_steps, self.place_lower_steps)
                latent = policy.project_z((1.0 - blend) * z["align"] + blend * z["lower"])
                return place_observation, latent, f"place_lower_{blend:.2f}"
            blend = _smooth_place_blend(
                self.phase_steps - self.release_started_step,
                0,
                self.place_release_steps,
            )
            latent = policy.project_z((1.0 - blend) * z["lower"] + blend * z["place"])
            return place_observation, latent, f"place_release_{blend:.2f}"
        return observation, z["place"], "placed_verify"

    def update(self, metrics: dict[str, float | bool], *, goal_height: float) -> None:
        self.phase_steps += 1
        root_height = float(metrics["root_height"])
        pelvis_up_z = float(metrics["pelvis_up_z"])
        fallen = root_height < self.args.fall_root_height or (
            pelvis_up_z < self.args.fall_upright_threshold
            and root_height < self.args.fall_tilted_root_height
        )
        self.fall_detect_steps = self.fall_detect_steps + 1 if fallen else 0

        if self.phase == "get_up":
            recovered = (
                float(metrics["pelvis_up_z"]) >= self.args.get_up_upright_threshold
                and float(metrics["root_height"]) >= self.args.get_up_root_height
            )
            self.get_up_stable_steps = self.get_up_stable_steps + 1 if recovered else 0
            if self.get_up_stable_steps >= self.args.get_up_stable_steps:
                next_phase = (
                    "lift"
                    if float(metrics["horizontal_distance"]) <= self.args.approach_switch_distance
                    else "approach"
                )
                self._change_phase(next_phase)
                self.fall_detect_steps = 0
                self.get_up_stable_steps = 0
            return

        if self.fall_detect_steps >= self.args.fall_detect_steps:
            if self.phase != "get_up":
                self.fall_event_count += 1
                self._change_phase("get_up")
                self.get_up_stable_steps = 0
            return

        self.get_up_stable_steps = 0
        if self.phase == "approach":
            if float(metrics["horizontal_distance"]) <= self.args.approach_switch_distance:
                self.phase = "lift"
                self.phase_steps = 0
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
            placed = (
                float(metrics["goal_distance"]) <= self.args.place_success_distance
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


class ManualKeyboardCarryController(PhaseController):
    """Carry controller whose locomotion and lift/place transitions are user-driven."""

    def reset(self) -> None:
        super().reset()
        self.manual_forward = False
        self.manual_turn = 0

    def set_motion(self, *, forward: bool, turn: int) -> None:
        self.manual_forward = bool(forward)
        self.manual_turn = int(np.clip(turn, -1, 1))

    def request_lift(self, metrics: dict[str, float | bool]) -> tuple[bool, str]:
        if self.phase != "approach":
            return False, f"lift ignored while phase={self.phase}"
        distance = float(metrics["horizontal_distance"])
        bearing = abs(float(metrics["push_box_bearing_deg"]))
        self._change_phase("lift")
        return True, (
            f"lift started without pose gating: robot-box distance {distance:.3f}m, "
            f"bearing {bearing:.1f}deg"
        )

    def request_place(self) -> tuple[bool, str]:
        if self.phase != "transport":
            return False, f"place ignored while phase={self.phase}"
        self._change_phase("place")
        self.place_attempt += 1
        self.place_drift_steps = 0
        self.placed_steps = 0
        self.failure_steps = 0
        return True, "place started at the box's current XY position"

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
        del metrics, base_rotation, box_direction, goal_direction
        self.last_transport_turn_weight = 0.0
        if self.phase == "get_up":
            sequence = z["get_up"]
            sequence_step = min(self.phase_steps, int(sequence.shape[0]) - 1)
            mode = "get_up_reward" if int(sequence.shape[0]) == 1 else f"get_up_{sequence_step:03d}"
            return (
                _masked_object_observation(observation),
                sequence[sequence_step : sequence_step + 1],
                mode,
            )
        if self.phase == "lift":
            return observation, z["lift"], "manual_lift"
        if self.phase == "place":
            place_observation = _goal_lookahead_observation(observation, self.args.align_goal_lookahead)
            if self.release_started_step is None:
                blend = _smooth_place_blend(self.phase_steps, self.place_settle_steps, self.place_lower_steps)
                latent = policy.project_z((1.0 - blend) * z["align"] + blend * z["lower"])
                return place_observation, latent, f"manual_place_lower_{blend:.2f}"
            blend = _smooth_place_blend(
                self.phase_steps - self.release_started_step,
                0,
                self.place_release_steps,
            )
            latent = policy.project_z((1.0 - blend) * z["lower"] + blend * z["place"])
            return place_observation, latent, f"manual_place_release_{blend:.2f}"

        carrying = self.phase == "transport"
        idle_latent = z["hold"] if carrying else z["place"]
        forward_latent = z["transport"] if carrying else z["approach"]
        turn_bank = z["transport_turn"] if carrying else z["turn"]
        prefix = "manual_carry" if carrying else "manual_empty"
        policy_observation = observation if carrying else _masked_object_observation(observation)
        if self.manual_turn:
            turn_latent = turn_bank[self.manual_turn]
            if self.manual_forward:
                weight = float(self.args.manual_turn_blend)
                self.last_transport_turn_weight = weight if carrying else 0.0
                latent = policy.project_z((1.0 - weight) * forward_latent + weight * turn_latent)
                direction = "left" if self.manual_turn > 0 else "right"
                return policy_observation, latent, f"{prefix}_forward_{direction}"
            direction = "left" if self.manual_turn > 0 else "right"
            self.last_transport_turn_weight = 1.0 if carrying else 0.0
            return policy_observation, turn_latent, f"{prefix}_turn_{direction}"
        if self.manual_forward:
            return policy_observation, forward_latent, f"{prefix}_forward"
        idle_name = "hold" if carrying else "place"
        return policy_observation, idle_latent, f"{prefix}_idle_{idle_name}"

    def update(self, metrics: dict[str, float | bool], *, goal_height: float) -> None:
        # Reuse the tested lowering/release/retry implementation. A successful
        # manual placement returns to empty-hand mode instead of staying complete.
        if self.phase == "place":
            super().update(metrics, goal_height=goal_height)
            if self.phase in ("complete", "lift"):
                self._change_phase("approach")
            return

        self.phase_steps += 1
        root_height = float(metrics["root_height"])
        pelvis_up_z = float(metrics["pelvis_up_z"])
        fallen = root_height < self.args.fall_root_height or (
            pelvis_up_z < self.args.fall_upright_threshold
            and root_height < self.args.fall_tilted_root_height
        )
        self.fall_detect_steps = self.fall_detect_steps + 1 if fallen else 0
        if self.phase == "get_up":
            recovered = (
                pelvis_up_z >= self.args.get_up_upright_threshold
                and root_height >= self.args.get_up_root_height
            )
            self.get_up_stable_steps = self.get_up_stable_steps + 1 if recovered else 0
            if self.get_up_stable_steps >= self.args.get_up_stable_steps:
                holding = (
                    bool(metrics["bilateral_contact"])
                    and float(metrics["lift"]) >= self.args.hold_release_lift
                )
                self._change_phase("transport" if holding else "approach")
                self.fall_detect_steps = 0
                self.get_up_stable_steps = 0
            return
        if self.fall_detect_steps >= self.args.fall_detect_steps:
            self.fall_event_count += 1
            self._change_phase("get_up")
            self.get_up_stable_steps = 0
            return

        self.get_up_stable_steps = 0
        if self.phase == "approach":
            return
        if self.phase == "lift":
            ready = (
                bool(metrics["bilateral_contact"])
                and float(metrics["lift"]) >= self.args.hold_switch_lift
                and float(metrics["robot_upright"]) >= self.args.hold_switch_upright
            )
            self.ready_steps = self.ready_steps + 1 if ready else 0
            if self.ready_steps >= self.args.phase_hold_steps:
                self._change_phase("transport")
            return
        if self.phase == "transport":
            lost = (
                not bool(metrics["bilateral_contact"])
                or float(metrics["lift"]) < self.args.hold_release_lift
            )
            self.lost_steps = self.lost_steps + 1 if lost else 0
            if self.lost_steps >= self.args.phase_hold_steps:
                self._change_phase("approach")


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
    """Continuously blend transport and terminal reward-inferred latents."""

    def reset(self) -> None:
        super().reset()
        self.phase = "place_direct"
        self.last_direct_place_terminal_weight = 0.0
        self.direct_place_lower_commitment = 0.0
        self.direct_place_lower_drive_weight = 0.0
        self.direct_place_recovery_active = False
        self.direct_place_recovery_stable_steps = 0
        self.direct_place_release_latched = False

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
        direct_place_base = z["direct_place"]
        direct_turn = z.get("direct_place_turn")
        turn_mode = ""
        if direct_turn is not None:
            distance = float(metrics["goal_distance"])
            bearing = _bearing_deg(base_rotation, goal_direction)
            fade_start = float(self.args.direct_place_turn_fade_start_distance)
            fade_end = float(self.args.direct_place_turn_fade_end_distance)
            turn_fade = float(
                np.clip((distance - fade_end) / (fade_start - fade_end), 0.0, 1.0)
            )
            turn_fade = turn_fade * turn_fade * (3.0 - 2.0 * turn_fade)
            angle_weight = float(
                np.clip(
                    abs(bearing) / float(self.args.direct_place_turn_heading_full_deg),
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
        profile_names = (
            "direct_place_slow",
            "direct_place_lower",
            "direct_place_release",
        )
        if all(name in z for name in profile_names):
            start = float(self.args.direct_place_terminal_blend_start_distance)
            end = float(self.args.direct_place_terminal_blend_end_distance)
            distance = float(metrics["goal_distance"])
            slow_progress = float(np.clip((start - distance) / (start - end), 0.0, 1.0))
            slow_progress = slow_progress * slow_progress * (3.0 - 2.0 * slow_progress)

            lower_full_distance = float(self.args.direct_place_profile_lower_full_distance)
            lower_progress = float(
                np.clip((end - distance) / (end - lower_full_distance), 0.0, 1.0)
            )
            lower_progress = lower_progress * lower_progress * (3.0 - 2.0 * lower_progress)
            commit_seconds = float(
                getattr(self.args, "direct_place_profile_commit_seconds", 0.0)
            )
            if commit_seconds > 0.0:
                commit_distance = float(
                    getattr(self.args, "direct_place_profile_commit_distance", end)
                )
                commit_speed = float(
                    getattr(self.args, "direct_place_profile_commit_speed", 0.20)
                )
                reset_distance = float(
                    getattr(self.args, "direct_place_profile_commit_reset_distance", start)
                )
                can_commit = (
                    distance <= commit_distance
                    and float(metrics.get("object_speed", 0.0)) <= commit_speed
                )
                delta = self.dt / commit_seconds
                if can_commit:
                    self.direct_place_lower_commitment = min(
                        1.0, self.direct_place_lower_commitment + delta
                    )
                elif distance > reset_distance:
                    self.direct_place_lower_commitment = max(
                        0.0, self.direct_place_lower_commitment - delta
                    )
                if getattr(self.args, "direct_place_profile_commit_only", False):
                    lower_progress = self.direct_place_lower_commitment
                else:
                    lower_progress = max(
                        lower_progress, self.direct_place_lower_commitment
                    )
                slow_progress = max(
                    slow_progress, self.direct_place_lower_commitment
                )
            lift_start = float(self.args.direct_place_release_lift_start)
            lift_end = float(self.args.direct_place_release_lift_end)
            release_progress = float(
                np.clip((lift_start - float(metrics["lift"])) / (lift_start - lift_end), 0.0, 1.0)
            )
            release_progress = release_progress * release_progress * (3.0 - 2.0 * release_progress)
            release_latch_lift = float(
                getattr(self.args, "direct_place_release_latch_lift", 0.0)
            )
            if (
                release_latch_lift > 0.0
                and distance <= float(
                    getattr(self.args, "direct_place_release_latch_distance", 0.35)
                )
                and float(metrics["lift"]) <= release_latch_lift
                and abs(float(metrics.get("object_vertical_speed", 0.0)))
                <= float(
                    getattr(
                        self.args,
                        "direct_place_release_latch_max_vertical_speed",
                        0.20,
                    )
                )
                and float(metrics.get("box_upright", 1.0))
                >= float(
                    getattr(self.args, "direct_place_release_latch_min_upright", 0.90)
                )
            ):
                self.direct_place_release_latched = True
            if self.direct_place_release_latched:
                release_progress = 1.0

            bilateral_contact = bool(metrics.get("bilateral_contact", False))
            lift = float(metrics["lift"])
            vertical_speed = float(metrics.get("object_vertical_speed", 0.0))
            recovery_latent = z.get("direct_place_recover")
            if recovery_latent is not None:
                airborne_for_recovery = lift > float(
                    self.args.direct_place_recovery_lift_threshold
                )
                unsafe_descent = (
                    vertical_speed < float(self.args.direct_place_recovery_fall_speed)
                    or not bilateral_contact
                ) and airborne_for_recovery
                if unsafe_descent:
                    self.direct_place_recovery_active = True
                    self.direct_place_recovery_stable_steps = 0
                    self.direct_place_lower_drive_weight = 0.0
                if self.direct_place_recovery_active:
                    recovered = (
                        bilateral_contact
                        and abs(vertical_speed)
                        <= float(self.args.direct_place_recovery_stable_speed)
                    )
                    self.direct_place_recovery_stable_steps = (
                        self.direct_place_recovery_stable_steps + 1 if recovered else 0
                    )
                    if (
                        self.direct_place_recovery_stable_steps
                        < int(self.args.direct_place_recovery_stable_steps)
                    ):
                        return observation, recovery_latent, "place_direct_recover"
                    self.direct_place_recovery_active = False
                    self.direct_place_recovery_stable_steps = 0

            lower_latent = z["direct_place_lower"]
            lower_height_names = (
                "direct_place_lower_high",
                "direct_place_lower_mid",
                "direct_place_lower_low",
            )
            if all(name in z for name in lower_height_names):
                high_lift = float(self.args.direct_place_lower_high_lift)
                mid_lift = float(self.args.direct_place_lower_mid_lift)
                low_lift = float(self.args.direct_place_lower_low_lift)
                if lift >= high_lift:
                    lower_latent = z["direct_place_lower_high"]
                elif lift >= mid_lift:
                    blend = (high_lift - lift) / (high_lift - mid_lift)
                    lower_latent = policy.project_z(
                        (1.0 - blend) * z["direct_place_lower_high"]
                        + blend * z["direct_place_lower_mid"]
                    )
                elif lift >= low_lift:
                    blend = (mid_lift - lift) / (mid_lift - low_lift)
                    lower_latent = policy.project_z(
                        (1.0 - blend) * z["direct_place_lower_mid"]
                        + blend * z["direct_place_lower_low"]
                    )
                else:
                    lower_latent = z["direct_place_lower_low"]
            lower_drive = z.get("direct_place_lower_drive")
            if lower_drive is not None and lower_progress > 0.0:
                target_speed = float(self.args.direct_place_lower_target_speed)
                deadband = float(self.args.direct_place_lower_speed_deadband)
                if bilateral_contact and vertical_speed > target_speed + deadband:
                    self.direct_place_lower_drive_weight = min(
                        float(self.args.direct_place_lower_drive_max_blend),
                        self.direct_place_lower_drive_weight
                        + self.dt / float(self.args.direct_place_lower_drive_rise_seconds),
                    )
                elif vertical_speed < target_speed - deadband or not bilateral_contact:
                    self.direct_place_lower_drive_weight = max(
                        0.0,
                        self.direct_place_lower_drive_weight
                        - self.dt / float(self.args.direct_place_lower_drive_fall_seconds),
                    )
                lower_latent = policy.project_z(
                    (1.0 - self.direct_place_lower_drive_weight) * lower_latent
                    + self.direct_place_lower_drive_weight * lower_drive
                )

            lower_or_release = policy.project_z(
                (1.0 - release_progress) * lower_latent
                + release_progress * z["direct_place_release"]
            )
            near_latent = policy.project_z(
                (1.0 - lower_progress) * z["direct_place_slow"]
                + lower_progress * lower_or_release
            )
            latent = policy.project_z(
                (1.0 - slow_progress) * direct_place_base
                + slow_progress * near_latent
            )
            # Keep the legacy controller behaviour for programmatic callers that
            # build a minimal Namespace without the new descent-guard options.
            # The command-line parser always defines these attributes, so normal
            # sim2sim runs still enable the guard by default.
            guard = 0.0
            if hasattr(self.args, "direct_place_descent_guard_start_speed"):
                guard_start = float(self.args.direct_place_descent_guard_start_speed)
                guard_full = float(self.args.direct_place_descent_guard_full_speed)
                guard = float(
                    np.clip(
                        (-vertical_speed - guard_start) / (guard_full - guard_start),
                        0.0,
                        1.0,
                    )
                )
                guard = guard * guard * (3.0 - 2.0 * guard)
                # Missing contact data must not be interpreted as a physical
                # contact loss.  Only the full simulator metrics provide this
                # signal; lightweight callers and unit tests may omit it.
                if (
                    "bilateral_contact" in metrics
                    and not bilateral_contact
                    and lift
                    > float(
                        getattr(
                            self.args,
                            "direct_place_recovery_lift_threshold",
                            0.10,
                        )
                    )
                ):
                    guard = 1.0
            if self.direct_place_release_latched:
                latent = z["direct_place_release"]
                guard = 0.0
            elif guard > 0.0:
                guard_latent = z.get(
                    "direct_place_guard", z["direct_place_slow"]
                )
                latent = policy.project_z(
                    (1.0 - guard) * latent
                    + guard * guard_latent
                )
            self.last_direct_place_terminal_weight = slow_progress
            return (
                observation,
                latent,
                "place_direct_profile_"
                f"s{slow_progress:.2f}_l{lower_progress:.2f}_r{release_progress:.2f}_"
                f"c{self.direct_place_lower_commitment:.2f}_"
                f"d{self.direct_place_lower_drive_weight:.2f}_g{guard:.2f}_"
                f"rl{int(self.direct_place_release_latched)}{turn_mode}",
            )
        terminal = z.get("direct_place_terminal")
        if terminal is None:
            self.last_direct_place_terminal_weight = 0.0
            return observation, direct_place_base, f"place_direct{turn_mode}"
        distance = float(metrics["goal_distance"])
        blend_start = getattr(self.args, "direct_place_terminal_blend_start_distance", None)
        blend_end = getattr(self.args, "direct_place_terminal_blend_end_distance", 0.30)
        blend_center = self.args.direct_place_terminal_blend_center
        if blend_start is not None:
            blend_progress = float(
                np.clip(
                    (float(blend_start) - distance)
                    / (float(blend_start) - float(blend_end)),
                    0.0,
                    1.0,
                )
            )
            blend = blend_progress * blend_progress * (3.0 - 2.0 * blend_progress)
        elif blend_center is None:
            blend_distance = float(self.args.direct_place_terminal_blend_distance)
            blend = math.exp(-((distance / blend_distance) ** 2))
        else:
            width = float(self.args.direct_place_terminal_blend_width)
            exponent = float(np.clip((distance - float(blend_center)) / width, -60.0, 60.0))
            blend = 1.0 / (1.0 + math.exp(exponent))
        speed_sigma = self.args.direct_place_terminal_speed_sigma
        if speed_sigma is not None:
            object_speed = float(metrics["object_speed"])
            blend *= math.exp(-((object_speed / float(speed_sigma)) ** 2))
        blend *= float(self.args.direct_place_terminal_max_blend)
        self.last_direct_place_terminal_weight = blend
        latent = policy.project_z(
            (1.0 - blend) * direct_place_base
            + blend * terminal
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
            self.completion_lost_steps = 0 if placed else self.completion_lost_steps + 1
            if self.completion_lost_steps >= self.place_retry_steps:
                self.phase = "place_direct"
                self.phase_steps = 0
                self.placed_steps = 0
                self.completion_lost_steps = 0
            return
        if self.placed_steps >= self.args.place_success_steps:
            self.phase = "complete"
            self.phase_steps = 0
            self.completion_lost_steps = 0


class DirectPushController(PhaseController):
    """Apply a direct push latent with optional goal-directed steering."""

    def reset(self) -> None:
        super().reset()
        self.phase = "push_direct"
        self.reposition_side = 0.0
        self.push_attempt = 1
        self.push_attempt_best_distance = float("inf")
        self.push_overshoot_steps = 0

    def _start_push_attempt(self, initial_goal_distance: float) -> None:
        self.push_attempt += 1
        self.push_attempt_best_distance = float(initial_goal_distance)
        self.push_overshoot_steps = 0

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
        if self.phase == "push_direct_stop":
            self.last_transport_turn_weight = 0.0
            return observation, z["direct_stop"], "push_direct_stop"
        if self.phase == "push_direct_reposition":
            goal_norm = max(float(np.linalg.norm(goal_direction)), 1e-8)
            goal_unit = np.asarray(goal_direction, dtype=np.float64) / goal_norm
            lateral_unit = np.asarray((-goal_unit[1], goal_unit[0]), dtype=np.float64)
            if self.reposition_side == 0.0:
                robot_from_box = -np.asarray(box_direction, dtype=np.float64)
                lateral_projection = float(np.dot(robot_from_box, lateral_unit))
                self.reposition_side = 1.0 if lateral_projection >= 0.0 else -1.0

            behind = float(metrics["push_behind_projection"])
            spatially_ready = (
                float(metrics["push_staging_distance"])
                <= self.args.push_staging_tolerance
                and behind >= self.args.push_behind_min
                and float(metrics["push_lateral_error"]) <= self.args.push_lateral_max
            )
            if behind < self.args.push_behind_min:
                # Do not take the straight path through the box when it was moved
                # across the robot. First route around one side to a behind-box
                # waypoint, then converge to the central staging point.
                target_direction = (
                    np.asarray(box_direction, dtype=np.float64)
                    - goal_unit * float(self.args.direct_push_reposition_behind_offset)
                    + self.reposition_side
                    * lateral_unit
                    * float(self.args.direct_push_reposition_side_clearance)
                )
                mode_prefix = "push_reposition_bypass"
            elif spatially_ready:
                target_direction = np.asarray(box_direction, dtype=np.float64)
                mode_prefix = "push_reposition_align"
            else:
                target_direction = np.asarray(box_direction, dtype=np.float64) - (
                    goal_unit * float(self.args.push_standoff_distance)
                )
                mode_prefix = "push_reposition"
            if float(np.linalg.norm(target_direction)) < 1e-6:
                target_direction = np.asarray(box_direction, dtype=np.float64)
            bearing = _bearing_deg(base_rotation, target_direction)
            turn_weight = (
                min(abs(bearing) / self.args.heading_blend_full_deg, 1.0)
                * self.args.max_turn_blend
            )
            self.last_transport_turn_weight = turn_weight
            direction = 1 if bearing >= 0.0 else -1
            latent = policy.project_z(
                (1.0 - turn_weight) * z["direct_approach"]
                + turn_weight * z["direct_approach_turn"][direction]
            )
            mode = mode_prefix if turn_weight < 1e-3 else (
                f"{mode_prefix}_left" if direction > 0 else f"{mode_prefix}_right"
            )
            policy_observation = (
                _masked_object_observation(observation)
                if self.args.mask_approach_object_context
                else observation
            )
            return policy_observation, latent, mode
        if "direct_turn" not in z:
            self.last_transport_turn_weight = 0.0
            return observation, z["push_direct"], "push_direct"

        bearing = _bearing_deg(base_rotation, goal_direction)
        turn_weight = (
            min(abs(bearing) / self.args.direct_push_heading_blend_full_deg, 1.0)
            * self.args.direct_push_max_turn_blend
        )
        self.last_transport_turn_weight = turn_weight
        direction = 1 if bearing >= 0.0 else -1
        latent = policy.project_z(
            (1.0 - turn_weight) * z["push_direct"]
            + turn_weight * z["direct_turn"][direction]
        )
        mode = "push_direct" if turn_weight < 1e-3 else (
            "push_direct_left" if direction > 0 else "push_direct_right"
        )
        return observation, latent, mode

    def update(self, metrics: dict[str, float | bool], *, goal_height: float) -> None:
        del goal_height
        self.phase_steps += 1
        if self.phase == "push_direct_stop":
            outside_goal = (
                float(metrics["goal_distance"]) > self.args.direct_push_stop_exit_distance
            )
            self.lost_steps = self.lost_steps + 1 if outside_goal else 0
            if self.lost_steps >= self.args.direct_push_stop_exit_steps:
                recovery_phase = (
                    "push_direct_reposition"
                    if getattr(self.args, "direct_push_turn_latent_dir", None) is not None
                    else "push_direct"
                )
                self._change_phase(recovery_phase)
                self.reposition_side = 0.0
                self.recovery_count += 1
            return
        if self.phase == "push_direct_reposition":
            ready = (
                float(metrics["push_staging_distance"])
                <= self.args.push_staging_tolerance
                and float(metrics["push_behind_projection"]) >= self.args.push_behind_min
                and float(metrics["push_lateral_error"]) <= self.args.push_lateral_max
                and abs(float(metrics["push_box_bearing_deg"]))
                <= self.args.direct_push_reposition_heading_tolerance_deg
            )
            self.ready_steps = self.ready_steps + 1 if ready else 0
            if self.ready_steps >= self.args.direct_push_reposition_enter_steps:
                self._change_phase("push_direct")
                self._start_push_attempt(float(metrics["goal_distance"]))
            return
        if self.phase != "push_direct":
            return

        stop_enabled = getattr(self.args, "direct_push_stop_latent_dir", None) is not None
        overshoot_configured = (
            getattr(self.args, "direct_push_turn_latent_dir", None) is not None
            and getattr(self.args, "direct_push_overshoot_margin", 0.0) > 0.0
        )
        if not stop_enabled and not overshoot_configured:
            return
        goal_distance = float(metrics["goal_distance"])
        if stop_enabled:
            inside_goal = goal_distance <= self.args.direct_push_stop_distance
            self.ready_steps = self.ready_steps + 1 if inside_goal else 0
            if self.ready_steps >= self.args.direct_push_stop_enter_steps:
                self._change_phase("push_direct_stop")
                return

        if goal_distance < self.push_attempt_best_distance:
            self.push_attempt_best_distance = goal_distance
        overshot = (
            overshoot_configured
            and goal_distance
            > self.push_attempt_best_distance + self.args.direct_push_overshoot_margin
        )
        self.push_overshoot_steps = self.push_overshoot_steps + 1 if overshot else 0
        if self.push_overshoot_steps >= self.args.direct_push_overshoot_steps:
            self._change_phase("push_direct_reposition")
            self.reposition_side = 0.0
            self.recovery_count += 1


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
        if args.show_coordinate_frames:
            label_parts.append("Frames: robot base + box; XYZ = RGB")
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
    privileged_body_ids = [model.body(name).id for name in motion["body_names"]]
    privileged_torso_body_id = model.body("torso_link").id
    try:
        privileged_dim = int(policy.obs_space["privileged_state"].shape[0])
    except (KeyError, TypeError):
        privileged_dim = 0
    max_local_self_dim = 15 * (len(privileged_body_ids) + 1) - 2
    privileged_object_context_dim = privileged_dim - max_local_self_dim
    if privileged_dim and privileged_object_context_dim <= 0:
        raise ValueError(
            f"Unexpected privileged observation dimension {privileged_dim} for "
            f"{len(privileged_body_ids) + 1} tracked bodies"
        )
    if privileged_dim:
        print(
            f"[sphere-log] privileged_state={privileged_dim} "
            f"max_local_self={max_local_self_dim} "
            f"object_context={privileged_object_context_dim}",
            flush=True,
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
        if args.direct_place_terminal_latent_dir is not None:
            profile_files = {
                name: args.direct_place_terminal_latent_dir / f"z_{name}.pkl"
                for name in (
                    "direct_place_slow",
                    "direct_place_lower",
                    "direct_place_release",
                )
            }
            if all(path.is_file() for path in profile_files.values()):
                for name in profile_files:
                    z[name] = _load_latent(
                        args.direct_place_terminal_latent_dir,
                        f"z_{name}",
                        device,
                    )
                for name in (
                    "direct_place_lower_drive",
                    "direct_place_recover",
                    "direct_place_guard",
                    "direct_place_lower_high",
                    "direct_place_lower_mid",
                    "direct_place_lower_low",
                ):
                    optional_path = args.direct_place_terminal_latent_dir / f"z_{name}.pkl"
                    if optional_path.is_file():
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
        if args.direct_push_turn_latent_dir is not None:
            z["direct_turn"] = _load_latent_bank(
                args.direct_push_turn_latent_dir,
                "z_push_turn",
                device,
            )
            z["direct_approach"] = _load_latent(
                args.direct_push_turn_latent_dir,
                "z_approach",
                device,
            )
            z["direct_approach_turn"] = _load_latent_bank(
                args.direct_push_turn_latent_dir,
                "z_turn",
                device,
            )
        if args.direct_push_stop_latent_dir is not None:
            z["direct_stop"] = _load_latent(
                args.direct_push_stop_latent_dir,
                "z_push_stop",
                device,
            )
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
            for name in ("approach", "lift", "transport", "align", "lower", "place")
        }
        if args.manual_keyboard:
            z["hold"] = _load_latent(latent_dir, "z_hold", device)
        z["get_up"] = _load_get_up_latent(args.get_up_latent, device)
        if int(z["get_up"].shape[1]) != int(z["approach"].shape[1]):
            raise ValueError(
                f"Get-up latent dim {int(z['get_up'].shape[1])} does not match policy latent dim "
                f"{int(z['approach'].shape[1])}"
            )
        z["turn"] = _load_latent_bank(latent_dir, "z_turn", device)
        z["transport_turn"] = _load_latent_bank(latent_dir, "z_transport_turn", device)

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
        preserve_root_object_relative_pose=args.controller_mode == "place-direct",
    )
    _set_initial_box_center_height(
        model,
        data,
        qpos_address=box_qpos_address,
        qvel_address=box_qvel_address,
        center_height=args.initial_box_center_height,
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
        "lift": False,
        "place": False,
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
    if args.manual_keyboard:
        controller = ManualKeyboardCarryController(args, dt=CONTROL_DT)
    elif args.controller_mode == "place-direct":
        controller = DirectPlaceController(args, dt=CONTROL_DT)
    elif args.controller_mode == "push-direct":
        controller = DirectPushController(args, dt=CONTROL_DT)
    elif args.controller_mode == "push":
        controller = PushPhaseController(args, dt=CONTROL_DT)
    else:
        controller = PhaseController(args, dt=CONTROL_DT)
    current_action = np.clip((initial_dof_pos - defaults) / target_scale, -args.action_clip, args.action_clip)
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
            "manual_forward",
            "manual_turn",
            "lift_requested",
            "place_requested",
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
            "left_gap_m",
            "right_gap_m",
            "box_x",
            "box_y",
            "box_z",
            "root_x",
            "root_y",
            "root_z",
            "push_staging_distance_m",
            "push_behind_projection_m",
            "push_lateral_error_m",
            "push_box_bearing_deg",
            "push_attempt",
            "push_attempt_best_goal_distance_m",
            "push_overshoot_steps",
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

    manual_keyboard = X11KeyboardState() if args.manual_keyboard else None
    previous_manual_held = (
        {name: False for name in manual_keyboard._KEYSYMS}
        if manual_keyboard is not None
        else {}
    )
    if args.manual_keyboard:
        print(
            "[manual] hold Up=forward, Left/Right=turn, Up+Left/Right=curve; "
            "release arrows=stop; Down/Space=stop; B=lift, F=place, R=reset, P=status",
            flush=True,
        )
        print(
            "[manual] empty idle uses z_place (it may retain a bent/place posture); "
            "carry idle uses z_hold",
            flush=True,
        )
    else:
        print("[interactive] W/S: goal +/-X, A/D: goal +/-Y, H: restore goal, R: reset, P: status")
    print(f"[interactive] green marker goal={configured_goal.tolist()} step={args.goal_step:.2f}m")
    if args.show_coordinate_frames:
        print(
            f"[visualization] coordinate frames: robot={robot_training.robot.base_body}, "
            f"box={BOX_BODY_NAME}; X=red Y=green Z=blue "
            f"positive-axis length={args.coordinate_frame_size:.3f}m "
            f"width={args.coordinate_frame_width:.3f}m; arrow tails=frame origins",
            flush=True,
        )
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
        mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
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
            manual_held = manual_keyboard.poll() if manual_keyboard is not None else {}
            lift_key_pressed = bool(
                manual_held.get("b", False) and not previous_manual_held.get("b", False)
            )
            place_key_pressed = bool(
                manual_held.get("f", False) and not previous_manual_held.get("f", False)
            )
            previous_manual_held = manual_held
            with lock:
                if interaction["restore_goal"]:
                    interaction["goal"][:] = configured_goal
                    interaction["goal_changed"] = True
                    interaction["restore_goal"] = False
                should_reset = bool(interaction["reset"])
                should_print = bool(interaction["print"])
                should_lift = bool(interaction["lift"]) or lift_key_pressed
                should_place = bool(interaction["place"]) or place_key_pressed
                interaction["reset"] = False
                interaction["print"] = False
                interaction["lift"] = False
                interaction["place"] = False
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
                    preserve_root_object_relative_pose=args.controller_mode == "place-direct",
                )
                _set_initial_box_center_height(
                    model,
                    data,
                    qpos_address=box_qpos_address,
                    qvel_address=box_qvel_address,
                    center_height=args.initial_box_center_height,
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
                observer.reset_history()
                controller.reset()
                contact_hysteresis.reset()
                data.xfrc_applied[:] = 0.0
                print("[interactive] simulation reset", flush=True)
            if args.manual_keyboard and should_place:
                accepted, message = controller.request_place()
                print(f"[manual] {message}", flush=True)
                event_log.append(
                    {
                        "event": "manual_place_request",
                        "time_s": step * CONTROL_DT,
                        "accepted": bool(accepted),
                        "message": message,
                    }
                )
                if accepted:
                    goal = data.qpos[box_qpos_address : box_qpos_address + 3].copy()
                    goal[2] = ground_height
                    goal_changed = True
                    with lock:
                        interaction["goal"][:] = goal
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
            if privileged_dim:
                object_context = observation["state"][:, -privileged_object_context_dim:]
                privileged_state = _build_privileged_state(
                    model,
                    data,
                    body_ids=privileged_body_ids,
                    torso_body_id=privileged_torso_body_id,
                    object_context=object_context,
                    device=device,
                )
                if int(privileged_state.shape[-1]) != privileged_dim:
                    raise RuntimeError(
                        f"Unexpected privileged_state shape {tuple(privileged_state.shape)}; "
                        f"expected (*, {privileged_dim})"
                    )
                observation["privileged_state"] = privileged_state
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
            if args.manual_keyboard and should_lift:
                accepted, message = controller.request_lift(metrics)
                print(f"[manual] {message}", flush=True)
                event_log.append(
                    {
                        "event": "manual_lift_request",
                        "time_s": step * CONTROL_DT,
                        "accepted": bool(accepted),
                        "message": message,
                        "robot_box_distance_m": float(metrics["horizontal_distance"]),
                        "box_bearing_deg": float(metrics["push_box_bearing_deg"]),
                    }
                )
            if manual_keyboard is not None:
                stop = manual_held["down"] or manual_held["space"]
                turn = int(manual_held["left"]) - int(manual_held["right"])
                controller.set_motion(
                    forward=manual_held["up"] and not stop,
                    turn=0 if stop else turn,
                )
            active_phase = controller.phase
            box_pos = data.qpos[box_qpos_address : box_qpos_address + 3]
            policy_observation, latent, mode = controller.latent(
                policy,
                observation,
                metrics,
                base_rotation=data.xmat[base_body_id].reshape(3, 3),
                box_direction=box_pos[:2] - data.xpos[base_body_id, :2],
                goal_direction=goal[:2] - box_pos[:2],
                z=z,
            )
            raw_action = policy.act(policy_observation, latent, mean=True)[0].detach().cpu().numpy()
            current_action = np.clip(raw_action * float(robot_training.normalize_action_to), -args.action_clip, args.action_clip)
            q_target = np.clip(defaults + current_action * target_scale, limits[:, 0], limits[:, 1])

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
            phase_before_update = controller.phase
            controller.update(post_metrics, goal_height=ground_height)
            if args.manual_keyboard and controller.phase != phase_before_update:
                transition = f"{phase_before_update} -> {controller.phase}"
                print(f"[manual] phase {transition}", flush=True)
                event_log.append(
                    {
                        "event": "manual_phase_transition",
                        "time_s": (step + 1) * CONTROL_DT,
                        "transition": transition,
                    }
                )

            if trace_writer is not None:
                post_object_lin_vel, post_object_ang_vel = _world_body_velocity(model, data, box_body_id)
                trace_writer.writerow(
                    {
                        "step": step + 1,
                        "time_s": (step + 1) * CONTROL_DT,
                        "phase_before": active_phase,
                        "phase_after": controller.phase,
                        "mode": mode,
                        "manual_forward": int(getattr(controller, "manual_forward", False)),
                        "manual_turn": int(getattr(controller, "manual_turn", 0)),
                        "lift_requested": int(should_lift),
                        "place_requested": int(should_place),
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
                        "left_gap_m": post_surface_distance[0],
                        "right_gap_m": post_surface_distance[1],
                        "box_x": data.qpos[box_qpos_address],
                        "box_y": data.qpos[box_qpos_address + 1],
                        "box_z": data.qpos[box_qpos_address + 2],
                        "root_x": data.xpos[base_body_id, 0],
                        "root_y": data.xpos[base_body_id, 1],
                        "root_z": data.xpos[base_body_id, 2],
                        "push_staging_distance_m": post_metrics["push_staging_distance"],
                        "push_behind_projection_m": post_metrics["push_behind_projection"],
                        "push_lateral_error_m": post_metrics["push_lateral_error"],
                        "push_box_bearing_deg": post_metrics["push_box_bearing_deg"],
                        "push_attempt": getattr(controller, "push_attempt", 0),
                        "push_attempt_best_goal_distance_m": (
                            ""
                            if not np.isfinite(
                                getattr(controller, "push_attempt_best_distance", float("inf"))
                            )
                            else getattr(controller, "push_attempt_best_distance")
                        ),
                        "push_overshoot_steps": getattr(
                            controller, "push_overshoot_steps", 0
                        ),
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
                completed_phase = controller.phase in ("complete", "push_direct_stop")
                complete_stable_steps = complete_stable_steps + 1 if completed_phase else 0
                if (
                    completed_phase
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
                    f"attempt={getattr(controller, 'push_attempt', controller.place_attempt)} "
                    f"best_goal={getattr(controller, 'push_attempt_best_distance', float('nan')):.3f}m "
                    f"overshoot={getattr(controller, 'push_overshoot_steps', 0)} "
                    f"recovery={controller.recovery_count}",
                    flush=True,
                )
            if viewer is not None:
                if args.show_coordinate_frames or args.show_hand_capsules:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        if args.show_coordinate_frames:
                            _append_robot_box_coordinate_frames(
                                viewer.user_scn,
                                data,
                                base_body_id=base_body_id,
                                box_body_id=box_body_id,
                                size=args.coordinate_frame_size,
                                width=args.coordinate_frame_width,
                            )
                        if args.show_hand_capsules:
                            _append_hand_contact_capsules(
                                viewer.user_scn,
                                data,
                                hand_body_ids=hand_body_ids,
                                capsule_segments=contact_hysteresis.capsule_segments.cpu().numpy(),
                                capsule_radius=contact_hysteresis.capsule_radius,
                                contact=post_hysteresis_contact,
                            )
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
                    if args.show_coordinate_frames:
                        _append_robot_box_coordinate_frames(
                            renderer.scene,
                            data,
                            base_body_id=base_body_id,
                            box_body_id=box_body_id,
                            size=args.coordinate_frame_size,
                            width=args.coordinate_frame_width,
                        )
                    if args.show_hand_capsules:
                        _append_hand_contact_capsules(
                            renderer.scene,
                            data,
                            hand_body_ids=hand_body_ids,
                            capsule_segments=contact_hysteresis.capsule_segments.cpu().numpy(),
                            capsule_radius=contact_hysteresis.capsule_radius,
                            contact=post_hysteresis_contact,
                        )
                    frame = np.asarray(renderer.render()).copy()
                    video_writer.add_image(_annotate_video_frame(frame, video_label))
                    capture_accumulator -= 1.0

    if video_writer is not None:
        video_writer.close()
        print(f"[recording] saved MuJoCo video to {args.output_path}", flush=True)
    if renderer is not None:
        renderer.close()
    if manual_keyboard is not None:
        manual_keyboard.close()
    if trace_stream is not None:
        trace_stream.close()
        print(f"[trace] saved per-step CSV to {args.trace_csv}", flush=True)
    if event_log and args.save_mp4:
        event_path = args.output_path.with_suffix(".events.json")
        event_path.write_text(json.dumps(event_log, indent=2), encoding="utf-8")
        print(f"[disturbance] event log saved to {event_path}", flush=True)
    if args.summary_path is not None:
        summary = _summarize_speed_rollout(rollout_history, final_phase=controller.phase)
        summary.update(
            {
                "model_folder": str(args.model_folder),
                "latent_dir": str(args.latent_dir),
                "initial_distance": float(args.initial_distance),
                "goal_offset": [float(args.goal_offset_x), float(args.goal_offset_y)],
                "manual_keyboard": bool(args.manual_keyboard),
                "manual_turn_blend": float(args.manual_turn_blend),
                "manual_lift_pose_gating": False,
                "manual_command_events": [
                    event for event in event_log if event["event"].startswith("manual_")
                ],
            }
        )
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[metrics] saved rollout summary to {args.summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument(
        "--controller-mode",
        choices=("carry", "place-direct", "push", "push-direct"),
        default="carry",
        help=(
            "Use the original phased Carry controller, one fixed direct-place latent, "
            "the phased ground-push controller, or one fixed direct-push latent."
        ),
    )
    parser.add_argument(
        "--direct-push-turn-latent-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing z_push_turn.pkl. When supplied, push-direct "
            "steers toward the current box-to-goal direction."
        ),
    )
    parser.add_argument(
        "--direct-place-terminal-latent-dir",
        type=Path,
        default=None,
        help=(
            "Optional reward-inferred stable-placement latent. place-direct blends toward "
            "it continuously as the box approaches the goal, without a phase switch."
        ),
    )
    parser.add_argument(
        "--direct-place-turn-latent-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing z_transport_turn.pkl. Direct placement uses "
            "it for far-field goal steering and fades it out before lowering."
        ),
    )
    parser.add_argument("--direct-place-turn-heading-full-deg", type=float, default=45.0)
    parser.add_argument("--direct-place-turn-max-blend", type=float, default=0.45)
    parser.add_argument("--direct-place-turn-fade-start-distance", type=float, default=0.80)
    parser.add_argument("--direct-place-turn-fade-end-distance", type=float, default=0.60)
    parser.add_argument("--direct-place-terminal-blend-distance", type=float, default=0.35)
    parser.add_argument(
        "--direct-place-terminal-blend-start-distance",
        type=float,
        default=None,
        help=(
            "Optional smoothstep start distance for continuous deceleration blending; "
            "when supplied it takes precedence over Gaussian/sigmoid blending."
        ),
    )
    parser.add_argument(
        "--direct-place-terminal-blend-end-distance",
        type=float,
        default=0.30,
        help="Distance where smoothstep terminal blending reaches one.",
    )
    parser.add_argument("--direct-place-profile-release-distance", type=float, default=0.20)
    parser.add_argument("--direct-place-profile-lower-full-distance", type=float, default=0.25)
    parser.add_argument("--direct-place-release-lift-start", type=float, default=0.08)
    parser.add_argument("--direct-place-release-lift-end", type=float, default=0.02)
    parser.add_argument(
        "--direct-place-release-latch-lift",
        type=float,
        default=0.0,
        help="Latch full release once normalized lift falls below this value; zero disables.",
    )
    parser.add_argument(
        "--direct-place-release-latch-distance",
        type=float,
        default=0.35,
        help="Require the box to be this close to the goal before latching release.",
    )
    parser.add_argument(
        "--direct-place-release-latch-max-vertical-speed",
        type=float,
        default=0.20,
        help="Require near-ground vertical speed below this magnitude before release.",
    )
    parser.add_argument(
        "--direct-place-release-latch-min-upright",
        type=float,
        default=0.90,
        help="Require box uprightness above this value before latching release.",
    )
    parser.add_argument(
        "--direct-place-profile-commit-seconds",
        type=float,
        default=5.0,
        help=(
            "Seconds needed to latch the lower latent after reaching the near-goal region; "
            "zero disables stateful commitment."
        ),
    )
    parser.add_argument(
        "--direct-place-profile-commit-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Drive lowering only from the time-filtered commitment state instead of "
            "allowing the geometric distance schedule to jump directly to full lowering."
        ),
    )
    parser.add_argument(
        "--direct-place-profile-commit-distance",
        type=float,
        default=0.32,
        help=(
            "Near-goal distance where a sufficiently slow box begins accumulating lower intent; "
            "slightly outside the 0.30 m target prevents stopping on its numerical boundary."
        ),
    )
    parser.add_argument("--direct-place-profile-commit-speed", type=float, default=0.20)
    parser.add_argument("--direct-place-profile-commit-reset-distance", type=float, default=0.60)
    parser.add_argument("--direct-place-lower-target-speed", type=float, default=-0.08)
    parser.add_argument("--direct-place-lower-high-lift", type=float, default=0.75)
    parser.add_argument("--direct-place-lower-mid-lift", type=float, default=0.45)
    parser.add_argument("--direct-place-lower-low-lift", type=float, default=0.15)
    parser.add_argument("--direct-place-lower-speed-deadband", type=float, default=0.04)
    parser.add_argument("--direct-place-lower-drive-max-blend", type=float, default=0.40)
    parser.add_argument("--direct-place-lower-drive-rise-seconds", type=float, default=3.0)
    parser.add_argument("--direct-place-lower-drive-fall-seconds", type=float, default=0.25)
    parser.add_argument("--direct-place-recovery-fall-speed", type=float, default=-0.30)
    parser.add_argument("--direct-place-recovery-lift-threshold", type=float, default=0.10)
    parser.add_argument("--direct-place-recovery-stable-speed", type=float, default=0.12)
    parser.add_argument("--direct-place-recovery-stable-steps", type=int, default=25)
    parser.add_argument(
        "--direct-place-descent-guard-start-speed",
        type=float,
        default=0.10,
        help="Start blending toward the stable slow latent above this downward speed.",
    )
    parser.add_argument(
        "--direct-place-descent-guard-full-speed",
        type=float,
        default=0.25,
        help="Use the stable slow latent fully above this downward speed.",
    )
    parser.add_argument(
        "--direct-place-terminal-blend-center",
        type=float,
        default=None,
        help="Optional sigmoid centre; when set it replaces the Gaussian blend curve.",
    )
    parser.add_argument("--direct-place-terminal-blend-width", type=float, default=0.03)
    parser.add_argument(
        "--direct-place-terminal-speed-sigma",
        type=float,
        default=None,
        help="Optional smooth low-box-speed gate multiplied into terminal blending.",
    )
    parser.add_argument("--direct-place-terminal-max-blend", type=float, default=1.0)
    parser.add_argument(
        "--direct-place-success-distance",
        type=float,
        default=0.30,
        help=(
            "Goal-distance tolerance used to mark a stable direct placement complete. "
            "Ground height, speed, uprightness, release, and dwell checks still apply."
        ),
    )
    parser.add_argument("--direct-push-heading-blend-full-deg", type=float, default=45.0)
    parser.add_argument("--direct-push-max-turn-blend", type=float, default=0.65)
    parser.add_argument(
        "--direct-push-stop-latent-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing z_push_stop.pkl. When supplied, direct push "
            "enters the standing stop latent inside the goal region and resumes via a "
            "behind-box reposition phase if the box later leaves it."
        ),
    )
    parser.add_argument("--direct-push-stop-distance", type=float, default=0.20)
    parser.add_argument("--direct-push-stop-enter-steps", type=int, default=5)
    parser.add_argument("--direct-push-stop-exit-distance", type=float, default=0.35)
    parser.add_argument("--direct-push-stop-exit-steps", type=int, default=10)
    parser.add_argument("--direct-push-reposition-enter-steps", type=int, default=5)
    parser.add_argument("--direct-push-reposition-side-clearance", type=float, default=0.75)
    parser.add_argument("--direct-push-reposition-behind-offset", type=float, default=0.40)
    parser.add_argument(
        "--direct-push-reposition-heading-tolerance-deg",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--direct-push-overshoot-margin",
        type=float,
        default=0.10,
        help="Distance regression beyond the attempt best before counting an overshoot.",
    )
    parser.add_argument(
        "--direct-push-overshoot-arm-distance",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--direct-push-overshoot-steps", type=int, default=10)
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
        "--initial-box-center-height",
        type=float,
        default=None,
        help="Optionally override the reset box centre height and zero its free-joint velocity.",
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
    parser.add_argument(
        "--show-coordinate-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show robot-base and box-body local frames (X red, Y green, Z blue) in the viewer and MP4.",
    )
    parser.add_argument("--coordinate-frame-size", type=float, default=0.65)
    parser.add_argument("--coordinate-frame-width", type=float, default=0.012)
    parser.add_argument(
        "--show-hand-capsules",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Show the exact contact-hysteresis hand capsules in the viewer and MP4; "
            "left is cyan, right is orange, and an estimated contact turns green."
        ),
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--manual-keyboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable held-arrow carry control: Up moves, Left/Right turn, releasing stops; "
            "B starts lift and F starts placement. Requires an X11/XWayland viewer."
        ),
    )
    parser.add_argument(
        "--manual-turn-blend",
        type=float,
        default=0.65,
        help="Turn-latent weight for Up+Left/Right in manual keyboard mode.",
    )
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
        help="Deprecated compatibility option; fallen robots now stay in get-up until stably upright.",
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
    parser.add_argument("--place-release-seconds", type=float, default=2.0)
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
    if args.initial_distance <= 0.0 or args.goal_step <= 0.0 or args.start_fallen_root_height <= 0.0:
        parser.error("--initial-distance, --goal-step, and --start-fallen-root-height must be positive")
    if args.initial_box_center_height is not None and args.initial_box_center_height <= 0.0:
        parser.error("--initial-box-center-height must be positive")
    if args.save_mp4 and not args.viewer and args.max_steps <= 0:
        parser.error("headless --save-mp4 requires a positive --max-steps")
    if args.manual_keyboard and not args.viewer:
        parser.error("--manual-keyboard requires --viewer")
    if args.manual_keyboard and args.controller_mode != "carry":
        parser.error("--manual-keyboard is only compatible with --controller-mode carry")
    if not 0.0 <= args.manual_turn_blend <= 1.0:
        parser.error("--manual-turn-blend must be in [0, 1]")
    if args.render_size <= 0 or args.video_fps <= 0.0:
        parser.error("--render-size and --video-fps must be positive")
    if args.coordinate_frame_size <= 0.0 or args.coordinate_frame_width <= 0.0:
        parser.error("--coordinate-frame-size and --coordinate-frame-width must be positive")
    if min(
        args.box_drop_force_n,
        args.box_drop_force_seconds,
        args.box_drop_transport_delay_seconds,
        args.relocate_after_complete_seconds,
    ) < 0.0:
        parser.error("disturbance force and timing values must be non-negative")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("--device must be cpu or cuda[:index]")
    if args.direct_push_heading_blend_full_deg <= 0.0:
        parser.error("--direct-push-heading-blend-full-deg must be positive")
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
    if args.direct_place_terminal_blend_start_distance is not None and not (
        args.direct_place_terminal_blend_start_distance
        > args.direct_place_terminal_blend_end_distance
        > 0.0
    ):
        parser.error("direct-place smoothstep distances must satisfy start > end > 0")
    if not (
        args.direct_place_terminal_blend_end_distance
        > args.direct_place_profile_lower_full_distance
        >= args.direct_place_profile_release_distance
        > 0.0
    ):
        parser.error(
            "direct-place profile distances must satisfy blend end > lower full >= release > 0"
        )
    if not 0.0 <= args.direct_place_release_lift_end < args.direct_place_release_lift_start:
        parser.error("direct-place release lift values must satisfy 0 <= end < start")
    if not 0.0 <= args.direct_place_release_latch_lift <= 1.0:
        parser.error("--direct-place-release-latch-lift must be in [0, 1]")
    if args.direct_place_release_latch_distance <= 0.0:
        parser.error("--direct-place-release-latch-distance must be positive")
    if args.direct_place_release_latch_max_vertical_speed <= 0.0:
        parser.error("--direct-place-release-latch-max-vertical-speed must be positive")
    if not 0.0 <= args.direct_place_release_latch_min_upright <= 1.0:
        parser.error("--direct-place-release-latch-min-upright must be in [0, 1]")
    if args.direct_place_profile_commit_seconds < 0.0:
        parser.error("--direct-place-profile-commit-seconds must be non-negative")
    if not (
        0.0
        < args.direct_place_profile_commit_distance
        < args.direct_place_profile_commit_reset_distance
    ):
        parser.error("direct-place commitment distances must satisfy 0 < enter < reset")
    if args.direct_place_profile_commit_speed <= 0.0:
        parser.error("--direct-place-profile-commit-speed must be positive")
    if args.direct_place_lower_target_speed >= 0.0:
        parser.error("--direct-place-lower-target-speed must be negative")
    if not (
        0.0
        < args.direct_place_lower_low_lift
        < args.direct_place_lower_mid_lift
        < args.direct_place_lower_high_lift
        <= 1.0
    ):
        parser.error("direct-place lower lift anchors must satisfy 0 < low < mid < high <= 1")
    if args.direct_place_lower_speed_deadband <= 0.0:
        parser.error("--direct-place-lower-speed-deadband must be positive")
    if not 0.0 <= args.direct_place_lower_drive_max_blend <= 1.0:
        parser.error("--direct-place-lower-drive-max-blend must be in [0, 1]")
    if min(
        args.direct_place_lower_drive_rise_seconds,
        args.direct_place_lower_drive_fall_seconds,
        args.direct_place_recovery_lift_threshold,
        args.direct_place_recovery_stable_speed,
    ) <= 0.0:
        parser.error("direct-place lower-drive/recovery scales must be positive")
    if args.direct_place_recovery_fall_speed >= 0.0:
        parser.error("--direct-place-recovery-fall-speed must be negative")
    if args.direct_place_recovery_stable_steps <= 0:
        parser.error("--direct-place-recovery-stable-steps must be positive")
    if not (
        0.0
        < args.direct_place_descent_guard_start_speed
        < args.direct_place_descent_guard_full_speed
    ):
        parser.error("direct-place descent guard speeds must satisfy 0 < start < full")
    if args.direct_place_terminal_blend_center is not None and (
        args.direct_place_terminal_blend_center <= 0.0
        or args.direct_place_terminal_blend_width <= 0.0
    ):
        parser.error("direct-place sigmoid blend centre and width must be positive")
    if (
        args.direct_place_terminal_speed_sigma is not None
        and args.direct_place_terminal_speed_sigma <= 0.0
    ):
        parser.error("--direct-place-terminal-speed-sigma must be positive")
    if not 0.0 <= args.direct_place_terminal_max_blend <= 1.0:
        parser.error("--direct-place-terminal-max-blend must be in [0, 1]")
    if args.direct_place_success_distance <= 0.0:
        parser.error("--direct-place-success-distance must be positive")
    if not 0.0 <= args.direct_push_max_turn_blend < 1.0:
        parser.error("--direct-push-max-turn-blend must be in [0, 1)")
    if not 0.0 < args.direct_push_stop_distance < args.direct_push_stop_exit_distance:
        parser.error("direct-push stop distances must satisfy 0 < enter < exit")
    if min(
        args.direct_push_stop_enter_steps,
        args.direct_push_stop_exit_steps,
        args.direct_push_reposition_enter_steps,
    ) <= 0:
        parser.error("direct-push stop/reposition step counts must be positive")
    if min(
        args.direct_push_reposition_side_clearance,
        args.direct_push_reposition_behind_offset,
        args.direct_push_reposition_heading_tolerance_deg,
    ) <= 0.0:
        parser.error("direct-push reposition geometry and heading values must be positive")
    if args.direct_push_reposition_behind_offset < args.push_behind_min:
        parser.error("--direct-push-reposition-behind-offset must be >= --push-behind-min")
    if args.direct_push_overshoot_margin < 0.0 or args.direct_push_overshoot_steps <= 0:
        parser.error("direct-push overshoot margin/steps must be non-negative/positive")
    if not 0.0 <= args.contact_enter_distance < args.contact_exit_distance:
        parser.error("contact thresholds must satisfy 0 <= enter < exit")
    if args.transport_heading_blend_full_deg <= 0.0:
        parser.error("--transport-heading-blend-full-deg must be positive")
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
