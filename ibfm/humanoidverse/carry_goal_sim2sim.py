"""Run a fixed Carry goal policy in a pure MuJoCo simulation loop.

Unlike ``goal_recovery_inference``, this module does not build an MJLab
environment.  It reconstructs the actor observation from MuJoCo state and
applies the same normalized-action and joint-PD conventions used for training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio_ffmpeg
import joblib
import mediapy as media
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.envs.carry_observations import (
    build_object_context,
    compute_carry_object_observation,
    compute_hand_contact_points,
    compute_hand_spatial_observation,
)
from humanoidverse.envs.motion_observations import compute_humanoid_observations_max
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import quat_rotate_inverse

BOX_BODY_NAME = "sim2sim_carry_box"
BOX_JOINT_NAME = "sim2sim_carry_box_joint"
BOX_GEOM_NAME = "sim2sim_carry_box_geom"
FLOOR_GEOM_NAME = "sim2sim_floor"
GOAL_GEOM_NAME = "sim2sim_carry_goal"
HAND_GEOM_NAMES = ("left_hand_collision", "right_hand_collision")
HISTORY_LENGTH = 4
CONTROL_DT = 0.02
SIM_DT = 0.005


def _reflected_inertia(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    return rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2 + rotor_inertia[1] * gear_ratio[2] ** 2 + rotor_inertia[2]


_ARMATURE_5020 = _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + (46 / 18), 1 + (56 / 16)))
_ARMATURE_7520_14 = _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + (48 / 22)))
_ARMATURE_7520_22 = _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
_ARMATURE_5010 = _reflected_inertia((0.084e-4, 0.015e-4, 0.068e-4), (1, 4, 4))


def _as_numpy(value: Any, *, dtype: np.dtype = np.float64) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _frame_or_static(value: Any, frame: int, *, width: int) -> np.ndarray:
    array = _as_numpy(value)
    if array.shape == (width,):
        return array.copy()
    if array.ndim == 2 and array.shape[1] == width:
        return array[int(frame)].copy()
    raise ValueError(f"Expected a static [{width}] or framed [T, {width}] value, got shape {array.shape}")


def _match_joint_parameter(joint_name: str, values: dict[str, float]) -> float:
    matches = [(key, float(value)) for key, value in values.items() if key in joint_name]
    if not matches:
        raise KeyError(f"No parameter matches joint {joint_name!r}; available keys={sorted(values)}")
    return max(matches, key=lambda item: len(item[0]))[1]


def _load_motion(cache_path: Path, motion_id: int) -> tuple[str, dict[str, Any]]:
    motions = joblib.load(cache_path)
    if not isinstance(motions, dict):
        raise TypeError(f"Expected a motion dictionary in {cache_path}, got {type(motions)}")
    keys = list(motions)
    if not 0 <= int(motion_id) < len(keys):
        raise IndexError(f"motion_id={motion_id} outside [0, {len(keys)})")
    key = keys[int(motion_id)]
    return key, motions[key]


def _load_box_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}, got {type(config)}")
    return config


def _build_model(
    robot_xml: Path,
    box_cfg: dict[str, Any],
    *,
    render_size: int,
    show_goal_marker: bool = False,
    goal_marker_radius: float = 0.20,
) -> mujoco.MjModel:
    half_size = np.asarray(box_cfg["half_size"], dtype=np.float64)
    volume = float(8.0 * np.prod(half_size))
    if volume <= 0.0:
        raise ValueError(f"Invalid box half-size {half_size.tolist()}")
    default_friction = tuple(float(value) for value in box_cfg.get("friction", (0.9, 0.01, 0.001)))
    box_friction = tuple(float(value) for value in box_cfg.get("sim2sim_box_friction", default_friction))
    box_condim = int(box_cfg.get("sim2sim_box_condim", 3))

    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.option.timestep = SIM_DT
    spec.worldbody.add_geom(
        name=FLOOR_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, 0.0),
        size=(20.0, 20.0, 0.05),
        rgba=(0.42, 0.44, 0.47, 1.0),
        friction=default_friction,
        contype=1,
        conaffinity=1,
    )
    box = spec.worldbody.add_body(name=BOX_BODY_NAME)
    box.add_freejoint(name=BOX_JOINT_NAME)
    box.add_geom(
        name=BOX_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=tuple(float(value) for value in half_size),
        pos=tuple(float(value) for value in box_cfg.get("geom_pos", (0.0, 0.0, 0.0))),
        quat=tuple(float(value) for value in box_cfg.get("geom_quat_wxyz", (1.0, 0.0, 0.0, 0.0))),
        rgba=(0.62, 0.43, 0.24, 1.0),
        friction=box_friction,
        condim=box_condim,
        density=float(box_cfg.get("mass", 1.25)) / volume,
        contype=1,
        conaffinity=1,
    )
    if show_goal_marker:
        spec.worldbody.add_geom(
            name=GOAL_GEOM_NAME,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(0.0, 0.0, 0.006),
            size=(float(goal_marker_radius), 0.006, 0.0),
            rgba=(0.12, 0.82, 0.25, 0.45),
            contype=0,
            conaffinity=0,
        )
    spec.worldbody.add_light(
        name="sim2sim_key_light",
        pos=(0.0, -3.0, 4.0),
        dir=(0.2, 0.5, -1.0),
        diffuse=(0.8, 0.8, 0.8),
        ambient=(0.35, 0.35, 0.35),
        specular=(0.1, 0.1, 0.1),
    )
    model = spec.compile()
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(render_size))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(render_size))
    return model


def _joint_addresses(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = np.asarray([model.joint(name).id for name in joint_names], dtype=np.int32)
    return model.jnt_qposadr[joint_ids].astype(np.int64), model.jnt_dofadr[joint_ids].astype(np.int64)


def _actuator_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    actuator_joint_ids = model.actuator_trnid[:, 0]
    result = []
    for name in joint_names:
        joint_id = model.joint(name).id
        matches = np.flatnonzero(actuator_joint_ids == joint_id)
        if matches.size != 1:
            raise ValueError(f"Expected one actuator for {name}, found {matches.tolist()}")
        result.append(int(matches[0]))
    return np.asarray(result, dtype=np.int64)


def _mode15_armature(joint_name: str) -> float:
    if "_hip_pitch_joint" in joint_name or "_hip_roll_joint" in joint_name or "_knee_joint" in joint_name:
        return _ARMATURE_7520_22
    if "_hip_yaw_joint" in joint_name or joint_name == "waist_yaw_joint":
        return _ARMATURE_7520_14
    if "_ankle_pitch_joint" in joint_name or "_ankle_roll_joint" in joint_name:
        return 2.0 * _ARMATURE_5020
    if joint_name in ("waist_pitch_joint", "waist_roll_joint"):
        return 2.0 * _ARMATURE_5020
    if "_wrist_pitch_joint" in joint_name or "_wrist_yaw_joint" in joint_name:
        return _ARMATURE_5010
    if any(
        token in joint_name
        for token in ("_shoulder_pitch_joint", "_shoulder_roll_joint", "_shoulder_yaw_joint", "_elbow_joint", "_wrist_roll_joint")
    ):
        return _ARMATURE_5020
    raise ValueError(f"No G1 mode-15 armature for joint {joint_name}")


def _patch_mode15_joint_physics(model: mujoco.MjModel, joint_names: list[str]) -> None:
    _qpos, qvel = _joint_addresses(model, joint_names)
    model.dof_armature[qvel] = np.asarray([_mode15_armature(name) for name in joint_names])
    model.dof_frictionloss[qvel] = 0.01


def _world_body_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), velocity, 0)
    return velocity[3:6].copy(), velocity[:3].copy()


def _physical_hand_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_geom_id: int,
    hand_body_ids: tuple[int, int],
) -> np.ndarray:
    found = np.zeros(2, dtype=np.float32)
    body_to_hand = {int(body_id): index for index, body_id in enumerate(hand_body_ids)}
    for contact in data.contact[: data.ncon]:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 == box_geom_id:
            other = geom2
        elif geom2 == box_geom_id:
            other = geom1
        else:
            continue
        hand = body_to_hand.get(int(model.geom_bodyid[other]))
        if hand is not None:
            found[hand] = 1.0
    return found


class ActorObservation:
    def __init__(
        self,
        *,
        device: torch.device,
        default_dof_pos: np.ndarray,
        target_object_pos: np.ndarray,
        hand_contact_offsets: np.ndarray,
        include_hand_spatial: bool,
    ) -> None:
        self.device = device
        self.default_dof_pos = torch.as_tensor(default_dof_pos, device=device, dtype=torch.float32).reshape(1, -1)
        self.target_object_pos = torch.as_tensor(target_object_pos, device=device, dtype=torch.float32).reshape(1, 3)
        self.hand_contact_offsets = torch.as_tensor(
            hand_contact_offsets, device=device, dtype=torch.float32
        ).reshape(2, 3)
        self.include_hand_spatial = bool(include_hand_spatial)
        self.num_dof = int(self.default_dof_pos.shape[-1])
        self.history = {
            "actions": torch.zeros(1, HISTORY_LENGTH, self.num_dof, device=device),
            "base_ang_vel": torch.zeros(1, HISTORY_LENGTH, 3, device=device),
            "dof_pos": torch.zeros(1, HISTORY_LENGTH, self.num_dof, device=device),
            "dof_vel": torch.zeros(1, HISTORY_LENGTH, self.num_dof, device=device),
            "projected_gravity": torch.zeros(1, HISTORY_LENGTH, 3, device=device),
        }

    def set_target_object_pos(self, target_object_pos: np.ndarray) -> None:
        self.target_object_pos.copy_(
            torch.as_tensor(target_object_pos, device=self.device, dtype=torch.float32).reshape(1, 3)
        )

    def reset_history(self) -> None:
        for value in self.history.values():
            value.zero_()

    @torch.no_grad()
    def build(
        self,
        *,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        base_pos: np.ndarray,
        base_quat_wxyz: np.ndarray,
        base_ang_vel_world: np.ndarray,
        object_pos: np.ndarray,
        object_quat_wxyz: np.ndarray,
        object_lin_vel: np.ndarray,
        object_ang_vel: np.ndarray,
        hand_body_pos: np.ndarray,
        hand_body_quat_wxyz: np.ndarray,
        hand_contact: np.ndarray,
        current_action: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        tensor = lambda value: torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(1, -1)
        base_quat = tensor(np.roll(base_quat_wxyz, -1))
        base_pos_t = tensor(base_pos)
        object_pos_t = tensor(object_pos)
        object_quat = tensor(np.roll(object_quat_wxyz, -1))
        dof_pos_t = tensor(dof_pos)
        dof_vel_t = tensor(dof_vel)
        action_t = tensor(current_action)
        gravity = tensor((0.0, 0.0, -1.0))
        projected_gravity = quat_rotate_inverse(base_quat, gravity, w_last=True)
        base_ang_vel = 0.25 * quat_rotate_inverse(base_quat, tensor(base_ang_vel_world), w_last=True)
        dof_pos_rel = dof_pos_t - self.default_dof_pos
        object_observation = compute_carry_object_observation(
            base_pos_t,
            base_quat,
            object_pos_t,
            object_quat,
            tensor(object_lin_vel),
            tensor(object_ang_vel),
            self.target_object_pos,
        )
        object_context = build_object_context(
            object_observation,
            torch.ones(1, 1, device=self.device),
            tensor(hand_contact),
            compute_hand_spatial_observation(
                base_pos_t,
                base_quat,
                compute_hand_contact_points(
                    torch.as_tensor(hand_body_pos, device=self.device, dtype=torch.float32).reshape(1, 2, 3),
                    torch.as_tensor(
                        np.roll(hand_body_quat_wxyz, -1, axis=-1), device=self.device, dtype=torch.float32
                    ).reshape(1, 2, 4),
                    self.hand_contact_offsets,
                ),
                object_pos_t,
            ),
            include_hand_spatial=self.include_hand_spatial,
            observe_target_delta=True,
            observe_object_present=True,
            observe_task_id=True,
        )
        state = torch.cat((dof_pos_rel, dof_vel_t, projected_gravity, base_ang_vel, object_context), dim=-1)
        raw = {
            "actions": action_t,
            "base_ang_vel": base_ang_vel,
            "dof_pos": dof_pos_rel,
            "dof_vel": dof_vel_t,
            "projected_gravity": projected_gravity,
        }
        history_actor = torch.cat([self.history[key].reshape(1, -1) for key in sorted(self.history)], dim=-1)
        for key, value in raw.items():
            self.history[key][:, 1:] = self.history[key][:, :-1].clone()
            self.history[key][:, 0] = value
        expected_state_dim = 2 * self.num_dof + (40 if self.include_hand_spatial else 28)
        expected_history_dim = HISTORY_LENGTH * (3 * self.num_dof + 6)
        if state.shape[-1] != expected_state_dim or history_actor.shape[-1] != expected_history_dim:
            raise RuntimeError(f"Unexpected actor observation dimensions state={state.shape}, history={history_actor.shape}")
        return {"state": state, "last_action": action_t, "history_actor": history_actor}


def _backward_observation_from_mujoco(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source_motion: dict[str, Any],
    actor_observation: dict[str, torch.Tensor],
    include_hand_spatial: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build the current B observation used by INTACT displacement commands."""

    body_names = list(source_motion["body_names"])
    body_pos: list[np.ndarray] = []
    body_quat_xyzw: list[np.ndarray] = []
    body_vel: list[np.ndarray] = []
    body_ang_vel: list[np.ndarray] = []
    for name in body_names:
        body_id = model.body(name).id
        linear, angular = _world_body_velocity(model, data, body_id)
        body_pos.append(data.xpos[body_id].copy())
        body_quat_xyzw.append(np.roll(data.xquat[body_id], -1).copy())
        body_vel.append(linear)
        body_ang_vel.append(angular)

    torso_index = body_names.index("torso_link")
    torso_rotation = np.asarray(data.xmat[model.body("torso_link").id]).reshape(3, 3)
    head_offset = torso_rotation @ np.asarray((0.0, 0.0, 0.35), dtype=np.float64)
    head_pos = body_pos[torso_index] + head_offset
    head_ang_vel = body_ang_vel[torso_index]
    head_vel = body_vel[torso_index] + np.cross(head_ang_vel, head_offset)
    body_pos.append(head_pos)
    body_quat_xyzw.append(body_quat_xyzw[torso_index])
    body_vel.append(head_vel)
    body_ang_vel.append(head_ang_vel)

    tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float32).unsqueeze(0)
    privileged_parts = compute_humanoid_observations_max(
        tensor(np.asarray(body_pos)),
        tensor(np.asarray(body_quat_xyzw)),
        tensor(np.asarray(body_vel)),
        tensor(np.asarray(body_ang_vel)),
        local_root_obs=True,
        root_height_obs=True,
    )
    privileged_state = torch.cat(list(privileged_parts.values()), dim=-1)
    object_context_dim = 34 if include_hand_spatial else 22
    object_context = actor_observation["state"][:, -object_context_dim:]
    privileged_state = torch.cat((privileged_state, object_context), dim=-1)
    expected_dim = 497 if include_hand_spatial else 485
    if privileged_state.shape != (1, expected_dim):
        raise RuntimeError(f"Unexpected current B privileged_state shape {privileged_state.shape}")
    return {
        "state": actor_observation["state"],
        "privileged_state": privileged_state,
    }


def _initialise_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: dict[str, Any],
    *,
    initial_frame: int,
    initial_distance: float,
    joint_names: list[str],
    qpos_addresses: np.ndarray,
    standing_init: bool = False,
    standing_root_height: float = 0.80,
    default_dof_pos: np.ndarray | None = None,
    initial_bearing_deg: float | None = None,
    initial_box_center_height: float | None = None,
    preserve_root_object_relative_pose: bool = False,
) -> np.ndarray:
    root_pos = _as_numpy(motion["root_trans_offset"])[initial_frame].copy()
    root_quat_xyzw = _as_numpy(motion["root_quat"])[initial_frame]
    object_pos = _as_numpy(motion["object_pos"])[initial_frame].copy()
    object_quat_xyzw = _as_numpy(motion["object_quat"])[initial_frame]
    if initial_box_center_height is not None:
        object_pos[2] = float(initial_box_center_height)
    if standing_init:
        if default_dof_pos is None:
            raise ValueError("standing_init requires default_dof_pos")
        # Keep reference XY/yaw so the object remains in the same root-heading
        # frame, but remove reference roll/pitch and use the default joint pose.
        x, y, z, w = root_quat_xyzw
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        root_quat_wxyz = np.asarray(
            [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float64
        )
        root_pos[2] = float(standing_root_height)
    else:
        direction = root_pos[:2] - object_pos[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-6:
            direction = np.asarray((-1.0, 0.0))
            norm = 1.0
        if not preserve_root_object_relative_pose:
            root_pos[:2] = object_pos[:2] + direction / norm * float(initial_distance)
        root_quat_wxyz = np.roll(root_quat_xyzw, 1)

    # Optional evaluation-only placement.  Bearing is expressed in the
    # robot's yaw frame: 0 deg puts the box directly in front, positive
    # angles put it to the robot's left.  Keeping this opt-in preserves the
    # historical initialization used by existing recordings.
    if initial_bearing_deg is not None:
        root_quat_xyzw_effective = np.roll(root_quat_wxyz, -1)
        x, y, z, w = root_quat_xyzw_effective
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        bearing = yaw + math.radians(float(initial_bearing_deg))
        robot_to_box = np.asarray((math.cos(bearing), math.sin(bearing)), dtype=np.float64)
        root_pos[:2] = object_pos[:2] - float(initial_distance) * robot_to_box

    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat_wxyz
    motion_joint_names = list(motion.get("joint_names", joint_names))
    motion_index = {name: index for index, name in enumerate(motion_joint_names)}
    dof_pos = (
        np.asarray(default_dof_pos, dtype=np.float64)
        if standing_init
        else _as_numpy(motion["dof_pos"])[initial_frame]
    )
    ordered_dof_pos = (
        np.asarray(dof_pos, dtype=np.float64)
        if standing_init
        else np.asarray([dof_pos[motion_index[name]] for name in joint_names], dtype=np.float64)
    )
    data.qpos[qpos_addresses] = ordered_dof_pos
    box_qpos_address = int(model.jnt_qposadr[model.joint(BOX_JOINT_NAME).id])
    data.qpos[box_qpos_address : box_qpos_address + 3] = object_pos
    data.qpos[box_qpos_address + 3 : box_qpos_address + 7] = np.roll(object_quat_xyzw, 1)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return ordered_dof_pos.copy()


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    robot_training = load_robot_training_spec(args.robot_config)
    joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(joint_names)
    box_cfg = _load_box_config(args.box_config)
    if args.box_friction is not None:
        box_cfg = dict(box_cfg)
        box_cfg["sim2sim_box_friction"] = tuple(float(value) for value in args.box_friction)
    if args.box_condim is not None:
        box_cfg = dict(box_cfg)
        box_cfg["sim2sim_box_condim"] = int(args.box_condim)
    model = _build_model(Path(robot_training.robot.xml_path), box_cfg, render_size=args.render_size)
    if args.hand_friction is not None:
        for geom_name in HAND_GEOM_NAMES:
            geom_id = model.geom(geom_name).id
            model.geom_friction[geom_id] = np.asarray(args.hand_friction, dtype=np.float64)
    _patch_mode15_joint_physics(model, joint_names)
    data = mujoco.MjData(model)
    qpos_addresses, qvel_addresses = _joint_addresses(model, joint_names)
    actuator_indices = _actuator_indices(model, joint_names)
    motion_key, motion = _load_motion(args.motion_cache, args.motion_id)
    defaults = np.asarray([robot_training.default_joint_angles[name] for name in joint_names], dtype=np.float64)
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
    )

    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported inference device: {device}")
    # Checkpoint configs accept the device class ("cpu"/"cuda"), while the
    # loaded policy can subsequently be moved to a specific CUDA index.
    checkpoint_device = device.type
    # Pure policy inference does not use target F/B or target critic networks.
    # Skipping them avoids duplicating several gigabytes of training-only state.
    policy = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint",
        device=checkpoint_device,
        strict=False,
    )
    policy.to(device).eval()
    full_config_path = args.model_folder / "config.json"
    full_config = json.loads(full_config_path.read_text()) if full_config_path.exists() else {}
    saved_train_cfg = full_config.get("agent", {}).get("train", {})
    intact_goal_coef = float(saved_train_cfg.get("intact_goal_coef", 0.0))
    goal_latent_mode = args.goal_latent_mode
    if goal_latent_mode == "auto":
        goal_latent_mode = "displacement" if intact_goal_coef > 0.0 else "absolute"
    policy_state_dim = int(policy.obs_space["state"].shape[0])
    compact_state_dim = 2 * num_dof + 28
    spatial_state_dim = 2 * num_dof + 40
    if policy_state_dim not in (compact_state_dim, spatial_state_dim):
        raise ValueError(
            f"Unsupported Carry policy state dimension {policy_state_dim}; "
            f"expected {compact_state_dim} or {spatial_state_dim} for {num_dof} DoFs"
        )
    latent_value = joblib.load(args.tracking_latent or args.z_goal)
    latent = torch.as_tensor(latent_value, device=device, dtype=torch.float32)
    if args.tracking_latent is None:
        latent = latent.reshape(1, -1)
    elif latent.ndim != 2 or latent.shape[0] < 1:
        raise ValueError(f"tracking latent must be [T, Z], got {tuple(latent.shape)}")
    elif args.tracking_start_s is not None or args.tracking_end_s is not None:
        start_step = 0 if args.tracking_start_s is None else round(float(args.tracking_start_s) / CONTROL_DT)
        end_step = latent.shape[0] if args.tracking_end_s is None else round(float(args.tracking_end_s) / CONTROL_DT)
        start_step = max(0, min(start_step, latent.shape[0]))
        end_step = max(0, min(end_step, latent.shape[0]))
        if end_step <= start_step:
            raise ValueError(
                "tracking latent slice is empty: "
                f"start={args.tracking_start_s}s end={args.tracking_end_s}s "
                f"latent_duration={latent.shape[0] * CONTROL_DT:.3f}s"
            )
        latent = latent[start_step:end_step]
    if latent.shape[-1] != int(policy.cfg.archi.z_dim):
        raise ValueError(f"latent has dim {latent.shape[-1]}, expected {policy.cfg.archi.z_dim}")
    if args.tracking_latent is not None and goal_latent_mode == "displacement":
        raise ValueError("displacement goal mode requires --z-goal, not --tracking-latent")
    print(
        f"[sim2sim] goal_latent_mode={goal_latent_mode} intact_goal_coef={intact_goal_coef}",
        flush=True,
    )
    height_reward_latent = None
    if args.height_reward_latent is not None:
        height_reward_latent = torch.as_tensor(
            joblib.load(args.height_reward_latent), device=device, dtype=torch.float32
        ).reshape(1, -1)
        if height_reward_latent.shape[-1] != int(policy.cfg.archi.z_dim):
            raise ValueError(
                f"height reward latent has dim {height_reward_latent.shape[-1]}, "
                f"expected {policy.cfg.archi.z_dim}"
            )

    kp = np.asarray([_match_joint_parameter(name, robot_training.stiffness) for name in joint_names])
    kd = np.asarray([_match_joint_parameter(name, robot_training.damping) for name in joint_names])
    effort = np.asarray(robot_training.effort_limits, dtype=np.float64)
    hard_limits = np.asarray([robot_training.robot.joint_ranges[name] for name in joint_names], dtype=np.float64)
    base_scale = np.asarray([float(robot_training.action_scale_overrides.get(name, robot_training.action_scale)) for name in joint_names])
    target_scale = base_scale * effort / kp
    current_action = np.clip((initial_dof_pos - defaults) / target_scale, -args.action_clip, args.action_clip)
    target_object_pos = _frame_or_static(motion["object_target_pos"], args.initial_frame, width=3)
    if args.target_object_height is not None:
        target_object_pos = _as_numpy(motion["object_pos"])[args.initial_frame].copy()
        target_object_pos[2] = float(args.target_object_height)
    observer = ActorObservation(
        device=device,
        default_dof_pos=defaults,
        target_object_pos=target_object_pos,
        hand_contact_offsets=np.asarray(box_cfg["hand_contact_offsets"], dtype=np.float32),
        include_hand_spatial=policy_state_dim == spatial_state_dim,
    )

    base_body_id = model.body(robot_training.robot.base_body).id
    box_body_id = model.body(BOX_BODY_NAME).id
    box_geom_id = model.geom(BOX_GEOM_NAME).id
    applied_box_friction = model.geom_friction[box_geom_id].copy()
    applied_box_condim = int(model.geom_condim[box_geom_id])
    floor_geom_id = model.geom(FLOOR_GEOM_NAME).id
    applied_floor_friction = model.geom_friction[floor_geom_id].copy()
    applied_hand_friction = {
        geom_name: model.geom_friction[model.geom(geom_name).id].copy() for geom_name in HAND_GEOM_NAMES
    }
    print(
        f"[sim2sim] box friction={applied_box_friction.tolist()} condim={applied_box_condim} "
        f"floor friction={applied_floor_friction.tolist()}",
        flush=True,
    )
    print(
        "[sim2sim] hand friction="
        + str({name: value.tolist() for name, value in applied_hand_friction.items()}),
        flush=True,
    )
    hand_body_ids = tuple(model.body(name).id for name in robot_training.robot.hands)
    foot_body_ids = tuple(model.body(name).id for name in robot_training.robot.feet)
    if len(foot_body_ids) != 2:
        raise ValueError(f"Expected exactly two foot bodies, got {robot_training.robot.feet}")
    box_qpos_address = int(model.jnt_qposadr[model.joint(BOX_JOINT_NAME).id])
    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(model, height=args.render_size, width=args.render_size) if args.save_mp4 else None
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(args.camera_distance)
    camera.azimuth = float(args.camera_azimuth)
    camera.elevation = float(args.camera_elevation)
    if viewer is not None:
        viewer.cam.distance = float(args.camera_distance)
        viewer.cam.azimuth = float(args.camera_azimuth)
        viewer.cam.elevation = float(args.camera_elevation)
    substeps = round(CONTROL_DT / SIM_DT)
    episode_steps = max(1, round(float(args.episode_seconds) / CONTROL_DT))
    history: list[dict[str, float | bool]] = []

    try:
        for step in range(episode_steps):
            if viewer is not None and not viewer.is_running():
                print("[sim2sim] viewer closed; stopping rollout", flush=True)
                break
            control_started_at = time.monotonic()
            dof_pos = data.qpos[qpos_addresses].copy()
            dof_vel = data.qvel[qvel_addresses].copy()
            base_lin_vel, base_ang_vel = _world_body_velocity(model, data, base_body_id)
            object_lin_vel, object_ang_vel = _world_body_velocity(model, data, box_body_id)
            if args.target_follows_object:
                observer.set_target_object_pos(data.qpos[box_qpos_address : box_qpos_address + 3])
            contact = _physical_hand_contacts(
                model,
                data,
                box_geom_id=box_geom_id,
                hand_body_ids=hand_body_ids,
            )
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
            if args.tracking_latent is None:
                if goal_latent_mode == "displacement":
                    current_backward_observation = _backward_observation_from_mujoco(
                        model=model,
                        data=data,
                        source_motion=motion,
                        actor_observation=observation,
                        include_hand_spatial=policy_state_dim == spatial_state_dim,
                        device=device,
                    )
                    current_b = policy.backward_map(current_backward_observation)
                    active_latent = policy.project_z(latent - current_b)
                else:
                    active_latent = latent
            elif args.loop_tracking:
                active_latent = latent[step % latent.shape[0]].reshape(1, -1)
            else:
                active_latent = latent[min(step, latent.shape[0] - 1)].reshape(1, -1)
            if height_reward_latent is not None:
                weight = float(args.height_reward_weight)
                active_latent = policy.project_z(
                    (1.0 - weight) * active_latent + weight * height_reward_latent
                )
            raw_action = policy.act(observation, active_latent, mean=True)[0].detach().cpu().numpy()
            current_action = np.clip(raw_action * float(robot_training.normalize_action_to), -args.action_clip, args.action_clip)
            q_target = np.clip(defaults + current_action * target_scale, hard_limits[:, 0], hard_limits[:, 1])
            for _ in range(substeps):
                torque = np.clip(kp * (q_target - data.qpos[qpos_addresses]) - kd * data.qvel[qvel_addresses], -effort, effort)
                data.ctrl[actuator_indices] = torque
                mujoco.mj_step(model, data)

            box_pos = data.qpos[box_qpos_address : box_qpos_address + 3]
            base_pos = data.xpos[base_body_id]
            gravity_local = quat_rotate_inverse(
                torch.as_tensor(np.roll(data.xquat[base_body_id], -1), dtype=torch.float32).reshape(1, 4),
                torch.tensor([[0.0, 0.0, -1.0]]),
                w_last=True,
            )[0]
            history.append(
                {
                    "root_height": float(base_pos[2]),
                    "upright": float(np.clip(-gravity_local[2].item(), 0.0, 1.0)),
                    "object_height": float(box_pos[2]),
                    "horizontal_distance": float(np.linalg.norm(box_pos[:2] - base_pos[:2])),
                    "left_contact": bool(contact[0] > 0.5),
                    "right_contact": bool(contact[1] > 0.5),
                    "lower_body_joint_speed_rms": float(np.sqrt(np.mean(np.square(dof_vel[:12])))),
                    "waist_joint_speed_rms": float(np.sqrt(np.mean(np.square(dof_vel[12:15])))),
                    "arm_joint_speed_rms": float(np.sqrt(np.mean(np.square(dof_vel[15:])))),
                    "root_horizontal_speed": float(np.linalg.norm(base_lin_vel[:2])),
                    "left_foot_height": float(data.xpos[foot_body_ids[0], 2]),
                    "right_foot_height": float(data.xpos[foot_body_ids[1], 2]),
                }
            )
            if renderer is not None:
                camera.lookat[:] = (float(base_pos[0]), float(base_pos[1]), max(float(base_pos[2]), 0.75))
                renderer.update_scene(data, camera=camera)
                frames.append(np.asarray(renderer.render()).copy())
            if viewer is not None:
                viewer.cam.lookat[:] = (float(base_pos[0]), float(base_pos[1]), max(float(base_pos[2]), 0.75))
                viewer.sync()
                remaining = CONTROL_DT - (time.monotonic() - control_started_at)
                if remaining > 0.0:
                    time.sleep(remaining)
            if step == 0 or step + 1 == episode_steps or (step + 1) % int(args.log_every) == 0:
                print(
                    f"[sim2sim] step={step + 1}/{episode_steps} root_h={base_pos[2]:.3f} "
                    f"box_h={box_pos[2]:.3f} distance={history[-1]['horizontal_distance']:.3f} "
                    f"contact={int(contact[0])}/{int(contact[1])}",
                    flush=True,
                )
    finally:
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / f"goal_sim2sim_distance_{args.initial_distance:.1f}m.mp4"
    if frames:
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        media.write_video(str(video_path), frames, fps=round(1.0 / CONTROL_DT))
    final_window = history[-max(1, round(float(args.final_window_seconds) / CONTROL_DT)) :]
    left_foot_heights = np.asarray([row["left_foot_height"] for row in history])
    right_foot_heights = np.asarray([row["right_foot_height"] for row in history])
    left_foot_clearance = left_foot_heights - float(left_foot_heights.min())
    right_foot_clearance = right_foot_heights - float(right_foot_heights.min())
    single_foot_kick = (
        ((left_foot_clearance > 0.15) & (right_foot_clearance < 0.05))
        | ((right_foot_clearance > 0.15) & (left_foot_clearance < 0.05))
    )
    summary = {
        "model_folder": str(args.model_folder),
        "z_goal": None if args.z_goal is None else str(args.z_goal),
        "goal_latent_mode": goal_latent_mode,
        "intact_goal_coef": intact_goal_coef,
        "tracking_latent": None if args.tracking_latent is None else str(args.tracking_latent),
        "tracking_start_s": args.tracking_start_s,
        "tracking_end_s": args.tracking_end_s,
        "height_reward_latent": (
            None if args.height_reward_latent is None else str(args.height_reward_latent)
        ),
        "height_reward_weight": float(args.height_reward_weight),
        "height_threshold": float(args.height_threshold),
        "box_above_height_rate": float(
            np.mean([row["object_height"] > float(args.height_threshold) for row in history])
        ),
        "episode_motion_mean": {
            key: float(np.mean([row[key] for row in history]))
            for key in (
                "lower_body_joint_speed_rms",
                "waist_joint_speed_rms",
                "arm_joint_speed_rms",
                "root_horizontal_speed",
            )
        },
        "kick_motion": {
            "left_foot_max_clearance": float(left_foot_clearance.max()),
            "left_foot_max_clearance_time_s": float(left_foot_clearance.argmax() * CONTROL_DT),
            "right_foot_max_clearance": float(right_foot_clearance.max()),
            "right_foot_max_clearance_time_s": float(right_foot_clearance.argmax() * CONTROL_DT),
            "single_foot_kick_rate": float(single_foot_kick.mean()),
            "clearance_threshold": 0.15,
            "support_foot_threshold": 0.05,
        },
        "target_follows_object": bool(args.target_follows_object),
        "standing_init": bool(args.standing_init),
        "motion_cache": str(args.motion_cache),
        "motion_id": int(args.motion_id),
        "motion_key": motion_key,
        "initial_frame": int(args.initial_frame),
        "initial_distance": float(args.initial_distance),
        "initial_bearing_deg": args.initial_bearing_deg,
        "initial_box_center_height": args.initial_box_center_height,
        "target_object_pos": target_object_pos.tolist(),
        "episode_seconds": float(args.episode_seconds),
        "physics": "pure_mujoco",
        "sim_dt": SIM_DT,
        "control_dt": CONTROL_DT,
        "box_friction": applied_box_friction.tolist(),
        "box_condim": applied_box_condim,
        "floor_friction": applied_floor_friction.tolist(),
        "hand_friction": {name: value.tolist() for name, value in applied_hand_friction.items()},
        "video": str(video_path) if frames else None,
        "final": history[-1],
        "final_window_mean": {
            "root_height": float(np.mean([row["root_height"] for row in final_window])),
            "upright": float(np.mean([row["upright"] for row in final_window])),
            "object_height": float(np.mean([row["object_height"] for row in final_window])),
            "horizontal_distance": float(np.mean([row["horizontal_distance"] for row in final_window])),
            "left_contact_rate": float(np.mean([row["left_contact"] for row in final_window])),
            "right_contact_rate": float(np.mean([row["right_contact"] for row in final_window])),
            "bilateral_contact_rate": float(np.mean([row["left_contact"] and row["right_contact"] for row in final_window])),
        },
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"[sim2sim] saved summary {args.output_dir / 'summary.json'}")
    if frames:
        print(f"[sim2sim] saved video {video_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument(
        "--goal-latent-mode",
        choices=("auto", "absolute", "displacement"),
        default="auto",
        help=(
            "Goal command semantics. auto uses displacement for checkpoints trained with "
            "intact_goal_coef > 0 and absolute B(goal) otherwise."
        ),
    )
    latent_group = parser.add_mutually_exclusive_group(required=True)
    latent_group.add_argument("--z-goal", type=Path)
    latent_group.add_argument(
        "--tracking-latent",
        type=Path,
        help="Time-indexed [T, 256] latent sequence; the final row is held unless --loop-tracking is set.",
    )
    parser.add_argument(
        "--height-reward-latent",
        type=Path,
        default=None,
        help="Optional fixed reward latent blended with the goal/tracking latent at every control step.",
    )
    parser.add_argument(
        "--height-reward-weight",
        type=float,
        default=0.5,
        help="Convex blend weight assigned to --height-reward-latent before projecting z.",
    )
    parser.add_argument(
        "--height-threshold",
        type=float,
        default=0.5,
        help="World-frame box-center height used only for rollout reporting.",
    )
    parser.add_argument("--motion-cache", type=Path, required=True)
    parser.add_argument("--motion-id", type=int, default=1199)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--initial-distance", type=float, default=1.0)
    parser.add_argument(
        "--initial-box-center-height",
        type=float,
        default=None,
        help="Optional world-Z box center used only for the initial physical state.",
    )
    parser.add_argument(
        "--initial-bearing-deg",
        type=float,
        default=None,
        help=(
            "Optional box bearing in the initial robot yaw frame: 0 is directly ahead and "
            "positive is left. When omitted, preserve the reference-motion placement."
        ),
    )
    parser.add_argument(
        "--standing-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start from upright yaw-only root and default joint angles instead of the reference first pose.",
    )
    parser.add_argument("--standing-root-height", type=float, default=0.80)
    parser.add_argument("--loop-tracking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tracking-start-s", type=float, default=None)
    parser.add_argument("--tracking-end-s", type=float, default=None)
    parser.add_argument("--target-object-height", type=float)
    parser.add_argument(
        "--target-follows-object",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set target position equal to the current box position so target-delta observation stays zero.",
    )
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument(
        "--box-friction",
        type=float,
        nargs=3,
        metavar=("SLIDING", "TORSIONAL", "ROLLING"),
        default=None,
        help="Override only the box geom's MuJoCo sliding, torsional and rolling friction.",
    )
    parser.add_argument(
        "--box-condim",
        type=int,
        choices=(1, 3, 4, 6),
        default=None,
        help="Override only the box geom contact dimension; use 4 to enable torsional friction.",
    )
    parser.add_argument(
        "--hand-friction",
        type=float,
        nargs=3,
        metavar=("SLIDING", "TORSIONAL", "ROLLING"),
        default=None,
        help="Override only the left/right hand collision geom friction during sim2sim.",
    )
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--final-window-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=4.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--action-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-mp4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.initial_distance <= 0.0:
        parser.error("--initial-distance must be positive")
    if args.initial_box_center_height is not None and args.initial_box_center_height <= 0.0:
        parser.error("--initial-box-center-height must be positive")
    if args.episode_seconds <= 0.0:
        parser.error("--episode-seconds must be positive")
    if args.standing_root_height <= 0.0:
        parser.error("--standing-root-height must be positive")
    if args.action_clip <= 0.0:
        parser.error("--action-clip must be positive")
    if args.height_reward_latent is None and args.height_reward_weight != 0.5:
        parser.error("--height-reward-weight requires --height-reward-latent")
    if not 0.0 <= args.height_reward_weight <= 1.0:
        parser.error("--height-reward-weight must be in [0, 1]")
    if args.height_threshold <= 0.0:
        parser.error("--height-threshold must be positive")
    if args.tracking_start_s is not None and args.tracking_start_s < 0.0:
        parser.error("--tracking-start-s must be non-negative")
    if args.tracking_end_s is not None and args.tracking_end_s <= 0.0:
        parser.error("--tracking-end-s must be positive")
    if args.box_friction is not None and (
        args.box_friction[0] <= 0.0 or args.box_friction[1] < 0.0 or args.box_friction[2] < 0.0
    ):
        parser.error("--box-friction requires positive sliding and non-negative torsional/rolling values")
    if args.hand_friction is not None and (
        args.hand_friction[0] <= 0.0 or args.hand_friction[1] < 0.0 or args.hand_friction[2] < 0.0
    ):
        parser.error("--hand-friction requires positive sliding and non-negative torsional/rolling values")
    if args.viewer and args.save_mp4:
        parser.error("--viewer requires --no-save-mp4")
    return args


if __name__ == "__main__":
    run(parse_args())
