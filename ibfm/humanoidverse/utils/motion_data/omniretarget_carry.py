"""Import OmniRetarget 43D robot-object qpos archives for G1 carry training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from humanoidverse.utils.motion_data.carry_box import _angular_velocity
from humanoidverse.utils.motion_data.paths import expand_motion_paths
from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.motion_data.robot_state_convert import robot_state_to_ufo_motion
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import RobotSpec


OMNIRETARGET_G1_DOF = 29
OMNIRETARGET_CARRY_QPOS_DIM = 43
DEFAULT_CONTACT_METHOD = "mujoco_capsule_box_4p5cm_6cm_hysteresis"


def omniretarget_qpos_components(qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Decode OmniRetarget's quat-first G1/object qpos into UFO conventions."""

    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != OMNIRETARGET_CARRY_QPOS_DIM:
        raise ValueError(
            f"OmniRetarget carry qpos must have shape [T, {OMNIRETARGET_CARRY_QPOS_DIM}], got {qpos.shape}"
        )
    if qpos.shape[0] < 2 or not np.all(np.isfinite(qpos)):
        raise ValueError("OmniRetarget carry qpos must contain at least two finite frames")

    # OmniRetarget stores [qw, qx, qy, qz, x, y, z] for both free joints.
    return {
        "root_quat": qpos[:, [1, 2, 3, 0]].astype(np.float32),
        "root_pos": qpos[:, 4:7].astype(np.float32),
        "dof_pos": qpos[:, 7 : 7 + OMNIRETARGET_G1_DOF].astype(np.float32),
        "object_quat": qpos[:, 36:40][:, [1, 2, 3, 0]].astype(np.float32),
        "object_pos": qpos[:, 40:43].astype(np.float32),
    }


def omniretarget_qpos_to_mujoco(qpos: np.ndarray) -> np.ndarray:
    """Reorder quat-first OmniRetarget qpos into MuJoCo free-joint order."""

    parts = omniretarget_qpos_components(qpos)
    return np.concatenate(
        (
            parts["root_pos"],
            np.asarray(qpos)[:, :4],
            parts["dof_pos"],
            parts["object_pos"],
            np.asarray(qpos)[:, 36:40],
        ),
        axis=1,
    ).astype(np.float64)


def load_omniretarget_carry_npz(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
) -> dict[str, Any]:
    """Load labeled OmniRetarget carry NPZ files into UFO motion records."""

    converted: dict[str, Any] = {}
    paths = expand_motion_paths(path_spec, base_dir=base_dir, suffix=".npz")
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if "qpos" not in archive or "fps" not in archive:
                raise ValueError(f"OmniRetarget archive must contain qpos and fps: {path}")
            parts = omniretarget_qpos_components(archive["qpos"])
            motion_fps = float(fps) if fps is not None else float(np.asarray(archive["fps"]).reshape(-1)[0])
            if not np.isfinite(motion_fps) or motion_fps <= 0.0:
                raise ValueError(f"Invalid fps={motion_fps} in {path}")
            contact = np.asarray(archive["hand_box_contact"], dtype=np.float32) if "hand_box_contact" in archive else None
            surface_distance = (
                np.asarray(archive["hand_box_surface_distance"], dtype=np.float32)
                if "hand_box_surface_distance" in archive
                else None
            )
            contact_method = (
                str(np.asarray(archive["hand_box_contact_method"]).reshape(-1)[0])
                if "hand_box_contact_method" in archive
                else DEFAULT_CONTACT_METHOD
            )

        num_frames = parts["root_pos"].shape[0]
        if contact is None:
            raise ValueError(f"Labeled OmniRetarget archive is missing hand_box_contact: {path}")
        if contact.shape != (num_frames, 2) or not np.all(np.isin(contact, (0.0, 1.0))):
            raise ValueError(f"hand_box_contact must be binary [T, 2] in {path}, got {contact.shape}")
        if surface_distance is not None and surface_distance.shape != (num_frames, 2):
            raise ValueError(
                f"hand_box_surface_distance must have shape [T, 2] in {path}, got {surface_distance.shape}"
            )

        key = path.stem
        if key in converted:
            raise ValueError(f"Duplicate OmniRetarget motion key: {key}")
        robot_motion = RobotStateMotion(
            motion_key=key,
            root_pos=parts["root_pos"],
            root_quat=parts["root_quat"],
            dof_pos=parts["dof_pos"],
            fps=motion_fps,
            joint_names=list(robot_spec.control_joint_names),
            source=source_name,
            metadata={"source_npz": str(path), "contact_method": contact_method},
        )
        record = robot_state_to_ufo_motion(robot_motion, robot_spec, source_name)
        times = np.arange(num_frames, dtype=np.float64) / motion_fps
        record.update(
            {
                "object_pos": parts["object_pos"],
                "object_quat": parts["object_quat"],
                "object_lin_vel": np.gradient(parts["object_pos"], times, axis=0).astype(np.float32),
                "object_ang_vel": _angular_velocity(parts["object_quat"], times),
                "object_target_pos": parts["object_pos"][-1].copy(),
                "object_target_quat": parts["object_quat"][-1].copy(),
                "hand_box_contact": contact,
            }
        )
        if surface_distance is not None:
            record["hand_box_surface_distance"] = surface_distance
        converted[key] = record

    return validate_ufo_motion_dict(converted, source_name)
