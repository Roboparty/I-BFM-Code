"""Paired HHTools G1 and object-track import for carry tasks."""

from __future__ import annotations

import csv
import glob
import os
from pathlib import Path
from typing import Any

import numpy as np

from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.motion_data.robot_state_convert import robot_state_to_ufo_motion
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import RobotSpec


def _path_items(path_spec: str | os.PathLike[str] | list[str], base_dir: Path | None) -> list[Path]:
    raw_items = path_spec if isinstance(path_spec, list) else [path_spec]
    resolved: list[Path] = []
    for raw in raw_items:
        expanded = Path(os.path.expandvars(os.path.expanduser(str(raw))))
        candidates = [expanded] if expanded.is_absolute() else [base_dir / expanded if base_dir else expanded, Path.cwd() / expanded]
        matches: list[Path] = []
        for candidate in candidates:
            if glob.has_magic(str(candidate)):
                matches = [Path(item) for item in sorted(glob.glob(str(candidate)))]
            elif candidate.exists():
                matches = [candidate]
            if matches:
                break
        resolved.extend(matches)
    sample_dirs: list[Path] = []
    for item in resolved:
        if item.is_dir() and (item / "object_0_largebox.csv").exists():
            sample_dirs.append(item)
        elif item.is_dir():
            sample_dirs.extend(sorted(path for path in item.iterdir() if path.is_dir() and (path / "object_0_largebox.csv").exists()))
    if not sample_dirs:
        raise FileNotFoundError(f"No paired HHTools carry samples found: {path_spec}")
    return sample_dirs


def _read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="") as file:
        lines = [line for line in file if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"CSV has no header: {path}")
    rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"CSV needs at least two samples: {path}")
    return [str(name) for name in reader.fieldnames], rows


def _matrix(rows: list[dict[str, str]], columns: list[str], path: Path) -> np.ndarray:
    try:
        value = np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"CSV file={path} missing column {exc.args[0]!r}") from exc
    if not np.all(np.isfinite(value)):
        raise ValueError(f"CSV file={path} columns={columns} contain non-finite values")
    return value


def _time(rows: list[dict[str, str]], path: Path) -> np.ndarray:
    times = _matrix(rows, ["time"], path)[:, 0].astype(np.float64)
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError(f"CSV file={path} time must be strictly increasing")
    return times


def _normalize_quat(quat: np.ndarray, path: Path) -> np.ndarray:
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-8):
        raise ValueError(f"CSV file={path} contains zero quaternions")
    quat = quat / norms
    # q and -q are equivalent; continuity is required for angular differencing.
    for index in range(1, len(quat)):
        if float(np.dot(quat[index - 1], quat[index])) < 0.0:
            quat[index] *= -1.0
    return quat.astype(np.float32)


def _quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., :3] *= -1.0
    return result


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _angular_velocity(quat: np.ndarray, times: np.ndarray) -> np.ndarray:
    delta = _quat_multiply_xyzw(quat[1:], _quat_conjugate_xyzw(quat[:-1]))
    delta = _normalize_quat(delta.astype(np.float32), Path("<object quaternion track>"))
    vector_norm = np.linalg.norm(delta[:, :3], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[:, 3], -1.0, 1.0))
    axis = np.zeros_like(delta[:, :3])
    valid = vector_norm > 1.0e-8
    axis[valid] = delta[valid, :3] / vector_norm[valid, None]
    interval_velocity = axis * (angle / np.diff(times))[:, None]
    velocity = np.zeros_like(quat[:, :3])
    velocity[:-1] = interval_velocity
    velocity[-1] = interval_velocity[-1]
    return velocity.astype(np.float32)


def _robot_csv(sample_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in sample_dir.glob("*.csv")
        if not path.name.startswith("object_") and path.name != "hand_box_contact.csv"
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected one robot CSV in {sample_dir}, found {[path.name for path in candidates]}")
    return candidates[0]


def load_hhtools_carry_csv(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
) -> dict[str, Any]:
    """Load paired robot/object CSV directories into UFO motion records."""

    converted: dict[str, Any] = {}
    for sample_dir in _path_items(path_spec, base_dir):
        robot_path = _robot_csv(sample_dir)
        object_path = sample_dir / "object_0_largebox.csv"
        robot_fields, robot_rows = _read_table(robot_path)
        object_fields, object_rows = _read_table(object_path)
        del robot_fields, object_fields

        robot_time = _time(robot_rows, robot_path)
        object_time = _time(object_rows, object_path)
        if robot_time.shape != object_time.shape or not np.allclose(robot_time, object_time, atol=1.0e-6, rtol=0.0):
            raise ValueError(
                f"Robot/object timestamps do not match for {sample_dir.name}: "
                f"robot={robot_time.shape}, object={object_time.shape}"
            )
        inferred_fps = 1.0 / float(np.median(np.diff(robot_time)))
        motion_fps = float(fps) if fps is not None else inferred_fps

        joint_columns = [f"dof_{joint}" for joint in robot_spec.control_joint_names]
        motion = RobotStateMotion(
            motion_key=sample_dir.name,
            root_pos=_matrix(robot_rows, ["root_x", "root_y", "root_z"], robot_path),
            root_quat=_normalize_quat(_matrix(robot_rows, ["root_qx", "root_qy", "root_qz", "root_qw"], robot_path), robot_path),
            dof_pos=_matrix(robot_rows, joint_columns, robot_path),
            fps=motion_fps,
            joint_names=list(robot_spec.control_joint_names),
            source=source_name,
            metadata={"robot_csv": str(robot_path), "object_csv": str(object_path)},
        )
        record = robot_state_to_ufo_motion(motion, robot_spec, source_name)
        object_pos = _matrix(object_rows, ["pos_x", "pos_y", "pos_z"], object_path)
        object_quat = _normalize_quat(_matrix(object_rows, ["quat_x", "quat_y", "quat_z", "quat_w"], object_path), object_path)
        record.update(
            {
                "object_pos": object_pos,
                "object_quat": object_quat,
                "object_lin_vel": np.gradient(object_pos, robot_time, axis=0).astype(np.float32),
                "object_ang_vel": _angular_velocity(object_quat, robot_time),
                "object_target_pos": object_pos[-1].copy(),
                "object_target_quat": object_quat[-1].copy(),
                "object_mesh_path": str((sample_dir / "largebox_cleaned_simplified.obj").resolve()),
            }
        )
        contact_path = sample_dir / "hand_box_contact.csv"
        if contact_path.exists():
            _contact_fields, contact_rows = _read_table(contact_path)
            contact_time = _time(contact_rows, contact_path)
            if contact_time.shape != robot_time.shape or not np.allclose(
                contact_time, robot_time, atol=1.0e-6, rtol=0.0
            ):
                raise ValueError(
                    f"Robot/contact timestamps do not match for {sample_dir.name}: "
                    f"robot={robot_time.shape}, contact={contact_time.shape}"
                )
            contact = _matrix(
                contact_rows,
                ["left_hand_box_contact", "right_hand_box_contact"],
                contact_path,
            )
            if not np.all(np.isin(contact, (0.0, 1.0))):
                raise ValueError(f"Contact labels must be binary in {contact_path}")
            record["hand_box_contact"] = contact.astype(np.float32)
            distance_columns = [
                "left_hand_box_surface_distance",
                "right_hand_box_surface_distance",
            ]
            if all(column in _contact_fields for column in distance_columns):
                record["hand_box_surface_distance"] = _matrix(
                    contact_rows, distance_columns, contact_path
                ).astype(np.float32)
            record["metadata"]["contact_csv"] = str(contact_path)
            record["metadata"]["contact_method"] = "rp1_phase_lift_with_mujoco_wrist_gap_v1"
        if sample_dir.name in converted:
            raise ValueError(f"Duplicate carry motion key: {sample_dir.name}")
        converted[sample_dir.name] = record
    return validate_ufo_motion_dict(converted, source_name)
