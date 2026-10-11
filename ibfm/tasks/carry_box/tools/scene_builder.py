#!/usr/bin/env python3
"""Scene geometry helpers used by the Carry Box interactive launcher."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import joblib
import numpy as np

PLATFORM_SIZE_XY = 0.4
PLATFORM_THICKNESS = 0.02


def quaternion_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(float(x * x + y * y + z * z + w * w))
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def platform_top_for_box_center(
    center_z: float,
    full_size_xyz: np.ndarray,
    object_quaternion_xyzw: np.ndarray,
    box_config: dict[str, Any],
) -> float:
    body_rotation = quaternion_matrix_xyzw(object_quaternion_xyzw)
    geom_quaternion_xyzw = np.roll(
        np.asarray(box_config.get("geom_quat_wxyz", [1.0, 0.0, 0.0, 0.0])), -1
    )
    geom_rotation = quaternion_matrix_xyzw(geom_quaternion_xyzw)
    geom_position = np.asarray(box_config.get("geom_pos", [0.0, 0.0, 0.0]))
    geom_center_offset_z = float((body_rotation @ geom_position)[2])
    world_rotation = body_rotation @ geom_rotation
    vertical_extent = float(np.dot(np.abs(world_rotation[2]), full_size_xyz / 2.0))
    return float(center_z + geom_center_offset_z - vertical_extent)


def write_motion(
    template: dict[str, Any],
    sample: dict[str, Any],
    default_joint_angles: dict[str, float],
    path: Path,
) -> np.ndarray:
    payload = copy.deepcopy(template)
    if len(payload) != 1:
        raise ValueError(f"Expected one selected reference motion, found {len(payload)}")
    motion = next(iter(payload.values()))
    heading = float(sample["robot_heading_rad"])
    motion["root_trans_offset"][:] = np.asarray(
        [sample["robot_x"], sample["robot_y"], 0.8], dtype=np.float64
    )
    motion["root_quat"][:] = np.asarray(
        [0.0, 0.0, math.sin(heading / 2.0), math.cos(heading / 2.0)], dtype=np.float64
    )
    joint_names = list(motion["joint_names"])
    standing = np.asarray([default_joint_angles[name] for name in joint_names], dtype=np.float64)
    motion["dof_pos"][:] = standing
    motion["object_pos"][:] = np.asarray(
        [sample["box_x"], sample["box_y"], sample["box_z"]], dtype=np.float64
    )
    motion["object_lin_vel"][:] = 0.0
    motion["object_ang_vel"][:] = 0.0
    metadata = dict(motion.get("metadata", {}))
    metadata["random_platform_benchmark_sample"] = sample
    motion["metadata"] = metadata
    joblib.dump(payload, path)
    return np.asarray(motion["object_quat"])[0].copy()


def write_platform_robot_xml(
    source_xml: Path,
    path: Path,
    *,
    initial_xy: tuple[float, float],
    initial_top: float,
    target_xy: tuple[float, float],
    target_top: float,
) -> None:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"No compiler element in {source_xml}")
    mesh_dir = source_xml.parent / compiler.attrib.get("meshdir", "")
    compiler.set("meshdir", str(mesh_dir.resolve()))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No worldbody element in {source_xml}")

    half_height = PLATFORM_THICKNESS / 2.0
    specifications = (
        ("initial_box_support", initial_xy, initial_top, "0.10 0.62 0.82 1"),
        ("target_box_support", target_xy, target_top, "0.16 0.82 0.34 1"),
    )
    for name, xy, top, rgba in specifications:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": f"{xy[0]:.12g} {xy[1]:.12g} {top - half_height:.12g}",
                "size": f"{PLATFORM_SIZE_XY / 2:.12g} {PLATFORM_SIZE_XY / 2:.12g} {half_height:.12g}",
                "rgba": rgba,
                "friction": "0.9 0.01 0.001",
                "contype": "1",
                "conaffinity": "1",
            },
        )
    tree.write(path, encoding="utf-8", xml_declaration=True)
