#!/usr/bin/env python3
"""Prepare a standing robot, front box, and two support platforms."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import yaml

from humanoidverse.utils.robot_spec import load_robot_training_spec
from scene_builder import platform_top_for_box_center, write_motion, write_platform_robot_xml


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
REPO = BUNDLE.parents[1]
def main() -> None:
    output = BUNDLE / "random_height_scene"
    output.mkdir(parents=True, exist_ok=True)
    box_size = 0.30
    initial_z = 0.65
    target_z = float(os.environ.get("TARGET_HEIGHT", "0.65"))
    if not 0.15 <= target_z <= 0.80:
        raise ValueError("TARGET_HEIGHT must be in [0.15, 0.80] m")
    distance_min = float(os.environ.get("ROBOT_BOX_DISTANCE_MIN", "1.5"))
    distance_max = float(os.environ.get("ROBOT_BOX_DISTANCE_MAX", "2.0"))
    if not 0.0 < distance_min <= distance_max:
        raise ValueError("ROBOT_BOX_DISTANCE_MIN/MAX must satisfy 0 < min <= max")
    scene_seed = int(os.environ.get("SCENE_SEED", "166559744"))
    rng = np.random.default_rng(scene_seed)
    robot_distance = float(rng.uniform(distance_min, distance_max))
    robot_bearing = float(rng.uniform(-math.pi, math.pi))
    robot_xy = np.asarray([1.0, 0.0]) + robot_distance * np.asarray(
        [math.cos(robot_bearing), math.sin(robot_bearing)]
    )
    sample = {
        "episode_id": 0,
        "box_x": 1.0,
        "box_y": 0.0,
        "box_z": initial_z,
        "target_x": 2.5,
        "target_y": 0.0,
        "target_z": target_z,
        "robot_x": float(robot_xy[0]),
        "robot_y": float(robot_xy[1]),
        "robot_heading_rad": float(rng.uniform(-math.pi, math.pi)),
        "box_size_m": box_size,
        "box_mass_kg": 0.70,
        "initial_robot_box_distance": robot_distance,
        "initial_box_target_error_3d": float(
            (1.5**2 + (initial_z - target_z) ** 2) ** 0.5
        ),
    }
    (output / "scene.json").write_text(
        json.dumps(sample, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    motion_source = (
        REPO
        / "cache/motion_data/g1_carry_box_omniretarget_no_foot/"
        "g1_carry_box_omniretarget_no_foot_full_grounded_mujoco_hysteresis_4p5_6cm_ufo.pkl"
    )
    motions = joblib.load(motion_source)
    selected_key = list(motions)[67]
    template = {selected_key: motions[selected_key]}
    robot_config_path = REPO / "configs/robots/g1_29dof.yaml"
    robot_training = load_robot_training_spec(robot_config_path)
    default_angles = dict(robot_training.default_joint_angles)
    motion_path = output / "standing_front_box.pkl"
    write_motion(template, sample, default_angles, motion_path)

    box_config_path = REPO / "configs/tasks/carry_box.yaml"
    box_config = yaml.safe_load(box_config_path.read_text())
    geom_wxyz = np.asarray(box_config["geom_quat_wxyz"], dtype=np.float64)
    quaternion = np.asarray(
        [-geom_wxyz[1], -geom_wxyz[2], -geom_wxyz[3], geom_wxyz[0]],
        dtype=np.float64,
    )
    prepared_motion = joblib.load(motion_path)
    next(iter(prepared_motion.values()))["object_quat"][:] = quaternion
    joblib.dump(prepared_motion, motion_path)
    box_size_xyz = np.full(3, box_size, dtype=np.float64)
    initial_top = platform_top_for_box_center(
        initial_z, box_size_xyz, quaternion, box_config
    )
    target_top = platform_top_for_box_center(
        target_z, box_size_xyz, quaternion, box_config
    )
    robot_xml = output / "g1_random_height_supports.xml"
    write_platform_robot_xml(
        Path(robot_training.robot.xml_path),
        robot_xml,
        initial_xy=(1.0, 0.0),
        initial_top=initial_top,
        target_xy=(2.5, 0.0),
        target_top=target_top,
    )
    tree = ET.parse(robot_xml)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No worldbody in {robot_xml}")
    initial_support = worldbody.find("geom[@name='initial_box_support']")
    if initial_support is None:
        raise RuntimeError("scene builder did not create initial_box_support")
    support_position = initial_support.attrib.pop("pos")
    worldbody.remove(initial_support)
    movable_support = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "initial_box_support_body",
            "mocap": "true",
            "pos": support_position,
        },
    )
    initial_support.set("pos", "0 0 0")
    movable_support.append(initial_support)
    tree.write(robot_xml, encoding="utf-8", xml_declaration=True)
    robot_config = yaml.safe_load(robot_config_path.read_text())
    robot_config["xml_path"] = str(robot_xml)
    (output / "robot_config.yaml").write_text(
        yaml.safe_dump(robot_config, sort_keys=False)
    )
    box_config = copy.deepcopy(box_config)
    box_config["ground_center_height"] = target_z
    (output / "box_config.yaml").write_text(
        yaml.safe_dump(box_config, sort_keys=False)
    )
    print(output)


if __name__ == "__main__":
    main()
