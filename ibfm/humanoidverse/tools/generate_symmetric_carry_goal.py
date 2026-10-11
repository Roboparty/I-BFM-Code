"""Generate a kinematic, symmetric G1 box-hold goal motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from humanoidverse.carry_goal_sim2sim import _build_model, _joint_addresses, _load_box_config
from humanoidverse.utils.robot_spec import load_robot_training_spec


ARM_PARTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
LEFT_ARM = np.asarray(
    (-0.46393482, 0.34941934, -0.10956539, 0.34412098, -0.04069618, 0.05796013, 0.01534813),
    dtype=np.float64,
)
# Mirror signs follow the left/right joint axes in the G1 MJCF.
RIGHT_MIRROR_SIGNS = np.asarray((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0), dtype=np.float64)


def _pose_axis_angle(robot_training, dof_pos: np.ndarray, root_quat_xyzw: np.ndarray) -> np.ndarray:
    body_names = list(robot_training.robot.body_names)
    body_index = {name: index for index, name in enumerate(body_names)}
    pose = np.zeros((len(body_names), 3), dtype=np.float32)
    pose[body_index[robot_training.robot.base_body]] = Rotation.from_quat(root_quat_xyzw).as_rotvec()
    for index, joint_name in enumerate(robot_training.robot.control_joint_names):
        body_name = robot_training.robot.joint_body_names[joint_name]
        axis = np.asarray(robot_training.robot.joint_axes[joint_name], dtype=np.float32)
        pose[body_index[body_name]] = axis * np.float32(dof_pos[index])
    return pose


def run(args: argparse.Namespace) -> None:
    robot_training = load_robot_training_spec(args.robot_config)
    joint_names = list(robot_training.robot.control_joint_names)
    joint_index = {name: index for index, name in enumerate(joint_names)}
    dof_pos = np.asarray([robot_training.default_joint_angles[name] for name in joint_names], dtype=np.float64)
    for part, left_value, mirror_sign in zip(ARM_PARTS, LEFT_ARM, RIGHT_MIRROR_SIGNS):
        dof_pos[joint_index[f"left_{part}_joint"]] = left_value
        dof_pos[joint_index[f"right_{part}_joint"]] = mirror_sign * left_value

    box_cfg = _load_box_config(args.box_config)
    model = _build_model(Path(robot_training.robot.xml_path), box_cfg, render_size=480)
    data = mujoco.MjData(model)
    qpos_addresses, _ = _joint_addresses(model, joint_names)
    root_pos = np.asarray((0.0, 0.0, float(args.root_height)), dtype=np.float64)
    root_quat_xyzw = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    data.qpos[:3] = root_pos
    data.qpos[3:7] = np.roll(root_quat_xyzw, 1)
    data.qpos[qpos_addresses] = dof_pos
    mujoco.mj_forward(model, data)

    hand_offsets = np.asarray(box_cfg["hand_contact_offsets"], dtype=np.float64)
    hand_points = []
    for hand_name, offset in zip(robot_training.robot.hands, hand_offsets):
        body_id = model.body(hand_name).id
        hand_points.append(data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset)
    hand_points = np.asarray(hand_points)

    # Cancel the marker-to-geom rotation so the visible box axes equal the world axes.
    geom_rotation = Rotation.from_quat(np.roll(np.asarray(box_cfg["geom_quat_wxyz"], dtype=np.float64), -1))
    object_rotation = geom_rotation.inv()
    visible_box_center = hand_points.mean(axis=0)
    visible_box_center[1] = 0.0
    object_pos = visible_box_center - object_rotation.apply(np.asarray(box_cfg["geom_pos"], dtype=np.float64))
    object_quat_xyzw = object_rotation.as_quat()

    half_y = float(box_cfg["half_size"][1])
    desired_hand_points = np.asarray(
        (
            (visible_box_center[0], visible_box_center[1] + half_y, visible_box_center[2]),
            (visible_box_center[0], visible_box_center[1] - half_y, visible_box_center[2]),
        )
    )
    contact_error = np.linalg.norm(hand_points - desired_hand_points, axis=-1)
    if float(contact_error.max()) > 1.0e-3:
        raise RuntimeError(f"Symmetric hand-to-box contact error is too large: {contact_error.tolist()}")

    frame_count = max(1, round(float(args.duration) * float(args.fps)))
    repeat = lambda value: np.repeat(np.asarray(value, dtype=np.float32)[None], frame_count, axis=0)
    pose_aa = _pose_axis_angle(robot_training, dof_pos, root_quat_xyzw)
    motion_key = "g1_symmetric_static_box_hold"
    motion = {
        "root_trans_offset": repeat(root_pos),
        "pose_aa": repeat(pose_aa),
        "fps": float(args.fps),
        "dof_pos": repeat(dof_pos),
        "root_quat": repeat(root_quat_xyzw),
        "joint_names": joint_names,
        "body_names": list(robot_training.robot.body_names),
        "robot_name": robot_training.robot.name,
        "motion_key": motion_key,
        "source": "synthetic_symmetric_static_carry_goal",
        "object_pos": repeat(object_pos),
        "object_quat": repeat(object_quat_xyzw),
        "object_lin_vel": np.zeros((frame_count, 3), dtype=np.float32),
        "object_ang_vel": np.zeros((frame_count, 3), dtype=np.float32),
        "object_target_pos": object_pos.astype(np.float32),
        "object_target_quat": object_quat_xyzw.astype(np.float32),
        "hand_box_contact": np.ones((frame_count, 2), dtype=np.float32),
        "hand_box_surface_distance": np.zeros((frame_count, 2), dtype=np.float32),
        "metadata": {
            "synthetic": True,
            "all_velocities_zero": True,
            "default_body_and_leg_joints": True,
            "mirrored_arm_joints": True,
            "visible_box_world_aligned": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({motion_key: motion}, args.output)
    z_goal_path = None
    if args.encode_goal:
        if args.model_folder is None:
            raise ValueError("--model-folder is required with --encode-goal")
        from humanoidverse.static_carry_pose_mujoco import _encode_goal

        pose = {
            "root_pos": root_pos,
            "root_quat_xyzw": root_quat_xyzw,
            "dof_pos": dof_pos,
            "object_pos": object_pos,
            "object_quat_xyzw": object_quat_xyzw,
        }
        _encode_goal(
            model=model,
            data=data,
            pose=pose,
            source=motion,
            defaults=np.asarray(
                [robot_training.default_joint_angles[name] for name in joint_names], dtype=np.float64
            ),
            hand_body_names=tuple(robot_training.robot.hands),
            hand_contact_offsets=hand_offsets,
            model_folder=args.model_folder,
            output=args.output_z_goal,
        )
        z_goal_path = str(args.output_z_goal.resolve())
    summary = {
        "output": str(args.output.resolve()),
        "motion_key": motion_key,
        "frames": frame_count,
        "fps": float(args.fps),
        "root_position": root_pos.tolist(),
        "left_arm": dict(zip(ARM_PARTS, LEFT_ARM.tolist())),
        "right_arm": dict(zip(ARM_PARTS, (RIGHT_MIRROR_SIGNS * LEFT_ARM).tolist())),
        "hand_contact_points_world": hand_points.tolist(),
        "visible_box_center_world": visible_box_center.tolist(),
        "visible_box_world_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "box_marker_position_world": object_pos.tolist(),
        "box_marker_quat_xyzw": object_quat_xyzw.tolist(),
        "hand_contact_error_m": contact_error.tolist(),
        "model_folder": None if args.model_folder is None else str(args.model_folder.resolve()),
        "z_goal": z_goal_path,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[symmetric-carry-goal] saved {args.output}")
    print(f"[symmetric-carry-goal] saved {args.summary}")
    print(f"[symmetric-carry-goal] hand contact error metres: {contact_error.tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument("--root-height", type=float, default=0.8)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/symmetric_static_carry_goal/symmetric_static_carry_goal.pkl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/symmetric_static_carry_goal/summary.json"),
    )
    parser.add_argument("--encode-goal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-folder", type=Path)
    parser.add_argument(
        "--output-z-goal",
        type=Path,
        default=Path("analysis/symmetric_static_carry_goal/z_goal_step73670656.pkl"),
    )
    args = parser.parse_args()
    if args.duration <= 0.0 or args.fps <= 0.0:
        parser.error("--duration and --fps must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
