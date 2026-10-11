"""Display one Carry reference frame in MuJoCo without running dynamics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from humanoidverse.carry_goal_sim2sim import (
    BOX_JOINT_NAME,
    _as_numpy,
    _build_model,
    _joint_addresses,
    _load_box_config,
    _load_motion,
    _patch_mode15_joint_physics,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec


def run(args: argparse.Namespace) -> None:
    robot_training = load_robot_training_spec(args.robot_config)
    joint_names = list(robot_training.robot.control_joint_names)
    box_cfg = _load_box_config(args.box_config)
    model = _build_model(Path(robot_training.robot.xml_path), box_cfg, render_size=480)
    _patch_mode15_joint_physics(model, joint_names)
    data = mujoco.MjData(model)

    motion_key, motion = _load_motion(args.motion_cache, args.motion_id)
    frame_count = int(_as_numpy(motion["dof_pos"]).shape[0])
    frame = int(args.frame if args.frame >= 0 else frame_count + args.frame)
    if not 0 <= frame < frame_count:
        raise IndexError(f"frame={args.frame} resolves to {frame}, outside [0, {frame_count})")

    qpos_addresses, _ = _joint_addresses(model, joint_names)
    motion_joint_names = list(motion.get("joint_names", joint_names))
    motion_index = {name: index for index, name in enumerate(motion_joint_names)}
    dof_pos = _as_numpy(motion["dof_pos"])[frame]

    data.qpos[:3] = _as_numpy(motion["root_trans_offset"])[frame]
    data.qpos[3:7] = np.roll(_as_numpy(motion["root_quat"])[frame], 1)
    data.qpos[qpos_addresses] = np.asarray([dof_pos[motion_index[name]] for name in joint_names])

    box_qpos_address = int(model.jnt_qposadr[model.joint(BOX_JOINT_NAME).id])
    data.qpos[box_qpos_address : box_qpos_address + 3] = _as_numpy(motion["object_pos"])[frame]
    data.qpos[box_qpos_address + 3 : box_qpos_address + 7] = np.roll(
        _as_numpy(motion["object_quat"])[frame], 1
    )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print(
        f"[goal-pose] motion_id={args.motion_id} motion={motion_key} "
        f"frame={frame}/{frame_count - 1}; dynamics disabled",
        flush=True,
    )
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = data.qpos[:3]
        viewer.cam.lookat[2] = float(args.camera_height)
        viewer.cam.distance = float(args.camera_distance)
        viewer.cam.azimuth = float(args.camera_azimuth)
        viewer.cam.elevation = float(args.camera_elevation)
        while viewer.is_running():
            # Recompute kinematics only. No mj_step is called, so the pose cannot fall or drift.
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-cache",
        type=Path,
        default=Path(
            "cache/motion_data/g1_carry_box_omniretarget_no_foot/"
            "g1_carry_box_omniretarget_no_foot_full_ufo.pkl"
        ),
    )
    parser.add_argument("--motion-id", type=int, default=1199)
    parser.add_argument("--frame", type=int, default=154)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument("--camera-distance", type=float, default=3.5)
    parser.add_argument("--camera-height", type=float, default=0.8)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
