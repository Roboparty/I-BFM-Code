"""Build and test a static upright two-hand box-hold pose in MuJoCo."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from humanoidverse.carry_goal_sim2sim import (
    BOX_BODY_NAME,
    BOX_GEOM_NAME,
    BOX_JOINT_NAME,
    CONTROL_DT,
    SIM_DT,
    _actuator_indices,
    _as_numpy,
    _build_model,
    _joint_addresses,
    _load_box_config,
    _match_joint_parameter,
    _patch_mode15_joint_physics,
    _physical_hand_contacts,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _load_source(path: Path, motion_id: int) -> tuple[str, dict[str, Any]]:
    motions = joblib.load(path)
    keys = list(motions)
    if not 0 <= motion_id < len(keys):
        raise IndexError(f"motion_id={motion_id} outside [0, {len(keys)})")
    return keys[motion_id], motions[keys[motion_id]]


def _build_pose(
    source: dict[str, Any],
    *,
    frame: int,
    joint_names: list[str],
    defaults: np.ndarray,
    root_height: float,
    squeeze_angle: float,
) -> dict[str, np.ndarray]:
    source_joint_names = list(source.get("joint_names", joint_names))
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    source_dof = _as_numpy(source["dof_pos"])[frame]
    dof = defaults.copy()

    # Keep the demonstrated waist and arm geometry, but use the nominal standing legs.
    for index, name in enumerate(joint_names):
        if name.startswith("waist_") or any(token in name for token in ("shoulder", "elbow", "wrist")):
            dof[index] = source_dof[source_index[name]]
    dof[joint_names.index("left_shoulder_roll_joint")] -= squeeze_angle
    dof[joint_names.index("right_shoulder_roll_joint")] += squeeze_angle

    source_root_pos = _as_numpy(source["root_trans_offset"])[frame]
    source_root_rotation = Rotation.from_quat(_as_numpy(source["root_quat"])[frame])
    source_object_pos = _as_numpy(source["object_pos"])[frame]
    source_object_rotation = Rotation.from_quat(_as_numpy(source["object_quat"])[frame])
    object_pos_root = source_root_rotation.inv().apply(source_object_pos - source_root_pos)
    object_rotation_root = source_root_rotation.inv() * source_object_rotation

    root_pos = np.asarray((0.0, 0.0, root_height), dtype=np.float64)
    root_quat_xyzw = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    return {
        "root_pos": root_pos,
        "root_quat_xyzw": root_quat_xyzw,
        "dof_pos": dof,
        "object_pos": root_pos + object_pos_root,
        "object_quat_xyzw": object_rotation_root.as_quat(),
    }


def _export_static_motion(
    output: Path,
    pose: dict[str, np.ndarray],
    *,
    source_key: str,
    source_motion_id: int,
    source_frame: int,
    joint_names: list[str],
    duration: float,
) -> None:
    frame_count = max(1, round(duration / CONTROL_DT))
    repeat = lambda value: np.repeat(np.asarray(value, dtype=np.float32)[None], frame_count, axis=0)
    motion = {
        "fps": round(1.0 / CONTROL_DT),
        "robot_name": "g1_29dof",
        "motion_key": "synthetic_static_upright_horizontal_hold",
        "joint_names": joint_names,
        "root_trans_offset": repeat(pose["root_pos"]),
        "root_quat": repeat(pose["root_quat_xyzw"]),
        "dof_pos": repeat(pose["dof_pos"]),
        "object_pos": repeat(pose["object_pos"]),
        "object_quat": repeat(pose["object_quat_xyzw"]),
        "object_lin_vel": np.zeros((frame_count, 3), dtype=np.float32),
        "object_ang_vel": np.zeros((frame_count, 3), dtype=np.float32),
        "object_target_pos": repeat(pose["object_pos"]),
        "object_target_quat": repeat(pose["object_quat_xyzw"]),
        "hand_box_contact": np.ones((frame_count, 2), dtype=np.float32),
        "metadata": {
            "synthetic": True,
            "description": "Upright G1 with both arms raised to hold a horizontal box",
            "source_motion_key": source_key,
            "source_motion_id": source_motion_id,
            "source_frame": source_frame,
            "all_velocities_zero": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({motion["motion_key"]: motion}, output)


def _encode_goal(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: dict[str, np.ndarray],
    source: dict[str, Any],
    defaults: np.ndarray,
    hand_body_names: tuple[str, str],
    hand_contact_offsets: np.ndarray,
    model_folder: Path,
    output: Path,
) -> np.ndarray:
    import torch

    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
    from humanoidverse.envs.carry_observations import (
        build_object_context,
        compute_carry_object_observation,
        compute_hand_contact_points,
        compute_hand_spatial_observation,
    )
    from humanoidverse.envs.motion_observations import compute_humanoid_observations_max
    from humanoidverse.utils.torch_utils import quat_rotate_inverse

    body_names = list(source["body_names"])
    body_pos = np.asarray([data.xpos[model.body(name).id] for name in body_names], dtype=np.float32)
    body_quat_xyzw = np.asarray([np.roll(data.xquat[model.body(name).id], -1) for name in body_names], dtype=np.float32)
    torso_index = body_names.index("torso_link")
    torso_rotation = Rotation.from_quat(body_quat_xyzw[torso_index])
    head_pos = body_pos[torso_index] + torso_rotation.apply((0.0, 0.0, 0.35))
    body_pos = np.concatenate((body_pos, head_pos[None].astype(np.float32)), axis=0)
    body_quat_xyzw = np.concatenate((body_quat_xyzw, body_quat_xyzw[torso_index : torso_index + 1]), axis=0)

    body_pos_t = torch.from_numpy(body_pos).unsqueeze(0)
    body_quat_t = torch.from_numpy(body_quat_xyzw).unsqueeze(0)
    body_vel_t = torch.zeros_like(body_pos_t)
    body_ang_vel_t = torch.zeros_like(body_pos_t)
    privileged_parts = compute_humanoid_observations_max(
        body_pos_t,
        body_quat_t,
        body_vel_t,
        body_ang_vel_t,
        local_root_obs=True,
        root_height_obs=True,
    )
    privileged_state = torch.cat(list(privileged_parts.values()), dim=-1)

    base_pos = body_pos_t[:, 0]
    base_quat = body_quat_t[:, 0]
    object_pos = torch.as_tensor(pose["object_pos"], dtype=torch.float32).reshape(1, 3)
    object_quat = torch.as_tensor(pose["object_quat_xyzw"], dtype=torch.float32).reshape(1, 4)
    policy = load_model_from_checkpoint_dir(model_folder / "checkpoint", device="cpu")
    policy.eval()
    policy_state_dim = int(policy.obs_space["state"].shape[0])
    if policy_state_dim not in (86, 98):
        raise ValueError(f"Unsupported Carry policy state dimension {policy_state_dim}; expected 86 or 98")
    include_hand_spatial = policy_state_dim == 98
    hand_body_pos = torch.as_tensor(
        np.asarray([data.xpos[model.body(name).id] for name in hand_body_names]), dtype=torch.float32
    ).reshape(1, 2, 3)
    hand_body_quat = torch.as_tensor(
        np.asarray([np.roll(data.xquat[model.body(name).id], -1) for name in hand_body_names]), dtype=torch.float32
    ).reshape(1, 2, 4)
    object_context = build_object_context(
        compute_carry_object_observation(
            base_pos,
            base_quat,
            object_pos,
            object_quat,
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            object_pos,
        ),
        torch.ones(1, 1),
        torch.ones(1, 2),
        compute_hand_spatial_observation(
            base_pos,
            base_quat,
            compute_hand_contact_points(
                hand_body_pos,
                hand_body_quat,
                torch.as_tensor(hand_contact_offsets, dtype=torch.float32),
            ),
            object_pos,
        ),
        include_hand_spatial=include_hand_spatial,
        observe_target_delta=True,
        observe_object_present=True,
        observe_task_id=True,
    )
    dof_pos_rel = torch.as_tensor(pose["dof_pos"] - defaults, dtype=torch.float32).reshape(1, -1)
    projected_gravity = quat_rotate_inverse(base_quat, torch.tensor([[0.0, 0.0, -1.0]]), w_last=True)
    state = torch.cat((dof_pos_rel, torch.zeros_like(dof_pos_rel), projected_gravity, torch.zeros(1, 3), object_context), dim=-1)
    privileged_state = torch.cat((privileged_state, object_context), dim=-1)
    expected_privileged_dim = 497 if include_hand_spatial else 485
    if state.shape != (1, policy_state_dim) or privileged_state.shape != (1, expected_privileged_dim):
        raise RuntimeError(f"Unexpected B observation shapes state={state.shape}, privileged_state={privileged_state.shape}")

    z_goal = policy.goal_inference({"state": state, "privileged_state": privileged_state}).cpu().numpy().astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(z_goal, output)
    print(f"[static-hold] encoded z_goal {output} shape={z_goal.shape} norm={np.linalg.norm(z_goal[0]):.4f}")
    return z_goal


def run(args: argparse.Namespace) -> dict[str, Any]:
    robot_training = load_robot_training_spec(args.robot_config)
    joint_names = list(robot_training.robot.control_joint_names)
    defaults = np.asarray([robot_training.default_joint_angles[name] for name in joint_names], dtype=np.float64)
    source_key, source = _load_source(args.motion_cache, args.motion_id)
    pose = _build_pose(
        source,
        frame=args.frame,
        joint_names=joint_names,
        defaults=defaults,
        root_height=args.root_height,
        squeeze_angle=args.squeeze_angle,
    )
    _export_static_motion(
        args.output_motion,
        pose,
        source_key=source_key,
        source_motion_id=args.motion_id,
        source_frame=args.frame,
        joint_names=joint_names,
        duration=args.duration,
    )

    box_cfg = _load_box_config(args.box_config)
    model = _build_model(Path(robot_training.robot.xml_path), box_cfg, render_size=480)
    _patch_mode15_joint_physics(model, joint_names)
    data = mujoco.MjData(model)
    qpos_addresses, qvel_addresses = _joint_addresses(model, joint_names)
    actuator_indices = _actuator_indices(model, joint_names)
    box_qpos_address = int(model.jnt_qposadr[model.joint(BOX_JOINT_NAME).id])
    box_qvel_address = int(model.jnt_dofadr[model.joint(BOX_JOINT_NAME).id])
    data.qpos[:3] = pose["root_pos"]
    data.qpos[3:7] = np.roll(pose["root_quat_xyzw"], 1)
    data.qpos[qpos_addresses] = pose["dof_pos"]
    data.qpos[box_qpos_address : box_qpos_address + 3] = pose["object_pos"]
    data.qpos[box_qpos_address + 3 : box_qpos_address + 7] = np.roll(pose["object_quat_xyzw"], 1)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if args.encode_goal:
        _encode_goal(
            model=model,
            data=data,
            pose=pose,
            source=source,
            defaults=defaults,
            hand_body_names=tuple(robot_training.robot.hands),
            hand_contact_offsets=np.asarray(box_cfg["hand_contact_offsets"], dtype=np.float32),
            model_folder=args.model_folder,
            output=args.output_z_goal,
        )

    kp = np.asarray([_match_joint_parameter(name, robot_training.stiffness) for name in joint_names])
    kd = np.asarray([_match_joint_parameter(name, robot_training.damping) for name in joint_names])
    effort = np.asarray(robot_training.effort_limits, dtype=np.float64)
    hand_body_ids = tuple(model.body(name).id for name in robot_training.robot.hands)
    box_geom_id = model.geom(BOX_GEOM_NAME).id
    box_body_id = model.body(BOX_BODY_NAME).id
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None
    if viewer is not None:
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -15.0

    history: list[dict[str, Any]] = []
    steps = max(1, round(args.duration / CONTROL_DT))
    substeps = round(CONTROL_DT / SIM_DT)
    try:
        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                break
            started_at = time.monotonic()
            for _ in range(substeps):
                torque = np.clip(
                    kp * (pose["dof_pos"] - data.qpos[qpos_addresses]) - kd * data.qvel[qvel_addresses],
                    -effort,
                    effort,
                )
                data.ctrl[actuator_indices] = torque
                mujoco.mj_step(model, data)
                if args.fix_base:
                    data.qpos[:3] = pose["root_pos"]
                    data.qpos[3:7] = np.roll(pose["root_quat_xyzw"], 1)
                    data.qvel[:6] = 0.0
                if args.fix_box:
                    data.qpos[box_qpos_address : box_qpos_address + 3] = pose["object_pos"]
                    data.qpos[box_qpos_address + 3 : box_qpos_address + 7] = np.roll(pose["object_quat_xyzw"], 1)
                    data.qvel[box_qvel_address : box_qvel_address + 6] = 0.0
                if args.fix_base or args.fix_box:
                    mujoco.mj_forward(model, data)
            contact = _physical_hand_contacts(
                model,
                data,
                box_geom_id=box_geom_id,
                hand_body_ids=hand_body_ids,
            )
            row = {
                "time": (step + 1) * CONTROL_DT,
                "root_height": float(data.qpos[2]),
                "box_height": float(data.xpos[box_body_id, 2]),
                "left_contact": bool(contact[0]),
                "right_contact": bool(contact[1]),
            }
            history.append(row)
            if step == 0 or step + 1 == steps or (step + 1) % args.log_every == 0:
                print(
                    f"[static-hold] t={row['time']:.2f}s root_h={row['root_height']:.3f} "
                    f"box_h={row['box_height']:.3f} contact={int(contact[0])}/{int(contact[1])}",
                    flush=True,
                )
            if viewer is not None:
                viewer.cam.lookat[:] = data.xpos[0]
                viewer.cam.lookat[2] = 0.8
                viewer.sync()
                remaining = CONTROL_DT - (time.monotonic() - started_at)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()

    final = history[-1]
    summary = {
        "pose": {key: value.tolist() for key, value in pose.items()},
        "source_motion_key": source_key,
        "source_motion_id": args.motion_id,
        "source_frame": args.frame,
        "squeeze_angle": args.squeeze_angle,
        "output_motion": str(args.output_motion),
        "box_mass": float(box_cfg["mass"]),
        "fix_base": bool(args.fix_base),
        "fix_box": bool(args.fix_box),
        "final": final,
        "bilateral_contact_rate": float(np.mean([row["left_contact"] and row["right_contact"] for row in history])),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(f"[static-hold] saved motion {args.output_motion}")
    print(f"[static-hold] saved summary {args.output_summary}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-cache",
        type=Path,
        default=Path("cache/motion_data/g1_carry_box_omniretarget_no_foot/g1_carry_box_omniretarget_no_foot_full_ufo.pkl"),
    )
    parser.add_argument("--motion-id", type=int, default=1187)
    parser.add_argument("--frame", type=int, default=120)
    parser.add_argument("--root-height", type=float, default=0.8)
    parser.add_argument("--squeeze-angle", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument(
        "--output-motion",
        type=Path,
        default=Path("analysis/synthetic_static_upright_horizontal_hold/static_hold_motion.pkl"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("analysis/synthetic_static_upright_horizontal_hold/mujoco_summary.json"),
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--encode-goal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("runs/ufo_carry_nofoot_lafan_50_50_baseline_8gpu_dr_192m_snapshot_step83279872"),
    )
    parser.add_argument(
        "--output-z-goal",
        type=Path,
        default=Path("analysis/synthetic_static_upright_horizontal_hold/z_goal.pkl"),
    )
    parser.add_argument("--fix-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fix-box", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
