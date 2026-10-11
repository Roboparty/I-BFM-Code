"""Audit and visualize collisions in object-aware LAFAN approach motions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import joblib
import mediapy as media
import mujoco
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from humanoidverse.carry_goal_sim2sim import (
    BOX_GEOM_NAME,
    BOX_JOINT_NAME,
    FLOOR_GEOM_NAME,
    _build_model,
    _joint_addresses,
)
from humanoidverse.utils.robot_spec.training_spec import load_robot_training_spec


DEFAULT_VISUALIZATIONS = (
    "walk1_subject1_clip23_approach_box",
    "walk2_subject1_clip21_approach_box",
    "sprint1_subject4_clip21_approach_box",
    "fallAndGetUp1_subject1_clip9_approach_box",
    "fallAndGetUp3_subject1_clip3_approach_box",
)


def _motion_qpos(
    model: mujoco.MjModel,
    motion: dict[str, Any],
    frame: int,
    *,
    root_qpos_address: int,
    joint_qpos_addresses: np.ndarray,
    box_qpos_address: int,
) -> np.ndarray:
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    root_quat_xyzw = np.asarray(motion.get("root_rot", motion.get("root_quat")), dtype=np.float64)
    dof_pos = np.asarray(motion.get("dof_pos", motion.get("dof")), dtype=np.float64)
    object_pos = np.asarray(motion["object_pos"], dtype=np.float64)
    object_quat_xyzw = np.asarray(motion["object_quat"], dtype=np.float64)

    qpos = model.qpos0.copy()
    qpos[root_qpos_address : root_qpos_address + 3] = root_pos[frame]
    qpos[root_qpos_address + 3 : root_qpos_address + 7] = root_quat_xyzw[frame, [3, 0, 1, 2]]
    qpos[joint_qpos_addresses] = dof_pos[frame]
    qpos[box_qpos_address : box_qpos_address + 3] = object_pos[frame]
    qpos[box_qpos_address + 3 : box_qpos_address + 7] = object_quat_xyzw[frame, [3, 0, 1, 2]]
    return qpos


def _box_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_geom_id: int,
    floor_geom_id: int,
) -> tuple[bool, bool, float, float, list[str]]:
    robot_contact = False
    ground_contact = False
    max_robot_penetration = 0.0
    max_ground_penetration = 0.0
    robot_geoms: list[str] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if box_geom_id not in (geom1, geom2):
            continue
        other = geom2 if geom1 == box_geom_id else geom1
        penetration = max(0.0, -float(contact.dist))
        if other == floor_geom_id:
            ground_contact = True
            max_ground_penetration = max(max_ground_penetration, penetration)
        else:
            robot_contact = True
            max_robot_penetration = max(max_robot_penetration, penetration)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
            robot_geoms.append(name or f"geom_{other}")
    return robot_contact, ground_contact, max_robot_penetration, max_ground_penetration, robot_geoms


def _robot_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_geom_id: int,
    floor_geom_id: int,
    foot_body_names: set[str],
) -> tuple[list[str], list[str], list[str], float, float]:
    ground_bodies: list[str] = []
    non_foot_ground_bodies: list[str] = []
    self_pairs: list[str] = []
    max_non_foot_ground_penetration = 0.0
    max_self_penetration = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if box_geom_id in (geom1, geom2):
            continue
        if floor_geom_id in (geom1, geom2):
            other = geom2 if geom1 == floor_geom_id else geom1
            body_id = int(model.geom_bodyid[other])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            ground_bodies.append(body_name)
            if body_name not in foot_body_names:
                non_foot_ground_bodies.append(body_name)
                max_non_foot_ground_penetration = max(
                    max_non_foot_ground_penetration, max(0.0, -float(contact.dist))
                )
            continue
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or f"geom_{geom1}"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or f"geom_{geom2}"
        self_pairs.append(" <-> ".join(sorted((name1, name2))))
        max_self_penetration = max(max_self_penetration, max(0.0, -float(contact.dist)))
    return (
        ground_bodies,
        non_foot_ground_bodies,
        self_pairs,
        max_non_foot_ground_penetration,
        max_self_penetration,
    )


def _model_context(robot_config: Path, box_config: Path, render_size: int):
    robot_training = load_robot_training_spec(robot_config)
    box_cfg = OmegaConf.to_container(OmegaConf.load(box_config), resolve=True)
    if not isinstance(box_cfg, dict):
        raise TypeError(f"Expected mapping in {box_config}")
    model = _build_model(Path(robot_training.robot.xml_path), box_cfg, render_size=render_size)
    if robot_training.robot.free_joint is None:
        raise ValueError("Robot MJCF requires a free root joint")
    root_qpos_address = int(model.jnt_qposadr[model.joint(robot_training.robot.free_joint).id])
    joint_qpos_addresses, _ = _joint_addresses(model, list(robot_training.robot.control_joint_names))
    box_qpos_address = int(model.jnt_qposadr[model.joint(BOX_JOINT_NAME).id])
    return model, root_qpos_address, joint_qpos_addresses, box_qpos_address, set(robot_training.robot.feet)


def audit_dataset(
    motions: dict[str, dict[str, Any]],
    model: mujoco.MjModel,
    *,
    root_qpos_address: int,
    joint_qpos_addresses: np.ndarray,
    box_qpos_address: int,
    foot_body_names: set[str],
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    box_geom_id = model.geom(BOX_GEOM_NAME).id
    floor_geom_id = model.geom(FLOOR_GEOM_NAME).id
    rows: list[dict[str, Any]] = []
    geom_pairs: Counter[str] = Counter()
    ground_bodies: Counter[str] = Counter()
    non_foot_ground_bodies: Counter[str] = Counter()
    self_pairs: Counter[str] = Counter()
    total_frames = robot_frames = ground_frames = 0
    robot_ground_frames = non_foot_ground_frames = self_collision_frames = 0
    max_robot_penetration = max_ground_penetration = 0.0
    max_non_foot_ground_penetration = max_self_penetration = 0.0

    for key, motion in motions.items():
        num_frames = int(np.asarray(motion["root_trans_offset"]).shape[0])
        motion_robot_frames = motion_ground_frames = 0
        motion_robot_penetration = motion_ground_penetration = 0.0
        motion_geoms: Counter[str] = Counter()
        motion_non_foot_ground_frames = motion_self_collision_frames = 0
        for frame in range(num_frames):
            data.qpos[:] = _motion_qpos(
                model,
                motion,
                frame,
                root_qpos_address=root_qpos_address,
                joint_qpos_addresses=joint_qpos_addresses,
                box_qpos_address=box_qpos_address,
            )
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            robot, ground, robot_pen, ground_pen, robot_geoms = _box_contacts(
                model, data, box_geom_id=box_geom_id, floor_geom_id=floor_geom_id
            )
            motion_robot_frames += int(robot)
            motion_ground_frames += int(ground)
            motion_robot_penetration = max(motion_robot_penetration, robot_pen)
            motion_ground_penetration = max(motion_ground_penetration, ground_pen)
            motion_geoms.update(robot_geoms)
            (
                frame_ground_bodies,
                frame_non_foot_ground,
                frame_self_pairs,
                frame_non_foot_penetration,
                frame_self_penetration,
            ) = _robot_contacts(
                model,
                data,
                box_geom_id=box_geom_id,
                floor_geom_id=floor_geom_id,
                foot_body_names=foot_body_names,
            )
            robot_ground_frames += int(bool(frame_ground_bodies))
            non_foot_ground_frames += int(bool(frame_non_foot_ground))
            self_collision_frames += int(bool(frame_self_pairs))
            motion_non_foot_ground_frames += int(bool(frame_non_foot_ground))
            motion_self_collision_frames += int(bool(frame_self_pairs))
            ground_bodies.update(frame_ground_bodies)
            non_foot_ground_bodies.update(frame_non_foot_ground)
            self_pairs.update(frame_self_pairs)
            max_non_foot_ground_penetration = max(
                max_non_foot_ground_penetration, frame_non_foot_penetration
            )
            max_self_penetration = max(max_self_penetration, frame_self_penetration)

        total_frames += num_frames
        robot_frames += motion_robot_frames
        ground_frames += motion_ground_frames
        max_robot_penetration = max(max_robot_penetration, motion_robot_penetration)
        max_ground_penetration = max(max_ground_penetration, motion_ground_penetration)
        geom_pairs.update(motion_geoms)
        root_pos = np.asarray(motion["root_trans_offset"])
        object_pos = np.asarray(motion["object_pos"])
        planar_distance = np.linalg.norm(object_pos[:, :2] - root_pos[:, :2], axis=1)
        rows.append(
            {
                "motion_key": key,
                "frames": num_frames,
                "robot_box_contact_frames": motion_robot_frames,
                "robot_box_contact_fraction": motion_robot_frames / num_frames,
                "box_ground_contact_frames": motion_ground_frames,
                "box_ground_contact_fraction": motion_ground_frames / num_frames,
                "max_robot_box_penetration_m": motion_robot_penetration,
                "max_box_ground_penetration_m": motion_ground_penetration,
                "robot_contact_geoms": dict(sorted(motion_geoms.items())),
                "non_foot_ground_contact_frames": motion_non_foot_ground_frames,
                "self_collision_frames": motion_self_collision_frames,
                "initial_planar_distance_m": float(planar_distance[0]),
                "final_planar_distance_m": float(planar_distance[-1]),
            }
        )

    return {
        "motion_count": len(rows),
        "total_frames": total_frames,
        "motions_with_robot_box_contact": sum(row["robot_box_contact_frames"] > 0 for row in rows),
        "robot_box_contact_frames": robot_frames,
        "robot_box_contact_fraction": robot_frames / total_frames,
        "box_ground_contact_frames": ground_frames,
        "box_ground_contact_fraction": ground_frames / total_frames,
        "max_robot_box_penetration_m": max_robot_penetration,
        "max_box_ground_penetration_m": max_ground_penetration,
        "robot_contact_geoms": dict(sorted(geom_pairs.items())),
        "robot_ground_contact_frames": robot_ground_frames,
        "non_foot_ground_contact_frames": non_foot_ground_frames,
        "motions_with_non_foot_ground_contact": sum(row["non_foot_ground_contact_frames"] > 0 for row in rows),
        "self_collision_frames": self_collision_frames,
        "motions_with_self_collision": sum(row["self_collision_frames"] > 0 for row in rows),
        "max_non_foot_ground_penetration_m": max_non_foot_ground_penetration,
        "max_self_collision_penetration_m": max_self_penetration,
        "ground_contact_bodies": dict(sorted(ground_bodies.items())),
        "non_foot_ground_contact_bodies": dict(sorted(non_foot_ground_bodies.items())),
        "self_collision_geom_pairs": dict(sorted(self_pairs.items())),
        "motions": rows,
    }


def _camera_for_motion(motion: dict[str, Any]) -> mujoco.MjvCamera:
    root = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    box = np.asarray(motion["object_pos"], dtype=np.float64)
    points = np.concatenate((root[:, :2], box[:, :2]), axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = ((lower[0] + upper[0]) / 2.0, (lower[1] + upper[1]) / 2.0, 0.65)
    camera.distance = max(3.0, float(np.linalg.norm(upper - lower)) * 1.15 + 1.8)
    camera.azimuth = 135.0
    camera.elevation = -24.0
    return camera


def _overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    line_height = 21
    draw.rectangle((8, 8, 540, 18 + line_height * len(lines)), fill=(0, 0, 0, 185))
    for index, (line, color) in enumerate(lines):
        draw.text((16, 14 + index * line_height), line, fill=color, font=font)
    return np.asarray(image)


def render_motion(
    key: str,
    motion: dict[str, Any],
    model: mujoco.MjModel,
    output_path: Path,
    *,
    render_size: int,
    root_qpos_address: int,
    joint_qpos_addresses: np.ndarray,
    box_qpos_address: int,
    foot_body_names: set[str],
) -> None:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=render_size, width=render_size)
    camera = _camera_for_motion(motion)
    box_geom_id = model.geom(BOX_GEOM_NAME).id
    floor_geom_id = model.geom(FLOOR_GEOM_NAME).id
    root = np.asarray(motion["root_trans_offset"])
    box = np.asarray(motion["object_pos"])
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    frames: list[np.ndarray] = []
    try:
        for frame_index in range(root.shape[0]):
            data.qpos[:] = _motion_qpos(
                model,
                motion,
                frame_index,
                root_qpos_address=root_qpos_address,
                joint_qpos_addresses=joint_qpos_addresses,
                box_qpos_address=box_qpos_address,
            )
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            robot, ground, robot_pen, ground_pen, robot_geoms = _box_contacts(
                model, data, box_geom_id=box_geom_id, floor_geom_id=floor_geom_id
            )
            (
                _,
                non_foot_ground_bodies,
                self_pairs,
                non_foot_penetration,
                self_penetration,
            ) = _robot_contacts(
                model,
                data,
                box_geom_id=box_geom_id,
                floor_geom_id=floor_geom_id,
                foot_body_names=foot_body_names,
            )
            renderer.update_scene(data, camera=camera)
            distance = float(np.linalg.norm(box[frame_index, :2] - root[frame_index, :2]))
            frames.append(
                _overlay(
                    renderer.render(),
                    [
                        (f"{key}  t={frame_index / fps:.2f}s  frame={frame_index}", (240, 240, 240)),
                        (f"robot-box collision: {int(robot)}  max penetration: {robot_pen * 1000:.2f} mm", (255, 80, 80) if robot else (100, 255, 120)),
                        (f"box-ground contact: {int(ground)}  penetration: {ground_pen * 1000:.2f} mm", (255, 210, 90)),
                        (f"robot self-collision: {int(bool(self_pairs))}  penetration: {self_penetration * 1000:.2f} mm", (255, 80, 80) if self_pairs else (100, 255, 120)),
                        (f"non-foot ground contact: {int(bool(non_foot_ground_bodies))}  penetration: {non_foot_penetration * 1000:.2f} mm", (255, 170, 70) if non_foot_ground_bodies else (100, 255, 120)),
                        (f"root-box planar distance: {distance:.3f} m", (120, 200, 255)),
                        (f"box-contact geoms: {', '.join(robot_geoms) if robot_geoms else 'none'}", (240, 240, 240)),
                        (f"self-contact: {self_pairs[0] if self_pairs else 'none'}", (240, 240, 240)),
                    ],
                )
            )
    finally:
        renderer.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(output_path, frames, fps=fps, codec="h264", crf=18)


def _safe_filename(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("humanoidverse/data/lafan_29dof_10s-clipped_approach_box.pkl"))
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/lafan_approach_box/collision_visualizations"))
    parser.add_argument("--summary", type=Path, default=Path("analysis/lafan_approach_box/collision_summary.json"))
    parser.add_argument("--motions", nargs="*", default=list(DEFAULT_VISUALIZATIONS))
    parser.add_argument("--render-size", type=int, default=720)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    motions = joblib.load(args.input.expanduser().resolve())
    if not isinstance(motions, dict):
        raise TypeError(f"Expected motion mapping in {args.input}")
    model, root_address, joint_addresses, box_address, foot_body_names = _model_context(
        args.robot_config.expanduser().resolve(), args.box_config.expanduser().resolve(), args.render_size
    )
    summary = audit_dataset(
        motions,
        model,
        root_qpos_address=root_address,
        joint_qpos_addresses=joint_addresses,
        box_qpos_address=box_address,
        foot_body_names=foot_body_names,
    )
    summary_path = args.summary.expanduser()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "motions"}, indent=2, sort_keys=True))
    print(f"Wrote collision audit to {summary_path}")
    if args.audit_only:
        return
    missing = [key for key in args.motions if key not in motions]
    if missing:
        raise KeyError(f"Requested visualization motions not found: {missing}")
    for key in args.motions:
        output_path = args.output_dir / f"{_safe_filename(key)}_collision.mp4"
        render_motion(
            key,
            motions[key],
            model,
            output_path,
            render_size=args.render_size,
            root_qpos_address=root_address,
            joint_qpos_addresses=joint_addresses,
            box_qpos_address=box_address,
            foot_body_names=foot_body_names,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
