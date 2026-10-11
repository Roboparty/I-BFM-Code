"""Validate paired G1 carry data and optionally plot robot/object alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from humanoidverse.envs.carry_observations import (
    compute_hand_capsule_box_surface_distance,
    infer_carry_phase_contact,
)
from humanoidverse.mjlab_inference_utils import MujocoQposRenderer
from humanoidverse.utils.motion_data import prepare_motion_manifest
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _select_motion(data: dict, selector: str | None) -> tuple[str, dict]:
    keys = list(data)
    if selector is None:
        key = keys[0]
    elif selector in data:
        key = selector
    else:
        try:
            key = keys[int(selector)]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Unknown motion {selector!r}; expected a key or index in [0, {len(keys) - 1}]") from exc
    return key, data[key]


def _plot_alignment(key: str, motion: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    fps = float(motion["fps"])
    robot_pos = np.asarray(motion["root_trans_offset"])
    object_pos = np.asarray(motion["object_pos"])
    time = np.arange(len(robot_pos)) / fps

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(robot_pos[:, 0], robot_pos[:, 1], label="G1 root")
    axes[0].plot(object_pos[:, 0], object_pos[:, 1], label="box")
    axes[0].scatter(*object_pos[-1, :2], marker="x", s=70, label="box target")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="World XY trajectory")
    axes[0].axis("equal")
    axes[0].legend()

    axes[1].plot(time, robot_pos[:, 2], label="G1 root z")
    axes[1].plot(time, object_pos[:, 2], label="box z")
    axes[1].set(xlabel="time (s)", ylabel="z (m)", title="Lift timing")
    axes[1].legend()
    figure.suptitle(key)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    with path.open(errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
    if not vertices:
        raise ValueError(f"OBJ mesh has no vertices: {path}")
    vertices_array = np.asarray(vertices, dtype=np.float64)
    minimum = vertices_array.min(axis=0)
    maximum = vertices_array.max(axis=0)
    return maximum - minimum, (maximum + minimum) * 0.5


def _motion_qpos(motion: dict) -> np.ndarray:
    root_quat_wxyz = np.roll(np.asarray(motion["root_quat"]), 1, axis=1)
    object_quat_wxyz = np.roll(np.asarray(motion["object_quat"]), 1, axis=1)
    return np.concatenate(
        [
            np.asarray(motion["root_trans_offset"]),
            root_quat_wxyz,
            np.asarray(motion["dof_pos"]),
            np.asarray(motion["object_pos"]),
            object_quat_wxyz,
        ],
        axis=1,
    )


def _label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 30), fill=(0, 0, 0))
    draw.text((10, 8), text, fill=(255, 255, 255))
    return np.asarray(image)


def _label_contact_frame(
    frame: np.ndarray,
    *,
    motion_key: str,
    frame_index: int,
    fps: float,
    contact: np.ndarray,
    surface_distance: np.ndarray,
    lift_clearance: float,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 76), fill=(0, 0, 0))
    draw.text((10, 7), f"{motion_key}   t={frame_index / fps:.2f}s   frame={frame_index}", fill=(255, 255, 255))
    for hand_index, (name, x) in enumerate((("LEFT", 10), ("RIGHT", image.width // 2))):
        is_contact = bool(contact[hand_index])
        color = (80, 230, 100) if is_contact else (245, 90, 80)
        draw.text((x, 30), f"{name} CONTACT = {int(is_contact)}", fill=color)
        draw.text((x, 51), f"surface gap = {surface_distance[hand_index] * 100:.2f} cm", fill=(230, 230, 230))
    draw.text(
        (image.width - 210, 7),
        f"box bottom clearance > {lift_clearance * 100:.1f} cm",
        fill=(255, 210, 90),
    )
    return np.asarray(image)


def _render_contact_video(
    key: str,
    motion: dict,
    output: Path,
    *,
    robot_config: Path,
    box_config: Path,
    render_size: int,
) -> None:
    import imageio_ffmpeg
    import mediapy as media

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    training_spec = load_robot_training_spec(robot_config)
    carry_config = OmegaConf.load(box_config)
    box_half_size = torch.tensor(carry_config.half_size, dtype=torch.float32)
    box_geom_pos = torch.tensor(carry_config.get("geom_pos", [0.0, 0.0, 0.0]), dtype=torch.float32)
    geom_quat_wxyz = torch.tensor(
        carry_config.get("geom_quat_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=torch.float32
    )
    box_geom_quat = geom_quat_wxyz[[1, 2, 3, 0]]
    hand_capsule_segments = torch.tensor(carry_config.hand_capsule_segments, dtype=torch.float32)
    hand_radius = float(carry_config.hand_contact_radius)
    lift_clearance = float(carry_config.get("phase_lift_clearance", 0.005))
    expected_qpos_size = 7 + len(training_spec.robot.control_joint_names) + 7
    renderer = MujocoQposRenderer(
        Path(training_spec.robot.xml_path),
        render_size=render_size,
        expected_qpos_size=expected_qpos_size,
        box_half_size=tuple(float(value) for value in carry_config.half_size),
        box_geom_pos=tuple(float(value) for value in carry_config.get("geom_pos", [0.0, 0.0, 0.0])),
        box_geom_quat_wxyz=tuple(float(value) for value in geom_quat_wxyz),
    )
    qpos_frames = _motion_qpos(motion)
    hand_body_pos = []
    hand_body_quat = []
    hand_names = tuple(training_spec.robot.hands)
    if len(hand_names) != 2:
        raise ValueError(f"Contact visualization requires exactly two configured hands, got {hand_names}")
    try:
        hand_body_ids = [renderer.model.body(name).id for name in hand_names]
        for qpos in qpos_frames:
            renderer.data.qpos[:] = qpos
            renderer.data.qvel[:] = 0.0
            mujoco.mj_forward(renderer.model, renderer.data)
            hand_body_pos.append(renderer.data.xpos[hand_body_ids].copy())
            hand_body_quat.append(renderer.data.xquat[hand_body_ids][:, [1, 2, 3, 0]].copy())
        hand_body_pos_tensor = torch.from_numpy(np.asarray(hand_body_pos, dtype=np.float32))
        hand_body_quat_tensor = torch.from_numpy(np.asarray(hand_body_quat, dtype=np.float32))
        object_pos = torch.from_numpy(np.asarray(motion["object_pos"], dtype=np.float32))
        object_quat = torch.from_numpy(np.asarray(motion["object_quat"], dtype=np.float32))
        surface_distance = compute_hand_capsule_box_surface_distance(
            hand_body_pos_tensor,
            hand_body_quat_tensor,
            hand_capsule_segments,
            object_pos,
            object_quat,
            box_half_size,
            box_geom_pos,
            box_geom_quat,
            hand_contact_radius=hand_radius,
        )
        contact = infer_carry_phase_contact(
            object_pos,
            object_quat,
            torch.ones(len(object_pos), 1, dtype=object_pos.dtype),
            box_half_size,
            box_geom_pos,
            box_geom_quat,
            ground_surface_height=float(carry_config.get("ground_surface_height", 0.0)),
            lift_clearance=lift_clearance,
        )
        frames = [
            _label_contact_frame(
                renderer.render_qpos(qpos),
                motion_key=key,
                frame_index=frame_index,
                fps=float(motion["fps"]),
                contact=contact[frame_index].numpy(),
                surface_distance=surface_distance[frame_index].numpy(),
                lift_clearance=lift_clearance,
            )
            for frame_index, qpos in enumerate(qpos_frames)
        ]
    finally:
        renderer.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(output, frames, fps=float(motion["fps"]))
    states = contact[:, 0].to(torch.int64) + 2 * contact[:, 1].to(torch.int64)
    counts = torch.bincount(states, minlength=4)
    print(
        "contact_frames_none_left_right_both="
        f"{counts.tolist()} contact_percent={[(100.0 * value / len(states)).item() for value in counts]}"
    )


def _render_comparison_video(
    motion: dict,
    output: Path,
    *,
    robot_config: Path,
    box_config: Path,
    render_size: int,
) -> None:
    import imageio_ffmpeg
    import mediapy as media

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    training_spec = load_robot_training_spec(robot_config)
    mesh_path = Path(str(motion["object_mesh_path"]))
    carry_config = OmegaConf.load(box_config)
    box_half_size = tuple(float(value) for value in carry_config.half_size)
    box_geom_pos = tuple(float(value) for value in carry_config.get("geom_pos", [0.0, 0.0, 0.0]))
    box_geom_quat_wxyz = tuple(
        float(value) for value in carry_config.get("geom_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
    )
    mesh_size, _mesh_center = _mesh_bounds(mesh_path)
    expected_qpos_size = 7 + len(training_spec.robot.control_joint_names) + 7
    data_renderer = MujocoQposRenderer(
        Path(training_spec.robot.xml_path),
        render_size=render_size,
        expected_qpos_size=expected_qpos_size,
        box_mesh_path=mesh_path,
    )
    simulation_renderer = MujocoQposRenderer(
        Path(training_spec.robot.xml_path),
        render_size=render_size,
        expected_qpos_size=expected_qpos_size,
        box_half_size=box_half_size,
        box_geom_pos=box_geom_pos,
        box_geom_quat_wxyz=box_geom_quat_wxyz,
    )
    frames = []
    try:
        for qpos in _motion_qpos(motion):
            data_frame = _label_frame(
                data_renderer.render_qpos(qpos),
                "Dataset OBJ: " + " x ".join(f"{value * 100:.2f}" for value in mesh_size) + " cm",
            )
            simulation_frame = _label_frame(
                simulation_renderer.render_qpos(qpos),
                "Configured box: " + " x ".join(f"{2 * value * 100:.2f}" for value in box_half_size) + " cm",
            )
            frames.append(np.concatenate([data_frame, simulation_frame], axis=1))
    finally:
        data_renderer.close()
        simulation_renderer.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(output, frames, fps=float(motion["fps"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/data/g1_carry_box.yaml"))
    parser.add_argument("--motion", default=None, help="Motion key or zero-based index; defaults to the first motion.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional output PNG for robot/box trajectory alignment.")
    parser.add_argument("--video", type=Path, default=None, help="Optional side-by-side dataset OBJ/current simulation MP4.")
    parser.add_argument("--contact-video", type=Path, default=None, help="Optional MP4 with left/right contact labels and surface gaps.")
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    prepared = prepare_motion_manifest(args.manifest, rebuild_cache=args.rebuild_cache)
    if not prepared.inference_paths:
        raise ValueError(f"Manifest has no full-sequence data: {args.manifest}")
    full_path = next(iter(prepared.inference_paths.values()))
    data = joblib.load(full_path)
    key, motion = _select_motion(data, args.motion)

    frame_counts = np.asarray([len(item["object_pos"]) for item in data.values()])
    fps_values = np.asarray([float(item["fps"]) for item in data.values()])
    lift_values = np.asarray(
        [np.max(np.asarray(item["object_pos"])[:, 2]) - np.asarray(item["object_pos"])[0, 2] for item in data.values()]
    )
    object_pos = np.asarray(motion["object_pos"])
    root_pos = np.asarray(motion["root_trans_offset"])
    if object_pos.shape[0] != root_pos.shape[0]:
        raise ValueError(f"Motion {key} robot/object frame mismatch: {root_pos.shape[0]} vs {object_pos.shape[0]}")
    print(
        f"motions={len(data)} frames=[{frame_counts.min()}, {frame_counts.max()}] "
        f"fps=[{fps_values.min():.3f}, {fps_values.max():.3f}] min_peak_lift={lift_values.min():.3f}m"
    )
    print(
        f"selected={key} frames={len(object_pos)} duration={(len(object_pos) - 1) / float(motion['fps']):.3f}s "
        f"box_start={object_pos[0].tolist()} box_target={np.asarray(motion['object_target_pos']).tolist()}"
    )
    mesh_size, mesh_center = _mesh_bounds(Path(str(motion["object_mesh_path"])))
    print(f"mesh_size_m={mesh_size.tolist()} mesh_center_m={mesh_center.tolist()}")
    if args.plot is not None:
        _plot_alignment(key, motion, args.plot)
        print(f"plot={args.plot.resolve()}")
    if args.video is not None:
        _render_comparison_video(
            motion,
            args.video,
            robot_config=args.robot_config,
            box_config=args.box_config,
            render_size=args.render_size,
        )
        print(f"video={args.video.resolve()}")
    if args.contact_video is not None:
        _render_contact_video(
            key,
            motion,
            args.contact_video,
            robot_config=args.robot_config,
            box_config=args.box_config,
            render_size=args.render_size,
        )
        print(f"contact_video={args.contact_video.resolve()}")


if __name__ == "__main__":
    main()
