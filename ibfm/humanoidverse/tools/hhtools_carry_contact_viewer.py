"""Render RP1 HHTools Carry scenes with stored hand-contact labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from humanoidverse.mjlab_inference_utils import MujocoQposRenderer
from humanoidverse.utils.motion_data.carry_box import _matrix, _read_table, _robot_csv, _time
from humanoidverse.utils.robot_spec import load_robot_spec


def _scene_qpos(scene: Path, renderer: MujocoQposRenderer, robot_config: Path) -> tuple[np.ndarray, np.ndarray]:
    robot = load_robot_spec(robot_config)
    robot_path = _robot_csv(scene)
    object_path = scene / "object_0_largebox.csv"
    contact_path = scene / "hand_box_contact.csv"
    _robot_fields, robot_rows = _read_table(robot_path)
    _object_fields, object_rows = _read_table(object_path)
    contact_fields, contact_rows = _read_table(contact_path)

    times = _time(robot_rows, robot_path)
    if not np.allclose(times, _time(object_rows, object_path), atol=1.0e-6, rtol=0.0):
        raise ValueError(f"Robot/object timestamps do not match in {scene}")
    if not np.allclose(times, _time(contact_rows, contact_path), atol=1.0e-6, rtol=0.0):
        raise ValueError(f"Robot/contact timestamps do not match in {scene}")

    root_pos = _matrix(robot_rows, ["root_x", "root_y", "root_z"], robot_path)
    root_quat = _matrix(robot_rows, ["root_qx", "root_qy", "root_qz", "root_qw"], robot_path)
    dof = _matrix(robot_rows, [f"dof_{name}" for name in robot.control_joint_names], robot_path)
    object_pos = _matrix(object_rows, ["pos_x", "pos_y", "pos_z"], object_path)
    object_quat = _matrix(object_rows, ["quat_x", "quat_y", "quat_z", "quat_w"], object_path)

    model = renderer.model
    root_qpos = int(model.joint(robot.free_joint).qposadr[0])
    box_qpos = int(model.joint("inference_carry_box_joint").qposadr[0])
    joint_qpos = np.asarray(
        [int(model.joint(name).qposadr[0]) for name in robot.control_joint_names], dtype=np.int32
    )
    qpos = np.broadcast_to(model.qpos0, (len(times), model.nq)).copy()
    qpos[:, root_qpos : root_qpos + 3] = root_pos
    qpos[:, root_qpos + 3 : root_qpos + 7] = root_quat[:, [3, 0, 1, 2]]
    qpos[:, joint_qpos] = dof
    qpos[:, box_qpos : box_qpos + 3] = object_pos
    qpos[:, box_qpos + 3 : box_qpos + 7] = object_quat[:, [3, 0, 1, 2]]

    columns = [
        "left_hand_box_contact",
        "right_hand_box_contact",
        "left_hand_box_surface_distance",
        "right_hand_box_surface_distance",
    ]
    if not all(column in contact_fields for column in columns):
        raise ValueError(f"Missing contact visualization columns in {contact_path}")
    contact = _matrix(contact_rows, columns, contact_path)
    return qpos, contact


def render_scene(
    scene: Path,
    output: Path,
    *,
    robot_config: Path,
    box_config: Path,
    render_size: int,
) -> None:
    import imageio_ffmpeg
    import mediapy as media
    from PIL import Image, ImageDraw

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    robot = load_robot_spec(robot_config)
    box = OmegaConf.load(box_config)
    renderer = MujocoQposRenderer(
        Path(robot.xml_path),
        render_size=render_size,
        camera_distance=2.3,
        camera_azimuth=135.0,
        camera_elevation=-16.0,
        expected_qpos_size=robot.nq + 7,
        box_half_size=tuple(float(value) for value in box.half_size),
        box_geom_pos=tuple(float(value) for value in box.geom_pos),
        box_geom_quat_wxyz=tuple(float(value) for value in box.geom_quat_wxyz),
        box_collisions=True,
    )
    try:
        qpos, contact = _scene_qpos(scene, renderer, robot_config)
        frames = []
        for index, frame_qpos in enumerate(qpos):
            image = Image.fromarray(renderer.render_qpos(frame_qpos))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 94), fill=(0, 0, 0))
            draw.text(
                (10, 7),
                f"{scene.name}  t={index / 30.0:.2f}s  frame={index}",
                fill=(255, 255, 255),
            )
            for hand, (name, x) in enumerate((("LEFT", 10), ("RIGHT", image.width // 2))):
                active = bool(contact[index, hand])
                color = (70, 230, 100) if active else (245, 85, 75)
                draw.text((x, 34), f"{name} CONTACT={int(active)}", fill=color)
                draw.text(
                    (x, 61),
                    f"MuJoCo wrist gap={contact[index, hand + 2] * 100.0:.2f} cm",
                    fill=(225, 225, 225),
                )
            frames.append(np.asarray(image))
        output.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(output, frames, fps=30.0)
    finally:
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--motions", required=True, help="Comma-separated scene names")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/rp1_28dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box_rp1.yaml"))
    parser.add_argument("--render-size", type=int, default=600)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    for motion in (item.strip() for item in args.motions.split(",") if item.strip()):
        scene = data_root / motion
        if not scene.is_dir():
            raise FileNotFoundError(f"Unknown RP1 Carry scene: {scene}")
        output = args.output_dir / f"{motion}_hand_contact.mp4"
        render_scene(
            scene,
            output,
            robot_config=args.robot_config,
            box_config=args.box_config,
            render_size=args.render_size,
        )
        print(f"[rp1-contact-video] {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
