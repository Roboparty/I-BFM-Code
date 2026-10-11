"""Render proxy, strict MuJoCo, and diagnostic distance-contact labels."""

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
    compute_hand_box_surface_distance,
    compute_hand_contact_points,
    infer_hand_box_contact,
)
from humanoidverse.mjlab_inference_utils import MujocoQposRenderer
from humanoidverse.tools.carry_data_inspect import _motion_qpos, _select_motion
from humanoidverse.utils.motion_data import prepare_motion_manifest
from humanoidverse.utils.robot_spec import load_robot_training_spec


DIAGNOSTIC_CONTACT_ENTER_DISTANCE = 0.015
DIAGNOSTIC_CONTACT_EXIT_DISTANCE = 0.03


def _hysteresis(distance: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    contact = np.zeros_like(distance, dtype=bool)
    if len(contact) == 0:
        return contact
    contact[0] = distance[0] < enter
    for frame in range(1, len(contact)):
        contact[frame] = np.where(
            distance[frame] < enter,
            True,
            np.where(distance[frame] > exit_, False, contact[frame - 1]),
        )
    return contact


def _bits(value: np.ndarray) -> str:
    return f"L={int(value[0])} R={int(value[1])}"


def _color(value: np.ndarray) -> tuple[int, int, int]:
    return (80, 230, 100) if bool(np.all(value)) else (245, 90, 80)


def render_comparison(
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
    from PIL import Image, ImageDraw

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    training_spec = load_robot_training_spec(robot_config)
    config = OmegaConf.load(box_config)
    qpos_frames = _motion_qpos(motion)
    renderer = MujocoQposRenderer(
        Path(training_spec.robot.xml_path),
        render_size=render_size,
        expected_qpos_size=qpos_frames.shape[1],
        box_half_size=tuple(float(value) for value in config.half_size),
        box_geom_pos=tuple(float(value) for value in config.geom_pos),
        box_geom_quat_wxyz=tuple(float(value) for value in config.geom_quat_wxyz),
        box_collisions=True,
    )
    hand_body_ids = [renderer.model.body(name).id for name in training_spec.robot.hands]
    hand_geom_ids = [[renderer.model.geom(f"{side}_hand_collision").id] for side in ("left", "right")]
    box_geom_id = renderer.model.geom("inference_carry_box_geom").id
    hand_pos = []
    hand_quat = []
    mujoco_distance = []
    try:
        for qpos in qpos_frames:
            renderer.data.qpos[:] = qpos
            renderer.data.qvel[:] = 0.0
            mujoco.mj_forward(renderer.model, renderer.data)
            hand_pos.append(renderer.data.xpos[hand_body_ids].copy())
            hand_quat.append(renderer.data.xquat[hand_body_ids][:, [1, 2, 3, 0]].copy())
            mujoco_distance.append(
                [
                    min(
                        mujoco.mj_geomDistance(
                            renderer.model,
                            renderer.data,
                            geom_id,
                            box_geom_id,
                            1.0,
                            np.zeros(6),
                        )
                        for geom_id in side_geom_ids
                    )
                    for side_geom_ids in hand_geom_ids
                ]
            )
        hand_points = compute_hand_contact_points(
            torch.tensor(np.asarray(hand_pos), dtype=torch.float32),
            torch.tensor(np.asarray(hand_quat), dtype=torch.float32),
            torch.tensor(config.hand_contact_offsets, dtype=torch.float32),
        )
        geom_quat_wxyz = torch.tensor(config.geom_quat_wxyz, dtype=torch.float32)
        common = (
            hand_points,
            torch.tensor(motion["object_pos"], dtype=torch.float32),
            torch.tensor(motion["object_quat"], dtype=torch.float32),
            torch.tensor(config.half_size, dtype=torch.float32),
            torch.tensor(config.geom_pos, dtype=torch.float32),
            geom_quat_wxyz[[1, 2, 3, 0]],
        )
        proxy_distance = compute_hand_box_surface_distance(
            *common,
            hand_contact_radius=float(config.hand_contact_radius),
        ).numpy()
        proxy_contact = infer_hand_box_contact(
            *common,
            hand_contact_radius=float(config.hand_contact_radius),
            enter_distance=DIAGNOSTIC_CONTACT_ENTER_DISTANCE,
            exit_distance=DIAGNOSTIC_CONTACT_EXIT_DISTANCE,
        ).numpy().astype(bool)
        capsule_distance = compute_hand_capsule_box_surface_distance(
            torch.tensor(np.asarray(hand_pos), dtype=torch.float32),
            torch.tensor(np.asarray(hand_quat), dtype=torch.float32),
            torch.tensor(config.hand_capsule_segments, dtype=torch.float32),
            torch.tensor(motion["object_pos"], dtype=torch.float32),
            torch.tensor(motion["object_quat"], dtype=torch.float32),
            torch.tensor(config.half_size, dtype=torch.float32),
            torch.tensor(config.geom_pos, dtype=torch.float32),
            geom_quat_wxyz[[1, 2, 3, 0]],
            hand_contact_radius=float(config.hand_contact_radius),
        ).numpy()
        mujoco_distance = np.asarray(mujoco_distance)
        strict_contact = mujoco_distance <= 0.0
        mujoco_contact = _hysteresis(
            mujoco_distance,
            DIAGNOSTIC_CONTACT_ENTER_DISTANCE,
            DIAGNOSTIC_CONTACT_EXIT_DISTANCE,
        )
        frames = []
        fps = float(motion["fps"])
        for frame_index, qpos in enumerate(qpos_frames):
            image = Image.fromarray(renderer.render_qpos(qpos))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 116), fill=(0, 0, 0))
            draw.text((10, 6), f"{key}  t={frame_index / fps:.2f}s frame={frame_index}", fill="white")
            rows = (
                (
                    f"PROXY {DIAGNOSTIC_CONTACT_ENTER_DISTANCE * 100:g}/{DIAGNOSTIC_CONTACT_EXIT_DISTANCE * 100:g}cm",
                    proxy_contact[frame_index],
                    proxy_distance[frame_index],
                ),
                ("MUJOCO STRICT", strict_contact[frame_index], mujoco_distance[frame_index]),
                (
                    f"MUJOCO {DIAGNOSTIC_CONTACT_ENTER_DISTANCE * 100:g}/{DIAGNOSTIC_CONTACT_EXIT_DISTANCE * 100:g}cm",
                    mujoco_contact[frame_index],
                    mujoco_distance[frame_index],
                ),
            )
            for row, (name, contact, distance) in enumerate(rows):
                y = 29 + 26 * row
                draw.text((10, y), name, fill=(180, 180, 180))
                draw.text((155, y), _bits(contact), fill=_color(contact))
                draw.text(
                    (280, y),
                    f"gap L={distance[0] * 100:.2f} R={distance[1] * 100:.2f} cm",
                    fill="white",
                )
            draw.text((10, 103), "negative MuJoCo gap = actual penetration/contact", fill=(255, 210, 90))
            frames.append(np.asarray(image))
    finally:
        renderer.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(output, frames, fps=float(motion["fps"]))
    print(f"motion={key} output={output.resolve()}")
    for name, contact in (
        ("proxy", proxy_contact),
        ("mujoco_strict", strict_contact),
        ("mujoco_hysteresis", mujoco_contact),
    ):
        state = contact[:, 0].astype(int) + 2 * contact[:, 1].astype(int)
        print(f"  {name} none/left/right/both={np.bincount(state, minlength=4).tolist()}")
    np.testing.assert_allclose(capsule_distance, mujoco_distance, atol=2.0e-5, rtol=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/data/g1_carry_box.yaml"))
    parser.add_argument("--motions", required=True, help="Comma-separated motion indices or keys.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/contact_labels/comparisons"))
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument("--render-size", type=int, default=720)
    args = parser.parse_args()

    prepared = prepare_motion_manifest(args.manifest)
    data = joblib.load(next(iter(prepared.inference_paths.values())))
    for selector in (item.strip() for item in args.motions.split(",")):
        key, motion = _select_motion(data, selector)
        safe_key = key.replace("/", "_")
        render_comparison(
            key,
            motion,
            args.output_dir / f"motion_{selector}_{safe_key}_contact_compare.mp4",
            robot_config=args.robot_config,
            box_config=args.box_config,
            render_size=args.render_size,
        )


if __name__ == "__main__":
    main()
