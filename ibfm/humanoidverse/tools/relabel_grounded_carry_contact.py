"""Relabel a grounded G1 Carry cache from MuJoCo hand-box distances."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from omegaconf import OmegaConf

from humanoidverse.mjlab_inference_utils import MujocoQposRenderer
from humanoidverse.tools.carry_data_inspect import _motion_qpos, _select_motion
from humanoidverse.utils.motion_data.adapters import dump_ufo_pkl
from humanoidverse.utils.motion_data.clip import clip_ufo_motion_dict
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import load_robot_training_spec


STRICT_CONTACT_METHOD = "mujoco_strict_capsule_box_grounded_v1"


def contact_hysteresis(distance: np.ndarray, *, enter: float, exit_: float) -> np.ndarray:
    if distance.ndim != 2 or distance.shape[1] != 2:
        raise ValueError(f"distance must have shape [T, 2], got {distance.shape}")
    if not 0.0 <= enter <= exit_:
        raise ValueError(f"Expected 0 <= enter <= exit, got enter={enter}, exit={exit_}")
    contact = np.zeros(distance.shape, dtype=bool)
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


class GroundedCarryContactRelabeler:
    def __init__(
        self,
        robot_config: Path,
        box_config: Path,
        *,
        render_size: int = 600,
        contact_enter_distance: float = 0.0,
        contact_exit_distance: float = 0.0,
    ):
        if not 0.0 <= contact_enter_distance <= contact_exit_distance:
            raise ValueError(
                "Expected 0 <= contact_enter_distance <= contact_exit_distance, got "
                f"{contact_enter_distance}, {contact_exit_distance}"
            )
        self.contact_enter_distance = float(contact_enter_distance)
        self.contact_exit_distance = float(contact_exit_distance)
        self.strict_contact = self.contact_enter_distance == self.contact_exit_distance == 0.0
        self.contact_method = (
            STRICT_CONTACT_METHOD
            if self.strict_contact
            else (
                "mujoco_capsule_box_grounded_"
                f"{self.contact_enter_distance:g}m_{self.contact_exit_distance:g}m_hysteresis_v1"
            )
        )
        training_spec = load_robot_training_spec(robot_config)
        self.box_config = OmegaConf.load(box_config)
        self.renderer = MujocoQposRenderer(
            Path(training_spec.robot.xml_path),
            render_size=render_size,
            expected_qpos_size=7 + len(training_spec.robot.control_joint_names) + 7,
            box_half_size=tuple(float(value) for value in self.box_config.half_size),
            box_geom_pos=tuple(float(value) for value in self.box_config.geom_pos),
            box_geom_quat_wxyz=tuple(float(value) for value in self.box_config.geom_quat_wxyz),
            box_collisions=True,
        )
        self.hand_geom_ids = np.asarray(
            [
                self.renderer.model.geom("left_hand_collision").id,
                self.renderer.model.geom("right_hand_collision").id,
            ],
            dtype=np.int32,
        )
        self.box_geom_id = self.renderer.model.geom("inference_carry_box_geom").id

    def close(self) -> None:
        self.renderer.close()

    def relabel_motion(self, motion: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        qpos_frames = _motion_qpos(motion)
        distances = np.empty((len(qpos_frames), 2), dtype=np.float32)
        for frame_index, qpos in enumerate(qpos_frames):
            self.renderer.data.qpos[:] = qpos
            self.renderer.data.qvel[:] = 0.0
            mujoco.mj_forward(self.renderer.model, self.renderer.data)
            for hand_index, hand_geom_id in enumerate(self.hand_geom_ids):
                distances[frame_index, hand_index] = mujoco.mj_geomDistance(
                    self.renderer.model,
                    self.renderer.data,
                    int(hand_geom_id),
                    self.box_geom_id,
                    1.0,
                    np.zeros(6, dtype=np.float64),
                )

        contact = (
            distances <= 0.0
            if self.strict_contact
            else contact_hysteresis(
                distances,
                enter=self.contact_enter_distance,
                exit_=self.contact_exit_distance,
            )
        )
        result = deepcopy(motion)
        result["hand_box_surface_distance"] = distances
        result["hand_box_contact"] = contact.astype(np.float32)
        metadata = dict(result.get("metadata") or {})
        metadata["hand_box_contact"] = {
            "method": self.contact_method,
            "criterion": (
                "mj_geomDistance <= 0"
                if self.strict_contact
                else (
                    f"enter < {self.contact_enter_distance:g} m, "
                    f"exit > {self.contact_exit_distance:g} m, otherwise retain previous state"
                )
            ),
            "enter_distance_m": self.contact_enter_distance,
            "exit_distance_m": self.contact_exit_distance,
            "robot_geoms": ["left_hand_collision", "right_hand_collision"],
            "box_geom": "inference_carry_box_geom",
        }
        result["metadata"] = metadata
        return result, qpos_frames

    def render_video(self, key: str, motion: dict[str, Any], output: Path) -> None:
        import imageio_ffmpeg
        import mediapy as media
        from PIL import Image, ImageDraw

        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        qpos_frames = _motion_qpos(motion)
        contact = np.asarray(motion["hand_box_contact"], dtype=bool)
        distance = np.asarray(motion["hand_box_surface_distance"], dtype=np.float32)
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        frames = []
        for frame_index, qpos in enumerate(qpos_frames):
            image = Image.fromarray(self.renderer.render_qpos(qpos))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 85), fill=(0, 0, 0))
            draw.text(
                (10, 7),
                f"{key}  t={frame_index / fps:.2f}s frame={frame_index}",
                fill=(255, 255, 255),
            )
            for hand_index, (name, x) in enumerate((("LEFT", 10), ("RIGHT", image.width // 2))):
                active = bool(contact[frame_index, hand_index])
                color = (80, 230, 100) if active else (245, 90, 80)
                mode = "STRICT" if self.strict_contact else "HYSTERESIS"
                draw.text((x, 31), f"{name} {mode} CONTACT={int(active)}", fill=color)
                draw.text(
                    (x, 56),
                    f"MuJoCo gap={distance[frame_index, hand_index] * 100:.2f} cm",
                    fill=(230, 230, 230),
                )
            frames.append(np.asarray(image))
        output.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(output, frames, fps=fps)


def relabel_dataset(
    motions: dict[str, dict[str, Any]],
    relabeler: GroundedCarryContactRelabeler,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validated = validate_ufo_motion_dict(motions, "grounded_carry_contact_input")
    output: dict[str, dict[str, Any]] = {}
    state_counts = np.zeros(4, dtype=np.int64)
    minimum_gaps = np.full(2, np.inf, dtype=np.float64)
    maximum_gaps = np.full(2, -np.inf, dtype=np.float64)
    for index, (key, motion) in enumerate(validated.items(), start=1):
        labeled, _qpos = relabeler.relabel_motion(motion)
        output[key] = labeled
        contact = np.asarray(labeled["hand_box_contact"], dtype=np.int64)
        distance = np.asarray(labeled["hand_box_surface_distance"], dtype=np.float64)
        states = contact[:, 0] + 2 * contact[:, 1]
        state_counts += np.bincount(states, minlength=4)
        minimum_gaps = np.minimum(minimum_gaps, distance.min(axis=0))
        maximum_gaps = np.maximum(maximum_gaps, distance.max(axis=0))
        if index == 1 or index % 100 == 0 or index == len(validated):
            print(f"[grounded-contact] {index}/{len(validated)} {key}", flush=True)
    validate_ufo_motion_dict(output, "grounded_carry_contact_output")
    return output, {
        "method": relabeler.contact_method,
        "enter_distance_m": relabeler.contact_enter_distance,
        "exit_distance_m": relabeler.contact_exit_distance,
        "motion_count": len(output),
        "frame_count": int(state_counts.sum()),
        "none_left_right_both": state_counts.tolist(),
        "minimum_gap_m_left_right": minimum_gaps.tolist(),
        "maximum_gap_m_left_right": maximum_gaps.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("cache/motion_data/g1_carry_box_omniretarget_no_foot")
    parser.add_argument(
        "--input-full",
        type=Path,
        default=base / "g1_carry_box_omniretarget_no_foot_full_grounded_ufo.pkl",
    )
    parser.add_argument(
        "--output-full",
        type=Path,
        default=base / "g1_carry_box_omniretarget_no_foot_full_grounded_mujoco_strict_ufo.pkl",
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        default=base / "g1_carry_box_omniretarget_no_foot_train_near10s_grounded_mujoco_strict_ufo.pkl",
    )
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/contact_labels/grounded_mujoco_strict_1321/summary.json"),
    )
    parser.add_argument("--render-motions", default="", help="Comma-separated keys or zero-based indices.")
    parser.add_argument(
        "--video-output-dir",
        type=Path,
        default=Path("analysis/contact_labels/grounded_mujoco_strict_1321"),
    )
    parser.add_argument("--render-size", type=int, default=600)
    parser.add_argument("--contact-enter-distance", type=float, default=0.0)
    parser.add_argument("--contact-exit-distance", type=float, default=0.0)
    parser.add_argument("--output-tag", default="mujoco_strict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motions = joblib.load(args.input_full.expanduser().resolve())
    relabeler = GroundedCarryContactRelabeler(
        args.robot_config.expanduser().resolve(),
        args.box_config.expanduser().resolve(),
        render_size=args.render_size,
        contact_enter_distance=args.contact_enter_distance,
        contact_exit_distance=args.contact_exit_distance,
    )
    try:
        labeled, summary = relabel_dataset(motions, relabeler)
        train = clip_ufo_motion_dict(
            labeled,
            clip_seconds=10.0,
            stride_seconds=10.0,
            keep_short=True,
            min_clip_seconds=1.0,
            source_name="g1_carry_box_grounded_mujoco_strict_train",
        )
        dump_ufo_pkl(labeled, args.output_full.expanduser(), "g1_carry_box_grounded_mujoco_strict_full")
        dump_ufo_pkl(train, args.output_train.expanduser(), "g1_carry_box_grounded_mujoco_strict_train")
        summary["train_motion_count"] = len(train)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for selector in (value.strip() for value in args.render_motions.split(",") if value.strip()):
            key, motion = _select_motion(labeled, selector)
            safe_key = key.replace("/", "_")
            output = args.video_output_dir / f"{safe_key}_grounded_{args.output_tag}_contact.mp4"
            relabeler.render_video(key, motion, output)
            print(f"[contact-video] {output.resolve()}")
    finally:
        relabeler.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
