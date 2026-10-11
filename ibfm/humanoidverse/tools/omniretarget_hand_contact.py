"""Generate and visualize G1 hand-box labels for OmniRetarget carry NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from humanoidverse.mjlab_inference_utils import MujocoQposRenderer
from humanoidverse.utils.motion_data.omniretarget_carry import omniretarget_qpos_to_mujoco
from humanoidverse.utils.robot_spec import load_robot_training_spec


CONTACT_ENTER_DISTANCE = 0.045
CONTACT_EXIT_DISTANCE = 0.06
CONTACT_METHOD = "mujoco_capsule_box_4p5cm_6cm_hysteresis"


def contact_hysteresis(distance: np.ndarray, *, enter: float, exit_: float) -> np.ndarray:
    distance = np.asarray(distance)
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


class HandBoxLabeler:
    def __init__(self, robot_config: Path, box_config: Path, *, render_size: int):
        training_spec = load_robot_training_spec(robot_config)
        config = OmegaConf.load(box_config)
        self.renderer = MujocoQposRenderer(
            Path(training_spec.robot.xml_path),
            render_size=render_size,
            expected_qpos_size=43,
            box_half_size=tuple(float(value) for value in config.half_size),
            box_geom_pos=tuple(float(value) for value in config.geom_pos),
            box_geom_quat_wxyz=tuple(float(value) for value in config.geom_quat_wxyz),
            box_collisions=True,
        )
        self.hand_geom_ids = [
            self.renderer.model.geom("left_hand_collision").id,
            self.renderer.model.geom("right_hand_collision").id,
        ]
        self.box_geom_id = self.renderer.model.geom("inference_carry_box_geom").id
        for geom_id in self.hand_geom_ids:
            if self.renderer.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                raise ValueError(f"Expected hand geom {geom_id} to be a capsule")
        if self.renderer.model.geom_type[self.box_geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Expected inference carry-box geom to be a box")

    def close(self) -> None:
        self.renderer.close()

    def distances(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mujoco_qpos = omniretarget_qpos_to_mujoco(qpos)
        distance = np.empty((len(mujoco_qpos), 2), dtype=np.float32)
        for frame_index, frame_qpos in enumerate(mujoco_qpos):
            self.renderer.data.qpos[:] = frame_qpos
            self.renderer.data.qvel[:] = 0.0
            mujoco.mj_forward(self.renderer.model, self.renderer.data)
            for hand_index, hand_geom_id in enumerate(self.hand_geom_ids):
                distance[frame_index, hand_index] = mujoco.mj_geomDistance(
                    self.renderer.model,
                    self.renderer.data,
                    hand_geom_id,
                    self.box_geom_id,
                    1.0,
                    np.zeros(6, dtype=np.float64),
                )
        return mujoco_qpos, distance


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if "qpos" not in archive or "fps" not in archive:
            raise ValueError(f"Archive must contain qpos and fps: {path}")
        return {key: np.asarray(archive[key]) for key in archive.files}


def _write_labeled_archive(
    source: Path,
    output: Path,
    *,
    contact: np.ndarray,
    distance: np.ndarray,
) -> None:
    payload = _load_archive(source)
    payload.update(
        {
            "hand_box_contact": contact.astype(np.uint8),
            "hand_box_surface_distance": distance.astype(np.float32),
            "hand_box_contact_enter_distance": np.asarray(CONTACT_ENTER_DISTANCE, dtype=np.float32),
            "hand_box_contact_exit_distance": np.asarray(CONTACT_EXIT_DISTANCE, dtype=np.float32),
            "hand_box_contact_method": np.asarray(CONTACT_METHOD),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)


def _render_contact_video(
    labeler: HandBoxLabeler,
    archive_path: Path,
    output: Path,
) -> None:
    import imageio_ffmpeg
    import mediapy as media
    from PIL import Image, ImageDraw

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    archive = _load_archive(archive_path)
    qpos = np.asarray(archive["qpos"])
    contact = np.asarray(archive["hand_box_contact"], dtype=bool)
    distance = np.asarray(archive["hand_box_surface_distance"], dtype=np.float32)
    fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
    mujoco_qpos = omniretarget_qpos_to_mujoco(qpos)
    frames = []
    for frame_index, frame_qpos in enumerate(mujoco_qpos):
        image = Image.fromarray(labeler.renderer.render_qpos(frame_qpos))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 82), fill=(0, 0, 0))
        draw.text(
            (10, 7),
            f"{archive_path.stem}  t={frame_index / fps:.2f}s frame={frame_index}",
            fill=(255, 255, 255),
        )
        for hand_index, (name, x) in enumerate((("LEFT", 10), ("RIGHT", image.width // 2))):
            active = bool(contact[frame_index, hand_index])
            color = (80, 230, 100) if active else (245, 90, 80)
            draw.text((x, 31), f"{name} CONTACT={int(active)}", fill=color)
            draw.text(
                (x, 54),
                f"MuJoCo gap={distance[frame_index, hand_index] * 100:.2f} cm",
                fill=(230, 230, 230),
            )
        frames.append(np.asarray(image))
    output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(output, frames, fps=fps)


def generate_labels(
    source_dir: Path,
    output_dir: Path,
    *,
    robot_config: Path,
    box_config: Path,
    render_size: int,
    overwrite: bool,
) -> dict:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    paths = sorted(source_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ files found in {source_dir}")
    if source_dir == output_dir:
        raise ValueError("Output directory must differ from source directory")

    labeler = HandBoxLabeler(robot_config, box_config, render_size=render_size)
    per_motion = []
    total_states = np.zeros(4, dtype=np.int64)
    total_frames = 0
    try:
        for index, source in enumerate(paths, start=1):
            output = output_dir / source.name
            if output.exists() and not overwrite:
                archive = _load_archive(output)
                contact = np.asarray(archive["hand_box_contact"], dtype=bool)
                distance = np.asarray(archive["hand_box_surface_distance"], dtype=np.float32)
            else:
                archive = _load_archive(source)
                _mujoco_qpos, distance = labeler.distances(archive["qpos"])
                contact = contact_hysteresis(
                    distance,
                    enter=CONTACT_ENTER_DISTANCE,
                    exit_=CONTACT_EXIT_DISTANCE,
                )
                _write_labeled_archive(source, output, contact=contact, distance=distance)
            states = contact[:, 0].astype(np.int64) + 2 * contact[:, 1].astype(np.int64)
            counts = np.bincount(states, minlength=4)
            total_states += counts
            total_frames += len(contact)
            per_motion.append(
                {
                    "motion": source.stem,
                    "frames": int(len(contact)),
                    "none_left_right_both": counts.tolist(),
                    "minimum_gap_cm": (100.0 * distance.min(axis=0)).tolist(),
                }
            )
            if index == 1 or index % 100 == 0 or index == len(paths):
                print(f"[contact-labels] {index}/{len(paths)} {source.name}", flush=True)
    finally:
        labeler.close()

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "method": CONTACT_METHOD,
        "enter_distance_m": CONTACT_ENTER_DISTANCE,
        "exit_distance_m": CONTACT_EXIT_DISTANCE,
        "motions": len(paths),
        "frames": total_frames,
        "none_left_right_both": total_states.tolist(),
        "per_motion": per_motion,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_selected(
    data_dir: Path,
    selectors: list[str],
    output_dir: Path,
    *,
    robot_config: Path,
    box_config: Path,
    render_size: int,
) -> None:
    paths = sorted(data_dir.expanduser().resolve().glob("*.npz"))
    by_stem = {path.stem: path for path in paths}
    selected = []
    for selector in selectors:
        if selector in by_stem:
            selected.append(by_stem[selector])
        else:
            try:
                selected.append(paths[int(selector)])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Unknown motion selector {selector!r}") from exc
    labeler = HandBoxLabeler(robot_config, box_config, render_size=render_size)
    try:
        for path in selected:
            output = output_dir / f"{path.stem}_hand_contact.mp4"
            _render_contact_video(labeler, path, output)
            print(f"[contact-video] {output.resolve()}", flush=True)
    finally:
        labeler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument("--render-size", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render-motions", help="Comma-separated motion keys or sorted indices.")
    parser.add_argument("--video-output-dir", type=Path, help="Video directory; defaults to OUTPUT_DIR/contact_videos.")
    args = parser.parse_args()

    if args.source_dir is not None:
        summary = generate_labels(
            args.source_dir,
            args.output_dir,
            robot_config=args.robot_config,
            box_config=args.box_config,
            render_size=args.render_size,
            overwrite=bool(args.overwrite),
        )
        print(json.dumps({key: value for key, value in summary.items() if key != "per_motion"}, indent=2))
    if args.render_motions:
        render_selected(
            args.output_dir,
            [item.strip() for item in args.render_motions.split(",") if item.strip()],
            args.video_output_dir or args.output_dir / "contact_videos",
            robot_config=args.robot_config,
            box_config=args.box_config,
            render_size=args.render_size,
        )
    if args.source_dir is None and not args.render_motions:
        parser.error("Provide --source-dir to label data and/or --render-motions to create videos")


if __name__ == "__main__":
    main()
