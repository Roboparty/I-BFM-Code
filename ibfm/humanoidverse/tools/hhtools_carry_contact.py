"""Create portable HHTools Carry scenes with UFO hand-contact sidecars."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from humanoidverse.envs.carry_observations import infer_carry_phase_contact
from humanoidverse.utils.motion_data.carry_box import _matrix, _read_table, _robot_csv, _time
from humanoidverse.utils.robot_spec import load_robot_spec

CONTACT_FILENAME = "hand_box_contact.csv"
CONTACT_METHOD = "rp1_phase_lift_with_mujoco_wrist_gap_v1"


def _quat_rotate_xyzw(quat: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    xyz = quat[:3]
    w = quat[3]
    cross = np.cross(np.broadcast_to(xyz, vectors.shape), vectors)
    return vectors + 2.0 * (w * cross + np.cross(np.broadcast_to(xyz, vectors.shape), cross))


def _align_box_track_to_ground(scene: Path, config: OmegaConf, clearance: float) -> float:
    """Raise an entire object track so its first physical box bottom clears the ground."""

    object_path = scene / "object_0_largebox.csv"
    fields, rows = _read_table(object_path)
    required = {"pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Object CSV {object_path} is missing columns: {missing}")

    half_size = np.asarray(config.half_size, dtype=np.float64)
    corners = np.asarray(
        [
            [x, y, z]
            for x in (-half_size[0], half_size[0])
            for y in (-half_size[1], half_size[1])
            for z in (-half_size[2], half_size[2])
        ],
        dtype=np.float64,
    )
    geom_quat = np.asarray(config.geom_quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    local_corners = _quat_rotate_xyzw(geom_quat, corners) + np.asarray(config.geom_pos, dtype=np.float64)
    first = rows[0]
    object_quat = np.asarray(
        [first["quat_x"], first["quat_y"], first["quat_z"], first["quat_w"]], dtype=np.float64
    )
    object_pos = np.asarray([first["pos_x"], first["pos_y"], first["pos_z"]], dtype=np.float64)
    first_bottom = float((_quat_rotate_xyzw(object_quat, local_corners) + object_pos).min(axis=0)[2])
    target_bottom = float(config.ground_surface_height) + float(clearance)
    correction = max(0.0, target_bottom - first_bottom)
    if correction <= 0.0:
        return 0.0

    comments = [line for line in object_path.read_text().splitlines() if line.lstrip().startswith("#")]
    with object_path.open("w", newline="", encoding="utf-8") as stream:
        for line in comments:
            stream.write(line + "\n")
        stream.write(f"# first_frame_ground_correction_z_m: {correction:.9f}\n")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["pos_z"] = f"{float(row['pos_z']) + correction:.9f}"
            writer.writerow(row)
    return correction


class Rp1ContactLabeler:
    def __init__(self, robot_config: Path, box_config: Path):
        self.robot = load_robot_spec(robot_config)
        config = OmegaConf.load(box_config)
        self.config = config
        spec = mujoco.MjSpec.from_file(str(self.robot.xml_path))
        box_body = spec.worldbody.add_body(name="inference_carry_box")
        box_body.add_freejoint(name="inference_carry_box_joint")
        box_body.add_geom(
            name="inference_carry_box_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=tuple(float(value) for value in config.half_size),
            pos=tuple(float(value) for value in config.geom_pos),
            quat=tuple(float(value) for value in config.geom_quat_wxyz),
            contype=1,
            conaffinity=1,
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        model = self.model
        if model.nq != self.robot.nq + 7:
            raise ValueError(f"Expected RP1+box nq={self.robot.nq + 7}, got {model.nq}")
        self.root_qpos = int(model.joint(self.robot.free_joint).qposadr[0])
        self.box_qpos = int(model.joint("inference_carry_box_joint").qposadr[0])
        self.joint_qpos = np.asarray(
            [int(model.joint(name).qposadr[0]) for name in self.robot.control_joint_names],
            dtype=np.int32,
        )
        self.hand_geoms = np.asarray(
            [
                model.geom("left_wrist_pitch_collision").id,
                model.geom("right_wrist_pitch_collision").id,
            ],
            dtype=np.int32,
        )
        self.box_geom = int(model.geom("inference_carry_box_geom").id)
        if any(model.geom_type[index] != mujoco.mjtGeom.mjGEOM_CAPSULE for index in self.hand_geoms):
            raise ValueError("RP1 hand collision geoms must be capsules")

    def close(self) -> None:
        pass

    def label(self, scene: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        robot_path = _robot_csv(scene)
        object_path = scene / "object_0_largebox.csv"
        _robot_fields, robot_rows = _read_table(robot_path)
        _object_fields, object_rows = _read_table(object_path)
        robot_time = _time(robot_rows, robot_path)
        object_time = _time(object_rows, object_path)
        if robot_time.shape != object_time.shape or not np.allclose(
            robot_time, object_time, atol=1.0e-6, rtol=0.0
        ):
            raise ValueError(f"Robot/object timestamps do not match in {scene}")

        root_pos = _matrix(robot_rows, ["root_x", "root_y", "root_z"], robot_path)
        root_quat = _matrix(robot_rows, ["root_qx", "root_qy", "root_qz", "root_qw"], robot_path)
        dof = _matrix(robot_rows, [f"dof_{name}" for name in self.robot.control_joint_names], robot_path)
        object_pos = _matrix(object_rows, ["pos_x", "pos_y", "pos_z"], object_path)
        object_quat = _matrix(object_rows, ["quat_x", "quat_y", "quat_z", "quat_w"], object_path)

        distances = np.empty((len(robot_time), 2), dtype=np.float32)
        qpos = self.model.qpos0.copy()
        for frame in range(len(robot_time)):
            qpos[self.root_qpos : self.root_qpos + 3] = root_pos[frame]
            qpos[self.root_qpos + 3 : self.root_qpos + 7] = root_quat[frame, [3, 0, 1, 2]]
            qpos[self.joint_qpos] = dof[frame]
            qpos[self.box_qpos : self.box_qpos + 3] = object_pos[frame]
            qpos[self.box_qpos + 3 : self.box_qpos + 7] = object_quat[frame, [3, 0, 1, 2]]
            self.data.qpos[:] = qpos
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            for hand, geom_id in enumerate(self.hand_geoms):
                distances[frame, hand] = mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    int(geom_id),
                    self.box_geom,
                    1.0,
                    np.zeros(6, dtype=np.float64),
                )

        geom_quat_wxyz = torch.tensor(self.config.geom_quat_wxyz, dtype=torch.float32)
        phase_contact = infer_carry_phase_contact(
            torch.from_numpy(object_pos),
            torch.from_numpy(object_quat),
            torch.ones(len(object_pos), dtype=torch.bool),
            torch.tensor(self.config.half_size, dtype=torch.float32),
            torch.tensor(self.config.geom_pos, dtype=torch.float32),
            geom_quat_wxyz[[1, 2, 3, 0]],
            ground_surface_height=float(self.config.ground_surface_height),
            lift_clearance=float(self.config.phase_lift_clearance),
        ).numpy().astype(np.uint8)
        return robot_time, phase_contact, distances


def _write_contact(path: Path, times: np.ndarray, contact: np.ndarray, distances: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write(f"# format: ufo_hand_box_contact_v1\n# method: {CONTACT_METHOD}\n")
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "left_hand_box_contact",
                "right_hand_box_contact",
                "left_hand_box_surface_distance",
                "right_hand_box_surface_distance",
            ]
        )
        for time, labels, gaps in zip(times, contact, distances):
            writer.writerow(
                [
                    f"{time:.6f}",
                    int(labels[0]),
                    int(labels[1]),
                    f"{gaps[0]:.8f}",
                    f"{gaps[1]:.8f}",
                ]
            )


def generate(
    source_root: Path,
    output_root: Path,
    robot_config: Path,
    box_config: Path,
    *,
    align_box_ground: bool = False,
    ground_clearance: float = 0.0,
) -> dict[str, object]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_root == output_root:
        raise ValueError("Output directory must differ from source directory")
    scenes = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not scenes:
        raise FileNotFoundError(f"No scene directories found in {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    labeler = Rp1ContactLabeler(robot_config, box_config)
    records: list[dict[str, object]] = []
    try:
        for index, scene in enumerate(scenes, start=1):
            robot_path = _robot_csv(scene)
            required = [robot_path, scene / "object_0_largebox.csv", scene / "largebox_cleaned_simplified.obj"]
            if not all(path.is_file() for path in required):
                raise FileNotFoundError(f"Incomplete HHTools scene: {scene}")
            destination = output_root / scene.name
            destination.mkdir(exist_ok=True)
            for source in required:
                shutil.copy2(source, destination / source.name)
            ground_correction = (
                _align_box_track_to_ground(destination, labeler.config, ground_clearance)
                if align_box_ground
                else 0.0
            )
            times, contact, distances = labeler.label(destination)
            _write_contact(destination / CONTACT_FILENAME, times, contact, distances)
            records.append(
                {
                    "motion": scene.name,
                    "frames": len(times),
                    "contact_frames": contact.sum(axis=0).astype(int).tolist(),
                    "minimum_gap_cm": (distances.min(axis=0) * 100.0).tolist(),
                    "ground_correction_z_m": ground_correction,
                }
            )
            if index == 1 or index % 25 == 0 or index == len(scenes):
                print(f"[rp1-contact] {index}/{len(scenes)} {scene.name}", flush=True)
    finally:
        labeler.close()
    summary: dict[str, object] = {
        "method": CONTACT_METHOD,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "motions": len(records),
        "frames": sum(int(record["frames"]) for record in records),
        "ground_aligned": align_box_ground,
        "ground_clearance_m": ground_clearance,
        "corrected_motions": sum(float(record["ground_correction_z_m"]) > 0.0 for record in records),
        "records": records,
    }
    (output_root / "contact_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/rp1_28dof.yaml"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box_rp1.yaml"))
    parser.add_argument("--align-box-ground", action="store_true")
    parser.add_argument("--ground-clearance", type=float, default=0.0)
    args = parser.parse_args()
    if args.ground_clearance < 0.0:
        parser.error("--ground-clearance must be non-negative")
    summary = generate(
        args.source_root,
        args.output_root,
        args.robot_config,
        args.box_config,
        align_box_ground=args.align_box_ground,
        ground_clearance=args.ground_clearance,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
