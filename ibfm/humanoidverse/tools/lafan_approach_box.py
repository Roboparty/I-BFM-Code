"""Build object-aware LAFAN approach motions with a static box ahead."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from humanoidverse.utils.motion_data.adapters import dump_ufo_pkl
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict


DEFAULT_MOTION_GROUPS = (
    "walk1",
    "walk2",
    "walk3",
    "walk4",
    "run1",
    "run2",
    "sprint1",
    "fallAndGetUp1",
    "fallAndGetUp2",
    "fallAndGetUp3",
)


@dataclass(frozen=True)
class ApproachBuildConfig:
    initial_distance_min: float = 2.0
    initial_distance_max: float = 5.0
    final_distance_min: float = 1.0
    final_distance_max: float = 1.5
    smoothing_seconds: float = 0.5
    min_decreasing_fraction: float = 0.55
    max_checkpoint_increase: float = 0.25
    min_final_root_height: float = 0.5
    min_final_upright: float = 0.65
    max_recovery_root_height: float = 0.35
    motion_groups: tuple[str, ...] = DEFAULT_MOTION_GROUPS


def _motion_group(key: str, motion: dict[str, Any]) -> str:
    name = str(motion.get("motion_name", key))
    name = re.sub(r"_clip\d+$", "", name)
    return re.sub(r"_subject\d+$", "", name)


def _deterministic_final_distance(key: str, config: ApproachBuildConfig) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return config.final_distance_min + unit * (config.final_distance_max - config.final_distance_min)


def _smoothed_distance(distance: np.ndarray, window_frames: int) -> np.ndarray:
    window_frames = max(1, int(window_frames))
    if window_frames == 1:
        return distance.copy()
    left = window_frames // 2
    right = window_frames - 1 - left
    padded = np.pad(distance, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window_frames, dtype=np.float64) / window_frames, mode="valid")


def _upright_static_box_pose(box_config: dict[str, Any]) -> tuple[np.ndarray, float]:
    geom_wxyz = np.asarray(box_config["geom_quat_wxyz"], dtype=np.float64)
    geom_xyzw = geom_wxyz[[1, 2, 3, 0]]
    marker_rotation = Rotation.from_quat(geom_xyzw).inv()
    # Match the height used by the current Carry reset path. The marker-to-geom
    # transform remains encoded in carry_box.yaml and is not baked into XY.
    marker_height = float(box_config["ground_center_height"])
    return marker_rotation.as_quat().astype(np.float32), marker_height


def _approach_record(
    key: str,
    motion: dict[str, Any],
    *,
    motion_group: str,
    box_quat_xyzw: np.ndarray,
    box_height: float,
    config: ApproachBuildConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    root_quat = np.asarray(motion.get("root_rot", motion.get("root_quat")), dtype=np.float64)
    if root_quat.shape != (root_pos.shape[0], 4):
        raise ValueError(f"motion {key!r} requires root_rot/root_quat with shape {(root_pos.shape[0], 4)}")
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    root_rotation = Rotation.from_quat(root_quat)
    final_forward = root_rotation[-1].apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))[:2]
    forward_norm = float(np.linalg.norm(final_forward))
    if forward_norm <= 1.0e-8:
        return None, {"reason": "degenerate_final_forward"}
    final_forward /= forward_norm

    final_distance = _deterministic_final_distance(key, config)
    object_position = np.array(
        [
            root_pos[-1, 0] + final_distance * final_forward[0],
            root_pos[-1, 1] + final_distance * final_forward[1],
            box_height,
        ],
        dtype=np.float32,
    )
    planar_distance = np.linalg.norm(object_position[None, :2] - root_pos[:, :2], axis=1)
    window_frames = max(1, round(config.smoothing_seconds * fps))
    smooth_distance = _smoothed_distance(planar_distance, window_frames)
    smooth_delta = np.diff(smooth_distance)
    checkpoint_delta = np.diff(smooth_distance[::window_frames])
    decreasing_fraction = float(np.mean(smooth_delta <= 0.0)) if smooth_delta.size else 1.0
    max_checkpoint_increase = float(np.max(checkpoint_delta)) if checkpoint_delta.size else 0.0
    final_upright = float(root_rotation[-1].apply(np.array([0.0, 0.0, 1.0]))[2])
    initial_distance = float(planar_distance[0])

    diagnostics = {
        "initial_distance": initial_distance,
        "final_distance": float(planar_distance[-1]),
        "decreasing_fraction": decreasing_fraction,
        "max_checkpoint_increase": max_checkpoint_increase,
        "final_root_height": float(root_pos[-1, 2]),
        "minimum_root_height": float(np.min(root_pos[:, 2])),
        "final_upright": final_upright,
    }
    contains_recovery = (
        not motion_group.startswith("fallAndGetUp")
        or float(np.min(root_pos[:, 2])) <= config.max_recovery_root_height
    )
    accepted = (
        config.initial_distance_min <= initial_distance <= config.initial_distance_max
        and decreasing_fraction >= config.min_decreasing_fraction
        and max_checkpoint_increase <= config.max_checkpoint_increase
        and float(root_pos[-1, 2]) >= config.min_final_root_height
        and final_upright >= config.min_final_upright
        and contains_recovery
    )
    if not accepted:
        diagnostics["reason"] = "approach_filter"
        return None, diagnostics

    num_frames = root_pos.shape[0]
    result = dict(motion)
    result["object_pos"] = np.repeat(object_position[None], num_frames, axis=0).astype(np.float32)
    result["object_quat"] = np.repeat(box_quat_xyzw[None], num_frames, axis=0).astype(np.float32)
    result["object_lin_vel"] = np.zeros((num_frames, 3), dtype=np.float32)
    result["object_ang_vel"] = np.zeros((num_frames, 3), dtype=np.float32)
    result["object_target_pos"] = object_position.copy()
    result["object_target_quat"] = box_quat_xyzw.copy()
    result["hand_box_contact"] = np.zeros((num_frames, 2), dtype=np.float32)
    result["carry_evaluation_eligible"] = False
    result["source"] = "g1_lafan_approach_box"
    metadata = dict(result.get("metadata") or {})
    metadata["approach_box"] = diagnostics
    metadata["approach_box"]["object_position"] = object_position.tolist()
    result["metadata"] = metadata
    return result, diagnostics


def build_lafan_approach_dataset(
    motions: dict[str, dict[str, Any]],
    box_config: dict[str, Any],
    config: ApproachBuildConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validate_ufo_motion_dict(motions, "lafan_approach_input")
    box_quat, box_height = _upright_static_box_pose(box_config)
    allowed = set(config.motion_groups)
    output: dict[str, dict[str, Any]] = {}
    accepted_groups: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    diagnostics: list[dict[str, Any]] = []

    for key, motion in motions.items():
        group = _motion_group(str(key), motion)
        if group not in allowed:
            rejected_reasons["motion_group"] += 1
            continue
        record, row = _approach_record(
            str(key),
            motion,
            motion_group=group,
            box_quat_xyzw=box_quat,
            box_height=box_height,
            config=config,
        )
        row = {"motion_key": str(key), "motion_group": group, **row}
        if record is None:
            rejected_reasons[str(row.get("reason", "unknown"))] += 1
            continue
        output_key = f"{key}_approach_box"
        output[output_key] = record
        accepted_groups[group] += 1
        diagnostics.append(row)

    if not output:
        raise ValueError("LAFAN approach filtering produced no motions")
    validate_ufo_motion_dict(output, "g1_lafan_approach_box")
    summary = {
        "input_motion_count": len(motions),
        "output_motion_count": len(output),
        "motion_groups": dict(sorted(accepted_groups.items())),
        "rejected_reasons": dict(sorted(rejected_reasons.items())),
        "config": {
            name: (list(value) if isinstance(value, tuple) else value)
            for name, value in vars(config).items()
        },
        "box_marker_quat_xyzw": box_quat.tolist(),
        "box_marker_height": box_height,
        "initial_distance_range": [
            min(row["initial_distance"] for row in diagnostics),
            max(row["initial_distance"] for row in diagnostics),
        ],
        "final_distance_range": [
            min(row["final_distance"] for row in diagnostics),
            max(row["final_distance"] for row in diagnostics),
        ],
        "motions": diagnostics,
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("humanoidverse/data/lafan_29dof_10s-clipped.pkl"))
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("humanoidverse/data/lafan_29dof_10s-clipped_approach_box.pkl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/lafan_approach_box/summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motions = joblib.load(args.input.expanduser().resolve())
    box_config = OmegaConf.to_container(OmegaConf.load(args.box_config.expanduser().resolve()), resolve=True)
    if not isinstance(motions, dict) or not isinstance(box_config, dict):
        raise TypeError("input motion pkl and box config must both contain mappings")
    config = ApproachBuildConfig()
    output, summary = build_lafan_approach_dataset(motions, box_config, config)
    output_path = dump_ufo_pkl(output, args.output.expanduser(), "g1_lafan_approach_box")
    summary_path = args.summary.expanduser()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(output)} approach motions to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Motion groups: {summary['motion_groups']}")
    print(f"Initial distance range: {summary['initial_distance_range']}")
    print(f"Final distance range: {summary['final_distance_range']}")


if __name__ == "__main__":
    main()
