"""Create collision-grounded Carry caches from an existing full UFO cache."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from humanoidverse.utils.motion_data.adapters import dump_ufo_pkl
from humanoidverse.utils.motion_data.clip import clip_ufo_motion_dict
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict


def box_bottom_height(
    object_pos: np.ndarray,
    object_quat_xyzw: np.ndarray,
    *,
    half_size: np.ndarray,
    geom_pos: np.ndarray,
    geom_quat_wxyz: np.ndarray,
) -> np.ndarray:
    marker_rotation = Rotation.from_quat(object_quat_xyzw)
    geom_quat_xyzw = geom_quat_wxyz[[1, 2, 3, 0]]
    world_geom_rotation = marker_rotation * Rotation.from_quat(geom_quat_xyzw)
    geom_center = object_pos + marker_rotation.apply(np.broadcast_to(geom_pos, object_pos.shape))
    vertical_extent = np.abs(world_geom_rotation.as_matrix()[:, 2, :]) @ half_size
    return geom_center[:, 2] - vertical_extent


def ground_carry_motion(
    motion: dict[str, Any],
    *,
    half_size: np.ndarray,
    geom_pos: np.ndarray,
    geom_quat_wxyz: np.ndarray,
    ground_height: float = 0.0,
    tolerance: float = 1.0e-6,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    result = deepcopy(motion)
    object_pos = np.asarray(motion["object_pos"], dtype=np.float64)
    object_quat = np.asarray(motion["object_quat"], dtype=np.float64)
    bottom_before = box_bottom_height(
        object_pos,
        object_quat,
        half_size=half_size,
        geom_pos=geom_pos,
        geom_quat_wxyz=geom_quat_wxyz,
    )
    correction = np.maximum(float(ground_height) - bottom_before, 0.0)
    corrected_pos = object_pos.copy()
    corrected_pos[:, 2] += correction
    result["object_pos"] = corrected_pos.astype(np.float32)

    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    corrected_velocity = np.asarray(motion["object_lin_vel"], dtype=np.float64).copy()
    corrected_velocity[:, 2] = np.gradient(corrected_pos[:, 2], 1.0 / fps, edge_order=1)
    result["object_lin_vel"] = corrected_velocity.astype(np.float32)

    target_pos = np.asarray(motion["object_target_pos"], dtype=np.float64).copy()
    if target_pos.shape != (3,):
        raise ValueError(f"object_target_pos must have shape (3,), got {target_pos.shape}")
    target_pos[2] += correction[-1]
    result["object_target_pos"] = target_pos.astype(np.float32)
    result["carry_evaluation_eligible"] = True
    # This diagnostic was computed against the ungrounded box pose. The binary
    # expert labels remain unchanged; consumers must not use stale distances.
    result.pop("hand_box_surface_distance", None)

    bottom_after = box_bottom_height(
        np.asarray(result["object_pos"], dtype=np.float64),
        object_quat,
        half_size=half_size,
        geom_pos=geom_pos,
        geom_quat_wxyz=geom_quat_wxyz,
    )
    if float(np.min(bottom_after)) < float(ground_height) - tolerance:
        raise ValueError(
            f"Grounding failed: minimum bottom={float(np.min(bottom_after))}, ground={ground_height}"
        )
    metadata = dict(result.get("metadata") or {})
    metadata["box_grounding"] = {
        "method": "minimum_per_frame_marker_z_v1",
        "ground_height": float(ground_height),
        "corrected_frames": int(np.sum(correction > tolerance)),
        "max_correction_m": float(np.max(correction)),
        "minimum_bottom_before_m": float(np.min(bottom_before)),
        "minimum_bottom_after_m": float(np.min(bottom_after)),
        "removed_stale_hand_box_surface_distance": "hand_box_surface_distance" in motion,
    }
    result["metadata"] = metadata
    return result, {
        "frames": len(correction),
        "corrected_frames": int(np.sum(correction > tolerance)),
        "max_correction_m": float(np.max(correction)),
        "minimum_bottom_before_m": float(np.min(bottom_before)),
        "minimum_bottom_after_m": float(np.min(bottom_after)),
    }


def ground_carry_dataset(
    motions: dict[str, dict[str, Any]],
    box_config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validated = validate_ufo_motion_dict(motions, "carry_grounding_input")
    half_size = np.asarray(box_config["half_size"], dtype=np.float64)
    geom_pos = np.asarray(box_config["geom_pos"], dtype=np.float64)
    geom_quat_wxyz = np.asarray(box_config["geom_quat_wxyz"], dtype=np.float64)
    ground_height = float(box_config.get("ground_surface_height", 0.0))
    output: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for key, motion in validated.items():
        grounded, stats = ground_carry_motion(
            motion,
            half_size=half_size,
            geom_pos=geom_pos,
            geom_quat_wxyz=geom_quat_wxyz,
            ground_height=ground_height,
        )
        output[key] = grounded
        rows.append({"motion_key": key, **stats})
    validate_ufo_motion_dict(output, "carry_grounding_output")
    summary = {
        "motion_count": len(output),
        "frame_count": sum(int(row["frames"]) for row in rows),
        "motions_corrected": sum(int(row["corrected_frames"]) > 0 for row in rows),
        "frames_corrected": sum(int(row["corrected_frames"]) for row in rows),
        "max_correction_m": max(float(row["max_correction_m"]) for row in rows),
        "minimum_bottom_before_m": min(float(row["minimum_bottom_before_m"]) for row in rows),
        "minimum_bottom_after_m": min(float(row["minimum_bottom_after_m"]) for row in rows),
        "motions": rows,
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-full",
        type=Path,
        default=Path(
            "cache/motion_data/g1_carry_box_omniretarget_no_foot/"
            "g1_carry_box_omniretarget_no_foot_full_ufo.pkl"
        ),
    )
    parser.add_argument("--box-config", type=Path, default=Path("configs/tasks/carry_box.yaml"))
    parser.add_argument(
        "--output-full",
        type=Path,
        default=Path(
            "cache/motion_data/g1_carry_box_omniretarget_no_foot/"
            "g1_carry_box_omniretarget_no_foot_full_grounded_ufo.pkl"
        ),
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        default=Path(
            "cache/motion_data/g1_carry_box_omniretarget_no_foot/"
            "g1_carry_box_omniretarget_no_foot_train_near10s_grounded_ufo.pkl"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/g1_carry_box_grounding/summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motions = joblib.load(args.input_full.expanduser().resolve())
    box_config = OmegaConf.to_container(OmegaConf.load(args.box_config.expanduser().resolve()), resolve=True)
    if not isinstance(motions, dict) or not isinstance(box_config, dict):
        raise TypeError("input cache and box config must contain mappings")
    grounded, summary = ground_carry_dataset(motions, box_config)
    train = clip_ufo_motion_dict(
        grounded,
        clip_seconds=10.0,
        stride_seconds=10.0,
        keep_short=True,
        min_clip_seconds=1.0,
        source_name="g1_carry_box_omniretarget_no_foot_grounded_train",
    )
    dump_ufo_pkl(grounded, args.output_full.expanduser(), "g1_carry_box_omniretarget_no_foot_grounded_full")
    dump_ufo_pkl(train, args.output_train.expanduser(), "g1_carry_box_omniretarget_no_foot_grounded_train")
    summary.update({"train_motion_count": len(train)})
    summary_path = args.summary.expanduser()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "motions"}, indent=2, sort_keys=True))
    print(f"Wrote {args.output_full}")
    print(f"Wrote {args.output_train}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
