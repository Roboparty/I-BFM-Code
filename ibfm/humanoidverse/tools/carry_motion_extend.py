"""Create recovery-evaluation motions by repeating or pausing carry data."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict


FRAME_FIELDS = (
    "root_trans_offset",
    "pose_aa",
    "dof_pos",
    "root_quat",
    "object_pos",
    "object_quat",
    "object_lin_vel",
    "object_ang_vel",
)


def _select_motion(data: dict[str, Any], selector: str) -> tuple[str, dict[str, Any]]:
    if selector in data:
        return selector, data[selector]
    try:
        key = list(data)[int(selector)]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unknown motion {selector!r}; expected a key or index in [0, {len(data) - 1}]") from exc
    return key, data[key]


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64)
    return (result / np.linalg.norm(result, axis=-1, keepdims=True)).astype(np.float32)


def _slerp_xyzw(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    start = _normalize_quaternions(start)
    end = _normalize_quaternions(end)
    dot = np.sum(start * end, axis=-1, keepdims=True)
    end = np.where(dot < 0.0, -end, end)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    linear = (1.0 - alpha) * start + alpha * end
    spherical = (
        np.sin((1.0 - alpha) * angle) / np.maximum(sin_angle, 1.0e-8) * start
        + np.sin(alpha * angle) / np.maximum(sin_angle, 1.0e-8) * end
    )
    return _normalize_quaternions(np.where(sin_angle < 1.0e-6, linear, spherical))


def _angular_velocity_xyzw(quaternions: np.ndarray, dt: float) -> np.ndarray:
    quaternions = _normalize_quaternions(quaternions)
    velocity = np.zeros((len(quaternions), 3), dtype=np.float32)
    for index in range(len(quaternions) - 1):
        first = quaternions[index]
        second = quaternions[index + 1]
        if np.dot(first, second) < 0.0:
            second = -second
        x1, y1, z1, w1 = first
        x2, y2, z2, w2 = second
        relative = np.array(
            [
                w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2,
                w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2,
                w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2,
                w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2,
            ]
        )
        vector_norm = np.linalg.norm(relative[:3])
        if vector_norm > 1.0e-8:
            angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[3], -1.0, 1.0))
            velocity[index] = relative[:3] / vector_norm * angle / dt
    if len(velocity) > 1:
        velocity[-1] = velocity[-2]
    return velocity


def extend_carry_motion(
    motion: dict[str, Any],
    *,
    walk_start: int,
    walk_end: int,
    repeats: int,
    transition_frames: int,
    preserve_target: bool = True,
) -> dict[str, Any]:
    """Repeat [walk_start, walk_end] while preserving a continuous world path."""

    frame_count = len(np.asarray(motion["root_trans_offset"]))
    if not 0 <= walk_start < walk_end < frame_count:
        raise ValueError(f"Invalid walk range [{walk_start}, {walk_end}] for {frame_count} frames")
    if repeats < 1 or transition_frames < 0:
        raise ValueError("repeats must be >= 1 and transition_frames must be >= 0")
    missing = [field for field in FRAME_FIELDS if field not in motion]
    if missing:
        raise ValueError(f"Motion is missing frame fields required for carry extension: {missing}")

    arrays = {field: np.asarray(motion[field]) for field in FRAME_FIELDS}
    root_cycle_delta = arrays["root_trans_offset"][walk_end] - arrays["root_trans_offset"][walk_start]
    root_cycle_delta = root_cycle_delta.astype(np.float32)
    root_cycle_delta[2] = 0.0
    pieces: dict[str, list[np.ndarray]] = {field: [value[:walk_start]] for field, value in arrays.items()}

    for repeat in range(repeats):
        translation = repeat * root_cycle_delta
        for field, value in arrays.items():
            cycle = value[walk_start : walk_end + 1].copy()
            if field in ("root_trans_offset", "object_pos"):
                cycle = cycle + translation
            pieces[field].append(cycle)

        if repeat == repeats - 1 or transition_frames == 0:
            continue
        next_translation = (repeat + 1) * root_cycle_delta
        for field, value in arrays.items():
            start_value = value[walk_end].copy()
            end_value = value[walk_start].copy()
            if field in ("root_trans_offset", "object_pos"):
                start_value = start_value + translation
                end_value = end_value + next_translation
            transition = []
            for transition_index in range(1, transition_frames + 1):
                alpha = transition_index / float(transition_frames + 1)
                if field in ("root_quat", "object_quat"):
                    transition.append(_slerp_xyzw(start_value, end_value, alpha))
                else:
                    transition.append((1.0 - alpha) * start_value + alpha * end_value)
            pieces[field].append(np.asarray(transition, dtype=value.dtype))

    suffix_translation = (repeats - 1) * root_cycle_delta
    for field, value in arrays.items():
        suffix = value[walk_end + 1 :].copy()
        if field in ("root_trans_offset", "object_pos"):
            suffix = suffix + suffix_translation
        pieces[field].append(suffix)

    extended = deepcopy(motion)
    for field, field_pieces in pieces.items():
        extended[field] = np.concatenate(field_pieces, axis=0).astype(arrays[field].dtype)

    dt = 1.0 / float(motion["fps"])
    extended["object_lin_vel"] = np.gradient(extended["object_pos"], dt, axis=0).astype(np.float32)
    extended["object_ang_vel"] = _angular_velocity_xyzw(extended["object_quat"], dt)
    # Preserve the untouched prefix bit-for-bit. This matters for seq-based
    # tracking because the latent reads future object velocities as well as pose.
    extended["object_lin_vel"][:walk_start] = arrays["object_lin_vel"][:walk_start]
    extended["object_ang_vel"][:walk_start] = arrays["object_ang_vel"][:walk_start]
    if preserve_target:
        # The target delta is part of the actor observation. Moving it changes
        # the policy from frame zero, even though the reference prefix is the
        # same. Recovery-only extensions therefore keep the original task goal.
        extended["object_target_pos"] = np.asarray(motion["object_target_pos"], dtype=np.float32).copy()
    else:
        extended["object_target_pos"] = (np.asarray(motion["object_target_pos"]) + suffix_translation).astype(np.float32)
    extended["motion_key"] = f"{motion.get('motion_key', 'carry')}_walk_extended_x{repeats}"
    metadata = dict(motion.get("metadata") or {})
    metadata["recovery_extension"] = {
        "walk_start": walk_start,
        "walk_end": walk_end,
        "repeats": repeats,
        "transition_frames": transition_frames,
        "cycle_translation": root_cycle_delta.tolist(),
        "source_frames": frame_count,
        "preserve_target": preserve_target,
    }
    extended["metadata"] = metadata
    return extended


def pause_carry_motion(
    motion: dict[str, Any],
    *,
    pause_frame: int,
    pause_frames: int,
) -> dict[str, Any]:
    """Hold one reference frame in place before continuing the motion."""

    frame_count = len(np.asarray(motion["root_trans_offset"]))
    if not 0 <= pause_frame < frame_count:
        raise ValueError(f"Invalid pause frame {pause_frame} for {frame_count} frames")
    if pause_frames < 1:
        raise ValueError("pause_frames must be >= 1")
    missing = [field for field in FRAME_FIELDS if field not in motion]
    if missing:
        raise ValueError(f"Motion is missing frame fields required for carry pause: {missing}")

    paused = deepcopy(motion)
    for field in FRAME_FIELDS:
        value = np.asarray(motion[field])
        held = np.repeat(value[pause_frame : pause_frame + 1], pause_frames, axis=0)
        if field in ("object_lin_vel", "object_ang_vel"):
            held.fill(0.0)
        paused[field] = np.concatenate(
            [value[: pause_frame + 1], held, value[pause_frame + 1 :]], axis=0
        ).astype(value.dtype)

    paused["object_target_pos"] = np.asarray(motion["object_target_pos"], dtype=np.float32).copy()
    paused["object_target_quat"] = np.asarray(motion["object_target_quat"], dtype=np.float32).copy()
    paused["motion_key"] = f"{motion.get('motion_key', 'carry')}_pause_f{pause_frame}_n{pause_frames}"
    metadata = dict(motion.get("metadata") or {})
    metadata["recovery_pause"] = {
        "pause_frame": pause_frame,
        "pause_frames": pause_frames,
        "source_frames": frame_count,
    }
    paused["metadata"] = metadata
    return paused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion", nargs="+", default=["45"], help="One or more motion keys or zero-based indices.")
    parser.add_argument("--walk-start", type=int, default=53)
    parser.add_argument("--walk-end", type=int, default=85)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--transition-frames", type=int, default=8)
    parser.add_argument("--pause-at-s", type=float, default=None, help="Pause the reference at this time in seconds.")
    parser.add_argument("--pause-duration-s", type=float, default=5.0)
    parser.add_argument(
        "--translate-target",
        action="store_true",
        help="Translate the final object goal with the repeated path. Recovery tests should leave this disabled.",
    )
    args = parser.parse_args()

    data = joblib.load(args.input)
    output_data = {}
    summaries = []
    for selector in args.motion:
        key, motion = _select_motion(data, selector)
        if args.pause_at_s is None:
            extended = extend_carry_motion(
                motion,
                walk_start=args.walk_start,
                walk_end=args.walk_end,
                repeats=args.repeats,
                transition_frames=args.transition_frames,
                preserve_target=not args.translate_target,
            )
        else:
            fps = float(motion["fps"])
            pause_frame = int(round(args.pause_at_s * fps))
            pause_frames = int(round(args.pause_duration_s * fps))
            extended = pause_carry_motion(motion, pause_frame=pause_frame, pause_frames=pause_frames)
        output_key = str(extended["motion_key"])
        if output_key in output_data:
            raise ValueError(f"Duplicate extended motion key: {output_key}")
        output_data[output_key] = extended
        summaries.append((selector, key, output_key, extended))
    output_data = validate_ufo_motion_dict(output_data, "extended-batch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output_data, args.output)
    for selector, key, output_key, extended in summaries:
        fps = float(extended["fps"])
        print(
            f"selector={selector} source={key} output={output_key} frames={len(extended['object_pos'])} "
            f"duration={(len(extended['object_pos']) - 1) / fps:.3f}s"
        )
    print(f"motions={len(output_data)} path={args.output.resolve()}")


if __name__ == "__main__":
    main()
