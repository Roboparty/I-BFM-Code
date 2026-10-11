"""Evaluate a fixed carry goal from several deterministic robot poses."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import mediapy as media
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.envs.carry_observations import LEGACY_OBJECT_CONTEXT_DIM, OBJECT_CONTEXT_DIM
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    add_bool_arg,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    render_policy_frame,
)
from humanoidverse.tracking_inference import (
    DEFAULT_ROBOT_CONFIG,
    _control_to_qpos_order_indices,
    _expert_qpos_from_obs,
    _resize_nearest,
    _resolve_tracking_robot_config,
    _target_states_from_obs,
)
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import quat_from_angle_axis, quat_mul, quat_rotate_inverse

INITIAL_POSES = ("standing", "left_side", "right_side", "supine", "prone")


def _clip_observed_object_distance(
    observation: Any,
    max_distance: float | None,
) -> Any:
    """Clip root-local box XY distance in an actor-observation copy.

    For the spatial Carry schema, hand-minus-box vectors are recomputed against
    the virtual clipped box position so the actor does not receive contradictory
    object and hand geometry. Simulation state and the original observation are
    left untouched.
    """

    if max_distance is None:
        return observation
    if float(max_distance) <= 0.0:
        raise ValueError("max observed object distance must be positive")
    if not isinstance(observation, Mapping):
        return observation
    state = observation.get("state")
    last_action = observation.get("last_action")
    if not torch.is_tensor(state) or not torch.is_tensor(last_action):
        return observation
    num_dof = int(last_action.shape[-1])
    context_dim = int(state.shape[-1]) - (2 * num_dof + 6)
    if context_dim not in (LEGACY_OBJECT_CONTEXT_DIM, OBJECT_CONTEXT_DIM):
        return observation

    result = dict(observation)
    for key in ("state", "privileged_state"):
        value = result.get(key)
        if not torch.is_tensor(value) or int(value.shape[-1]) < context_dim:
            continue
        value = value.clone()
        start = int(value.shape[-1]) - context_dim
        object_pos = value[..., start : start + 3]
        planar_distance = torch.linalg.vector_norm(object_pos[..., :2], dim=-1, keepdim=True)
        scale = torch.clamp(float(max_distance) / torch.clamp(planar_distance, min=1e-6), max=1.0)
        virtual_object_pos = object_pos.clone()
        virtual_object_pos[..., :2] *= scale
        value[..., start : start + 3] = virtual_object_pos

        if context_dim == OBJECT_CONTEXT_DIM:
            hand_root_start = start + 20
            hand_root = value[..., hand_root_start : hand_root_start + 6].reshape(*value.shape[:-1], 2, 3)
            hand_minus_box = hand_root - virtual_object_pos.unsqueeze(-2)
            value[..., hand_root_start + 6 : hand_root_start + 12] = hand_minus_box.flatten(start_dim=-2)
        result[key] = value
    return result


def _select_frame(values: Any, frame: int, num_frames: int) -> Any:
    def select(value: Any) -> Any:
        if hasattr(value, "shape") and len(value.shape) > 0 and int(value.shape[0]) == int(num_frames):
            return value[frame : frame + 1]
        return value

    return tree_map(select, values)


def _apply_initial_pose(
    target_states: dict[str, torch.Tensor],
    pose: str,
    *,
    lying_root_height: float,
) -> dict[str, torch.Tensor]:
    if pose not in INITIAL_POSES:
        raise ValueError(f"Unknown initial pose {pose!r}; expected one of {INITIAL_POSES}")
    result = {key: value.clone() if torch.is_tensor(value) else value for key, value in target_states.items()}
    result["root_states"][:, 7:13] = 0.0
    result["dof_states"][:, :, 1] = 0.0
    if "object_states" in result:
        result["object_states"][:, 7:13] = 0.0
    if pose == "standing":
        return result

    axis_and_angle = {
        "left_side": ((1.0, 0.0, 0.0), torch.pi / 2),
        "right_side": ((1.0, 0.0, 0.0), -torch.pi / 2),
        "supine": ((0.0, 1.0, 0.0), -torch.pi / 2),
        "prone": ((0.0, 1.0, 0.0), torch.pi / 2),
    }
    axis, angle = axis_and_angle[pose]
    root_states = result["root_states"]
    root_states[:, 2] = float(lying_root_height)
    rotation = quat_from_angle_axis(
        torch.as_tensor(angle, device=root_states.device, dtype=root_states.dtype),
        torch.as_tensor(axis, device=root_states.device, dtype=root_states.dtype),
        w_last=True,
    )
    root_states[:, 3:7] = quat_mul(
        rotation.expand_as(root_states[:, 3:7]),
        root_states[:, 3:7],
        w_last=True,
    )
    return result


def _apply_initial_distance(
    target_states: dict[str, torch.Tensor],
    distance: float,
) -> dict[str, torch.Tensor]:
    """Move a standing robot to an absolute planar distance from the box."""

    if float(distance) <= 0.0:
        raise ValueError("Initial root-to-box distance must be positive")
    result = _apply_initial_pose(target_states, "standing", lying_root_height=0.5)
    if "object_states" not in result:
        raise ValueError("Distance-based carry goal recovery requires object_states")
    root_xy = result["root_states"][:, :2]
    object_xy = result["object_states"][:, :2]
    direction = root_xy - object_xy
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if bool((norm < 1e-6).any()):
        raise ValueError("Robot root and box centers share the same XY position")
    result["root_states"][:, :2] = object_xy + direction / norm * float(distance)
    return result


def _rollout_metrics(
    env: Any,
    *,
    goal_dof_pos: torch.Tensor,
    goal_object_relative_pos: torch.Tensor,
) -> dict[str, object]:
    current_object_relative_pos = quat_rotate_inverse(
        env.base_quat[0:1],
        (env.object_pos[0] - env.robot_root_states[0, :3]).unsqueeze(0),
        w_last=True,
    )[0]
    return {
        "root_height": float(env.robot_root_states[0, 2].item()),
        "upright": float((-env.projected_gravity[0, 2]).item()),
        "joint_mae": float(torch.mean(torch.abs(env.dof_pos[0] - goal_dof_pos)).item()),
        "object_height": float(env.object_pos[0, 2].item()),
        "object_relative_position_error": float(
            torch.linalg.vector_norm(current_object_relative_pos - goal_object_relative_pos).item()
        ),
        "left_contact": bool(env.hand_box_contact[0, 0].item()),
        "right_contact": bool(env.hand_box_contact[0, 1].item()),
    }


def _mean_final_window(history: list[dict[str, object]], window: int) -> dict[str, object]:
    tail = history[-max(1, min(int(window), len(history))) :]
    numeric_keys = (
        "root_height",
        "upright",
        "joint_mae",
        "object_height",
        "object_relative_position_error",
    )
    result: dict[str, object] = {
        key: float(np.mean([float(row[key]) for row in tail])) for key in numeric_keys
    }
    result["left_contact_rate"] = float(np.mean([bool(row["left_contact"]) for row in tail]))
    result["right_contact_rate"] = float(np.mean([bool(row["right_contact"]) for row in tail]))
    result["bilateral_contact_rate"] = float(
        np.mean([bool(row["left_contact"]) and bool(row["right_contact"]) for row in tail])
    )
    return result


@torch.no_grad()
def run_goal_recovery_inference(args: argparse.Namespace) -> None:
    model_folder = args.model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    robot_config = _resolve_tracking_robot_config(args.robot_config, None)
    robot_training = load_robot_training_spec(robot_config)
    num_dof = len(robot_training.robot.control_joint_names)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    qpos_indices, _qpos_joint_names = _control_to_qpos_order_indices(robot_training)

    model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(args.device))
    model.to(args.device)
    model.eval()
    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=args.data_path,
        robot_config=robot_config,
        device=args.device,
        headless=args.headless,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        max_episode_length_s=max(float(args.episode_seconds) + 5.0, 100.0),
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    env._motion_lib.load_all_motions()
    env.is_evaluating = True

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_folder / "goal_recovery_inference"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    box_half_size = tuple(float(value) for value in env.carry_box_cfg.get("half_size", [0.2356, 0.2294, 0.1850]))
    renderer = (
        MujocoQposRenderer(
            robot_xml,
            render_size=args.render_size,
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
            expected_qpos_size=7 + num_dof + 7,
            box_half_size=box_half_size,
            box_geom_pos=tuple(float(value) for value in env.carry_box_cfg.get("geom_pos", [0.0, 0.0, 0.0])),
            box_geom_quat_wxyz=tuple(
                float(value) for value in env.carry_box_cfg.get("geom_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
            ),
        )
        if args.save_mp4
        else None
    )
    try:
        backward_obs, obs_dict = get_backward_observation(
            env,
            args.motion_id,
            use_root_height_obs=use_root_height_obs,
            velocity_multiplier=0,
            object_velocity_multiplier=0,
        )
        num_frames = int(next(iter(backward_obs.values())).shape[0])
        goal_frame = args.goal_frame if args.goal_frame >= 0 else num_frames + args.goal_frame
        initial_frame = args.initial_frame if args.initial_frame >= 0 else num_frames + args.initial_frame
        for name, frame in (("goal", goal_frame), ("initial", initial_frame)):
            if not 0 <= frame < num_frames:
                raise IndexError(f"{name}_frame={frame} outside motion length {num_frames}")

        goal_obs = _select_frame(backward_obs, goal_frame, num_frames)
        goal_obs = tree_map(lambda value: value.to(args.device) if hasattr(value, "to") else value, goal_obs)
        z_goal = model.goal_inference(goal_obs)
        initial_obs_dict = _select_frame(obs_dict, initial_frame, num_frames)
        initial_states = _target_states_from_obs(initial_obs_dict, device=args.device, num_dof=num_dof)

        expert_qpos = _expert_qpos_from_obs(
            obs_dict,
            num_dof=num_dof,
            dof_qpos_order_indices=qpos_indices,
        )
        goal_qpos = expert_qpos[goal_frame]
        goal_dof_pos = obs_dict["dof_pos"][goal_frame].to(args.device)
        goal_object_relative_pos = quat_rotate_inverse(
            obs_dict["ref_body_rots"][goal_frame : goal_frame + 1, 0].to(args.device),
            (obs_dict["object_pos"][goal_frame] - obs_dict["ref_body_pos"][goal_frame, 0])
            .to(args.device)
            .unsqueeze(0),
            w_last=True,
        )[0]
        episode_steps = max(1, round(float(args.episode_seconds) / float(env.dt)))
        final_window = max(1, round(float(args.final_window_seconds) / float(env.dt)))
        summaries: dict[str, object] = {}

        print(
            f"[INFO] fixed carry goal motion_id={args.motion_id} goal_frame={goal_frame} "
            f"initial_frame={initial_frame} steps={episode_steps} dt={env.dt:.6f}"
        )
        if args.initial_distances:
            initial_cases = [
                (f"distance_{float(distance):.1f}m", _apply_initial_distance(initial_states, float(distance)))
                for distance in args.initial_distances
            ]
            summary_group = "distances"
        else:
            initial_cases = [
                (
                    pose,
                    _apply_initial_pose(
                        initial_states,
                        pose,
                        lying_root_height=args.lying_root_height,
                    ),
                )
                for pose in args.initial_poses
            ]
            summary_group = "poses"

        for case_name, case_states in initial_cases:
            observation, _info = wrapped_env.reset(to_numpy=False, target_states=case_states)
            frames: list[np.ndarray] = []
            history: list[dict[str, object]] = []
            use_env_render = True
            for step in range(episode_steps):
                policy_observation = _clip_observed_object_distance(
                    observation,
                    args.max_observed_object_distance,
                )
                action = model.act(policy_observation, z_goal, mean=True)
                observation, _reward, _terminated, _truncated, _info = wrapped_env.step(action, to_numpy=False)
                history.append(
                    _rollout_metrics(
                        env,
                        goal_dof_pos=goal_dof_pos,
                        goal_object_relative_pos=goal_object_relative_pos,
                    )
                )
                if renderer is not None:
                    policy_frame, use_env_render = render_policy_frame(
                        wrapped_env,
                        renderer,
                        use_env_render=use_env_render,
                    )
                    target_frame = renderer.render_qpos(goal_qpos)
                    target_frame = _resize_nearest(target_frame, policy_frame.shape[0], policy_frame.shape[1])
                    frames.append(np.concatenate([target_frame, policy_frame], axis=1))
                if step == 0 or step + 1 == episode_steps or (step + 1) % 100 == 0:
                    print(f"[INFO] case={case_name} progress={step + 1}/{episode_steps}", flush=True)

            video_path = output_dir / f"goal_recovery_{case_name}.mp4"
            if renderer is not None:
                media.write_video(str(video_path), frames, fps=args.fps)
                print(f"[INFO] saved {video_path}")
            summaries[case_name] = {
                "video": str(video_path),
                "initial_distance": (
                    float(torch.linalg.vector_norm(case_states["root_states"][0, :2] - case_states["object_states"][0, :2]).item())
                    if "object_states" in case_states
                    else None
                ),
                "final": history[-1],
                "final_window_mean": _mean_final_window(history, final_window),
            }

        result = {
            "model_folder": str(model_folder),
            "motion_id": int(args.motion_id),
            "goal_frame": int(goal_frame),
            "initial_frame": int(initial_frame),
            "episode_seconds": float(args.episode_seconds),
            "goal_robot_velocity_multiplier": 0.0,
            "goal_object_velocity_multiplier": 0.0,
            "source_goal_object_lin_vel": obs_dict["object_lin_vel"][goal_frame].detach().cpu().tolist(),
            "source_goal_object_ang_vel": obs_dict["object_ang_vel"][goal_frame].detach().cpu().tolist(),
            "goal_object_relative_pos": goal_object_relative_pos.detach().cpu().tolist(),
            "max_observed_object_distance": args.max_observed_object_distance,
            summary_group: summaries,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        joblib.dump(z_goal.detach().cpu().numpy(), output_dir / "z_goal.pkl")
        print(f"[INFO] saved summary {summary_path}")
    finally:
        if renderer is not None:
            renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--data-manifest", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--rebuild-motion-cache", action="store_true")
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--motion-id", type=int, default=67)
    parser.add_argument("--goal-frame", type=int, default=126)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--initial-poses", nargs="+", choices=INITIAL_POSES, default=list(INITIAL_POSES))
    parser.add_argument(
        "--initial-distances",
        type=float,
        nargs="+",
        default=None,
        help="Use standing initial states at these absolute planar root-to-box distances.",
    )
    parser.add_argument("--lying-root-height", type=float, default=0.5)
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    parser.add_argument(
        "--max-observed-object-distance",
        type=float,
        default=None,
        help=(
            "Inference-only planar clipping radius for root-local box position. "
            "The spatial hand-minus-box features are kept consistent; simulation state is unchanged."
        ),
    )
    parser.add_argument("--final-window-seconds", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    add_bool_arg(parser, "--disable-dr", True, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", True, "Disable observation noise.")
    add_bool_arg(parser, "--save-mp4", True, "Render side-by-side goal and policy videos.")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.max_observed_object_distance is not None and args.max_observed_object_distance <= 0.0:
        parser.error("--max-observed-object-distance must be positive")

    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        if args.dataset is None:
            parser.error("--dataset is required with --data-manifest")
        args.data_path = Path(
            prepare_manifest_dataset_path(
                args.data_manifest,
                args.dataset,
                split="inference",
                rebuild_cache=bool(args.rebuild_motion_cache),
            )
        )
        manifest_robot_config = prepare_manifest_robot_config_path(args.data_manifest)
    args.robot_config = _resolve_tracking_robot_config(args.robot_config, manifest_robot_config)
    return args


def main() -> None:
    run_goal_recovery_inference(parse_args())


if __name__ == "__main__":
    main()
