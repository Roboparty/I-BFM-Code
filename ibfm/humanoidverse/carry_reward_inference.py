"""Infer and evaluate a Carry latent from replay-buffer auxiliary rewards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import mediapy as media
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.goal_recovery_inference import INITIAL_POSES, _apply_initial_pose, _select_frame
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    add_bool_arg,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    render_policy_frame,
)
from humanoidverse.mjlab_reward_relabel import get_next
from humanoidverse.reward_inference import _load_replay_buffer
from humanoidverse.tracking_inference import (
    DEFAULT_ROBOT_CONFIG,
    _resolve_tracking_robot_config,
    _target_states_from_obs,
)
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from humanoidverse.utils.robot_spec import load_robot_training_spec

CARRY_REWARD_KEYS = (
    "carry_bilateral_contact",
    "carry_lift",
    "carry_robot_upright",
    "carry_upright",
    "carry_drop",
)


def compute_carry_lift_upright_reward(aux_rewards: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return the bounded reward used to infer a stable upright carry latent."""

    missing = [key for key in CARRY_REWARD_KEYS if key not in aux_rewards]
    if missing:
        raise KeyError(f"Replay buffer is missing Carry auxiliary rewards: {missing}")

    values = {
        key: torch.as_tensor(aux_rewards[key], dtype=torch.float32)
        for key in CARRY_REWARD_KEYS
    }
    shape = values[CARRY_REWARD_KEYS[0]].shape
    if any(value.shape != shape for value in values.values()):
        shapes = {key: tuple(value.shape) for key, value in values.items()}
        raise ValueError(f"Carry auxiliary reward shapes must match, got {shapes}")

    contact = values["carry_bilateral_contact"]
    lift = values["carry_lift"]
    robot_upright = values["carry_robot_upright"]
    box_upright = values["carry_upright"]
    drop = values["carry_drop"]
    reward = robot_upright * box_upright * (0.25 * contact + 0.75 * contact * lift) - 0.25 * drop
    return reward.clamp(0.0, 1.0)


def _sample_carry_batch(
    dataset: Any,
    *,
    num_samples: int,
) -> tuple[Any, torch.Tensor, dict[str, float | int]]:
    if isinstance(dataset, TrajectoryDictBufferMultiDim) and "aux_rewards" not in dataset.output_key_t:
        dataset.output_key_t.append("aux_rewards")

    sample_count = min(int(num_samples), int(dataset.size()))
    if sample_count <= 0:
        raise ValueError("Replay buffer contains no transitions")
    if sample_count >= int(dataset.size()) and hasattr(dataset, "get_full_buffer"):
        data = dataset.get_full_buffer()
    else:
        data = dataset.sample(sample_count)

    if "aux_rewards" not in data:
        raise KeyError("Replay sample does not contain aux_rewards")
    next_obs = get_next("observation", data)
    if not isinstance(next_obs, Mapping) or "state" not in next_obs:
        raise KeyError("Replay next observation must contain a state tensor")

    state = torch.as_tensor(next_obs["state"])
    if state.ndim != 2 or state.shape[-1] < 2:
        raise ValueError(f"Expected replay state shape [N, D>=2], got {tuple(state.shape)}")
    # The complete Carry schema ends with [object_present, task_id].
    carry_mask = state[:, -2] > 0.5
    carry_count = int(carry_mask.sum().item())
    if carry_count == 0:
        raise ValueError("Sampled replay transitions contain no object_present Carry states")

    reward = compute_carry_lift_upright_reward(data["aux_rewards"])
    if reward.ndim == 1:
        reward = reward.unsqueeze(-1)
    if reward.ndim != 2 or reward.shape[0] != state.shape[0] or reward.shape[1] != 1:
        raise ValueError(f"Expected Carry reward shape [N, 1], got {tuple(reward.shape)}")

    next_obs = tree_map(lambda value: value[carry_mask], next_obs)
    reward = reward[carry_mask]
    positive = reward[:, 0] > 0.0
    high = reward[:, 0] >= 0.75
    stats: dict[str, float | int] = {
        "sampled_transitions": int(state.shape[0]),
        "carry_transitions": carry_count,
        "positive_reward_transitions": int(positive.sum().item()),
        "high_reward_transitions": int(high.sum().item()),
        "reward_mean": float(reward.mean().item()),
        "reward_max": float(reward.max().item()),
        "reward_p50": float(torch.quantile(reward[:, 0], 0.50).item()),
        "reward_p90": float(torch.quantile(reward[:, 0], 0.90).item()),
        "reward_p99": float(torch.quantile(reward[:, 0], 0.99).item()),
    }
    return next_obs, reward, stats


def _scalar(value: Any) -> float:
    tensor = torch.as_tensor(value)
    return float(tensor.reshape(-1)[0].item())


def _apply_standing_distance_offset(
    target_states: dict[str, torch.Tensor],
    distance_offset: float,
) -> tuple[dict[str, torch.Tensor], float]:
    """Move the standing robot along the horizontal box-to-root direction."""

    if "object_states" not in target_states:
        raise ValueError("Standing distance offsets require object_states")
    result = {key: value.clone() if torch.is_tensor(value) else value for key, value in target_states.items()}
    root_xy = result["root_states"][:, :2]
    object_xy = result["object_states"][:, :2]
    box_to_root = root_xy - object_xy
    distance = torch.linalg.vector_norm(box_to_root, dim=-1, keepdim=True)
    if bool((distance < 1e-6).any()):
        raise ValueError("Cannot change standing distance when robot root and box centers share the same XY position")
    requested_distance = distance + float(distance_offset)
    if bool((requested_distance <= 0.0).any()):
        raise ValueError(
            f"distance_offset={distance_offset} would move the robot through the box center; "
            f"nominal distance={float(distance.min().item())}"
        )
    result["root_states"][:, :2] = object_xy + box_to_root / distance * requested_distance
    return result, float(requested_distance[0, 0].item())


def _rollout_metrics(env: Any, info: Mapping[str, Any]) -> dict[str, float | bool]:
    aux = info.get("aux_rewards", {})
    required = set(CARRY_REWARD_KEYS)
    missing = required.difference(aux)
    if missing:
        raise KeyError(f"Carry rollout did not return auxiliary rewards: {sorted(missing)}")
    reward = compute_carry_lift_upright_reward({key: torch.as_tensor(aux[key]) for key in CARRY_REWARD_KEYS})
    return {
        "root_height": float(env.robot_root_states[0, 2].item()),
        "object_height": float(env.object_pos[0, 2].item()),
        "left_contact": bool(env.hand_box_contact[0, 0].item()),
        "right_contact": bool(env.hand_box_contact[0, 1].item()),
        "bilateral_contact": _scalar(aux["carry_bilateral_contact"]) > 0.5,
        "lift": _scalar(aux["carry_lift"]),
        "robot_upright": _scalar(aux["carry_robot_upright"]),
        "box_upright": _scalar(aux["carry_upright"]),
        "drop": _scalar(aux["carry_drop"]),
        "inference_reward": _scalar(reward),
    }


def _summarize_rollout(history: list[dict[str, float | bool]], final_window: int, dt: float) -> dict[str, Any]:
    if not history:
        raise ValueError("Cannot summarize an empty rollout")
    tail = history[-max(1, min(int(final_window), len(history))) :]
    numeric_keys = (
        "root_height",
        "object_height",
        "lift",
        "robot_upright",
        "box_upright",
        "drop",
        "inference_reward",
    )

    def means(rows: list[dict[str, float | bool]]) -> dict[str, float]:
        result = {key: float(np.mean([float(row[key]) for row in rows])) for key in numeric_keys}
        for key in ("left_contact", "right_contact", "bilateral_contact"):
            result[f"{key}_rate"] = float(np.mean([bool(row[key]) for row in rows]))
        return result

    stable = [
        bool(row["bilateral_contact"])
        and float(row["lift"]) >= 0.5
        and float(row["robot_upright"]) >= 0.85
        and float(row["box_upright"]) >= 0.9
        for row in history
    ]
    first_stable_step = next((index for index, value in enumerate(stable) if value), None)
    return {
        "final": history[-1],
        "final_window_mean": means(tail),
        "episode_mean": means(history),
        "max_object_height": float(max(float(row["object_height"]) for row in history)),
        "max_lift": float(max(float(row["lift"]) for row in history)),
        "stable_carry_rate": float(np.mean(stable)),
        "first_stable_carry_s": None if first_stable_step is None else float((first_stable_step + 1) * dt),
    }


@torch.no_grad()
def run_carry_reward_inference(args: argparse.Namespace) -> None:
    torch.manual_seed(int(args.seed))
    model_folder = args.model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    robot_config = _resolve_tracking_robot_config(args.robot_config, None)
    robot_training = load_robot_training_spec(robot_config)
    num_dof = len(robot_training.robot.control_joint_names)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()

    model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(args.device))
    model.to(args.device)
    model.eval()

    loaded_buffer_path: Path | None = None
    if args.z_path is not None:
        z_path = args.z_path.expanduser().resolve()
        if not z_path.is_file():
            raise FileNotFoundError(f"Missing reward latent: {z_path}")
        z_reward = torch.as_tensor(joblib.load(z_path), dtype=torch.float32, device=args.device).reshape(1, -1)
        expected_z_dim = int(model.cfg.archi.z_dim)
        if z_reward.shape[-1] != expected_z_dim:
            raise ValueError(f"Reward latent dim {z_reward.shape[-1]} does not match model z_dim={expected_z_dim}")
        reward_stats = None
        print(f"[INFO] Reusing Carry reward latent: {z_path}")
    else:
        print("[INFO] Loading Carry replay buffer...", end=" ", flush=True)
        dataset, loaded_buffer_path = _load_replay_buffer(
            model_folder,
            buffer_rank=int(args.buffer_rank),
            buffer_path=args.buffer_path,
        )
        print(f"done: {loaded_buffer_path}")
        next_obs, inference_reward, reward_stats = _sample_carry_batch(dataset, num_samples=int(args.num_samples))
        print(f"[INFO] Carry replay reward stats={reward_stats}")
        z_reward = model.reward_wr_inference(next_obs, inference_reward).reshape(1, -1)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_folder / "carry_reward_inference"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(z_reward.detach().cpu().numpy(), output_dir / "z_reward.pkl")

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
        _backward_obs, obs_dict = get_backward_observation(
            env,
            int(args.initial_motion_id),
            use_root_height_obs=use_root_height_obs,
        )
        num_frames = int(obs_dict["dof_pos"].shape[0])
        initial_frame = int(args.initial_frame)
        if initial_frame < 0:
            initial_frame += num_frames
        if not 0 <= initial_frame < num_frames:
            raise IndexError(f"initial_frame={initial_frame} outside motion length {num_frames}")
        initial_obs_dict = _select_frame(obs_dict, initial_frame, num_frames)
        initial_states = _target_states_from_obs(initial_obs_dict, device=args.device, num_dof=num_dof)

        episode_steps = max(1, round(float(args.episode_seconds) / float(env.dt)))
        final_window = max(1, round(float(args.final_window_seconds) / float(env.dt)))
        summaries: dict[str, Any] = {}
        print(
            f"[INFO] Carry reward rollout motion={args.initial_motion_id} frame={initial_frame} "
            f"steps={episode_steps} dt={env.dt:.6f} poses={args.initial_poses}"
        )

        if args.standing_distance_offsets is None:
            rollout_cases = [(pose, pose, None) for pose in args.initial_poses]
        else:
            rollout_cases = [
                (f"standing_distance_offset_{offset:+.2f}m", "standing", float(offset))
                for offset in args.standing_distance_offsets
            ]

        for case_name, pose, distance_offset in rollout_cases:
            pose_states = _apply_initial_pose(initial_states, pose, lying_root_height=float(args.lying_root_height))
            initial_horizontal_distance = float(
                torch.linalg.vector_norm(
                    pose_states["root_states"][0, :2] - pose_states["object_states"][0, :2]
                ).item()
            )
            if distance_offset is not None:
                pose_states, initial_horizontal_distance = _apply_standing_distance_offset(
                    pose_states,
                    distance_offset,
                )
            observation, _info = wrapped_env.reset(to_numpy=False, target_states=pose_states)
            frames: list[np.ndarray] = []
            history: list[dict[str, float | bool]] = []
            use_env_render = True
            for step in range(episode_steps):
                action = model.act(observation, z_reward, mean=True)
                observation, _reward, _terminated, _truncated, info = wrapped_env.step(action, to_numpy=False)
                history.append(_rollout_metrics(env, info))
                if renderer is not None:
                    frame, use_env_render = render_policy_frame(
                        wrapped_env,
                        renderer,
                        use_env_render=use_env_render,
                    )
                    frames.append(frame)
                if step == 0 or step + 1 == episode_steps or (step + 1) % 100 == 0:
                    print(f"[INFO] case={case_name} progress={step + 1}/{episode_steps}", flush=True)

            video_path = output_dir / f"carry_reward_{case_name}.mp4"
            if renderer is not None:
                media.write_video(str(video_path), frames, fps=int(args.fps))
                print(f"[INFO] saved {video_path}")
            summaries[case_name] = {
                "video": str(video_path),
                "pose": pose,
                "distance_offset": distance_offset,
                "initial_horizontal_root_box_distance": initial_horizontal_distance,
                **_summarize_rollout(history, final_window, float(env.dt)),
            }

        result = {
            "model_folder": str(model_folder),
            "buffer_path": None if loaded_buffer_path is None else str(loaded_buffer_path),
            "z_path": None if args.z_path is None else str(args.z_path.expanduser().resolve()),
            "reward_formula": (
                "clamp(robot_upright * box_upright * "
                "(0.25 * bilateral_contact + 0.75 * bilateral_contact * lift) - 0.25 * drop, 0, 1)"
            ),
            "reward_stats": reward_stats,
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "initial_motion_id": int(args.initial_motion_id),
            "initial_frame": initial_frame,
            "episode_seconds": float(args.episode_seconds),
            "poses": summaries,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[INFO] saved summary {summary_path}")
    finally:
        if renderer is not None:
            renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--z-path", type=Path, default=None, help="Reuse an existing z_reward.pkl instead of replay inference.")
    parser.add_argument("--buffer-path", type=Path, default=None)
    parser.add_argument("--buffer-rank", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--data-manifest", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--rebuild-motion-cache", action="store_true")
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--initial-motion-id", type=int, default=67)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--initial-poses", nargs="+", choices=INITIAL_POSES, default=list(INITIAL_POSES))
    parser.add_argument(
        "--standing-distance-offsets",
        type=float,
        nargs="+",
        default=None,
        help="Standing-only horizontal root-to-box distance offsets in meters; positive values move the robot away.",
    )
    parser.add_argument("--lying-root-height", type=float, default=0.5)
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    parser.add_argument("--final-window-seconds", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    add_bool_arg(parser, "--disable-dr", True, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", True, "Disable observation noise.")
    add_bool_arg(parser, "--save-mp4", True, "Render Carry reward-inference videos.")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

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
    run_carry_reward_inference(parse_args())


if __name__ == "__main__":
    main()
