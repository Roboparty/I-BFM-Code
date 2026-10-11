"""Build LAFAN dance and box-height-only reward latents for Carry sim2sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.carry_speed_latent_sweep import sample_replay_ranks
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg
from humanoidverse.mjlab_reward_relabel import get_next
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG, _tracking_z
from humanoidverse.utils.helpers import get_backward_observation


def compute_box_height_threshold_reward(
    aux_rewards: Mapping[str, torch.Tensor],
    *,
    threshold_world_height: float = 0.5,
    ground_center_height: float = 0.186028293,
    lift_height: float = 0.35,
) -> torch.Tensor:
    """Return exactly 1[box center world Z > threshold], using replay's lift signal."""

    if "carry_lift" not in aux_rewards:
        raise KeyError("Replay buffer is missing auxiliary reward 'carry_lift'")
    if lift_height <= 0.0:
        raise ValueError("lift_height must be positive")
    normalized_threshold = (float(threshold_world_height) - float(ground_center_height)) / float(lift_height)
    if not 0.0 <= normalized_threshold < 1.0:
        raise ValueError(
            "threshold_world_height must lie in the un-clipped carry_lift range "
            f"[{ground_center_height}, {ground_center_height + lift_height})"
        )
    lift = torch.as_tensor(aux_rewards["carry_lift"], dtype=torch.float32).reshape(-1)
    return (lift > normalized_threshold).to(torch.float32).unsqueeze(-1)


def infer_box_height_latent(
    model: Any,
    data: Mapping[str, Any],
    *,
    threshold_world_height: float,
    ground_center_height: float,
    lift_height: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    reward = compute_box_height_threshold_reward(
        data["aux_rewards"],
        threshold_world_height=threshold_world_height,
        ground_center_height=ground_center_height,
        lift_height=lift_height,
    )
    positive = reward[:, 0] > 0.5
    positive_count = int(positive.sum().item())
    if positive_count == 0:
        raise ValueError("Replay sample contains no states whose box center is above the requested height")
    next_obs = get_next("observation", data)
    positive_obs = tree_map(lambda value: value[positive], next_obs)
    positive_reward = reward[positive]
    latent = model.reward_wr_inference(positive_obs, positive_reward).reshape(1, -1)
    rank_ids = torch.as_tensor(data["rank"])
    stats = {
        "reward_formula": f"1[object_center_world_z > {float(threshold_world_height):g}]",
        "sample_count": int(reward.shape[0]),
        "positive_count": positive_count,
        "positive_rate": float(positive.float().mean().item()),
        "positive_count_per_rank": {
            str(int(rank)): int((positive & (rank_ids == rank)).sum().item())
            for rank in torch.unique(rank_ids, sorted=True)
        },
        "ground_center_height": float(ground_center_height),
        "lift_height": float(lift_height),
        "normalized_lift_threshold": float(
            (threshold_world_height - ground_center_height) / lift_height
        ),
        "excluded_reward_terms": [
            "hand_contact",
            "robot_upright",
            "box_upright",
            "object_velocity",
            "target_distance",
            "drop",
        ],
    }
    return latent, stats


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.manual_seed(int(args.seed))
    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint",
        device=checkpoint_load_device(args.device),
        strict=False,
    )
    model.to(args.device).eval()

    data = sample_replay_ranks(
        args.buffer_paths,
        sample_count=int(args.num_samples),
        seed=int(args.seed),
    )
    z_height, height_stats = infer_box_height_latent(
        model,
        data,
        threshold_world_height=float(args.height_threshold),
        ground_center_height=float(args.ground_center_height),
        lift_height=float(args.lift_height),
    )
    del data

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        args.model_folder,
        data_path=args.dance_data,
        robot_config=args.robot_config,
        device=args.device,
        headless=True,
        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=100.0,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    try:
        env = wrapped_env._env
        env._motion_lib.load_all_motions()
        backward_obs, _reference = get_backward_observation(
            env,
            int(args.dance_motion_id),
            use_root_height_obs=use_root_height_obs,
        )
        dance_obs = tree_map(lambda value: value[1:].to(args.device), backward_obs)
        z_dance = _tracking_z(model, dance_obs, seq_length=int(args.tracking_seq_length))
    finally:
        wrapped_env.close()

    motion_data = joblib.load(args.dance_data)
    motion_keys: Sequence[str] = list(motion_data)
    if not 0 <= int(args.dance_motion_id) < len(motion_keys):
        raise IndexError(f"dance_motion_id={args.dance_motion_id} outside [0, {len(motion_keys)})")
    motion_key = motion_keys[int(args.dance_motion_id)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(z_dance.detach().cpu().numpy(), args.output_dir / "z_dance_motion20.pkl")
    joblib.dump(z_height.detach().cpu().numpy(), args.output_dir / "z_box_height_gt_0p5.pkl")
    metadata = {
        "model_folder": str(args.model_folder),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "dance_data": str(args.dance_data),
        "dance_motion_id": int(args.dance_motion_id),
        "dance_motion_key": motion_key,
        "dance_latent_shape": list(z_dance.shape),
        "tracking_seq_length": int(args.tracking_seq_length),
        "height_latent_shape": list(z_height.shape),
        "height_reward": height_stats,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[INFO] dance motion={args.dance_motion_id} key={motion_key} z={tuple(z_dance.shape)}")
    print(f"[INFO] height reward stats={height_stats}")
    print(f"[INFO] saved dance/height latents to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--buffer-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--dance-data", type=Path, default=Path("humanoidverse/data/lafan_29dof_10s-clipped.pkl"))
    parser.add_argument("--dance-motion-id", type=int, default=20)
    parser.add_argument("--tracking-seq-length", type=int, default=8)
    parser.add_argument("--height-threshold", type=float, default=0.5)
    parser.add_argument("--ground-center-height", type=float, default=0.186028293)
    parser.add_argument("--lift-height", type=float, default=0.35)
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.tracking_seq_length <= 0:
        parser.error("--tracking-seq-length must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
