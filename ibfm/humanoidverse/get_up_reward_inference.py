"""Infer one state-conditioned get-up latent from all replay-buffer ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.carry_speed_latent_sweep import _rank_balanced_topk, _rank_counts, sample_replay_ranks
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG
from humanoidverse.utils.robot_spec import load_robot_training_spec


def compute_get_up_reward(
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    state: torch.Tensor,
    *,
    num_dof: int,
    root_height_threshold: float = 0.68,
    root_height_sigma: float = 0.04,
    upright_floor: float = 0.75,
    root_speed_sigma: float = 0.35,
    root_ang_speed_sigma: float = 0.50,
    joint_speed_sigma: float = 0.80,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reward stable upright states, not a particular get-up trajectory."""

    qpos = torch.as_tensor(qpos, dtype=torch.float32)
    qvel = torch.as_tensor(qvel, dtype=torch.float32)
    state = torch.as_tensor(state, dtype=torch.float32)
    if qpos.ndim != 2 or qpos.shape[-1] < 7:
        raise ValueError(f"Expected qpos [N, >=7], got {tuple(qpos.shape)}")
    if qvel.ndim != 2 or qvel.shape[-1] < 6 + int(num_dof):
        raise ValueError(f"Expected qvel [N, >=6+num_dof], got {tuple(qvel.shape)}")
    if state.ndim != 2 or state.shape[0] != qpos.shape[0]:
        raise ValueError(f"Expected state aligned with qpos, got {tuple(state.shape)}")
    if root_height_sigma <= 0.0 or min(root_speed_sigma, root_ang_speed_sigma, joint_speed_sigma) <= 0.0:
        raise ValueError("Get-up reward sigmas must be positive")
    if not 0.0 <= upright_floor < 1.0:
        raise ValueError("upright_floor must be in [0, 1)")

    root_height = qpos[:, 2]
    pelvis_up_z = torch.clamp(-state[:, 2 * int(num_dof) + 2], min=-1.0, max=1.0)
    root_speed = torch.linalg.vector_norm(qvel[:, :3], dim=-1)
    root_ang_speed = torch.linalg.vector_norm(qvel[:, 3:6], dim=-1)
    joint_speed_rms = torch.sqrt(torch.mean(torch.square(qvel[:, 6 : 6 + int(num_dof)]), dim=-1))

    height_score = torch.sigmoid((root_height - float(root_height_threshold)) / float(root_height_sigma))
    upright_score = torch.clamp(
        (pelvis_up_z - float(upright_floor)) / (1.0 - float(upright_floor)),
        min=0.0,
        max=1.0,
    )
    stationary_score = torch.exp(
        -torch.square(root_speed / float(root_speed_sigma))
        -torch.square(root_ang_speed / float(root_ang_speed_sigma))
        -torch.square(joint_speed_rms / float(joint_speed_sigma))
    )
    reward = height_score * upright_score * stationary_score
    return reward.unsqueeze(-1), {
        "root_height": root_height,
        "pelvis_up_z": pelvis_up_z,
        "root_speed": root_speed,
        "root_ang_speed": root_ang_speed,
        "joint_speed_rms": joint_speed_rms,
        "height_score": height_score,
        "upright_score": upright_score,
        "stationary_score": stationary_score,
    }


def _weighted_mean(values: torch.Tensor, reward: torch.Tensor) -> float:
    weight = reward.reshape(-1)
    denominator = torch.clamp(weight.sum(), min=1e-12)
    return float((values.reshape(-1) * weight).sum().item() / denominator.item())


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    if args.num_samples <= 0 or args.topk <= 0:
        raise ValueError("--num-samples and --topk must be positive")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    data = sample_replay_ranks(args.buffer_paths, sample_count=args.num_samples, seed=args.seed)
    robot = load_robot_training_spec(args.robot_config)
    num_dof = len(robot.robot.control_joint_names)
    next_obs = data["next"]["observation"]
    state = torch.as_tensor(next_obs["state"], dtype=torch.float32)
    expected_state_dim = 2 * num_dof + 40
    if state.ndim != 2 or state.shape[-1] != expected_state_dim:
        raise ValueError(f"Expected non-ablated {expected_state_dim}D state, got {tuple(state.shape)}")

    reward, components = compute_get_up_reward(
        data["next"]["qpos"],
        data["next"]["qvel"],
        state,
        num_dof=num_dof,
        root_height_threshold=args.root_height_threshold,
        root_height_sigma=args.root_height_sigma,
        upright_floor=args.upright_floor,
        root_speed_sigma=args.root_speed_sigma,
        root_ang_speed_sigma=args.root_ang_speed_sigma,
        joint_speed_sigma=args.joint_speed_sigma,
    )
    rank_ids = torch.as_tensor(data["rank"], dtype=torch.long)
    object_absent = state[:, -2] < 0.5
    candidate_mask = object_absent & (reward[:, 0] > 0.0)
    selected = _rank_balanced_topk(candidate_mask, reward[:, 0], rank_ids, args.topk)
    selected_obs = tree_map(lambda value: torch.as_tensor(value)[selected], next_obs)
    selected_reward = reward[selected]

    load_device = checkpoint_load_device(args.device)
    model = load_model_from_checkpoint_dir(args.model_folder / "checkpoint", device=load_device)
    model.to(args.device).eval()
    selected_obs = tree_map(lambda value: value.to(args.device), selected_obs)
    z_get_up = model.reward_wr_inference(
        selected_obs,
        selected_reward.to(args.device),
    ).reshape(-1)
    z_get_up = model.project_z(z_get_up.reshape(1, -1))[0].detach().cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "z_get_up_reward.pkl"
    joblib.dump(z_get_up.numpy(), output_path)
    selected_components = {key: value[selected] for key, value in components.items()}
    summary: dict[str, Any] = {
        "model_folder": str(args.model_folder),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(state.shape[0]),
        "sample_count_per_rank": _rank_counts(torch.ones_like(object_absent), rank_ids),
        "object_absent_count": int(object_absent.sum().item()),
        "candidate_count": int(candidate_mask.sum().item()),
        "selected_count": int(selected.numel()),
        "selected_count_per_rank": _rank_counts(
            torch.isin(torch.arange(state.shape[0]), selected), rank_ids
        ),
        "reward": {
            "formula": "sigmoid((root_z-threshold)/sigma) * upright_ramp * stationary",
            "root_height_threshold": float(args.root_height_threshold),
            "root_height_sigma": float(args.root_height_sigma),
            "upright_floor": float(args.upright_floor),
            "root_speed_sigma": float(args.root_speed_sigma),
            "root_ang_speed_sigma": float(args.root_ang_speed_sigma),
            "joint_speed_sigma": float(args.joint_speed_sigma),
            "selected_min": float(selected_reward.min().item()),
            "selected_mean": float(selected_reward.mean().item()),
            "selected_max": float(selected_reward.max().item()),
        },
        "selected_weighted_means": {
            key: _weighted_mean(value, selected_reward) for key, value in selected_components.items()
        },
        "latent_shape": list(z_get_up.shape),
        "latent_norm": float(torch.linalg.vector_norm(z_get_up).item()),
        "output": str(output_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--buffer-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--topk", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root-height-threshold", type=float, default=0.68)
    parser.add_argument("--root-height-sigma", type=float, default=0.04)
    parser.add_argument("--upright-floor", type=float, default=0.75)
    parser.add_argument("--root-speed-sigma", type=float, default=0.35)
    parser.add_argument("--root-ang-speed-sigma", type=float, default=0.50)
    parser.add_argument("--joint-speed-sigma", type=float, default=0.80)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
