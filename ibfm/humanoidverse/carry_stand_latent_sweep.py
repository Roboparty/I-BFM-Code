"""Infer and export empirical empty-handed stand latents from Carry replay."""

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
from humanoidverse.carry_speed_latent_sweep import (
    _rank_balanced_topk,
    _rank_counts,
    sample_replay_ranks,
)
from humanoidverse.get_up_reward_inference import compute_get_up_reward
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    weights = weights.reshape(-1)
    denominator = torch.clamp(weights.sum(), min=1e-12)
    return float((values.reshape(-1) * weights).sum().item() / denominator.item())


def _global_topk(mask: torch.Tensor, score: torch.Tensor, topk: int) -> torch.Tensor:
    candidates = torch.nonzero(mask, as_tuple=False).reshape(-1)
    take = min(int(topk), int(candidates.numel()))
    if take <= 0:
        raise ValueError("No eligible stand candidates")
    return candidates[torch.topk(score[candidates], k=take, largest=True).indices]


def _stand_source_mask(
    state: torch.Tensor,
    aux: dict[str, torch.Tensor],
    *,
    num_dof: int,
    source: str,
    distance_min: float,
    distance_max: float,
) -> torch.Tensor:
    object_present = state[:, -2]
    if source == "object-free":
        return object_present < 0.5
    if source != "carry-precontact":
        raise ValueError(f"Unsupported stand source: {source!r}")
    object_start = 2 * int(num_dof) + 6
    hand_contact = state[:, object_start + 18 : object_start + 20]
    distance = torch.linalg.vector_norm(
        state[:, object_start : object_start + 2], dim=-1
    )
    return (
        (object_present > 0.5)
        & (torch.sum(hand_contact, dim=-1) < 0.5)
        & (aux["carry_lift"] <= 0.10)
        & (aux["carry_robot_upright"] >= 0.90)
        & (aux["carry_upright"] >= 0.90)
        & (aux["carry_drop"] < 0.5)
        & (distance >= float(distance_min))
        & (distance <= float(distance_max))
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    data = sample_replay_ranks(
        args.buffer_paths,
        sample_count=int(args.num_samples),
        seed=int(args.seed),
    )
    robot = load_robot_training_spec(args.robot_config)
    num_dof = len(robot.robot.control_joint_names)
    next_obs = data["next"]["observation"]
    state = torch.as_tensor(next_obs["state"], dtype=torch.float32)
    expected_state_dim = 2 * num_dof + 40
    if state.ndim != 2 or int(state.shape[-1]) != expected_state_dim:
        raise ValueError(
            f"Expected non-ablated {expected_state_dim}D state, got {tuple(state.shape)}"
        )

    reward, components = compute_get_up_reward(
        data["next"]["qpos"],
        data["next"]["qvel"],
        state,
        num_dof=num_dof,
        root_height_threshold=float(args.root_height_threshold),
        root_height_sigma=float(args.root_height_sigma),
        upright_floor=float(args.upright_floor),
        root_speed_sigma=float(args.root_speed_sigma),
        root_ang_speed_sigma=float(args.root_ang_speed_sigma),
        joint_speed_sigma=float(args.joint_speed_sigma),
    )
    score = reward[:, 0]
    rank_ids = torch.as_tensor(data["rank"], dtype=torch.long)
    aux = {
        key: torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        for key, value in data["aux_rewards"].items()
    }
    source_mask = _stand_source_mask(
        state,
        aux,
        num_dof=num_dof,
        source=args.source,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
    )
    candidate_mask = source_mask & (score > 0.0)

    load_device = checkpoint_load_device(args.device)
    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint", device=load_device
    )
    model.to(args.device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, Any] = {}
    selections = [
        ("balanced", int(topk), _rank_balanced_topk(candidate_mask, score, rank_ids, int(topk)))
        for topk in args.balanced_topks
    ]
    selections.extend(
        ("global", int(topk), _global_topk(candidate_mask, score, int(topk)))
        for topk in args.global_topks
    )
    row_ids = torch.arange(int(state.shape[0]))
    for selection, topk, selected in selections:
        selected_obs = tree_map(
            lambda value: torch.as_tensor(value)[selected].to(args.device),
            next_obs,
        )
        selected_reward = reward[selected].to(args.device)
        latent = model.reward_wr_inference(selected_obs, selected_reward).reshape(1, -1)
        latent = model.project_z(latent).detach().cpu()
        name = f"z_stand_{selection}_topk_{topk:04d}"
        output = args.output_dir / f"{name}.pkl"
        joblib.dump(latent.numpy(), output)

        selected_mask = torch.isin(row_ids, selected)
        selected_components = {key: value[selected] for key, value in components.items()}
        selected_reward_cpu = reward[selected]
        candidates[name] = {
            "output": str(output),
            "inference_api": "reward_wr_inference",
            "selection": selection,
            "requested_topk": topk,
            "selected_count": int(selected.numel()),
            "selected_count_per_rank": _rank_counts(selected_mask, rank_ids),
            "reward_min": float(selected_reward_cpu.min().item()),
            "reward_mean": float(selected_reward_cpu.mean().item()),
            "reward_max": float(selected_reward_cpu.max().item()),
            "weighted_means": {
                key: _weighted_mean(value, selected_reward_cpu)
                for key, value in selected_components.items()
            },
            "latent_shape": list(latent.shape),
            "latent_norm": float(torch.linalg.vector_norm(latent).item()),
        }

    # A reward direction need not hold one stationary pose.  Export the best
    # replay frame from every rank through the state-goal encoder as well as two
    # cross-rank aggregates so rollout validation can compare both semantics.
    ranks = torch.unique(rank_ids, sorted=True)
    best_per_rank = _rank_balanced_topk(
        candidate_mask, score, rank_ids, int(ranks.numel())
    )
    best_obs = tree_map(
        lambda value: torch.as_tensor(value)[best_per_rank].to(args.device),
        next_obs,
    )
    goal_latents = model.goal_inference(best_obs).detach().cpu()
    best_rewards = reward[best_per_rank, 0].detach().cpu()
    for row, selected_index in enumerate(best_per_rank):
        rank = int(rank_ids[selected_index].item())
        name = f"z_stand_goal_rank_{rank}_best"
        output = args.output_dir / f"{name}.pkl"
        latent = goal_latents[row : row + 1]
        joblib.dump(latent.numpy(), output)
        candidates[name] = {
            "output": str(output),
            "inference_api": "goal_inference",
            "selection": "rank_best",
            "rank": rank,
            "reward": float(best_rewards[row].item()),
            "root_height": float(components["root_height"][selected_index].item()),
            "pelvis_up_z": float(components["pelvis_up_z"][selected_index].item()),
            "root_speed": float(components["root_speed"][selected_index].item()),
            "root_ang_speed": float(components["root_ang_speed"][selected_index].item()),
            "joint_speed_rms": float(components["joint_speed_rms"][selected_index].item()),
            "latent_shape": list(latent.shape),
            "latent_norm": float(torch.linalg.vector_norm(latent).item()),
        }

    goal_aggregates = {
        "z_stand_goal_rank_mean": goal_latents.mean(dim=0, keepdim=True),
        "z_stand_goal_rank_reward_weighted": torch.sum(
            goal_latents * (best_rewards / best_rewards.sum()).reshape(-1, 1),
            dim=0,
            keepdim=True,
        ),
    }
    for name, latent in goal_aggregates.items():
        latent = model.project_z(latent.to(args.device)).detach().cpu()
        output = args.output_dir / f"{name}.pkl"
        joblib.dump(latent.numpy(), output)
        candidates[name] = {
            "output": str(output),
            "inference_api": "goal_inference",
            "selection": "rank_aggregate",
            "selected_count": int(best_per_rank.numel()),
            "selected_count_per_rank": _rank_counts(
                torch.isin(row_ids, best_per_rank), rank_ids
            ),
            "latent_shape": list(latent.shape),
            "latent_norm": float(torch.linalg.vector_norm(latent).item()),
        }

    summary = {
        "model_folder": str(args.model_folder),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(state.shape[0]),
        "sample_count_per_rank": _rank_counts(torch.ones_like(source_mask), rank_ids),
        "source": args.source,
        "source_candidate_count": int(source_mask.sum().item()),
        "distance_min": float(args.distance_min),
        "distance_max": float(args.distance_max),
        "candidate_count": int(candidate_mask.sum().item()),
        "reward": {
            "formula": "sigmoid((root_z-threshold)/sigma) * upright_ramp * stationary",
            "root_height_threshold": float(args.root_height_threshold),
            "root_height_sigma": float(args.root_height_sigma),
            "upright_floor": float(args.upright_floor),
            "root_speed_sigma": float(args.root_speed_sigma),
            "root_ang_speed_sigma": float(args.root_ang_speed_sigma),
            "joint_speed_sigma": float(args.joint_speed_sigma),
        },
        "candidates": candidates,
    }
    (args.output_dir / "stand_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--buffer-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source",
        choices=("object-free", "carry-precontact"),
        default="object-free",
    )
    parser.add_argument("--distance-min", type=float, default=1.0)
    parser.add_argument("--distance-max", type=float, default=2.5)
    parser.add_argument("--balanced-topks", type=int, nargs="+", default=(8, 16, 32, 64, 128, 256, 512))
    parser.add_argument("--global-topks", type=int, nargs="+", default=(1, 4, 8, 16, 32, 64, 128, 256, 512))
    parser.add_argument("--root-height-threshold", type=float, default=0.74)
    parser.add_argument("--root-height-sigma", type=float, default=0.03)
    parser.add_argument("--upright-floor", type=float, default=0.85)
    parser.add_argument("--root-speed-sigma", type=float, default=0.15)
    parser.add_argument("--root-ang-speed-sigma", type=float, default=0.20)
    parser.add_argument("--joint-speed-sigma", type=float, default=0.35)
    args = parser.parse_args()
    if args.num_samples <= 0 or not args.buffer_paths:
        parser.error("--num-samples must be positive and --buffer-paths cannot be empty")
    if min((*args.balanced_topks, *args.global_topks)) <= 0:
        parser.error("all top-k values must be positive")
    if args.distance_min < 0.0 or args.distance_min >= args.distance_max:
        parser.error("--distance-min must be non-negative and less than --distance-max")
    return args


if __name__ == "__main__":
    run(parse_args())
