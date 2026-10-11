"""Infer robust Carry lift/lower latents from balanced replay shards.

Unlike the generic phased inference, this utility treats bilateral hand contact
as a hard eligibility condition.  It exports a cross product of lift and lower
top-k candidates while preserving every unrelated phase latent from a baseline
bank.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _aux(data: Mapping[str, Any], name: str) -> torch.Tensor:
    try:
        return torch.as_tensor(data["aux_rewards"][name], dtype=torch.float32).reshape(-1)
    except KeyError as exc:
        raise KeyError(f"Replay sample is missing auxiliary signal {name!r}") from exc


def _weighted_mean(values: torch.Tensor, reward: torch.Tensor) -> float:
    weights = reward.reshape(-1).clamp_min(0.0)
    denominator = weights.sum().clamp_min(1e-12)
    return float((values.reshape(-1) * weights).sum().item() / denominator.item())


def _select_latent(
    model: Any,
    observation: Mapping[str, torch.Tensor],
    reward: torch.Tensor,
    candidate_mask: torch.Tensor,
    rank_ids: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = _rank_balanced_topk(candidate_mask, reward, rank_ids, int(topk))
    selected_obs = tree_map(lambda value: torch.as_tensor(value)[selected], observation)
    selected_obs = tree_map(lambda value: value.to(next(model.parameters()).device), selected_obs)
    selected_reward = reward[selected].reshape(-1, 1).to(next(model.parameters()).device)
    latent = model.reward_wr_inference(selected_obs, selected_reward).reshape(1, -1)
    return latent.detach().cpu(), selected


def _export_bank(
    output_dir: Path,
    baseline_dir: Path,
    *,
    z_lift: torch.Tensor,
    z_lower: torch.Tensor,
    metadata: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in baseline_dir.glob("z_*.pkl"):
        shutil.copy2(source, output_dir / source.name)
    joblib.dump(z_lift.numpy(), output_dir / "z_lift.pkl")
    joblib.dump(z_lower.numpy(), output_dir / "z_lower.pkl")
    (output_dir / "lift_lower_inference.json").write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n"
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    data = sample_replay_ranks(args.buffer_paths, sample_count=args.num_samples, seed=args.seed)
    observation = data["next"]["observation"]
    state = torch.as_tensor(observation["state"], dtype=torch.float32)
    rank_ids = torch.as_tensor(data["rank"], dtype=torch.long)

    robot = load_robot_training_spec(args.robot_config)
    num_dof = len(robot.robot.control_joint_names)
    expected_state_dim = 2 * num_dof + 40
    if state.ndim != 2 or int(state.shape[-1]) != expected_state_dim:
        raise ValueError(f"Expected non-ablated {expected_state_dim}D state, got {tuple(state.shape)}")

    object_start = 2 * num_dof + 6
    object_lin_vel = state[:, object_start + 9 : object_start + 12]
    object_ang_vel = state[:, object_start + 12 : object_start + 15]
    target_delta = state[:, object_start + 15 : object_start + 18]
    horizontal_speed = torch.linalg.vector_norm(object_lin_vel[:, :2], dim=-1)
    vertical_speed = object_lin_vel[:, 2]
    angular_speed = torch.linalg.vector_norm(object_ang_vel, dim=-1)
    target_distance = torch.linalg.vector_norm(target_delta, dim=-1)

    object_present = state[:, -2] > 0.5
    bilateral = _aux(data, "carry_bilateral_contact") > 0.5
    lift = _aux(data, "carry_lift").clamp(0.0, 1.0)
    robot_upright = _aux(data, "carry_robot_upright").clamp(0.0, 1.0)
    box_upright = _aux(data, "carry_upright").clamp(0.0, 1.0)
    not_dropped = _aux(data, "carry_drop") < 0.5
    upright = robot_upright * box_upright

    lift_motion_score = torch.exp(
        -torch.square(horizontal_speed / float(args.lift_horizontal_speed_sigma))
        -torch.square(vertical_speed / float(args.lift_vertical_speed_sigma))
        -torch.square(angular_speed / float(args.lift_angular_speed_sigma))
    )
    lift_reward = bilateral.float() * upright * (0.20 + 0.80 * lift) * lift_motion_score
    lift_mask = (
        object_present
        & bilateral
        & not_dropped
        & (lift >= float(args.lift_min_height))
        & (robot_upright >= float(args.lift_upright_min))
        & (box_upright >= float(args.lift_upright_min))
        & (horizontal_speed <= float(args.lift_horizontal_speed_max))
        & (vertical_speed.abs() <= float(args.lift_vertical_speed_max))
        & (angular_speed <= float(args.lift_angular_speed_max))
    )

    lower_motion_score = torch.exp(
        -torch.square(
            (vertical_speed - float(args.lower_target_vertical_speed))
            / float(args.lower_vertical_speed_sigma)
        )
        -torch.square(horizontal_speed / float(args.lower_horizontal_speed_sigma))
        -torch.square(angular_speed / float(args.lower_angular_speed_sigma))
    )
    lower_near_goal = torch.exp(-torch.square(target_distance / float(args.lower_target_distance_sigma)))
    lower_reward = bilateral.float() * upright * lift * lower_near_goal * lower_motion_score
    lower_mask = (
        object_present
        & bilateral
        & not_dropped
        & (lift >= float(args.lower_min_lift))
        & (robot_upright >= float(args.lower_upright_min))
        & (box_upright >= float(args.lower_upright_min))
        & (vertical_speed >= float(args.lower_vertical_speed_min))
        & (vertical_speed <= float(args.lower_vertical_speed_max))
        & (horizontal_speed <= float(args.lower_horizontal_speed_max))
        & (angular_speed <= float(args.lower_angular_speed_max))
        & (target_distance <= float(args.lower_target_distance_max))
    )

    if not bool(lift_mask.any()):
        raise ValueError("No replay samples satisfy the hard lift conditions")
    if not bool(lower_mask.any()):
        raise ValueError("No replay samples satisfy the hard lower conditions")

    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint", device=checkpoint_load_device(args.device)
    )
    model.to(args.device).eval()

    lift_candidates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for topk in args.lift_topks:
        lift_candidates[int(topk)] = _select_latent(
            model, observation, lift_reward, lift_mask, rank_ids, topk=int(topk)
        )
    lower_candidates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for topk in args.lower_topks:
        lower_candidates[int(topk)] = _select_latent(
            model, observation, lower_reward, lower_mask, rank_ids, topk=int(topk)
        )

    summary: dict[str, Any] = {
        "model_folder": str(args.model_folder),
        "baseline_latent_dir": str(args.baseline_latent_dir),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(state.shape[0]),
        "sample_count_per_rank": _rank_counts(torch.ones_like(object_present), rank_ids),
        "hard_conditions": {
            "bilateral_contact_required": True,
            "lift": {
                "min_lift": float(args.lift_min_height),
                "upright_min": float(args.lift_upright_min),
                "horizontal_speed_max": float(args.lift_horizontal_speed_max),
                "vertical_speed_abs_max": float(args.lift_vertical_speed_max),
                "angular_speed_max": float(args.lift_angular_speed_max),
            },
            "lower": {
                "min_lift": float(args.lower_min_lift),
                "upright_min": float(args.lower_upright_min),
                "vertical_speed_range": [
                    float(args.lower_vertical_speed_min),
                    float(args.lower_vertical_speed_max),
                ],
                "horizontal_speed_max": float(args.lower_horizontal_speed_max),
                "angular_speed_max": float(args.lower_angular_speed_max),
                "target_distance_max": float(args.lower_target_distance_max),
            },
        },
        "lift_candidate_count": int(lift_mask.sum().item()),
        "lift_candidates_per_rank": _rank_counts(lift_mask, rank_ids),
        "lower_candidate_count": int(lower_mask.sum().item()),
        "lower_candidates_per_rank": _rank_counts(lower_mask, rank_ids),
        "banks": {},
    }

    for lift_topk, (z_lift, lift_selected) in lift_candidates.items():
        for lower_topk, (z_lower, lower_selected) in lower_candidates.items():
            name = f"lift_topk_{lift_topk:04d}_lower_topk_{lower_topk:04d}"
            metadata = {
                "lift_topk": int(lift_topk),
                "lower_topk": int(lower_topk),
                "lift_selected_count": int(lift_selected.numel()),
                "lift_selected_per_rank": _rank_counts(
                    torch.isin(torch.arange(state.shape[0]), lift_selected), rank_ids
                ),
                "lift_selected_reward_mean": float(lift_reward[lift_selected].mean().item()),
                "lift_selected_lift_mean": float(lift[lift_selected].mean().item()),
                "lift_selected_vertical_speed_mean": _weighted_mean(
                    vertical_speed[lift_selected], lift_reward[lift_selected]
                ),
                "lower_selected_count": int(lower_selected.numel()),
                "lower_selected_per_rank": _rank_counts(
                    torch.isin(torch.arange(state.shape[0]), lower_selected), rank_ids
                ),
                "lower_selected_reward_mean": float(lower_reward[lower_selected].mean().item()),
                "lower_selected_vertical_speed_mean": _weighted_mean(
                    vertical_speed[lower_selected], lower_reward[lower_selected]
                ),
                "lower_selected_horizontal_speed_mean": _weighted_mean(
                    horizontal_speed[lower_selected], lower_reward[lower_selected]
                ),
                "lower_selected_target_distance_mean": _weighted_mean(
                    target_distance[lower_selected], lower_reward[lower_selected]
                ),
                "bilateral_contact_required": True,
            }
            _export_bank(
                args.output_dir / name,
                args.baseline_latent_dir,
                z_lift=z_lift,
                z_lower=z_lower,
                metadata=metadata,
            )
            summary["banks"][name] = metadata

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lift_lower_inference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--baseline-latent-dir", type=Path, required=True)
    parser.add_argument("--buffer-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lift-topks", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--lower-topks", type=int, nargs="+", default=[50, 100, 200])

    parser.add_argument("--lift-min-height", type=float, default=0.70)
    parser.add_argument("--lift-upright-min", type=float, default=0.85)
    parser.add_argument("--lift-horizontal-speed-max", type=float, default=0.35)
    parser.add_argument("--lift-vertical-speed-max", type=float, default=0.25)
    parser.add_argument("--lift-angular-speed-max", type=float, default=0.60)
    parser.add_argument("--lift-horizontal-speed-sigma", type=float, default=0.20)
    parser.add_argument("--lift-vertical-speed-sigma", type=float, default=0.15)
    parser.add_argument("--lift-angular-speed-sigma", type=float, default=0.40)

    parser.add_argument("--lower-min-lift", type=float, default=0.10)
    parser.add_argument("--lower-upright-min", type=float, default=0.80)
    parser.add_argument("--lower-target-vertical-speed", type=float, default=-0.12)
    parser.add_argument("--lower-vertical-speed-min", type=float, default=-0.25)
    parser.add_argument("--lower-vertical-speed-max", type=float, default=-0.03)
    parser.add_argument("--lower-horizontal-speed-max", type=float, default=0.25)
    parser.add_argument("--lower-angular-speed-max", type=float, default=0.60)
    parser.add_argument("--lower-target-distance-max", type=float, default=0.75)
    parser.add_argument("--lower-vertical-speed-sigma", type=float, default=0.08)
    parser.add_argument("--lower-horizontal-speed-sigma", type=float, default=0.15)
    parser.add_argument("--lower-angular-speed-sigma", type=float, default=0.30)
    parser.add_argument("--lower-target-distance-sigma", type=float, default=0.35)
    args = parser.parse_args()
    if args.num_samples <= 0 or min(args.lift_topks) <= 0 or min(args.lower_topks) <= 0:
        parser.error("--num-samples and all top-k values must be positive")
    if args.lower_vertical_speed_min >= args.lower_vertical_speed_max:
        parser.error("--lower-vertical-speed-min must be less than --lower-vertical-speed-max")
    return args


if __name__ == "__main__":
    run(parse_args())
