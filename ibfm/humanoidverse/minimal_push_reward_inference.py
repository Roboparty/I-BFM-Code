"""Infer one direct ground-push latent from two explicit reward terms.

This is an intentionally minimal ablation. It rewards only a box centre no
higher than ``max_center_height`` and box velocity along the box-to-goal
direction near ``target_speed``. Contact, robot posture, box orientation, and
relative robot-box placement do not enter the reward or candidate filter.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
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
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.push_reward_inference import _push_components
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG
from humanoidverse.utils.robot_spec import load_robot_training_spec


def compute_minimal_push_reward(
    aux_rewards: Mapping[str, torch.Tensor],
    state: torch.Tensor,
    *,
    num_dof: int,
    ground_center_height: float = 0.186028293,
    lift_height: float = 0.35,
    max_center_height: float = 0.20,
    target_speed: float = 0.25,
    height_sigma: float = 0.02,
    speed_sigma: float = 0.08,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the two-term reward and diagnostic components for each row."""

    if lift_height <= 0.0 or height_sigma <= 0.0 or speed_sigma <= 0.0:
        raise ValueError("lift height and reward sigmas must be positive")
    if target_speed <= 0.0:
        raise ValueError("target speed must be positive")

    components = _push_components(aux_rewards, state, num_dof=num_dof)
    box_center_height = (
        float(ground_center_height) + components["lift"] * float(lift_height)
    )
    height_excess = torch.relu(box_center_height - float(max_center_height))
    height_score = torch.exp(-torch.square(height_excess / float(height_sigma)))
    speed_score = torch.exp(
        -torch.square((components["toward_speed"] - float(target_speed)) / float(speed_sigma))
    )
    reward = components["object_present"] * height_score * speed_score
    return reward.unsqueeze(-1), {
        **components,
        "box_center_height": box_center_height,
        "height_score": height_score,
        "speed_score": speed_score,
    }


def _selection_stats(
    indices: torch.Tensor,
    reward: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    rank_ids: torch.Tensor,
) -> dict[str, Any]:
    selected_mask = torch.zeros_like(rank_ids, dtype=torch.bool)
    selected_mask[indices] = True
    result: dict[str, Any] = {
        "count": int(indices.numel()),
        "rank_counts": _rank_counts(selected_mask, rank_ids),
        "reward_mean": float(reward[indices].mean().item()),
        "reward_min": float(reward[indices].min().item()),
        "reward_max": float(reward[indices].max().item()),
    }
    for name in (
        "box_center_height",
        "toward_speed",
        "lateral_speed",
        "vertical_speed",
        "angular_speed",
        "target_distance",
        "contact_any",
        "contact_bilateral",
        "robot_upright",
        "box_upright",
        "behind_projection",
        "behind_lateral_error",
    ):
        values = components[name][indices]
        result[f"{name}_mean"] = float(values.mean().item())
        result[f"{name}_min"] = float(values.min().item())
        result[f"{name}_max"] = float(values.max().item())
    return result


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    robot = load_robot_training_spec(args.robot_config)
    num_dof = len(robot.robot.control_joint_names)
    data = sample_replay_ranks(
        args.buffer_paths,
        sample_count=args.num_samples,
        seed=args.seed,
    )
    observation = data["next"]["observation"]
    state = torch.as_tensor(observation["state"], dtype=torch.float32)
    rank_ids = torch.as_tensor(data["rank"], dtype=torch.long)

    expected_state_dim = 2 * num_dof + 40
    if state.ndim != 2 or state.shape[-1] != expected_state_dim:
        raise ValueError(
            f"Expected non-ablated Carry state {expected_state_dim}D, got {tuple(state.shape)}"
        )

    reward, components = compute_minimal_push_reward(
        data["aux_rewards"],
        state,
        num_dof=num_dof,
        ground_center_height=args.ground_center_height,
        lift_height=args.lift_height,
        max_center_height=args.max_center_height,
        target_speed=args.target_speed,
        height_sigma=args.height_sigma,
        speed_sigma=args.speed_sigma,
    )
    # A nonzero target direction is required to define "forward". This is a
    # validity condition, not an extra task preference.
    valid = (
        (components["object_present"] > 0.5)
        & (components["target_distance"] > 1e-6)
        & torch.isfinite(reward[:, 0])
    )
    indices = _rank_balanced_topk(valid, reward[:, 0], rank_ids, args.topk)

    device = torch.device(args.device)
    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint",
        device=checkpoint_load_device(args.device),
        strict=False,
    ).to(device).eval()
    selected_observation = tree_map(
        lambda value: torch.as_tensor(value)[indices].to(device),
        observation,
    )
    selected_reward = reward[indices].to(device)
    latent = model.reward_wr_inference(selected_observation, selected_reward).reshape(1, -1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(latent.detach().cpu().numpy(), args.output_dir / "z_push_direct.pkl")
    stats = {
        "model_folder": str(args.model_folder),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(state.shape[0]),
        "valid_candidate_count": int(valid.sum().item()),
        "reward_definition": {
            "terms": ["box_center_height_lte", "box_goalward_speed"],
            "max_center_height_m": float(args.max_center_height),
            "target_speed_mps": float(args.target_speed),
            "ground_center_height_m": float(args.ground_center_height),
            "lift_height_m": float(args.lift_height),
            "height_sigma_m": float(args.height_sigma),
            "speed_sigma_mps": float(args.speed_sigma),
            "excluded_terms": [
                "hand_contact",
                "robot_upright",
                "box_upright",
                "robot_box_alignment",
                "lateral_velocity",
                "angular_velocity",
            ],
        },
        "selected": _selection_stats(indices, reward[:, 0], components, rank_ids),
    }
    (args.output_dir / "minimal_push_inference.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[minimal-push] saved z_push_direct.pkl to {args.output_dir}", flush=True)
    print(json.dumps(stats["selected"], indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--buffer-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ground-center-height", type=float, default=0.186028293)
    parser.add_argument("--lift-height", type=float, default=0.35)
    parser.add_argument("--max-center-height", type=float, default=0.20)
    parser.add_argument("--target-speed", type=float, default=0.25)
    parser.add_argument("--height-sigma", type=float, default=0.02)
    parser.add_argument("--speed-sigma", type=float, default=0.08)
    args = parser.parse_args()
    if min(args.num_samples, args.topk) <= 0:
        parser.error("--num-samples and --topk must be positive")
    if args.topk > args.num_samples:
        parser.error("--topk cannot exceed --num-samples")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("--device must be cpu or cuda[:index]")
    if min(args.lift_height, args.target_speed, args.height_sigma, args.speed_sigma) <= 0.0:
        parser.error("lift height, target speed, and reward sigmas must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
