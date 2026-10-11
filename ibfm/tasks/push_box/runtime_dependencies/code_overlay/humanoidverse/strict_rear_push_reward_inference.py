"""Infer push latents from bilateral rear-face hand-contact replay states.

Unlike ``rear_push_reward_inference``, every selected state must have both
contact labels active and both hands inside an explicit rear-face window.  The
script emits several independently filtered profiles so MuJoCo can select the
best closed-loop compromise without overwriting any existing latent.
"""

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
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.push_reward_inference import _push_components
from humanoidverse.rear_push_reward_inference import compute_rear_push_reward
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG
from humanoidverse.utils.robot_spec import load_robot_training_spec


PROFILES: dict[str, dict[str, float]] = {
    # A useful fallback when the strict set does not cover every replay rank.
    "bilateral_moderate": {
        "upright_min": 0.78,
        "box_upright_min": 0.94,
        "max_lift": 0.05,
        "speed_window": 0.16,
        "lateral_speed_max": 0.25,
        "vertical_speed_max": 0.18,
        "angular_speed_max": 1.10,
        "root_behind_min": 0.25,
        "root_lateral_max": 0.35,
        "hand_rear_min": 0.04,
        "hand_rear_max": 0.38,
        "hand_height_abs_max": 0.20,
        "hand_center_abs_max": 0.14,
        "hand_separation_min": 0.10,
        "hand_separation_max": 0.50,
    },
    "bilateral_strict": {
        "upright_min": 0.82,
        "box_upright_min": 0.95,
        "max_lift": 0.04,
        "speed_window": 0.15,
        "lateral_speed_max": 0.20,
        "vertical_speed_max": 0.15,
        "angular_speed_max": 1.00,
        "root_behind_min": 0.30,
        "root_lateral_max": 0.28,
        "hand_rear_min": 0.08,
        "hand_rear_max": 0.34,
        "hand_height_abs_max": 0.16,
        "hand_center_abs_max": 0.10,
        "hand_separation_min": 0.16,
        "hand_separation_max": 0.44,
    },
    "bilateral_centered": {
        "upright_min": 0.85,
        "box_upright_min": 0.97,
        "max_lift": 0.035,
        "speed_window": 0.14,
        "lateral_speed_max": 0.16,
        "vertical_speed_max": 0.12,
        "angular_speed_max": 0.80,
        "root_behind_min": 0.32,
        "root_lateral_max": 0.22,
        "hand_rear_min": 0.10,
        "hand_rear_max": 0.32,
        "hand_height_abs_max": 0.14,
        "hand_center_abs_max": 0.08,
        "hand_separation_min": 0.18,
        "hand_separation_max": 0.42,
    },
    "bilateral_low_rear": {
        "upright_min": 0.82,
        "box_upright_min": 0.95,
        "max_lift": 0.04,
        "speed_window": 0.16,
        "lateral_speed_max": 0.22,
        "vertical_speed_max": 0.15,
        "angular_speed_max": 1.00,
        "root_behind_min": 0.30,
        "root_lateral_max": 0.30,
        "hand_rear_min": 0.08,
        "hand_rear_max": 0.34,
        "hand_height_abs_max": 0.16,
        "hand_height_mean_max": -0.02,
        "hand_center_abs_max": 0.10,
        "hand_separation_min": 0.16,
        "hand_separation_max": 0.44,
    },
}


def _geometry(components: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    hand_rear = components["hand_rear_depth"]
    hand_height = components["hand_height_from_box"]
    hand_lateral = components["hand_lateral_offset"]
    return {
        "hand_rear_min": hand_rear.min(dim=-1).values,
        "hand_rear_max": hand_rear.max(dim=-1).values,
        "hand_rear_mean": hand_rear.mean(dim=-1),
        "hand_height_abs_max": hand_height.abs().max(dim=-1).values,
        "hand_height_mean": hand_height.mean(dim=-1),
        "hand_center_abs": hand_lateral.mean(dim=-1).abs(),
        "hand_separation": (hand_lateral[:, 0] - hand_lateral[:, 1]).abs(),
    }


def _mask(
    components: dict[str, torch.Tensor],
    geometry: dict[str, torch.Tensor],
    *,
    target_speed: float,
    target_distance_min: float,
    profile: dict[str, float],
) -> torch.Tensor:
    result = (
        (components["object_present"] > 0.5)
        & (components["contact_bilateral"] > 0.5)
        & (components["target_distance"] >= target_distance_min)
        & (components["lift"] <= profile["max_lift"])
        & (components["robot_upright"] >= profile["upright_min"])
        & (components["box_upright"] >= profile["box_upright_min"])
        & (components["drop"] < 0.5)
        & ((components["toward_speed"] - target_speed).abs() <= profile["speed_window"])
        & (components["lateral_speed"].abs() <= profile["lateral_speed_max"])
        & (components["vertical_speed"].abs() <= profile["vertical_speed_max"])
        & (components["angular_speed"] <= profile["angular_speed_max"])
        & (components["behind_projection"] >= profile["root_behind_min"])
        & (components["behind_lateral_error"] <= profile["root_lateral_max"])
        & (geometry["hand_rear_min"] >= profile["hand_rear_min"])
        & (geometry["hand_rear_max"] <= profile["hand_rear_max"])
        & (geometry["hand_height_abs_max"] <= profile["hand_height_abs_max"])
        & (geometry["hand_center_abs"] <= profile["hand_center_abs_max"])
        & (geometry["hand_separation"] >= profile["hand_separation_min"])
        & (geometry["hand_separation"] <= profile["hand_separation_max"])
    )
    if "hand_height_mean_max" in profile:
        result &= geometry["hand_height_mean"] <= profile["hand_height_mean_max"]
    return result


def _selection_stats(
    indices: torch.Tensor,
    reward: torch.Tensor,
    components: dict[str, torch.Tensor],
    geometry: dict[str, torch.Tensor],
    rank_ids: torch.Tensor,
) -> dict[str, Any]:
    selected_mask = torch.zeros_like(rank_ids, dtype=torch.bool)
    selected_mask[indices] = True
    result: dict[str, Any] = {
        "count": int(indices.numel()),
        "rank_counts": _rank_counts(selected_mask, rank_ids),
        "reward_mean": float(reward[indices].mean()),
        "reward_min": float(reward[indices].min()),
        "reward_max": float(reward[indices].max()),
    }
    fields = {
        "toward_speed": components["toward_speed"],
        "lateral_speed_abs": components["lateral_speed"].abs(),
        "contact_bilateral": components["contact_bilateral"],
        "robot_upright": components["robot_upright"],
        "box_upright": components["box_upright"],
        "root_behind": components["behind_projection"],
        "root_lateral": components["behind_lateral_error"],
        **geometry,
    }
    for name, values in fields.items():
        chosen = values[indices]
        result[f"{name}_mean"] = float(chosen.mean())
        result[f"{name}_min"] = float(chosen.min())
        result[f"{name}_max"] = float(chosen.max())
    return result


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    robot = load_robot_training_spec(args.robot_config)
    num_dof = len(robot.robot.control_joint_names)
    data = sample_replay_ranks(
        args.buffer_paths, sample_count=args.num_samples, seed=args.seed
    )
    observation = data["next"]["observation"]
    state = torch.as_tensor(observation["state"], dtype=torch.float32)
    rank_ids = torch.as_tensor(data["rank"], dtype=torch.long)
    components = _push_components(data["aux_rewards"], state, num_dof=num_dof)
    geometry = _geometry(components)
    reward, _ = compute_rear_push_reward(
        components,
        target_speed=args.target_speed,
        speed_sigma=0.08,
        root_behind_target=0.52,
        root_behind_sigma=0.20,
        root_lateral_sigma=0.18,
        hand_rear_target=0.19,
        hand_rear_sigma=0.10,
        hand_height_target=0.0,
        hand_height_sigma=0.11,
        hand_center_sigma=0.10,
        hand_separation_target=0.30,
        hand_separation_sigma=0.12,
    )
    reward = reward * components["contact_bilateral"]

    device = torch.device(args.device)
    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint",
        device=checkpoint_load_device(args.device),
        strict=False,
    ).to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model_folder": str(args.model_folder),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(state.shape[0]),
        "target_speed_mps": args.target_speed,
        "selection_rule": "bilateral contact and both hands inside rear-face window",
        "profiles": {},
    }
    for profile_name, profile in PROFILES.items():
        candidate = _mask(
            components,
            geometry,
            target_speed=args.target_speed,
            target_distance_min=args.target_distance_min,
            profile=profile,
        ) & torch.isfinite(reward)
        candidate_count = int(candidate.sum())
        rank_candidate_counts = _rank_counts(candidate, rank_ids)
        profile_result: dict[str, Any] = {
            "parameters": profile,
            "candidate_count": candidate_count,
            "candidate_rank_counts": rank_candidate_counts,
            "topks": {},
        }
        if candidate_count:
            for topk in args.topks:
                indices = _rank_balanced_topk(candidate, reward, rank_ids, int(topk))
                if not int(indices.numel()):
                    continue
                selected_observation = tree_map(
                    lambda value: torch.as_tensor(value)[indices].to(device), observation
                )
                latent = model.reward_wr_inference(
                    selected_observation, reward[indices].reshape(-1, 1).to(device)
                ).reshape(1, -1)
                output = args.output_dir / profile_name / f"topk_{int(topk):04d}"
                output.mkdir(parents=True, exist_ok=True)
                joblib.dump(latent.detach().cpu().numpy(), output / "z_push_direct.pkl")
                stats = _selection_stats(indices, reward, components, geometry, rank_ids)
                (output / "strict_rear_push_inference.json").write_text(
                    json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                profile_result["topks"][str(int(topk))] = stats
        summary["profiles"][profile_name] = profile_result
        print(
            f"[strict-rear] profile={profile_name} candidates={candidate_count} "
            f"rank_counts={rank_candidate_counts}",
            flush=True,
        )
    (args.output_dir / "strict_rear_push_sweep.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--buffer-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--topks", type=int, nargs="+", default=(8, 16, 32, 64))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-speed", type=float, default=0.20)
    parser.add_argument("--target-distance-min", type=float, default=0.50)
    args = parser.parse_args()
    if args.num_samples <= 0 or min(args.topks) <= 0:
        parser.error("sample count and top-k values must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
