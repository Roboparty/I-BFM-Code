"""Infer ground-push latents with explicit rear-face hand geometry.

This keeps the successful 73.6M minimal-push objective (grounded box moving
toward its goal) but adds soft posture preferences.  The geometry terms remain
soft because fully bilateral, correctly placed 0.2 m/s pushes are extremely
sparse in both the 73.6M and 140.935M replay buffers.
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


def _gaussian(value: torch.Tensor, target: float, sigma: float) -> torch.Tensor:
    return torch.exp(-torch.square((value - float(target)) / float(sigma)))


def compute_rear_push_reward(
    components: Mapping[str, torch.Tensor],
    *,
    target_speed: float,
    speed_sigma: float,
    root_behind_target: float,
    root_behind_sigma: float,
    root_lateral_sigma: float,
    hand_rear_target: float,
    hand_rear_sigma: float,
    hand_height_target: float,
    hand_height_sigma: float,
    hand_center_sigma: float,
    hand_separation_target: float,
    hand_separation_sigma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score velocity and a two-hand pose on the box's rear face."""

    hand_rear = components["hand_rear_depth"]
    hand_height = components["hand_height_from_box"]
    hand_lateral = components["hand_lateral_offset"]
    hand_center = hand_lateral.mean(dim=-1).abs()
    hand_separation = (hand_lateral[:, 0] - hand_lateral[:, 1]).abs()

    speed_score = _gaussian(components["toward_speed"], target_speed, speed_sigma)
    root_score = _gaussian(
        components["behind_projection"], root_behind_target, root_behind_sigma
    ) * _gaussian(components["behind_lateral_error"], 0.0, root_lateral_sigma)
    # Mean over the two hands means one badly placed hand cannot be hidden by
    # the other hand's score.
    hand_rear_score = _gaussian(
        hand_rear, hand_rear_target, hand_rear_sigma
    ).mean(dim=-1)
    hand_height_score = _gaussian(
        hand_height, hand_height_target, hand_height_sigma
    ).mean(dim=-1)
    hand_layout_score = _gaussian(hand_center, 0.0, hand_center_sigma) * _gaussian(
        hand_separation, hand_separation_target, hand_separation_sigma
    )
    lateral_score = _gaussian(components["lateral_speed"], 0.0, 0.20)
    vertical_score = _gaussian(components["vertical_speed"], 0.0, 0.16)
    angular_score = _gaussian(components["angular_speed"], 0.0, 0.90)
    contact_score = 0.65 + 0.35 * components["contact_bilateral"]
    posture_score = torch.clamp(
        components["robot_upright"] * components["box_upright"], min=0.0
    )

    reward = (
        speed_score
        * root_score
        * hand_rear_score
        * hand_height_score
        * hand_layout_score
        * lateral_score
        * vertical_score
        * angular_score
        * contact_score
        * posture_score
    )
    return reward, {
        "speed_score": speed_score,
        "root_score": root_score,
        "hand_rear_score": hand_rear_score,
        "hand_height_score": hand_height_score,
        "hand_layout_score": hand_layout_score,
        "hand_rear_min": hand_rear.min(dim=-1).values,
        "hand_rear_mean": hand_rear.mean(dim=-1),
        "hand_height_mean": hand_height.mean(dim=-1),
        "hand_center_abs": hand_center,
        "hand_separation": hand_separation,
    }


def _stats(
    indices: torch.Tensor,
    reward: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    geometry: Mapping[str, torch.Tensor],
    rank_ids: torch.Tensor,
) -> dict[str, Any]:
    mask = torch.zeros_like(rank_ids, dtype=torch.bool)
    mask[indices] = True
    result: dict[str, Any] = {
        "count": int(indices.numel()),
        "rank_counts": _rank_counts(mask, rank_ids),
        "reward_mean": float(reward[indices].mean()),
        "reward_min": float(reward[indices].min()),
        "reward_max": float(reward[indices].max()),
    }
    fields = {
        "toward_speed": components["toward_speed"],
        "contact_any": components["contact_any"],
        "contact_bilateral": components["contact_bilateral"],
        "robot_upright": components["robot_upright"],
        "box_upright": components["box_upright"],
        "root_behind": components["behind_projection"],
        "root_lateral": components["behind_lateral_error"],
        "hand_rear_min": geometry["hand_rear_min"],
        "hand_rear_mean": geometry["hand_rear_mean"],
        "hand_height_mean": geometry["hand_height_mean"],
        "hand_center_abs": geometry["hand_center_abs"],
        "hand_separation": geometry["hand_separation"],
    }
    for name, values in fields.items():
        selected = values[indices]
        result[f"{name}_mean"] = float(selected.mean())
        result[f"{name}_min"] = float(selected.min())
        result[f"{name}_max"] = float(selected.max())
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
    reward, geometry = compute_rear_push_reward(
        components,
        target_speed=args.target_speed,
        speed_sigma=args.speed_sigma,
        root_behind_target=args.root_behind_target,
        root_behind_sigma=args.root_behind_sigma,
        root_lateral_sigma=args.root_lateral_sigma,
        hand_rear_target=args.hand_rear_target,
        hand_rear_sigma=args.hand_rear_sigma,
        hand_height_target=args.hand_height_target,
        hand_height_sigma=args.hand_height_sigma,
        hand_center_sigma=args.hand_center_sigma,
        hand_separation_target=args.hand_separation_target,
        hand_separation_sigma=args.hand_separation_sigma,
    )
    candidate = (
        (components["object_present"] > 0.5)
        & (components["contact_any"] > 0.5)
        & (components["target_distance"] >= args.target_distance_min)
        & (components["lift"] <= args.max_lift)
        & (components["robot_upright"] >= args.upright_min)
        & (components["box_upright"] >= args.box_upright_min)
        & (components["drop"] < 0.5)
        & ((components["toward_speed"] - args.target_speed).abs() <= args.speed_window)
        & (components["lateral_speed"].abs() <= args.lateral_speed_max)
        & (components["vertical_speed"].abs() <= args.vertical_speed_max)
        & (components["angular_speed"] <= args.angular_speed_max)
        & (components["behind_projection"] >= args.root_behind_min)
        & (components["behind_lateral_error"] <= args.root_lateral_max)
        & torch.isfinite(reward)
    )
    if not bool(candidate.any()):
        raise ValueError("Rear-push candidate filter produced no replay states")

    device = torch.device(args.device)
    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint",
        device=checkpoint_load_device(args.device),
        strict=False,
    ).to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for topk in args.topks:
        indices = _rank_balanced_topk(candidate, reward, rank_ids, int(topk))
        selected_observation = tree_map(
            lambda value: torch.as_tensor(value)[indices].to(device), observation
        )
        latent = model.reward_wr_inference(
            selected_observation, reward[indices].reshape(-1, 1).to(device)
        ).reshape(1, -1)
        output = args.output_dir / f"topk_{int(topk):04d}"
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(latent.detach().cpu().numpy(), output / "z_push_direct.pkl")
        selection = _stats(indices, reward, components, geometry, rank_ids)
        (output / "rear_push_inference.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        results[str(int(topk))] = selection

    summary = {
        "model_folder": str(args.model_folder),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(state.shape[0]),
        "candidate_count": int(candidate.sum()),
        "reward_design": {
            "heritage": "73.6M minimal grounded target-speed push",
            "target_speed_mps": args.target_speed,
            "soft_geometry": [
                "root behind box",
                "both hands on rear face",
                "hands near box centre height",
                "hands laterally centred and separated",
            ],
            "hard_contact": "at least one hand",
        },
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"buffer_paths", "model_folder", "output_dir", "robot_config"}
        },
        "results": results,
    }
    (args.output_dir / "rear_push_sweep.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[rear-push] candidates={int(candidate.sum())} topks={list(args.topks)} "
        f"output={args.output_dir}",
        flush=True,
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
    parser.add_argument("--speed-window", type=float, default=0.12)
    parser.add_argument("--speed-sigma", type=float, default=0.07)
    parser.add_argument("--max-lift", type=float, default=0.08)
    parser.add_argument("--upright-min", type=float, default=0.70)
    parser.add_argument("--box-upright-min", type=float, default=0.85)
    parser.add_argument("--target-distance-min", type=float, default=0.50)
    parser.add_argument("--lateral-speed-max", type=float, default=0.40)
    parser.add_argument("--vertical-speed-max", type=float, default=0.30)
    parser.add_argument("--angular-speed-max", type=float, default=1.80)
    parser.add_argument("--root-behind-min", type=float, default=0.10)
    parser.add_argument("--root-behind-target", type=float, default=0.55)
    parser.add_argument("--root-behind-sigma", type=float, default=0.28)
    parser.add_argument("--root-lateral-max", type=float, default=0.55)
    parser.add_argument("--root-lateral-sigma", type=float, default=0.28)
    parser.add_argument("--hand-rear-target", type=float, default=0.20)
    parser.add_argument("--hand-rear-sigma", type=float, default=0.20)
    parser.add_argument("--hand-height-target", type=float, default=0.00)
    parser.add_argument("--hand-height-sigma", type=float, default=0.18)
    parser.add_argument("--hand-center-sigma", type=float, default=0.22)
    parser.add_argument("--hand-separation-target", type=float, default=0.30)
    parser.add_argument("--hand-separation-sigma", type=float, default=0.20)
    args = parser.parse_args()
    positives = (
        args.num_samples,
        *args.topks,
        args.target_speed,
        args.speed_window,
        args.speed_sigma,
        args.root_behind_sigma,
        args.root_lateral_sigma,
        args.hand_rear_sigma,
        args.hand_height_sigma,
        args.hand_center_sigma,
        args.hand_separation_sigma,
    )
    if min(positives) <= 0:
        parser.error("sample counts, speeds, windows, and sigmas must be positive")
    if max(args.topks) > args.num_samples:
        parser.error("--topks cannot exceed --num-samples")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("--device must be cpu or cuda[:index]")
    return args


if __name__ == "__main__":
    run(parse_args())
