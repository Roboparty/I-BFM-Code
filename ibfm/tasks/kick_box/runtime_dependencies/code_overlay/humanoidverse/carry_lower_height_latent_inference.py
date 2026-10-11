"""Infer contact-preserving Carry lowering latents for three object-height bands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.carry_phased_reward_inference import _state_object_context_dim
from humanoidverse.carry_speed_latent_sweep import (
    _rank_balanced_topk,
    _rank_counts,
    sample_replay_ranks,
)
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    levels = torch.tensor((0.0, 0.1, 0.5, 0.9, 1.0), dtype=torch.float32)
    result = torch.quantile(value.float(), levels)
    return {
        f"q{int(level.item() * 100):02d}": float(item)
        for level, item in zip(levels, result)
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    replay = sample_replay_ranks(
        args.buffer_paths,
        sample_count=int(args.num_samples),
        seed=int(args.seed),
        include_current_state=True,
    )
    robot = load_robot_training_spec(args.robot_config)
    num_dof = len(robot.robot.control_joint_names)
    state = torch.as_tensor(replay["next"]["observation"]["state"], dtype=torch.float32)
    current_state = torch.as_tensor(replay["current_state"], dtype=torch.float32)
    expected_state_dim = 2 * num_dof + 40
    if int(state.shape[-1]) != expected_state_dim:
        raise ValueError(f"Expected {expected_state_dim}D state, got {state.shape[-1]}D")

    object_start = 2 * num_dof + 6
    context_dim = _state_object_context_dim(state, num_dof=num_dof)
    present = state[:, object_start + context_dim - 2] > 0.5
    current_present = current_state[:, object_start + context_dim - 2] > 0.5
    contact = state[:, object_start + 18 : object_start + 20] > 0.5
    current_contact = current_state[:, object_start + 18 : object_start + 20] > 0.5
    persistent_bilateral = contact.all(dim=-1) & current_contact.all(dim=-1)
    object_lin_vel = state[:, object_start + 9 : object_start + 12]
    object_ang_vel = state[:, object_start + 12 : object_start + 15]
    target_distance = torch.linalg.vector_norm(
        state[:, object_start + 15 : object_start + 17], dim=-1
    )
    horizontal_speed = torch.linalg.vector_norm(object_lin_vel[:, :2], dim=-1)
    angular_speed = torch.linalg.vector_norm(object_ang_vel, dim=-1)
    vertical_speed = object_lin_vel[:, 2]
    aux = replay["aux_rewards"]
    lift = torch.as_tensor(aux["carry_lift"], dtype=torch.float32).reshape(-1)
    upright = (
        torch.as_tensor(aux["carry_robot_upright"], dtype=torch.float32).reshape(-1)
        * torch.as_tensor(aux["carry_upright"], dtype=torch.float32).reshape(-1)
    )
    common = (
        present
        & current_present
        & persistent_bilateral
        & (target_distance <= float(args.max_goal_distance))
        & (vertical_speed >= float(args.min_vertical_speed))
        & (vertical_speed <= float(args.max_vertical_speed))
        & (horizontal_speed <= float(args.max_horizontal_speed))
        & (angular_speed <= float(args.max_angular_speed))
        & (upright >= float(args.min_upright))
    )
    score = upright * torch.exp(
        -torch.square(
            (vertical_speed - float(args.target_vertical_speed))
            / float(args.vertical_speed_sigma)
        )
        -torch.square(horizontal_speed / float(args.horizontal_speed_sigma))
        -torch.square(angular_speed / float(args.angular_speed_sigma))
    )

    bands = {
        "high": (float(args.high_lift), 1.01),
        "mid": (float(args.mid_lift), float(args.high_lift)),
        "low": (float(args.min_lift), float(args.mid_lift)),
    }
    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint",
        device=checkpoint_load_device(args.device),
    ).to(args.device).eval()
    rank_ids = replay["rank"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "sample_count": int(state.shape[0]),
        "common_candidate_count": int(common.sum().item()),
        "filter": (
            "object present; current+next bilateral contact; near goal; upright; "
            "bounded vertical, horizontal and angular velocity"
        ),
        "bands": {},
    }
    for name, (lower, upper) in bands.items():
        valid = common & (lift >= lower) & (lift < upper)
        if int(valid.sum().item()) < int(args.min_candidates):
            raise ValueError(
                f"Only {int(valid.sum().item())} candidates for {name} lift band "
                f"[{lower}, {upper}); need {args.min_candidates}"
            )
        if args.global_topk:
            valid_indices = torch.where(valid)[0]
            count = min(int(args.topk), int(valid_indices.numel()))
            indices = valid_indices[
                torch.topk(score[valid_indices], k=count, largest=True).indices
            ]
        else:
            indices = _rank_balanced_topk(valid, score, rank_ids, int(args.topk))
        observation = tree_map(lambda value: value[indices], replay["next"]["observation"])
        selected_score = score[indices].reshape(-1, 1)
        latent = model.reward_wr_inference(observation, selected_score).reshape(1, -1)
        path = args.output_dir / f"z_direct_place_lower_{name}.pkl"
        joblib.dump(latent.detach().cpu().numpy(), path)
        metadata["bands"][name] = {
            "lift_range": [lower, upper],
            "candidate_count": int(valid.sum().item()),
            "selected_count": int(indices.numel()),
            "selected_per_rank": _rank_counts(
                torch.isin(torch.arange(rank_ids.numel()), indices), rank_ids
            ),
            "score": _quantiles(selected_score[:, 0]),
            "lift": _quantiles(lift[indices]),
            "vertical_speed_mps": _quantiles(vertical_speed[indices]),
            "horizontal_speed_mps": _quantiles(horizontal_speed[indices]),
            "angular_speed_radps": _quantiles(angular_speed[indices]),
            "latent_path": str(path),
        }
    metadata_path = args.output_dir / "lower_height_inference.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] saved height-conditioned lowering latents to {args.output_dir}")
    print(f"[INFO] saved metadata to {metadata_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--buffer-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument(
        "--global-topk",
        action="store_true",
        help="Select the best transitions globally instead of forcing equal rank counts.",
    )
    parser.add_argument("--min-candidates", type=int, default=50)
    parser.add_argument("--max-goal-distance", type=float, default=0.50)
    parser.add_argument("--target-vertical-speed", type=float, default=-0.08)
    parser.add_argument("--min-vertical-speed", type=float, default=-0.20)
    parser.add_argument("--max-vertical-speed", type=float, default=-0.015)
    parser.add_argument("--vertical-speed-sigma", type=float, default=0.05)
    parser.add_argument("--max-horizontal-speed", type=float, default=0.18)
    parser.add_argument("--horizontal-speed-sigma", type=float, default=0.10)
    parser.add_argument("--max-angular-speed", type=float, default=0.50)
    parser.add_argument("--angular-speed-sigma", type=float, default=0.30)
    parser.add_argument("--min-upright", type=float, default=0.75)
    parser.add_argument("--high-lift", type=float, default=0.75)
    parser.add_argument("--mid-lift", type=float, default=0.45)
    parser.add_argument("--low-lift", type=float, default=0.15)
    parser.add_argument("--min-lift", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("--device must be cpu or cuda[:index]")
    if min(args.num_samples, args.topk, args.min_candidates) <= 0:
        parser.error("sample counts and top-k must be positive")
    if not 0.0 < args.min_lift < args.low_lift < args.mid_lift < args.high_lift <= 1.0:
        parser.error("lift bands must satisfy min < low < mid < high <= 1")
    if not args.min_vertical_speed < args.target_vertical_speed < args.max_vertical_speed < 0.0:
        parser.error("vertical speeds must satisfy min < target < max < 0")
    return args


if __name__ == "__main__":
    run(parse_args())
