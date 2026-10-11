"""Infer direct latents for stable, constant-speed ground pushing.

Unlike the deliberately minimal push ablation, this reward requires hand
contact and filters out fallen, lifted, tumbling, and badly misaligned replay
states.  Multiple requested speeds and the modal replay push speed are encoded
from one balanced replay sample and one model load.
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


def _speed_name(speed: float) -> str:
    return f"speed_{speed:.3f}".replace(".", "p")


def _stable_push_reward(
    components: Mapping[str, torch.Tensor],
    *,
    target_speed: float,
    speed_sigma: float,
    lateral_sigma: float,
    vertical_sigma: float,
    angular_sigma: float,
    behind_distance: float,
    behind_sigma: float,
    alignment_sigma: float,
) -> torch.Tensor:
    speed = torch.exp(
        -torch.square((components["toward_speed"] - target_speed) / speed_sigma)
    )
    lateral = torch.exp(-torch.square(components["lateral_speed"] / lateral_sigma))
    vertical = torch.exp(-torch.square(components["vertical_speed"] / vertical_sigma))
    angular = torch.exp(-torch.square(components["angular_speed"] / angular_sigma))
    behind = torch.sigmoid(
        (components["behind_projection"] - behind_distance) / behind_sigma
    )
    alignment = torch.exp(
        -torch.square(components["behind_lateral_error"] / alignment_sigma)
    )
    # Any physical hand contact is mandatory in the candidate mask.  This
    # factor gives bilateral states extra weight without making them so rare
    # that the posterior is estimated from only a handful of samples.
    contact = 0.50 + 0.50 * components["contact_bilateral"]
    return (
        speed
        * lateral
        * vertical
        * angular
        * behind
        * alignment
        * contact
        * components["robot_upright"]
        * components["box_upright"]
    )


def _stats(
    indices: torch.Tensor,
    reward: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    rank_ids: torch.Tensor,
) -> dict[str, Any]:
    mask = torch.zeros_like(rank_ids, dtype=torch.bool)
    mask[indices] = True
    result: dict[str, Any] = {
        "count": int(indices.numel()),
        "rank_counts": _rank_counts(mask, rank_ids),
        "reward_mean": float(reward[indices].mean().item()),
        "reward_min": float(reward[indices].min().item()),
        "reward_max": float(reward[indices].max().item()),
    }
    for name in (
        "toward_speed",
        "lateral_speed",
        "vertical_speed",
        "angular_speed",
        "contact_any",
        "contact_bilateral",
        "lift",
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


def _speed_histogram(
    speed: torch.Tensor,
    mask: torch.Tensor,
    *,
    bin_width: float,
    max_speed: float,
) -> tuple[list[dict[str, float | int]], float]:
    values = speed[mask & (speed >= 0.0) & (speed < max_speed)].cpu().numpy()
    edges = np.arange(0.0, max_speed + bin_width, bin_width, dtype=np.float64)
    counts, edges = np.histogram(values, bins=edges)
    rows = [
        {
            "low_mps": float(edges[i]),
            "high_mps": float(edges[i + 1]),
            "center_mps": float((edges[i] + edges[i + 1]) * 0.5),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]
    # Ignore the near-zero bin: the requested mode is the most common moving
    # push speed, not stationary hand contact with a box.
    moving_rows = [row for row in rows if row["high_mps"] > bin_width]
    modal = max(moving_rows, key=lambda row: int(row["count"]))
    return rows, float(modal["center_mps"])


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
            f"Expected non-ablated Carry state {expected_state_dim}D, got {state.shape[-1]}D"
        )

    components = _push_components(data["aux_rewards"], state, num_dof=num_dof)
    stable_contact = (
        (components["object_present"] > 0.5)
        & (components["contact_any"] > 0.5)
        & (components["target_distance"] >= args.target_distance_min)
        & (components["lift"] <= args.max_lift)
        & (components["robot_upright"] >= args.upright_min)
        & (components["box_upright"] >= args.upright_min)
        & (components["drop"] < 0.5)
        & (components["toward_speed"] > 0.0)
        & (torch.abs(components["lateral_speed"]) <= args.lateral_speed_max)
        & (torch.abs(components["vertical_speed"]) <= args.vertical_speed_max)
        & (components["angular_speed"] <= args.angular_speed_max)
        & (components["behind_projection"] >= args.behind_projection_min)
        & (components["behind_lateral_error"] <= args.behind_lateral_max)
    )
    histogram, modal_speed = _speed_histogram(
        components["toward_speed"],
        stable_contact,
        bin_width=args.histogram_bin_width,
        max_speed=args.histogram_max_speed,
    )
    targets = list(args.target_speeds)
    if args.include_modal_speed:
        targets.append(modal_speed)
    targets = sorted(set(round(float(speed), 6) for speed in targets))

    device = torch.device(args.device)
    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint",
        device=checkpoint_load_device(args.device),
        strict=False,
    ).to(device).eval()

    results: dict[str, Any] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target_speed in targets:
        reward = _stable_push_reward(
            components,
            target_speed=target_speed,
            speed_sigma=args.speed_sigma,
            lateral_sigma=args.lateral_sigma,
            vertical_sigma=args.vertical_sigma,
            angular_sigma=args.angular_sigma,
            behind_distance=args.behind_distance,
            behind_sigma=args.behind_sigma,
            alignment_sigma=args.alignment_sigma,
        )
        speed_window = (
            torch.abs(components["toward_speed"] - target_speed)
            <= args.candidate_speed_window
        )
        candidate = stable_contact & speed_window & torch.isfinite(reward)
        result = {
            "target_speed_mps": target_speed,
            "candidate_count": int(candidate.sum().item()),
            "topk": {},
        }
        for topk in args.topks:
            indices = _rank_balanced_topk(candidate, reward, rank_ids, topk)
            selected_observation = tree_map(
                lambda value: torch.as_tensor(value)[indices].to(device), observation
            )
            latent = model.reward_wr_inference(
                selected_observation, reward[indices].reshape(-1, 1).to(device)
            ).reshape(1, -1)
            output = args.output_dir / _speed_name(target_speed) / f"topk_{topk:04d}"
            output.mkdir(parents=True, exist_ok=True)
            joblib.dump(latent.detach().cpu().numpy(), output / "z_push_direct.pkl")
            selection = _stats(indices, reward, components, rank_ids)
            metadata = {
                "target_speed_mps": target_speed,
                "candidate_count": int(candidate.sum().item()),
                "requested_topk": int(topk),
                "selected": selection,
            }
            (output / "stable_push_inference.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result["topk"][str(int(topk))] = selection
        results[_speed_name(target_speed)] = result

    summary = {
        "model_folder": str(args.model_folder),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(state.shape[0]),
        "stable_contact_candidate_count": int(stable_contact.sum().item()),
        "modal_moving_push_speed_mps": modal_speed,
        "histogram_bin_width_mps": float(args.histogram_bin_width),
        "speed_histogram": histogram,
        "filters": {
            "contact": "at_least_one_hand",
            "max_lift": float(args.max_lift),
            "upright_min": float(args.upright_min),
            "lateral_speed_max": float(args.lateral_speed_max),
            "vertical_speed_max": float(args.vertical_speed_max),
            "angular_speed_max": float(args.angular_speed_max),
            "behind_projection_min": float(args.behind_projection_min),
            "behind_lateral_max": float(args.behind_lateral_max),
            "candidate_speed_window": float(args.candidate_speed_window),
        },
        "results": results,
    }
    (args.output_dir / "stable_push_sweep.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[stable-push] candidates={int(stable_contact.sum())} "
        f"modal_speed={modal_speed:.3f} targets={targets} output={args.output_dir}",
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
    parser.add_argument("--target-speeds", type=float, nargs="+", default=(0.2, 0.3))
    parser.add_argument(
        "--include-modal-speed", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--histogram-bin-width", type=float, default=0.025)
    parser.add_argument("--histogram-max-speed", type=float, default=0.8)
    parser.add_argument("--candidate-speed-window", type=float, default=0.10)
    parser.add_argument("--speed-sigma", type=float, default=0.06)
    parser.add_argument("--lateral-sigma", type=float, default=0.15)
    parser.add_argument("--vertical-sigma", type=float, default=0.12)
    parser.add_argument("--angular-sigma", type=float, default=0.60)
    parser.add_argument("--behind-distance", type=float, default=0.20)
    parser.add_argument("--behind-sigma", type=float, default=0.15)
    parser.add_argument("--alignment-sigma", type=float, default=0.40)
    parser.add_argument("--max-lift", type=float, default=0.08)
    parser.add_argument("--upright-min", type=float, default=0.75)
    parser.add_argument("--target-distance-min", type=float, default=0.50)
    parser.add_argument("--lateral-speed-max", type=float, default=0.35)
    parser.add_argument("--vertical-speed-max", type=float, default=0.25)
    parser.add_argument("--angular-speed-max", type=float, default=1.50)
    parser.add_argument("--behind-projection-min", type=float, default=0.0)
    parser.add_argument("--behind-lateral-max", type=float, default=0.75)
    args = parser.parse_args()
    positive = (
        args.num_samples,
        *args.topks,
        *args.target_speeds,
        args.histogram_bin_width,
        args.histogram_max_speed,
        args.candidate_speed_window,
        args.speed_sigma,
        args.lateral_sigma,
        args.vertical_sigma,
        args.angular_sigma,
        args.behind_sigma,
        args.alignment_sigma,
    )
    if min(positive) <= 0:
        parser.error("sample counts, speeds, windows, and reward sigmas must be positive")
    if max(args.topks) > args.num_samples:
        parser.error("--topks cannot exceed --num-samples")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("--device must be cpu or cuda[:index]")
    return args


if __name__ == "__main__":
    run(parse_args())
