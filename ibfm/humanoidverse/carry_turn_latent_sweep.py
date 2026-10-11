"""Infer stable, fast in-place turn latents from every replay rank.

The original phased Carry latent bank inferred turns from an unfiltered replay
sample.  Its selected states rotate at only about 0.3 rad/s even when the
requested yaw rate is 0.8 rad/s.  This tool balances all replay ranks, filters
for upright object-free states, and exports isolated turn-latent candidates.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.carry_phased_reward_inference import compute_turn_reward
from humanoidverse.carry_speed_latent_sweep import (
    _adaptive_mask,
    _even_counts,
    _rank_balanced_topk,
    _rank_counts,
    _weighted_mean,
    sample_replay_ranks,
)
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _rate_name(yaw_rate: float) -> str:
    return f"yaw_{float(yaw_rate):.1f}".replace(".", "p")


def infer_turn_candidates(
    model: Any,
    data: dict[str, Any],
    *,
    num_dof: int,
    yaw_rate: float,
    topks: Sequence[int],
    yaw_rate_half_widths: Sequence[float],
    upright_min: float,
    root_height_min: float,
    planar_speed_max: float,
    vertical_speed_max: float,
    yaw_rate_sigma: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    next_data = data["next"]
    observation = next_data["observation"]
    state = torch.as_tensor(observation["state"], dtype=torch.float32)
    qpos = torch.as_tensor(next_data["qpos"], dtype=torch.float32)
    qvel = torch.as_tensor(next_data["qvel"], dtype=torch.float32)
    rank_ids = torch.as_tensor(data["rank"], dtype=torch.long)
    rank_count = int(torch.unique(rank_ids).numel())
    required_per_rank = max(_even_counts(max(int(value) for value in topks), rank_count))

    projected_gravity_z = state[:, 2 * int(num_dof) + 2]
    root_yaw_rate = state[:, 2 * int(num_dof) + 5]
    object_present = state[:, -2]
    root_height = qpos[:, 2]
    vertical_speed = qvel[:, 2]

    candidates_by_topk: dict[int, dict[int, torch.Tensor]] = {int(topk): {} for topk in topks}
    direction_stats: dict[str, Any] = {}
    for direction in (-1, 1):
        target = float(direction) * float(yaw_rate)
        reward, components = compute_turn_reward(
            qpos,
            qvel,
            state,
            num_dof=num_dof,
            target_yaw_rate=target,
            yaw_rate_sigma=float(yaw_rate_sigma),
            planar_speed_sigma=float(planar_speed_max),
        )
        signed_rate = float(direction) * root_yaw_rate

        def build_mask(width: float) -> torch.Tensor:
            return (
                (object_present < 0.5)
                & (-projected_gravity_z >= float(upright_min))
                & (root_height >= float(root_height_min))
                & (torch.abs(signed_rate - float(yaw_rate)) <= float(width))
                & (components["planar_speed"] <= float(planar_speed_max))
                & (torch.abs(vertical_speed) <= float(vertical_speed_max))
            )

        valid, selected_width, counts = _adaptive_mask(
            build_mask,
            yaw_rate_half_widths,
            rank_ids=rank_ids,
            required_per_rank=required_per_rank,
        )
        key = "left" if direction > 0 else "right"
        topk_stats: dict[str, Any] = {}
        for topk in topks:
            indices = _rank_balanced_topk(valid, reward[:, 0], rank_ids, int(topk))
            selected_obs = tree_map(lambda value: value[indices], observation)
            selected_reward = reward[indices]
            latent = model.reward_wr_inference(selected_obs, selected_reward).reshape(1, -1)
            candidates_by_topk[int(topk)][direction] = latent.detach().cpu()
            selected_mask = torch.isin(torch.arange(rank_ids.numel()), indices)
            topk_stats[str(int(topk))] = {
                "count": int(indices.numel()),
                "rank_counts": _rank_counts(selected_mask, rank_ids),
                "reward_mean": float(selected_reward.mean().item()),
                "yaw_rate_mean": float(root_yaw_rate[indices].mean().item()),
                "yaw_rate_reward_weighted_mean": _weighted_mean(root_yaw_rate[indices], selected_reward),
                "yaw_rate_abs_mean": float(root_yaw_rate[indices].abs().mean().item()),
                "planar_speed_mean": float(components["planar_speed"][indices].mean().item()),
                "vertical_speed_abs_mean": float(vertical_speed[indices].abs().mean().item()),
                "upright_mean": float((-projected_gravity_z[indices]).mean().item()),
                "root_height_mean": float(root_height[indices].mean().item()),
            }
        direction_stats[key] = {
            "target_yaw_rate": target,
            "yaw_rate_half_width": selected_width,
            "candidate_count": int(valid.sum().item()),
            "candidates_per_rank": counts,
            "topk": topk_stats,
        }
    return candidates_by_topk, direction_stats


def export_candidate(
    output_dir: Path,
    baseline_latent_dir: Path,
    *,
    z_turn: dict[int, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in baseline_latent_dir.glob("z_*.pkl"):
        shutil.copy2(source, output_dir / source.name)
    joblib.dump({key: value.numpy() for key, value in z_turn.items()}, output_dir / "z_turn.pkl")
    (output_dir / "turn_inference.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    data = sample_replay_ranks(args.buffer_paths, sample_count=args.num_samples, seed=args.seed)
    robot = load_robot_training_spec(args.robot_config)
    num_dof = len(robot.robot.control_joint_names)
    state_dim = int(data["next"]["observation"]["state"].shape[-1])
    if state_dim != 2 * num_dof + 40:
        raise ValueError(f"Expected non-ablated Carry state, got {state_dim}D")

    load_device = checkpoint_load_device(args.device)
    model = load_model_from_checkpoint_dir(args.model_folder / "checkpoint", device=load_device)
    model.to(args.device).eval()
    summary: dict[str, Any] = {
        "model_folder": str(args.model_folder),
        "baseline_latent_dir": str(args.baseline_latent_dir),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(data["rank"].numel()),
        "sample_count_per_rank": _rank_counts(
            torch.ones_like(data["rank"], dtype=torch.bool), data["rank"]
        ),
        "filters": {
            "source": "object_present=0 (LAFAN)",
            "upright_min": args.upright_min,
            "root_height_min": args.root_height_min,
            "planar_speed_max": args.planar_speed_max,
            "vertical_speed_max": args.vertical_speed_max,
            "yaw_rate_sigma": args.yaw_rate_sigma,
        },
        "yaw_rates": {},
    }
    for yaw_rate in args.yaw_rates:
        print(f"[INFO] inferring turn yaw_rate={yaw_rate:.2f} rad/s", flush=True)
        candidates, stats = infer_turn_candidates(
            model,
            data,
            num_dof=num_dof,
            yaw_rate=float(yaw_rate),
            topks=args.topks,
            yaw_rate_half_widths=args.yaw_rate_half_widths,
            upright_min=args.upright_min,
            root_height_min=args.root_height_min,
            planar_speed_max=args.planar_speed_max,
            vertical_speed_max=args.vertical_speed_max,
            yaw_rate_sigma=args.yaw_rate_sigma,
        )
        rate_name = _rate_name(yaw_rate)
        summary["yaw_rates"][rate_name] = stats
        for topk, z_turn in candidates.items():
            export_candidate(
                args.output_dir / rate_name / f"topk_{topk:04d}",
                args.baseline_latent_dir,
                z_turn=z_turn,
                metadata={
                    "target_yaw_rate_abs": float(yaw_rate),
                    "topk": int(topk),
                    "directions": {
                        key: value["topk"][str(int(topk))] | {
                            "candidate_count": value["candidate_count"],
                            "yaw_rate_half_width": value["yaw_rate_half_width"],
                        }
                        for key, value in stats.items()
                    },
                    "preserved_from_baseline": [
                        "z_approach.pkl",
                        "z_lift.pkl",
                        "z_transport.pkl",
                        "z_transport_slow.pkl",
                        "z_transport_turn.pkl",
                        "z_align.pkl",
                        "z_lower.pkl",
                        "z_place.pkl",
                        "z_hold.pkl",
                    ],
                },
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "turn_inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[INFO] saved turn latent bank to {args.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--baseline-latent-dir", type=Path, required=True)
    parser.add_argument("--buffer-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--yaw-rates", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.5])
    parser.add_argument("--topks", type=int, nargs="+", default=[50, 100, 200, 500])
    parser.add_argument("--yaw-rate-half-widths", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--upright-min", type=float, default=0.95)
    parser.add_argument("--root-height-min", type=float, default=0.65)
    parser.add_argument("--planar-speed-max", type=float, default=0.20)
    parser.add_argument("--vertical-speed-max", type=float, default=0.15)
    parser.add_argument("--yaw-rate-sigma", type=float, default=0.15)
    args = parser.parse_args()
    if args.num_samples <= 0 or min(args.topks) <= 0:
        parser.error("--num-samples and --topks must be positive")
    if not args.yaw_rates or min(args.yaw_rates) <= 0.0:
        parser.error("--yaw-rates must contain positive values")
    if not args.yaw_rate_half_widths or min(args.yaw_rate_half_widths) <= 0.0:
        parser.error("--yaw-rate-half-widths must contain positive values")
    if args.yaw_rate_half_widths != sorted(args.yaw_rate_half_widths):
        parser.error("--yaw-rate-half-widths must be sorted from narrow to wide")
    if not 0.0 < args.upright_min <= 1.0:
        parser.error("--upright-min must be in (0, 1]")
    if min(args.root_height_min, args.planar_speed_max, args.vertical_speed_max, args.yaw_rate_sigma) <= 0.0:
        parser.error("turn stability thresholds and sigma must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
