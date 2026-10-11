"""Infer no-contact navigation turn latents from balanced replay samples."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.carry_speed_latent_sweep import (
    _rank_balanced_topk,
    _rank_counts,
    sample_replay_ranks,
)
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.push_reward_inference import _push_components, _select_latent
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import quat_rotate_inverse


def _name(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _stats(
    indices: torch.Tensor,
    *,
    reward: torch.Tensor,
    rank_ids: torch.Tensor,
    components: dict[str, torch.Tensor],
) -> dict[str, Any]:
    mask = torch.zeros_like(rank_ids, dtype=torch.bool)
    mask[indices] = True
    result: dict[str, Any] = {
        "count": int(indices.numel()),
        "rank_counts": _rank_counts(mask, rank_ids),
        "reward_mean": float(reward[indices].mean().item()),
        "reward_min": float(reward[indices].min().item()),
    }
    for name in (
        "root_yaw_rate",
        "root_planar_speed",
        "root_vertical_speed",
        "object_planar_speed",
        "vertical_speed",
        "angular_speed",
        "contact_any",
        "robot_upright",
        "box_upright",
        "lift",
        "target_distance",
    ):
        values = components[name][indices]
        result[f"{name}_mean"] = float(values.mean().item())
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
    qpos = torch.as_tensor(data["next"]["qpos"], dtype=torch.float32)
    qvel = torch.as_tensor(data["next"]["qvel"], dtype=torch.float32)
    aux = data["aux_rewards"]
    rank_ids = torch.as_tensor(data["rank"], dtype=torch.long)
    components = _push_components(aux, state, num_dof=num_dof)
    root_quat_xyzw = torch.roll(qpos[:, 3:7], shifts=-1, dims=-1)
    root_velocity = quat_rotate_inverse(root_quat_xyzw, qvel[:, :3], w_last=True)
    components["root_planar_speed"] = torch.linalg.vector_norm(
        root_velocity[:, :2], dim=-1
    )
    components["root_vertical_speed"] = root_velocity[:, 2].abs()
    components["object_planar_speed"] = torch.sqrt(
        components["toward_speed"].square()
        + components["lateral_speed"].square()
    )

    conditions = {
        "object_present": components["object_present"] > 0.5,
        "no_contact": components["contact_any"] < 0.5,
        "robot_upright": components["robot_upright"] >= args.upright_min,
        "box_upright": components["box_upright"] >= args.upright_min,
        "grounded": components["lift"] <= args.max_lift,
        "not_drop": components["drop"] < 0.5,
        "root_planar_speed": components["root_planar_speed"]
        <= args.root_planar_speed_max,
        "root_vertical_speed": components["root_vertical_speed"]
        <= args.root_vertical_speed_max,
        "object_planar_speed": components["object_planar_speed"]
        <= args.object_planar_speed_max,
        "object_vertical_speed": components["vertical_speed"].abs()
        <= args.object_vertical_speed_max,
        "object_angular_speed": components["angular_speed"]
        <= args.object_angular_speed_max,
        "target_distance": components["target_distance"] >= args.target_distance_min,
    }
    common = torch.ones_like(rank_ids, dtype=torch.bool)
    for name, condition in conditions.items():
        common &= condition
        print(
            f"[navigation-turn] filter={name} individual={int(condition.sum())} "
            f"cumulative={int(common.sum())}",
            flush=True,
        )
    print(
        "[navigation-turn] "
        f"samples={state.shape[0]} common={int(common.sum())} "
        f"no_contact={int((components['contact_any'] < 0.5).sum())} "
        f"upright={int((components['robot_upright'] >= args.upright_min).sum())} "
        f"root_speed={int((components['root_planar_speed'] <= args.root_planar_speed_max).sum())}",
        flush=True,
    )

    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint",
        device=checkpoint_load_device(args.device),
        strict=False,
    ).to(args.device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "model_folder": str(args.model_folder),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(state.shape[0]),
        "common_candidate_count": int(common.sum().item()),
        "filters": {
            "contact_any": 0,
            "upright_min": args.upright_min,
            "max_lift": args.max_lift,
            "root_planar_speed_max": args.root_planar_speed_max,
            "root_vertical_speed_max": args.root_vertical_speed_max,
            "object_planar_speed_max": args.object_planar_speed_max,
            "object_vertical_speed_max": args.object_vertical_speed_max,
            "object_angular_speed_max": args.object_angular_speed_max,
            "target_distance_min": args.target_distance_min,
        },
        "candidates": {},
    }

    for yaw_rate in args.yaw_rates:
        for topk in args.topks:
            candidate_name = f"yaw_{_name(yaw_rate)}_topk_{topk:04d}"
            bank: dict[int, np.ndarray] = {}
            candidate_stats: dict[str, Any] = {}
            for direction in (-1, 1):
                signed_yaw = components["root_yaw_rate"] * float(direction)
                mask = (
                    common
                    & (signed_yaw > 0.0)
                    & ((signed_yaw - yaw_rate).abs() <= args.yaw_half_width)
                )
                print(
                    f"[navigation-turn] yaw={yaw_rate:.2f} direction={direction:+d} "
                    f"candidates={int(mask.sum())}",
                    flush=True,
                )
                yaw_score = torch.exp(
                    -((signed_yaw - yaw_rate) / args.yaw_sigma).square()
                )
                translation_score = torch.exp(
                    -(components["root_planar_speed"] / args.root_speed_sigma).square()
                )
                stability_score = (
                    components["robot_upright"]
                    * components["box_upright"]
                    * torch.exp(
                        -(components["root_vertical_speed"] / 0.10).square()
                        -(components["object_planar_speed"] / 0.08).square()
                        -(components["angular_speed"] / 0.25).square()
                    )
                )
                reward = yaw_score * translation_score * stability_score
                latent, indices = _select_latent(
                    model,
                    observation,
                    reward.unsqueeze(-1),
                    mask,
                    rank_ids,
                    topk=topk,
                )
                bank[direction] = latent.numpy()
                candidate_stats["left" if direction > 0 else "right"] = {
                    "candidate_count": int(mask.sum().item()),
                    **_stats(
                        indices,
                        reward=reward,
                        rank_ids=rank_ids,
                        components=components,
                    ),
                }
            output = args.output_dir / candidate_name
            output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.baseline_latent_dir / "z_approach.pkl", output)
            joblib.dump(bank, output / "z_turn.pkl")
            (output / "selection.json").write_text(
                json.dumps(candidate_stats, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            results["candidates"][candidate_name] = candidate_stats
            print(f"[navigation-turn] saved {candidate_name}", flush=True)

    (args.output_dir / "navigation_turn_inference.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--baseline-latent-dir", type=Path, required=True)
    parser.add_argument("--buffer-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--seed", type=int, default=166560)
    parser.add_argument("--yaw-rates", type=float, nargs="+", default=(0.2, 0.3, 0.4))
    parser.add_argument("--topks", type=int, nargs="+", default=(50, 100, 200))
    parser.add_argument("--yaw-half-width", type=float, default=0.15)
    parser.add_argument("--yaw-sigma", type=float, default=0.08)
    parser.add_argument("--root-speed-sigma", type=float, default=0.08)
    parser.add_argument("--upright-min", type=float, default=0.95)
    parser.add_argument("--max-lift", type=float, default=0.05)
    parser.add_argument("--root-planar-speed-max", type=float, default=0.15)
    parser.add_argument("--root-vertical-speed-max", type=float, default=0.15)
    parser.add_argument("--object-planar-speed-max", type=float, default=0.10)
    parser.add_argument("--object-vertical-speed-max", type=float, default=0.10)
    parser.add_argument("--object-angular-speed-max", type=float, default=0.30)
    parser.add_argument("--target-distance-min", type=float, default=0.50)
    args = parser.parse_args()
    if args.num_samples <= 0 or min(args.yaw_rates) <= 0 or min(args.topks) <= 0:
        parser.error("sample counts, yaw rates, and top-k values must be positive")
    if max(args.topks) > args.num_samples:
        parser.error("--topks cannot exceed --num-samples")
    return args


if __name__ == "__main__":
    run(parse_args())
