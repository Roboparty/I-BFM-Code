"""Infer controlled Approach/Transport speed latents from all replay ranks.

The historical phased Carry inference sampled one replay shard and allowed lift,
transport, and place states to share the same low-speed candidate pool.  This
tool reads equal samples from every rank, applies phase-specific filters, and
exports several small-top-k candidates without modifying existing latents.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.carry_phased_reward_inference import (
    compute_approach_speed_reward,
    compute_transport_reward,
)
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG
from humanoidverse.utils.robot_spec import load_robot_training_spec

_OBSERVATION_KEYS = ("state", "privileged_state", "last_action")
_CARRY_AUX_KEYS = (
    "carry_bilateral_contact",
    "carry_lift",
    "carry_robot_upright",
    "carry_upright",
    "carry_drop",
)


def _even_counts(total: int, count: int) -> list[int]:
    result = [int(total) // int(count)] * int(count)
    for index in range(int(total) % int(count)):
        result[index] += 1
    return result


def _tree_cat(rows: Sequence[Any]) -> Any:
    if not rows:
        raise ValueError("Cannot concatenate an empty replay sample")
    first = rows[0]
    if isinstance(first, Mapping):
        return {key: _tree_cat([row[key] for row in rows]) for key in first}
    return torch.cat([torch.as_tensor(row) for row in rows], dim=0)


def _read_selected(dataset: h5py.Dataset, time_indices: np.ndarray, env_indices: np.ndarray) -> torch.Tensor:
    # Formal shards are contiguous 5000 x 1024 arrays. The sampler deliberately
    # clusters rows by time so this reads only the selected time slices.
    unique_times, inverse = np.unique(time_indices, return_inverse=True)
    values = np.asarray(dataset[unique_times, ...])
    selected = np.asarray(values[inverse, env_indices]).copy()
    del values
    return torch.from_numpy(selected)


def _clustered_valid_sample(
    valid: np.ndarray, *, sample_count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    time_count, env_count = valid.shape
    selected: list[np.ndarray] = []
    total = 0
    for time_index in rng.permutation(time_count):
        envs = np.flatnonzero(valid[int(time_index)])
        if envs.size == 0:
            continue
        selected.append(int(time_index) * env_count + envs)
        total += int(envs.size)
        if total >= int(sample_count):
            break
    if total < int(sample_count):
        raise ValueError(f"Requested {sample_count} samples but only collected {total}")
    flat = np.concatenate(selected)
    flat = rng.choice(flat, size=int(sample_count), replace=False)
    return flat // env_count, flat % env_count


def sample_replay_rank(
    buffer_path: Path,
    *,
    sample_count: int,
    seed: int,
    rank: int,
    include_current_state: bool = False,
    include_current_observation: bool = False,
) -> dict[str, Any]:
    """Uniformly sample valid next states from one trajectory-buffer HDF5 shard."""

    config_path = buffer_path / "config.json"
    h5_path = buffer_path / "buffer.hdf5"
    if not config_path.is_file() or not h5_path.is_file():
        raise FileNotFoundError(f"Replay rank is incomplete: {buffer_path}")
    config = json.loads(config_path.read_text())
    cursor = (int(config.get("_idx", 0)) - 1) % int(config["capacity"])
    rng = np.random.default_rng(int(seed))

    with h5py.File(h5_path, "r") as handle:
        truncated = np.asarray(handle["truncated"][..., 0], dtype=bool)
        time_count, env_count = truncated.shape
        next_truncated = np.roll(truncated, shift=-1, axis=0)
        valid = ~(truncated | next_truncated)
        valid[cursor, :] = False  # newest -> oldest is not a temporal transition
        valid_count = int(valid.sum())
        if sample_count > valid_count:
            raise ValueError(f"Rank {rank} requested {sample_count} samples but only {valid_count} valid transitions exist")
        time_indices, env_indices = _clustered_valid_sample(
            valid, sample_count=int(sample_count), rng=rng
        )
        next_time_indices = (time_indices + 1) % time_count

        observation = {key: _read_selected(handle[f"observation-{key}"], next_time_indices, env_indices) for key in _OBSERVATION_KEYS}
        result = {
            "next": {
                "observation": observation,
                "qpos": _read_selected(handle["qpos"], next_time_indices, env_indices),
                "qvel": _read_selected(handle["qvel"], next_time_indices, env_indices),
            },
            # aux_rewards at slot t describe the state reached by transition t.
            "aux_rewards": {key: _read_selected(handle[f"aux_rewards-{key}"], time_indices, env_indices) for key in _CARRY_AUX_KEYS},
            "rank": torch.full((int(sample_count),), int(rank), dtype=torch.long),
        }
        if include_current_observation:
            result["current"] = {
                "observation": {
                    key: _read_selected(handle[f"observation-{key}"], time_indices, env_indices)
                    for key in _OBSERVATION_KEYS
                }
            }
            if include_current_state:
                result["current_state"] = result["current"]["observation"]["state"]
        elif include_current_state:
            result["current_state"] = _read_selected(handle["observation-state"], time_indices, env_indices)
    del truncated, next_truncated, valid
    gc.collect()
    return result


def sample_replay_ranks(
    buffer_paths: Sequence[Path],
    *,
    sample_count: int,
    seed: int,
    include_current_state: bool = False,
    include_current_observation: bool = False,
) -> dict[str, Any]:
    counts = _even_counts(int(sample_count), len(buffer_paths))
    rows = []
    for rank, (path, count) in enumerate(zip(buffer_paths, counts)):
        print(f"[INFO] sampling rank={rank} count={count} path={path}", flush=True)
        rows.append(
            sample_replay_rank(
                path,
                sample_count=count,
                seed=seed + rank,
                rank=rank,
                include_current_state=include_current_state,
                include_current_observation=include_current_observation,
            )
        )
    return _tree_cat(rows)


def _rank_counts(mask: torch.Tensor, rank_ids: torch.Tensor) -> dict[str, int]:
    return {str(int(rank)): int((mask & (rank_ids == rank)).sum().item()) for rank in torch.unique(rank_ids, sorted=True)}


def _rank_balanced_topk(
    mask: torch.Tensor,
    score: torch.Tensor,
    rank_ids: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select high-score rows while giving every replay rank equal quota."""

    ranks = torch.unique(rank_ids, sorted=True)
    quotas = _even_counts(int(topk), int(ranks.numel()))
    selected_parts: list[torch.Tensor] = []
    selected_mask = torch.zeros_like(mask, dtype=torch.bool)
    for rank, quota in zip(ranks, quotas):
        candidates = torch.nonzero(mask & (rank_ids == rank), as_tuple=False).reshape(-1)
        take = min(int(quota), int(candidates.numel()))
        if take:
            chosen = candidates[torch.topk(score[candidates], k=take, largest=True).indices]
            selected_parts.append(chosen)
            selected_mask[chosen] = True
    selected_count = sum(int(part.numel()) for part in selected_parts)
    if selected_count < int(topk):
        remaining = torch.nonzero(mask & ~selected_mask, as_tuple=False).reshape(-1)
        take = min(int(topk) - selected_count, int(remaining.numel()))
        if take:
            selected_parts.append(remaining[torch.topk(score[remaining], k=take, largest=True).indices])
    if not selected_parts:
        raise ValueError("Controlled filter produced no candidates")
    return torch.cat(selected_parts)


def _weighted_mean(values: torch.Tensor, reward: torch.Tensor) -> float:
    reward = reward.reshape(-1)
    weights = reward * torch.softmax(10.0 * reward, dim=0)
    denominator = torch.clamp(weights.sum(), min=1e-12)
    return float((values.reshape(-1) * weights / denominator).sum().item())


def _adaptive_mask(
    build_mask: Any,
    widths: Sequence[float],
    *,
    rank_ids: torch.Tensor,
    required_per_rank: int,
) -> tuple[torch.Tensor, float, dict[str, int]]:
    last_mask = None
    last_counts = None
    for width in widths:
        mask = build_mask(float(width))
        counts = _rank_counts(mask, rank_ids)
        last_mask, last_counts = mask, counts
        if counts and min(counts.values()) >= int(required_per_rank):
            return mask, float(width), counts
    assert last_mask is not None and last_counts is not None
    return last_mask, float(widths[-1]), last_counts


def infer_speed_candidates(
    model: Any,
    data: dict[str, Any],
    *,
    num_dof: int,
    speed: float,
    topks: Sequence[int],
    speed_widths: Sequence[float],
    approach_upright_min: float,
    transport_upright_min: float,
    transport_lift_min: float,
    lateral_speed_max: float,
    vertical_speed_max: float,
    yaw_rate_max: float,
    transport_target_distance_min: float,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    next_data = data["next"]
    observation = next_data["observation"]
    state = torch.as_tensor(observation["state"], dtype=torch.float32)
    qpos = torch.as_tensor(next_data["qpos"], dtype=torch.float32)
    qvel = torch.as_tensor(next_data["qvel"], dtype=torch.float32)
    rank_ids = torch.as_tensor(data["rank"], dtype=torch.long)
    max_topk_per_rank = max(_even_counts(max(int(value) for value in topks), int(torch.unique(rank_ids).numel())))

    approach_reward, root_velocity = compute_approach_speed_reward(
        qpos,
        qvel,
        state,
        num_dof=num_dof,
        target_speed=float(speed),
        box_heading_sigma_deg=None,
    )
    projected_gravity_z = state[:, 2 * int(num_dof) + 2]
    root_yaw_rate = state[:, 2 * int(num_dof) + 5]
    object_present = state[:, -2]

    def approach_mask(width: float) -> torch.Tensor:
        return (
            (object_present < 0.5)
            & (-projected_gravity_z >= float(approach_upright_min))
            & (torch.abs(root_velocity[:, 0] - float(speed)) <= width)
            & (torch.abs(root_velocity[:, 1]) <= float(lateral_speed_max))
            & (torch.abs(root_velocity[:, 2]) <= float(vertical_speed_max))
            & (torch.abs(root_yaw_rate) <= float(yaw_rate_max))
        )

    approach_valid, approach_width, approach_counts = _adaptive_mask(
        approach_mask,
        speed_widths,
        rank_ids=rank_ids,
        required_per_rank=max_topk_per_rank,
    )

    aux = {key: torch.as_tensor(value, dtype=torch.float32).reshape(-1) for key, value in data["aux_rewards"].items()}
    transport_reward, transport_components = compute_transport_reward(
        {key: value.unsqueeze(-1) for key, value in aux.items()},
        state,
        num_dof=num_dof,
        target_speed=float(speed),
    )
    object_start = 2 * int(num_dof) + 6
    object_velocity = state[:, object_start + 9 : object_start + 12]
    toward_speed = transport_components["toward_speed"]
    lateral_speed = transport_components["lateral_speed"]
    target_distance = transport_components["target_distance"]

    def transport_mask(width: float) -> torch.Tensor:
        return (
            (object_present > 0.5)
            & (aux["carry_bilateral_contact"] > 0.5)
            & (aux["carry_lift"] >= float(transport_lift_min))
            & (aux["carry_robot_upright"] >= float(transport_upright_min))
            & (aux["carry_upright"] >= float(transport_upright_min))
            & (aux["carry_drop"] < 0.5)
            & (target_distance >= float(transport_target_distance_min))
            & (torch.abs(toward_speed - float(speed)) <= width)
            & (torch.abs(lateral_speed) <= float(lateral_speed_max))
            & (torch.abs(object_velocity[:, 2]) <= float(vertical_speed_max))
            & (torch.abs(root_yaw_rate) <= float(yaw_rate_max))
        )

    transport_valid, transport_width, transport_counts = _adaptive_mask(
        transport_mask,
        speed_widths,
        rank_ids=rank_ids,
        required_per_rank=max_topk_per_rank,
    )

    candidates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    topk_stats: dict[str, Any] = {}
    for topk in topks:
        approach_indices = _rank_balanced_topk(approach_valid, approach_reward[:, 0], rank_ids, int(topk))
        transport_indices = _rank_balanced_topk(transport_valid, transport_reward[:, 0], rank_ids, int(topk))
        approach_obs = tree_map(lambda value: value[approach_indices], observation)
        transport_obs = tree_map(lambda value: value[transport_indices], observation)
        selected_approach_reward = approach_reward[approach_indices]
        selected_transport_reward = transport_reward[transport_indices]
        z_approach = model.reward_wr_inference(approach_obs, selected_approach_reward).reshape(1, -1)
        z_transport = model.reward_wr_inference(transport_obs, selected_transport_reward).reshape(1, -1)
        candidates[int(topk)] = (z_approach.detach().cpu(), z_transport.detach().cpu())
        topk_stats[str(int(topk))] = {
            "approach_count": int(approach_indices.numel()),
            "approach_rank_counts": _rank_counts(torch.isin(torch.arange(rank_ids.numel()), approach_indices), rank_ids),
            "approach_speed_mean": float(root_velocity[approach_indices, 0].mean().item()),
            "approach_speed_reward_weighted_mean": _weighted_mean(root_velocity[approach_indices, 0], selected_approach_reward),
            "approach_reward_mean": float(selected_approach_reward.mean().item()),
            "transport_count": int(transport_indices.numel()),
            "transport_rank_counts": _rank_counts(torch.isin(torch.arange(rank_ids.numel()), transport_indices), rank_ids),
            "transport_speed_mean": float(toward_speed[transport_indices].mean().item()),
            "transport_speed_reward_weighted_mean": _weighted_mean(toward_speed[transport_indices], selected_transport_reward),
            "transport_reward_mean": float(selected_transport_reward.mean().item()),
            "transport_vertical_speed_abs_mean": float(object_velocity[transport_indices, 2].abs().mean().item()),
        }
    stats = {
        "speed": float(speed),
        "approach_speed_half_width": approach_width,
        "approach_candidate_count": int(approach_valid.sum().item()),
        "approach_candidates_per_rank": approach_counts,
        "transport_speed_half_width": transport_width,
        "transport_candidate_count": int(transport_valid.sum().item()),
        "transport_candidates_per_rank": transport_counts,
        "topk": topk_stats,
    }
    return candidates, stats


def _speed_name(speed: float) -> str:
    return f"speed_{float(speed):.1f}".replace(".", "p")


def export_candidate(
    output_dir: Path,
    baseline_latent_dir: Path,
    *,
    z_approach: torch.Tensor,
    z_transport: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in baseline_latent_dir.glob("z_*.pkl"):
        shutil.copy2(source, output_dir / source.name)
    joblib.dump(z_approach.numpy(), output_dir / "z_approach.pkl")
    joblib.dump(z_transport.numpy(), output_dir / "z_transport.pkl")
    (output_dir / "controlled_inference.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
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
    all_stats: dict[str, Any] = {
        "model_folder": str(args.model_folder),
        "baseline_latent_dir": str(args.baseline_latent_dir),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(data["rank"].numel()),
        "sample_count_per_rank": _rank_counts(torch.ones_like(data["rank"], dtype=torch.bool), data["rank"]),
        "motion_id_available": False,
        "filters": {
            "approach_source": "object_present=0 (LAFAN)",
            "approach_upright_min": args.approach_upright_min,
            "transport_upright_min": args.transport_upright_min,
            "transport_lift_min": args.transport_lift_min,
            "lateral_speed_max": args.lateral_speed_max,
            "vertical_speed_max": args.vertical_speed_max,
            "yaw_rate_max": args.yaw_rate_max,
            "transport_target_distance_min": args.transport_target_distance_min,
        },
        "speeds": {},
    }
    for speed in args.speeds:
        print(f"[INFO] inferring controlled speed={speed:.1f} m/s", flush=True)
        candidates, stats = infer_speed_candidates(
            model,
            data,
            num_dof=num_dof,
            speed=float(speed),
            topks=args.topks,
            speed_widths=args.speed_half_widths,
            approach_upright_min=args.approach_upright_min,
            transport_upright_min=args.transport_upright_min,
            transport_lift_min=args.transport_lift_min,
            lateral_speed_max=args.lateral_speed_max,
            vertical_speed_max=args.vertical_speed_max,
            yaw_rate_max=args.yaw_rate_max,
            transport_target_distance_min=args.transport_target_distance_min,
        )
        all_stats["speeds"][_speed_name(speed)] = stats
        for topk, (z_approach, z_transport) in candidates.items():
            candidate_dir = args.output_dir / _speed_name(speed) / f"topk_{topk:04d}"
            export_candidate(
                candidate_dir,
                args.baseline_latent_dir,
                z_approach=z_approach,
                z_transport=z_transport,
                metadata={
                    "speed_mps": float(speed),
                    "topk": int(topk),
                    "controlled_stats": stats["topk"][str(int(topk))],
                    "preserved_from_baseline": [
                        "z_turn.pkl",
                        "z_lift.pkl",
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
    summary_path = args.output_dir / "controlled_inference_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] saved controlled inference bank to {args.output_dir}", flush=True)


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
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--topks", type=int, nargs="+", default=[200, 500, 1000])
    parser.add_argument("--speed-half-widths", type=float, nargs="+", default=[0.05, 0.075, 0.10, 0.15, 0.20])
    parser.add_argument("--approach-upright-min", type=float, default=0.90)
    parser.add_argument("--transport-upright-min", type=float, default=0.85)
    parser.add_argument("--transport-lift-min", type=float, default=0.50)
    parser.add_argument("--lateral-speed-max", type=float, default=0.15)
    parser.add_argument("--vertical-speed-max", type=float, default=0.20)
    parser.add_argument("--yaw-rate-max", type=float, default=0.30)
    parser.add_argument("--transport-target-distance-min", type=float, default=0.50)
    args = parser.parse_args()
    if args.num_samples <= 0 or min(args.topks) <= 0:
        parser.error("--num-samples and --topks must be positive")
    if not args.speeds or min(args.speeds) <= 0.0:
        parser.error("--speeds must contain positive values")
    if not args.speed_half_widths or min(args.speed_half_widths) <= 0.0:
        parser.error("--speed-half-widths must contain positive values")
    if args.speed_half_widths != sorted(args.speed_half_widths):
        parser.error("--speed-half-widths must be sorted from narrow to wide")
    return args


if __name__ == "__main__":
    run(parse_args())
