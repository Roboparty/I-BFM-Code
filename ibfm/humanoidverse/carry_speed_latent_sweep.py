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
_FOOT_STABILITY_AUX_KEYS = (
    "penalty_ankle_roll",
    "penalty_feet_ori",
    "penalty_slippage",
    "penalty_undesired_contact",
)
_MOTION_QUALITY_AUX_KEYS = (
    "penalty_action_rate",
    "penalty_torques",
    "carry_body_mpjpe",
    "carry_root_rot_err",
    "feet_heading_alignment",
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
    """Read only selected replay cells instead of materializing a whole field.

    One rank stores 5,000 x 1,024 environments.  Loading every value of every
    observation field caused avoidable memory and I/O pressure while a training
    job was using the same server.  Grouping by the circular-buffer time index
    keeps the result identical while reading only sampled environments.
    """

    time_indices = np.asarray(time_indices, dtype=np.int64).reshape(-1)
    env_indices = np.asarray(env_indices, dtype=np.int64).reshape(-1)
    if time_indices.shape != env_indices.shape:
        raise ValueError("time_indices and env_indices must have the same shape")
    selected = np.empty(
        (int(time_indices.size), *dataset.shape[2:]),
        dtype=dataset.dtype,
    )
    order = np.argsort(time_indices, kind="stable")
    sorted_time = time_indices[order]
    boundaries = np.flatnonzero(np.diff(sorted_time)) + 1
    for positions in np.split(order, boundaries):
        env_order = np.argsort(env_indices[positions], kind="stable")
        sorted_positions = positions[env_order]
        selected[sorted_positions] = np.asarray(
            dataset[int(time_indices[sorted_positions[0]]), env_indices[sorted_positions]]
        )
    return torch.from_numpy(selected)


def sample_replay_rank(
    buffer_path: Path,
    *,
    sample_count: int,
    seed: int,
    rank: int,
    include_current_state: bool = False,
    extra_aux_keys: Sequence[str] = (),
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
        valid_flat = np.flatnonzero(valid.reshape(-1))
        if sample_count > int(valid_flat.size):
            raise ValueError(f"Rank {rank} requested {sample_count} samples but only {valid_flat.size} valid transitions exist")
        flat = rng.choice(valid_flat, size=int(sample_count), replace=False)
        time_indices = flat // env_count
        env_indices = flat % env_count
        next_time_indices = (time_indices + 1) % time_count

        observation = {key: _read_selected(handle[f"observation-{key}"], next_time_indices, env_indices) for key in _OBSERVATION_KEYS}
        aux_keys = tuple(dict.fromkeys((*_CARRY_AUX_KEYS, *extra_aux_keys)))
        result = {
            "next": {
                "observation": observation,
                "qpos": _read_selected(handle["qpos"], next_time_indices, env_indices),
                "qvel": _read_selected(handle["qvel"], next_time_indices, env_indices),
            },
            # aux_rewards at slot t describe the state reached by transition t.
            "aux_rewards": {
                key: _read_selected(handle[f"aux_rewards-{key}"], time_indices, env_indices)
                for key in aux_keys
            },
            "rank": torch.full((int(sample_count),), int(rank), dtype=torch.long),
            # Preserve replay coordinates for optional temporal validation.
            # They are tiny compared with the observations and let downstream
            # inference tools inspect a short contiguous window without
            # materialising the 5,000 x 1,024 x state_dim buffer.
            "time_index": torch.from_numpy(time_indices.copy()),
            "env_index": torch.from_numpy(env_indices.copy()),
        }
        if include_current_state:
            # These values are the state immediately preceding ``next`` above.
            # Keeping them opt-in avoids the extra replay I/O for the historical
            # speed-latent path, while allowing transition-continuity filters for
            # lowering/place inference.
            result["current_state"] = _read_selected(
                handle["observation-state"], time_indices, env_indices
            )
            result["current_qpos"] = _read_selected(
                handle["qpos"], time_indices, env_indices
            )
            result["current_last_action"] = _read_selected(
                handle["observation-last_action"], time_indices, env_indices
            )
            result["current_qvel"] = _read_selected(
                handle["qvel"], time_indices, env_indices
            )
    del truncated, next_truncated, valid, valid_flat
    gc.collect()
    return result


def sample_replay_ranks(
    buffer_paths: Sequence[Path],
    *,
    sample_count: int,
    seed: int,
    include_current_state: bool = False,
    extra_aux_keys: Sequence[str] = (),
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
                extra_aux_keys=extra_aux_keys,
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


def _global_topk(mask: torch.Tensor, score: torch.Tensor, topk: int) -> torch.Tensor:
    """Select the globally highest scores without imposing replay-rank quotas."""

    candidates = torch.nonzero(mask, as_tuple=False).reshape(-1)
    take = min(int(topk), int(candidates.numel()))
    if not take:
        raise ValueError("Controlled filter produced no candidates")
    return candidates[torch.topk(score[candidates], k=take, largest=True).indices]


def _speed_closeness_score(
    speed: torch.Tensor,
    *,
    target_speed: float,
    sigma: float,
) -> torch.Tensor:
    """Return a score whose ordering depends only on distance to target speed."""

    return torch.exp(-torch.square((speed - float(target_speed)) / float(sigma)))


def _boxward_approach_kinematics(
    state: torch.Tensor,
    root_velocity_local: torch.Tensor,
    *,
    num_dof: int,
) -> dict[str, torch.Tensor]:
    """Measure root velocity toward a box represented in the root-local frame."""

    object_start = 2 * int(num_dof) + 6
    object_xy_local = state[:, object_start : object_start + 2]
    distance = torch.linalg.vector_norm(object_xy_local, dim=-1)
    direction = object_xy_local / torch.clamp(distance.unsqueeze(-1), min=1e-6)
    lateral_direction = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    return {
        "distance": distance,
        "toward_speed": torch.sum(root_velocity_local[:, :2] * direction, dim=-1),
        "lateral_speed": torch.sum(root_velocity_local[:, :2] * lateral_direction, dim=-1),
    }


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
    approach_foot_stability: bool = False,
    approach_ankle_roll_max: float = 0.10,
    approach_feet_orientation_max: float = 0.10,
    approach_slippage_max: float = 0.10,
    approach_undesired_contact_max: float = 0.0,
    approach_ankle_roll_sigma: float = 0.04,
    approach_feet_orientation_sigma: float = 0.03,
    approach_slippage_sigma: float = 0.05,
    approach_undesired_contact_sigma: float = 0.25,
    approach_motion_quality: bool = False,
    approach_root_height_min: float = 0.70,
    approach_root_height_max: float = 0.82,
    approach_target_root_height: float = 0.75,
    approach_root_height_sigma: float = 0.06,
    approach_action_rate_max: float = 15.0,
    approach_action_rate_sigma: float = 8.0,
    approach_torque_max: float = 8000.0,
    approach_torque_sigma: float = 5000.0,
    approach_body_mpjpe_max: float = 0.50,
    approach_body_mpjpe_sigma: float = 0.25,
    approach_root_rot_err_max: float = 60.0,
    approach_root_rot_err_sigma: float = 45.0,
    approach_feet_heading_max: float = 0.75,
    approach_feet_heading_sigma: float = 0.50,
    transport_foot_stability: bool = False,
    transport_ankle_roll_max: float = 0.10,
    transport_feet_orientation_max: float = 0.10,
    transport_slippage_max: float = 0.10,
    transport_undesired_contact_max: float = 0.0,
    transport_ankle_roll_sigma: float = 0.04,
    transport_feet_orientation_sigma: float = 0.03,
    transport_slippage_sigma: float = 0.05,
    transport_undesired_contact_sigma: float = 0.25,
    speed_closeness_only: bool = False,
    speed_score_sigma: float = 0.05,
    rank_balanced_topk: bool = True,
    approach_source: str = "lafan",
    approach_distance_min: float = 0.75,
    approach_distance_max: float = 3.0,
    approach_lift_max: float = 0.10,
    approach_box_upright_min: float = 0.90,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    next_data = data["next"]
    observation = next_data["observation"]
    state = torch.as_tensor(observation["state"], dtype=torch.float32)
    qpos = torch.as_tensor(next_data["qpos"], dtype=torch.float32)
    qvel = torch.as_tensor(next_data["qvel"], dtype=torch.float32)
    rank_ids = torch.as_tensor(data["rank"], dtype=torch.long)
    aux = {
        key: torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        for key, value in data["aux_rewards"].items()
    }
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
    object_start = 2 * int(num_dof) + 6
    approach_kinematics = _boxward_approach_kinematics(
        state, root_velocity, num_dof=num_dof
    )
    if approach_source == "lafan":
        approach_speed = root_velocity[:, 0]
        approach_lateral_speed = root_velocity[:, 1]
    elif approach_source == "carry-precontact":
        approach_speed = approach_kinematics["toward_speed"]
        approach_lateral_speed = approach_kinematics["lateral_speed"]
        approach_reward = (
            _speed_closeness_score(
                approach_speed,
                target_speed=speed,
                sigma=max(float(speed_widths[0]), 1e-6),
            )
            * torch.clamp(-projected_gravity_z, min=0.0, max=1.0)
        ).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported approach_source={approach_source!r}")

    foot_stability_enabled = approach_foot_stability or transport_foot_stability
    if foot_stability_enabled:
        ankle_roll = aux["penalty_ankle_roll"].clamp_min(0.0)
        feet_orientation = aux["penalty_feet_ori"].clamp_min(0.0)
        foot_slippage = aux["penalty_slippage"].clamp_min(0.0)
        undesired_contact = aux["penalty_undesired_contact"].clamp_min(0.0)
    else:
        ankle_roll = torch.zeros_like(approach_speed)
        feet_orientation = torch.zeros_like(approach_speed)
        foot_slippage = torch.zeros_like(approach_speed)
        undesired_contact = torch.zeros_like(approach_speed)
    approach_foot_stability_score = torch.exp(
        -ankle_roll / float(approach_ankle_roll_sigma)
        -feet_orientation / float(approach_feet_orientation_sigma)
        -foot_slippage / float(approach_slippage_sigma)
        -undesired_contact / float(approach_undesired_contact_sigma)
    )
    root_height = qpos[:, 2]
    if approach_motion_quality:
        action_rate = aux["penalty_action_rate"].clamp_min(0.0)
        torque = aux["penalty_torques"].clamp_min(0.0)
        body_mpjpe = aux["carry_body_mpjpe"].clamp_min(0.0)
        root_rot_err = aux["carry_root_rot_err"].clamp_min(0.0)
        feet_heading = aux["feet_heading_alignment"].clamp_min(0.0)
        motion_quality_penalty = (
            torch.abs(root_height - float(approach_target_root_height))
            / float(approach_root_height_sigma)
            + action_rate / float(approach_action_rate_sigma)
            + torque / float(approach_torque_sigma)
            + body_mpjpe / float(approach_body_mpjpe_sigma)
            + root_rot_err / float(approach_root_rot_err_sigma)
            + feet_heading / float(approach_feet_heading_sigma)
        ) / 6.0
        approach_motion_quality_score = torch.exp(-motion_quality_penalty)
    else:
        action_rate = torch.zeros_like(approach_speed)
        torque = torch.zeros_like(approach_speed)
        body_mpjpe = torch.zeros_like(approach_speed)
        root_rot_err = torch.zeros_like(approach_speed)
        feet_heading = torch.zeros_like(approach_speed)
        approach_motion_quality_score = torch.ones_like(approach_speed)

    def approach_mask(width: float) -> torch.Tensor:
        mask = (
            (-projected_gravity_z >= float(approach_upright_min))
            & (torch.abs(approach_speed - float(speed)) <= width)
            & (torch.abs(approach_lateral_speed) <= float(lateral_speed_max))
            & (torch.abs(root_velocity[:, 2]) <= float(vertical_speed_max))
            & (torch.abs(root_yaw_rate) <= float(yaw_rate_max))
        )
        if approach_source == "lafan":
            mask &= object_present < 0.5
        else:
            hand_contact = state[:, object_start + 18 : object_start + 20]
            mask &= (
                (object_present > 0.5)
                & (torch.sum(hand_contact, dim=-1) < 0.5)
                & (aux["carry_lift"] <= float(approach_lift_max))
                & (aux["carry_robot_upright"] >= float(approach_upright_min))
                & (aux["carry_upright"] >= float(approach_box_upright_min))
                & (aux["carry_drop"] < 0.5)
                & (approach_kinematics["distance"] >= float(approach_distance_min))
                & (approach_kinematics["distance"] <= float(approach_distance_max))
            )
        if approach_foot_stability:
            mask &= (
                (ankle_roll <= float(approach_ankle_roll_max))
                & (feet_orientation <= float(approach_feet_orientation_max))
                & (foot_slippage <= float(approach_slippage_max))
                & (undesired_contact <= float(approach_undesired_contact_max))
            )
        if approach_motion_quality:
            mask &= (
                (root_height >= float(approach_root_height_min))
                & (root_height <= float(approach_root_height_max))
                & (action_rate <= float(approach_action_rate_max))
                & (torque <= float(approach_torque_max))
                & (body_mpjpe <= float(approach_body_mpjpe_max))
                & (root_rot_err <= float(approach_root_rot_err_max))
                & (feet_heading <= float(approach_feet_heading_max))
            )
        return mask

    approach_valid, approach_width, approach_counts = _adaptive_mask(
        approach_mask,
        speed_widths,
        rank_ids=rank_ids,
        required_per_rank=max_topk_per_rank,
    )

    transport_reward, transport_components = compute_transport_reward(
        {key: value.unsqueeze(-1) for key, value in aux.items()},
        state,
        num_dof=num_dof,
        target_speed=float(speed),
    )
    object_velocity = state[:, object_start + 9 : object_start + 12]
    toward_speed = transport_components["toward_speed"]
    lateral_speed = transport_components["lateral_speed"]
    target_distance = transport_components["target_distance"]

    if transport_foot_stability:
        foot_stability_score = torch.exp(
            -ankle_roll / float(transport_ankle_roll_sigma)
            -feet_orientation / float(transport_feet_orientation_sigma)
            -foot_slippage / float(transport_slippage_sigma)
            -undesired_contact / float(transport_undesired_contact_sigma)
        )
        transport_reward = transport_reward * foot_stability_score.unsqueeze(-1)
    def transport_mask(width: float) -> torch.Tensor:
        mask = (
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
        if transport_foot_stability:
            mask &= (
                (ankle_roll <= float(transport_ankle_roll_max))
                & (feet_orientation <= float(transport_feet_orientation_max))
                & (foot_slippage <= float(transport_slippage_max))
                & (undesired_contact <= float(transport_undesired_contact_max))
            )
        return mask

    transport_valid, transport_width, transport_counts = _adaptive_mask(
        transport_mask,
        speed_widths,
        rank_ids=rank_ids,
        required_per_rank=max_topk_per_rank,
    )

    if speed_closeness_only:
        # Stability and task-state terms remain eligibility gates above.  The
        # scalar passed to reward inference now depends only on |v-target|.
        approach_selection_reward = _speed_closeness_score(
            approach_speed, target_speed=speed, sigma=speed_score_sigma
        ).unsqueeze(-1)
        if approach_foot_stability:
            approach_selection_reward *= approach_foot_stability_score.unsqueeze(-1)
        if approach_motion_quality:
            approach_selection_reward *= approach_motion_quality_score.unsqueeze(-1)
        transport_selection_reward = _speed_closeness_score(
            toward_speed, target_speed=speed, sigma=speed_score_sigma
        ).unsqueeze(-1)
    else:
        approach_selection_reward = approach_reward
        transport_selection_reward = transport_reward

    select_topk = (
        lambda mask, score, topk: _rank_balanced_topk(mask, score, rank_ids, topk)
        if rank_balanced_topk
        else _global_topk(mask, score, topk)
    )

    candidates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    topk_stats: dict[str, Any] = {}
    for topk in topks:
        approach_indices = select_topk(approach_valid, approach_selection_reward[:, 0], int(topk))
        transport_indices = select_topk(transport_valid, transport_selection_reward[:, 0], int(topk))
        approach_obs = tree_map(lambda value: value[approach_indices], observation)
        transport_obs = tree_map(lambda value: value[transport_indices], observation)
        selected_approach_reward = approach_selection_reward[approach_indices]
        selected_transport_reward = transport_selection_reward[transport_indices]
        z_approach = model.reward_wr_inference(approach_obs, selected_approach_reward).reshape(1, -1)
        z_transport = model.reward_wr_inference(transport_obs, selected_transport_reward).reshape(1, -1)
        candidates[int(topk)] = (z_approach.detach().cpu(), z_transport.detach().cpu())
        topk_stats[str(int(topk))] = {
            "approach_count": int(approach_indices.numel()),
            "approach_rank_counts": _rank_counts(torch.isin(torch.arange(rank_ids.numel()), approach_indices), rank_ids),
            "approach_speed_mean": float(approach_speed[approach_indices].mean().item()),
            "approach_speed_abs_error_mean": float(
                torch.abs(approach_speed[approach_indices] - float(speed)).mean().item()
            ),
            "approach_speed_abs_error_max": float(
                torch.abs(approach_speed[approach_indices] - float(speed)).max().item()
            ),
            "approach_speed_reward_weighted_mean": _weighted_mean(
                approach_speed[approach_indices], selected_approach_reward
            ),
            "approach_box_distance_mean": float(
                approach_kinematics["distance"][approach_indices].mean().item()
            ),
            "approach_reward_mean": float(selected_approach_reward.mean().item()),
            "approach_ankle_roll_mean": float(ankle_roll[approach_indices].mean().item()),
            "approach_feet_orientation_mean": float(
                feet_orientation[approach_indices].mean().item()
            ),
            "approach_slippage_mean": float(foot_slippage[approach_indices].mean().item()),
            "approach_undesired_contact_mean": float(
                undesired_contact[approach_indices].mean().item()
            ),
            "approach_root_height_mean": float(root_height[approach_indices].mean().item()),
            "approach_action_rate_mean": float(action_rate[approach_indices].mean().item()),
            "approach_torque_mean": float(torque[approach_indices].mean().item()),
            "approach_body_mpjpe_mean": float(body_mpjpe[approach_indices].mean().item()),
            "approach_root_rot_err_mean": float(root_rot_err[approach_indices].mean().item()),
            "approach_feet_heading_mean": float(feet_heading[approach_indices].mean().item()),
            "transport_count": int(transport_indices.numel()),
            "transport_rank_counts": _rank_counts(torch.isin(torch.arange(rank_ids.numel()), transport_indices), rank_ids),
            "transport_speed_mean": float(toward_speed[transport_indices].mean().item()),
            "transport_speed_abs_error_mean": float(
                torch.abs(toward_speed[transport_indices] - float(speed)).mean().item()
            ),
            "transport_speed_abs_error_max": float(
                torch.abs(toward_speed[transport_indices] - float(speed)).max().item()
            ),
            "transport_speed_reward_weighted_mean": _weighted_mean(toward_speed[transport_indices], selected_transport_reward),
            "transport_reward_mean": float(selected_transport_reward.mean().item()),
            "transport_vertical_speed_abs_mean": float(object_velocity[transport_indices, 2].abs().mean().item()),
            "transport_ankle_roll_mean": float(ankle_roll[transport_indices].mean().item()),
            "transport_feet_orientation_mean": float(feet_orientation[transport_indices].mean().item()),
            "transport_slippage_mean": float(foot_slippage[transport_indices].mean().item()),
            "transport_undesired_contact_mean": float(undesired_contact[transport_indices].mean().item()),
        }
    stats = {
        "speed": float(speed),
        "approach_source": approach_source,
        "approach_speed_semantics": (
            "robot_local_forward" if approach_source == "lafan" else "robot_boxward"
        ),
        "approach_speed_half_width": approach_width,
        "approach_candidate_count": int(approach_valid.sum().item()),
        "approach_candidates_per_rank": approach_counts,
        "approach_foot_stability_enabled": bool(approach_foot_stability),
        "approach_motion_quality_enabled": bool(approach_motion_quality),
        "approach_score_formula": (
            "speed_closeness * foot_stability * motion_quality"
            if speed_closeness_only
            and approach_foot_stability
            and approach_motion_quality
            else "speed_closeness * motion_quality"
            if speed_closeness_only and approach_motion_quality
            else "speed_closeness * exp(-ankle/sigma - feet_ori/sigma - slippage/sigma "
            "- undesired_contact/sigma)"
            if speed_closeness_only and approach_foot_stability
            else "speed_closeness_only"
            if speed_closeness_only
            else "legacy phase-specific combined reward"
        ),
        "transport_speed_half_width": transport_width,
        "transport_candidate_count": int(transport_valid.sum().item()),
        "transport_candidates_per_rank": transport_counts,
        "transport_foot_stability_enabled": bool(transport_foot_stability),
        "reward_mode": "speed_closeness_only" if speed_closeness_only else "legacy_combined",
        "speed_score_formula": (
            f"exp(-((v-{float(speed):g})/{float(speed_score_sigma):g})^2)"
            if speed_closeness_only
            else "legacy phase-specific combined reward"
        ),
        "speed_score_sigma": float(speed_score_sigma) if speed_closeness_only else None,
        "rank_balanced_topk": bool(rank_balanced_topk),
        "topk": topk_stats,
    }
    return candidates, stats


def _speed_name(speed: float) -> str:
    value = f"{float(speed):.6f}".rstrip("0").rstrip(".")
    return f"speed_{value.replace('.', 'p')}"


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
    data = sample_replay_ranks(
        args.buffer_paths,
        sample_count=args.num_samples,
        seed=args.seed,
        extra_aux_keys=tuple(
            dict.fromkeys(
                (
                    *(
                        _FOOT_STABILITY_AUX_KEYS
                        if args.approach_foot_stability or args.transport_foot_stability
                        else ()
                    ),
                    *(_MOTION_QUALITY_AUX_KEYS if args.approach_motion_quality else ()),
                )
            )
        ),
    )
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
        "latent_semantics": {
            "inference_api": "reward_wr_inference",
            "goal_latent_mode": "not_applicable",
            "note": "absolute/displacement is a goal-reaching option and is not applied to replay reward inference",
        },
        "filters": {
            "approach_source": args.approach_source,
            "approach_upright_min": args.approach_upright_min,
            "approach_distance_min": args.approach_distance_min,
            "approach_distance_max": args.approach_distance_max,
            "approach_lift_max": args.approach_lift_max,
            "approach_box_upright_min": args.approach_box_upright_min,
            "approach_foot_stability": args.approach_foot_stability,
            "approach_ankle_roll_max": args.approach_ankle_roll_max,
            "approach_feet_orientation_max": args.approach_feet_orientation_max,
            "approach_slippage_max": args.approach_slippage_max,
            "approach_undesired_contact_max": args.approach_undesired_contact_max,
            "approach_ankle_roll_sigma": args.approach_ankle_roll_sigma,
            "approach_feet_orientation_sigma": args.approach_feet_orientation_sigma,
            "approach_slippage_sigma": args.approach_slippage_sigma,
            "approach_undesired_contact_sigma": args.approach_undesired_contact_sigma,
            "approach_motion_quality": args.approach_motion_quality,
            "approach_root_height_min": args.approach_root_height_min,
            "approach_root_height_max": args.approach_root_height_max,
            "approach_target_root_height": args.approach_target_root_height,
            "approach_root_height_sigma": args.approach_root_height_sigma,
            "approach_action_rate_max": args.approach_action_rate_max,
            "approach_action_rate_sigma": args.approach_action_rate_sigma,
            "approach_torque_max": args.approach_torque_max,
            "approach_torque_sigma": args.approach_torque_sigma,
            "approach_body_mpjpe_max": args.approach_body_mpjpe_max,
            "approach_body_mpjpe_sigma": args.approach_body_mpjpe_sigma,
            "approach_root_rot_err_max": args.approach_root_rot_err_max,
            "approach_root_rot_err_sigma": args.approach_root_rot_err_sigma,
            "approach_feet_heading_max": args.approach_feet_heading_max,
            "approach_feet_heading_sigma": args.approach_feet_heading_sigma,
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
            approach_foot_stability=args.approach_foot_stability,
            approach_ankle_roll_max=args.approach_ankle_roll_max,
            approach_feet_orientation_max=args.approach_feet_orientation_max,
            approach_slippage_max=args.approach_slippage_max,
            approach_undesired_contact_max=args.approach_undesired_contact_max,
            approach_ankle_roll_sigma=args.approach_ankle_roll_sigma,
            approach_feet_orientation_sigma=args.approach_feet_orientation_sigma,
            approach_slippage_sigma=args.approach_slippage_sigma,
            approach_undesired_contact_sigma=args.approach_undesired_contact_sigma,
            approach_motion_quality=args.approach_motion_quality,
            approach_root_height_min=args.approach_root_height_min,
            approach_root_height_max=args.approach_root_height_max,
            approach_target_root_height=args.approach_target_root_height,
            approach_root_height_sigma=args.approach_root_height_sigma,
            approach_action_rate_max=args.approach_action_rate_max,
            approach_action_rate_sigma=args.approach_action_rate_sigma,
            approach_torque_max=args.approach_torque_max,
            approach_torque_sigma=args.approach_torque_sigma,
            approach_body_mpjpe_max=args.approach_body_mpjpe_max,
            approach_body_mpjpe_sigma=args.approach_body_mpjpe_sigma,
            approach_root_rot_err_max=args.approach_root_rot_err_max,
            approach_root_rot_err_sigma=args.approach_root_rot_err_sigma,
            approach_feet_heading_max=args.approach_feet_heading_max,
            approach_feet_heading_sigma=args.approach_feet_heading_sigma,
            transport_foot_stability=args.transport_foot_stability,
            transport_ankle_roll_max=args.transport_ankle_roll_max,
            transport_feet_orientation_max=args.transport_feet_orientation_max,
            transport_slippage_max=args.transport_slippage_max,
            transport_undesired_contact_max=args.transport_undesired_contact_max,
            transport_ankle_roll_sigma=args.transport_ankle_roll_sigma,
            transport_feet_orientation_sigma=args.transport_feet_orientation_sigma,
            transport_slippage_sigma=args.transport_slippage_sigma,
            transport_undesired_contact_sigma=args.transport_undesired_contact_sigma,
            speed_closeness_only=args.speed_closeness_only,
            speed_score_sigma=args.speed_score_sigma,
            rank_balanced_topk=args.rank_balanced_topk,
            approach_source=args.approach_source,
            approach_distance_min=args.approach_distance_min,
            approach_distance_max=args.approach_distance_max,
            approach_lift_max=args.approach_lift_max,
            approach_box_upright_min=args.approach_box_upright_min,
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
                    "latent_semantics": {
                        "inference_api": "reward_wr_inference",
                        "goal_latent_mode": "not_applicable",
                        "reward_mode": stats["reward_mode"],
                        "speed_score_formula": stats["speed_score_formula"],
                        "rank_balanced_topk": stats["rank_balanced_topk"],
                        "approach_source": stats["approach_source"],
                        "approach_speed_semantics": stats["approach_speed_semantics"],
                    },
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
    parser.add_argument(
        "--approach-source",
        choices=("lafan", "carry-precontact"),
        default="lafan",
        help="Infer Approach from object-free LAFAN rows or object-aware pre-contact Carry rows.",
    )
    parser.add_argument("--approach-distance-min", type=float, default=0.75)
    parser.add_argument("--approach-distance-max", type=float, default=3.0)
    parser.add_argument("--approach-lift-max", type=float, default=0.10)
    parser.add_argument("--approach-box-upright-min", type=float, default=0.90)
    parser.add_argument(
        "--approach-foot-stability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter approach replay states using foot/ankle stability penalties.",
    )
    parser.add_argument("--approach-ankle-roll-max", type=float, default=0.10)
    parser.add_argument("--approach-feet-orientation-max", type=float, default=0.10)
    parser.add_argument("--approach-slippage-max", type=float, default=0.10)
    parser.add_argument("--approach-undesired-contact-max", type=float, default=0.0)
    parser.add_argument("--approach-ankle-roll-sigma", type=float, default=0.04)
    parser.add_argument("--approach-feet-orientation-sigma", type=float, default=0.03)
    parser.add_argument("--approach-slippage-sigma", type=float, default=0.05)
    parser.add_argument("--approach-undesired-contact-sigma", type=float, default=0.25)
    parser.add_argument(
        "--approach-motion-quality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter and rank Approach rows by root pose, action smoothness, torque, and motion tracking quality.",
    )
    parser.add_argument("--approach-root-height-min", type=float, default=0.70)
    parser.add_argument("--approach-root-height-max", type=float, default=0.82)
    parser.add_argument("--approach-target-root-height", type=float, default=0.75)
    parser.add_argument("--approach-root-height-sigma", type=float, default=0.06)
    parser.add_argument("--approach-action-rate-max", type=float, default=15.0)
    parser.add_argument("--approach-action-rate-sigma", type=float, default=8.0)
    parser.add_argument("--approach-torque-max", type=float, default=8000.0)
    parser.add_argument("--approach-torque-sigma", type=float, default=5000.0)
    parser.add_argument("--approach-body-mpjpe-max", type=float, default=0.50)
    parser.add_argument("--approach-body-mpjpe-sigma", type=float, default=0.25)
    parser.add_argument("--approach-root-rot-err-max", type=float, default=60.0)
    parser.add_argument("--approach-root-rot-err-sigma", type=float, default=45.0)
    parser.add_argument("--approach-feet-heading-max", type=float, default=0.75)
    parser.add_argument("--approach-feet-heading-sigma", type=float, default=0.50)
    parser.add_argument("--transport-upright-min", type=float, default=0.85)
    parser.add_argument("--transport-lift-min", type=float, default=0.50)
    parser.add_argument("--lateral-speed-max", type=float, default=0.15)
    parser.add_argument("--vertical-speed-max", type=float, default=0.20)
    parser.add_argument("--yaw-rate-max", type=float, default=0.30)
    parser.add_argument("--transport-target-distance-min", type=float, default=0.50)
    parser.add_argument(
        "--transport-foot-stability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rank and filter transport replay states using foot/ankle stability penalties.",
    )
    parser.add_argument("--transport-ankle-roll-max", type=float, default=0.10)
    parser.add_argument("--transport-feet-orientation-max", type=float, default=0.10)
    parser.add_argument("--transport-slippage-max", type=float, default=0.10)
    parser.add_argument("--transport-undesired-contact-max", type=float, default=0.0)
    parser.add_argument("--transport-ankle-roll-sigma", type=float, default=0.04)
    parser.add_argument("--transport-feet-orientation-sigma", type=float, default=0.03)
    parser.add_argument("--transport-slippage-sigma", type=float, default=0.05)
    parser.add_argument("--transport-undesired-contact-sigma", type=float, default=0.25)
    parser.add_argument(
        "--speed-closeness-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only distance to target speed for top-k ranking and reward inference.",
    )
    parser.add_argument("--speed-score-sigma", type=float, default=0.05)
    parser.add_argument(
        "--rank-balanced-topk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reserve equal top-k quotas per replay rank; disable for globally closest speeds.",
    )
    args = parser.parse_args()
    if args.num_samples <= 0 or min(args.topks) <= 0:
        parser.error("--num-samples and --topks must be positive")
    if not args.speeds or min(args.speeds) <= 0.0:
        parser.error("--speeds must contain positive values")
    if not args.speed_half_widths or min(args.speed_half_widths) <= 0.0:
        parser.error("--speed-half-widths must contain positive values")
    if args.speed_half_widths != sorted(args.speed_half_widths):
        parser.error("--speed-half-widths must be sorted from narrow to wide")
    if min(
        args.approach_ankle_roll_sigma,
        args.approach_feet_orientation_sigma,
        args.approach_slippage_sigma,
        args.approach_undesired_contact_sigma,
        args.transport_ankle_roll_sigma,
        args.transport_feet_orientation_sigma,
        args.transport_slippage_sigma,
        args.transport_undesired_contact_sigma,
        args.approach_root_height_sigma,
        args.approach_action_rate_sigma,
        args.approach_torque_sigma,
        args.approach_body_mpjpe_sigma,
        args.approach_root_rot_err_sigma,
        args.approach_feet_heading_sigma,
    ) <= 0.0:
        parser.error("stability and motion-quality sigmas must be positive")
    if args.speed_score_sigma <= 0.0:
        parser.error("--speed-score-sigma must be positive")
    if min(
        args.approach_ankle_roll_max,
        args.approach_feet_orientation_max,
        args.approach_slippage_max,
        args.approach_undesired_contact_max,
    ) < 0.0:
        parser.error("approach foot-stability maxima must be non-negative")
    if args.approach_distance_min < 0.0 or args.approach_distance_min >= args.approach_distance_max:
        parser.error("--approach-distance-min must be non-negative and less than --approach-distance-max")
    if not 0.0 <= args.approach_lift_max <= 1.0:
        parser.error("--approach-lift-max must be in [0, 1]")
    if not 0.0 <= args.approach_box_upright_min <= 1.0:
        parser.error("--approach-box-upright-min must be in [0, 1]")
    if not (
        0.0
        < args.approach_root_height_min
        < args.approach_target_root_height
        < args.approach_root_height_max
    ):
        parser.error("approach root heights must satisfy 0 < min < target < max")
    if min(
        args.approach_action_rate_max,
        args.approach_torque_max,
        args.approach_body_mpjpe_max,
        args.approach_root_rot_err_max,
        args.approach_feet_heading_max,
    ) <= 0.0:
        parser.error("approach motion-quality maxima must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
