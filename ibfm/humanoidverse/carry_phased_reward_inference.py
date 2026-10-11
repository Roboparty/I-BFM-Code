"""Evaluate reward-inferred approach, lift, transport, place, and hold skills."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import mediapy as media
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_mjlab import _carry_box_upright
from humanoidverse.carry_reward_inference import (
    _rollout_metrics,
    _summarize_rollout,
    compute_carry_lift_upright_reward,
)
from humanoidverse.envs.carry_observations import LEGACY_OBJECT_CONTEXT_DIM, OBJECT_CONTEXT_DIM
from humanoidverse.goal_recovery_inference import _select_frame
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    add_bool_arg,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    render_policy_frame,
)
from humanoidverse.mjlab_reward_relabel import get_next
from humanoidverse.reward_inference import _load_replay_buffer
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG, _resolve_tracking_robot_config, _target_states_from_obs
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import quat_rotate_inverse


def _state_object_context_dim(state: torch.Tensor, *, num_dof: int) -> int:
    robot_state_dim = 2 * int(num_dof) + 6
    context_dim = int(state.shape[-1]) - robot_state_dim
    if context_dim not in (LEGACY_OBJECT_CONTEXT_DIM, OBJECT_CONTEXT_DIM):
        raise ValueError(
            f"Unsupported Carry object context dimension {context_dim}; "
            f"expected {LEGACY_OBJECT_CONTEXT_DIM} or {OBJECT_CONTEXT_DIM}"
        )
    return context_dim


def compute_approach_speed_reward(
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    state: torch.Tensor,
    *,
    num_dof: int,
    target_speed: float = 0.5,
    target_angle_deg: float = 0.0,
    forward_sigma: float = 0.15,
    lateral_sigma: float = 0.15,
    box_heading_sigma_deg: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score upright replay states whose local planar velocity matches a target."""

    qpos = torch.as_tensor(qpos, dtype=torch.float32)
    qvel = torch.as_tensor(qvel, dtype=torch.float32)
    state = torch.as_tensor(state, dtype=torch.float32)
    if qpos.ndim != 2 or qpos.shape[-1] < 7 or qvel.ndim != 2 or qvel.shape[-1] < 3:
        raise ValueError(f"Expected qpos/qvel shapes [N, >=7]/[N, >=3], got {qpos.shape}/{qvel.shape}")
    if state.ndim != 2 or state.shape[0] != qpos.shape[0]:
        raise ValueError(f"Expected state shape [N, D] aligned with qpos, got {state.shape}")
    if forward_sigma <= 0.0 or lateral_sigma <= 0.0:
        raise ValueError("Approach reward speed sigmas must be positive")

    # MuJoCo free-joint qpos stores wxyz; UFO quaternion helpers use xyzw.
    root_quat_xyzw = torch.roll(qpos[:, 3:7], shifts=-1, dims=-1)
    root_lin_vel_local = quat_rotate_inverse(root_quat_xyzw, qvel[:, :3], w_last=True)
    target_angle = torch.deg2rad(torch.as_tensor(float(target_angle_deg), dtype=torch.float32))
    target_forward = float(target_speed) * torch.cos(target_angle)
    target_lateral = float(target_speed) * torch.sin(target_angle)
    forward_error = (root_lin_vel_local[:, 0] - target_forward) / float(forward_sigma)
    lateral_error = (root_lin_vel_local[:, 1] - target_lateral) / float(lateral_sigma)
    projected_gravity_z = state[:, 2 * int(num_dof) + 2]
    upright = torch.clamp(-projected_gravity_z, min=0.0, max=1.0)
    reward = torch.exp(-torch.square(forward_error) - torch.square(lateral_error)) * upright
    if box_heading_sigma_deg is not None:
        if float(box_heading_sigma_deg) <= 0.0:
            raise ValueError("box_heading_sigma_deg must be positive")
        object_start = 2 * int(num_dof) + 6
        object_xy_local = state[:, object_start : object_start + 2]
        box_bearing = torch.atan2(object_xy_local[:, 1], object_xy_local[:, 0])
        heading_sigma = torch.deg2rad(torch.as_tensor(float(box_heading_sigma_deg), dtype=state.dtype))
        context_dim = _state_object_context_dim(state, num_dof=num_dof)
        object_present = state[:, object_start + context_dim - 2]
        heading_alignment = torch.exp(-torch.square(box_bearing / heading_sigma)) * object_present
        reward = reward * heading_alignment
    return reward.unsqueeze(-1), root_lin_vel_local


def compute_turn_reward(
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    state: torch.Tensor,
    *,
    num_dof: int,
    target_yaw_rate: float,
    yaw_rate_sigma: float = 0.2,
    planar_speed_sigma: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score upright, nearly in-place turns at the requested local yaw rate."""

    if yaw_rate_sigma <= 0.0 or planar_speed_sigma <= 0.0:
        raise ValueError("Turn reward sigmas must be positive")
    qpos = torch.as_tensor(qpos, dtype=torch.float32)
    qvel = torch.as_tensor(qvel, dtype=torch.float32)
    state = torch.as_tensor(state, dtype=torch.float32)
    root_quat_xyzw = torch.roll(qpos[:, 3:7], shifts=-1, dims=-1)
    root_lin_vel_local = quat_rotate_inverse(root_quat_xyzw, qvel[:, :3], w_last=True)
    planar_speed = torch.linalg.vector_norm(root_lin_vel_local[:, :2], dim=-1)
    root_yaw_rate = state[:, 2 * int(num_dof) + 5]
    projected_gravity_z = state[:, 2 * int(num_dof) + 2]
    upright = torch.clamp(-projected_gravity_z, min=0.0, max=1.0)
    reward = (
        torch.exp(-torch.square((root_yaw_rate - float(target_yaw_rate)) / float(yaw_rate_sigma)))
        * torch.exp(-torch.square(planar_speed / float(planar_speed_sigma)))
        * upright
    )
    return reward.unsqueeze(-1), {"root_yaw_rate": root_yaw_rate, "planar_speed": planar_speed}


def compute_static_carry_reward(
    aux_rewards: Mapping[str, torch.Tensor],
    state: torch.Tensor,
    *,
    num_dof: int,
    joint_speed_sigma: float = 1.0,
    root_ang_speed_sigma: float = 0.75,
    object_lin_speed_sigma: float = 0.5,
    object_ang_speed_sigma: float = 0.75,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer lifted bilateral-contact Carry states with near-zero velocities."""

    state = torch.as_tensor(state, dtype=torch.float32)
    object_start = 2 * int(num_dof) + 6
    if state.ndim != 2:
        raise ValueError(f"Expected Carry state shape [N, D], got {tuple(state.shape)}")
    _state_object_context_dim(state, num_dof=num_dof)
    sigmas = (joint_speed_sigma, root_ang_speed_sigma, object_lin_speed_sigma, object_ang_speed_sigma)
    if any(float(sigma) <= 0.0 for sigma in sigmas):
        raise ValueError("Static Carry reward speed sigmas must be positive")

    dof_vel = state[:, int(num_dof) : 2 * int(num_dof)]
    root_ang_vel = state[:, 2 * int(num_dof) + 3 : object_start]
    object_lin_vel = state[:, object_start + 9 : object_start + 12]
    object_ang_vel = state[:, object_start + 12 : object_start + 15]
    speeds = {
        "joint_speed_rms": torch.sqrt(torch.mean(torch.square(dof_vel), dim=-1)),
        "root_ang_speed": torch.linalg.vector_norm(root_ang_vel, dim=-1),
        "object_lin_speed": torch.linalg.vector_norm(object_lin_vel, dim=-1),
        "object_ang_speed": torch.linalg.vector_norm(object_ang_vel, dim=-1),
    }
    stationary = torch.exp(
        -torch.square(speeds["joint_speed_rms"] / float(joint_speed_sigma))
        - torch.square(speeds["root_ang_speed"] / float(root_ang_speed_sigma))
        - torch.square(speeds["object_lin_speed"] / float(object_lin_speed_sigma))
        - torch.square(speeds["object_ang_speed"] / float(object_ang_speed_sigma))
    )
    carry_reward = compute_carry_lift_upright_reward(aux_rewards).reshape(-1)
    return (carry_reward * stationary).unsqueeze(-1), {**speeds, "stationary": stationary}


def compute_transport_reward(
    aux_rewards: Mapping[str, torch.Tensor],
    state: torch.Tensor,
    *,
    num_dof: int,
    target_speed: float = 0.8,
    goal_distance_sigma: float | None = None,
    forward_sigma: float = 0.20,
    lateral_sigma: float = 0.20,
    target_yaw_rate: float = 0.0,
    yaw_rate_sigma: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer upright bilateral-contact box motion toward its observed target."""

    if forward_sigma <= 0.0 or lateral_sigma <= 0.0:
        raise ValueError("Transport reward speed sigmas must be positive")
    if goal_distance_sigma is not None and goal_distance_sigma <= 0.0:
        raise ValueError("Transport goal-distance sigma must be positive")
    if yaw_rate_sigma is not None and yaw_rate_sigma <= 0.0:
        raise ValueError("Transport yaw-rate sigma must be positive")
    state = torch.as_tensor(state, dtype=torch.float32)
    object_start = 2 * int(num_dof) + 6
    _state_object_context_dim(state, num_dof=num_dof)
    object_velocity = state[:, object_start + 9 : object_start + 12]
    target_delta = state[:, object_start + 15 : object_start + 18]
    target_xy = target_delta[:, :2]
    target_distance = torch.linalg.vector_norm(target_xy, dim=-1)
    target_direction = target_xy / torch.clamp(target_distance.unsqueeze(-1), min=1e-6)
    toward_speed = torch.sum(object_velocity[:, :2] * target_direction, dim=-1)
    lateral_direction = torch.stack((-target_direction[:, 1], target_direction[:, 0]), dim=-1)
    lateral_speed = torch.sum(object_velocity[:, :2] * lateral_direction, dim=-1)
    desired_speed = torch.full_like(target_distance, float(target_speed))
    motion_score = torch.exp(
        -torch.square((toward_speed - desired_speed) / float(forward_sigma))
        - torch.square(lateral_speed / float(lateral_sigma))
    )
    carry_score = compute_carry_lift_upright_reward(aux_rewards).reshape(-1)
    reward = carry_score * motion_score
    goal_proximity = torch.ones_like(target_distance)
    if goal_distance_sigma is not None:
        goal_proximity = torch.exp(-torch.square(target_distance / float(goal_distance_sigma)))
        reward = reward * goal_proximity
    root_yaw_rate = state[:, 2 * int(num_dof) + 5]
    if yaw_rate_sigma is not None:
        reward = reward * torch.exp(
            -torch.square((root_yaw_rate - float(target_yaw_rate)) / float(yaw_rate_sigma))
        )
    return reward.unsqueeze(-1), {
        "target_distance": target_distance,
        "toward_speed": toward_speed,
        "desired_speed": desired_speed,
        "lateral_speed": lateral_speed,
        "root_yaw_rate": root_yaw_rate,
        "motion_score": motion_score,
        "goal_proximity": goal_proximity,
    }


def compute_place_reward(
    aux_rewards: Mapping[str, torch.Tensor],
    state: torch.Tensor,
    *,
    num_dof: int,
    lift_height: float = 0.35,
    height_sigma: float = 0.08,
    target_distance_sigma: float = 0.35,
    object_speed_sigma: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer a released, upright, stationary box on the ground at its goal."""

    if min(lift_height, height_sigma, target_distance_sigma, object_speed_sigma) <= 0.0:
        raise ValueError("Place reward sigmas must be positive")
    state = torch.as_tensor(state, dtype=torch.float32)
    object_start = 2 * int(num_dof) + 6
    _state_object_context_dim(state, num_dof=num_dof)
    object_pos_local = state[:, object_start : object_start + 3]
    object_lin_vel = state[:, object_start + 9 : object_start + 12]
    object_ang_vel = state[:, object_start + 12 : object_start + 15]
    target_delta = state[:, object_start + 15 : object_start + 18]
    target_distance = torch.linalg.vector_norm(target_delta, dim=-1)
    object_speed = torch.linalg.vector_norm(object_lin_vel, dim=-1)
    object_ang_speed = torch.linalg.vector_norm(object_ang_vel, dim=-1)

    def aux(name: str) -> torch.Tensor:
        if name not in aux_rewards:
            raise KeyError(f"Replay buffer is missing Carry auxiliary reward {name!r}")
        return torch.as_tensor(aux_rewards[name], dtype=torch.float32).reshape(-1)

    # object_pos_local.z cannot give world height for a tilted root. The replay
    # lift signal is the reliable normalized measure of clearance above ground.
    on_ground = torch.exp(-torch.square(aux("carry_lift") / float(height_sigma / lift_height)))
    near_goal = torch.exp(-torch.square(target_distance / float(target_distance_sigma)))
    stationary = torch.exp(
        -torch.square(object_speed / float(object_speed_sigma))
        -torch.square(object_ang_speed / float(object_speed_sigma))
    )
    released = 1.0 - aux("carry_bilateral_contact")
    upright = aux("carry_robot_upright") * aux("carry_upright")
    placement_quality = on_ground * stationary * (0.25 + 0.75 * released)
    reward = upright * near_goal * (0.20 + 0.80 * placement_quality)
    return reward.unsqueeze(-1), {
        "target_distance": target_distance,
        "object_height_local": object_pos_local[:, 2],
        "object_speed": object_speed,
        "object_ang_speed": object_ang_speed,
        "on_ground": on_ground,
        "near_goal": near_goal,
        "released": released,
        "stationary": stationary,
        "placement_quality": placement_quality,
    }


def compute_lower_reward(
    aux_rewards: Mapping[str, torch.Tensor],
    state: torch.Tensor,
    *,
    num_dof: int,
    target_vertical_speed: float = -0.08,
    vertical_speed_sigma: float = 0.08,
    horizontal_speed_sigma: float = 0.15,
    angular_speed_sigma: float = 0.25,
    target_distance_sigma: float = 0.35,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer near-goal lowering while both hands still contact the box."""

    sigmas = (vertical_speed_sigma, horizontal_speed_sigma, angular_speed_sigma, target_distance_sigma)
    if any(float(sigma) <= 0.0 for sigma in sigmas):
        raise ValueError("Lowering reward sigmas must be positive")
    state = torch.as_tensor(state, dtype=torch.float32)
    object_start = 2 * int(num_dof) + 6
    _state_object_context_dim(state, num_dof=num_dof)
    object_lin_vel = state[:, object_start + 9 : object_start + 12]
    object_ang_vel = state[:, object_start + 12 : object_start + 15]
    target_delta = state[:, object_start + 15 : object_start + 18]

    def aux(name: str) -> torch.Tensor:
        if name not in aux_rewards:
            raise KeyError(f"Replay buffer is missing Carry auxiliary reward {name!r}")
        return torch.as_tensor(aux_rewards[name], dtype=torch.float32).reshape(-1)

    horizontal_speed = torch.linalg.vector_norm(object_lin_vel[:, :2], dim=-1)
    vertical_speed = object_lin_vel[:, 2]
    angular_speed = torch.linalg.vector_norm(object_ang_vel, dim=-1)
    target_distance = torch.linalg.vector_norm(target_delta, dim=-1)
    motion_score = torch.exp(
        -torch.square((vertical_speed - float(target_vertical_speed)) / float(vertical_speed_sigma))
        -torch.square(horizontal_speed / float(horizontal_speed_sigma))
        -torch.square(angular_speed / float(angular_speed_sigma))
    )
    contact_upright = aux("carry_bilateral_contact") * aux("carry_robot_upright") * aux("carry_upright")
    airborne = torch.clamp(aux("carry_lift"), min=0.0, max=1.0)
    near_goal = torch.exp(-torch.square(target_distance / float(target_distance_sigma)))
    reward = contact_upright * airborne * near_goal * motion_score
    return reward.unsqueeze(-1), {
        "target_distance": target_distance,
        "horizontal_speed": horizontal_speed,
        "vertical_speed": vertical_speed,
        "angular_speed": angular_speed,
        "near_goal": near_goal,
        "motion_score": motion_score,
    }


def _masked_object_observation(observation: Any) -> Any:
    """Present the approach actor with the in-distribution object-absent schema."""

    if not isinstance(observation, Mapping):
        return observation
    result = dict(observation)
    state = result.get("state")
    if not torch.is_tensor(state):
        return result
    last_action = result.get("last_action")
    if not torch.is_tensor(last_action):
        return result
    context_dim = int(state.shape[-1]) - (2 * int(last_action.shape[-1]) + 6)
    if context_dim not in (OBJECT_CONTEXT_DIM, LEGACY_OBJECT_CONTEXT_DIM):
        return result
    for key in ("state", "privileged_state"):
        value = result.get(key)
        if torch.is_tensor(value):
            if int(value.shape[-1]) < context_dim:
                continue
            value = value.clone()
            value[..., -context_dim:] = 0.0
            result[key] = value
    return result


def _goal_lookahead_observation(observation: Any, lookahead_distance: float) -> Any:
    """Move only the observed local XY goal farther ahead to cancel stopping bias."""

    if float(lookahead_distance) < 0.0:
        raise ValueError("Goal lookahead distance must be non-negative")
    if not isinstance(observation, Mapping) or float(lookahead_distance) == 0.0:
        return observation
    result = dict(observation)
    state = result.get("state")
    if not torch.is_tensor(state):
        return result
    last_action = result.get("last_action")
    if not torch.is_tensor(last_action):
        return result
    context_dim = int(state.shape[-1]) - (2 * int(last_action.shape[-1]) + 6)
    if context_dim not in (OBJECT_CONTEXT_DIM, LEGACY_OBJECT_CONTEXT_DIM):
        return result
    for key in ("state", "privileged_state"):
        value = result.get(key)
        if not torch.is_tensor(value) or int(value.shape[-1]) < context_dim:
            continue
        value = value.clone()
        context_start = int(value.shape[-1]) - context_dim
        target_delta = value[..., context_start + 15 : context_start + 18]
        target_norm = torch.linalg.vector_norm(target_delta[..., :2], dim=-1, keepdim=True)
        target_delta[..., :2] += (
            target_delta[..., :2]
            / torch.clamp(target_norm, min=1e-6)
            * float(lookahead_distance)
        )
        result[key] = value
    return result


class _ExpertInferenceDataset:
    """Minimal replay-compatible view over expert motion observations."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self._size = int(data["next"]["observation"]["state"].shape[0])

    def size(self) -> int:
        return self._size

    def sample(self, count: int) -> dict[str, Any]:
        count = min(int(count), self._size)
        indices = torch.randperm(self._size)[:count]
        return tree_map(lambda value: value[indices], self.data)


def _expert_inference_dataset(
    env: Any,
    *,
    num_samples: int,
    num_dof: int,
    use_root_height_obs: bool,
) -> _ExpertInferenceDataset:
    """Build B-network inputs and Carry signals directly from reference motions."""

    motion_lib = env._motion_lib
    motion_count = int(getattr(motion_lib, "_num_unique_motions", motion_lib.num_motions()))
    if motion_count <= 0:
        raise ValueError("Expert motion library is empty")
    rows_per_motion = max(1, int(np.ceil(float(num_samples) / motion_count)))
    observations: dict[str, list[torch.Tensor]] = {
        "state": [],
        "privileged_state": [],
        "last_action": [],
    }
    qpos_rows: list[torch.Tensor] = []
    qvel_rows: list[torch.Tensor] = []
    aux_rows: dict[str, list[torch.Tensor]] = {
        "carry_bilateral_contact": [],
        "carry_lift": [],
        "carry_upright": [],
        "carry_robot_upright": [],
        "carry_drop": [],
    }
    generator = torch.Generator(device="cpu").manual_seed(0)
    del generator  # Sampling is deterministic through evenly spaced frames.
    batch_size = min(64, motion_count)
    for batch_start in range(0, motion_count, batch_size):
        current_count = min(batch_size, motion_count - batch_start)
        motion_lib.load_motions(
            random_sample=False,
            num_motions_to_load=current_count,
            start_idx=batch_start,
        )
        for motion_id in range(current_count):
            obs, ref = get_backward_observation(
                env,
                motion_id,
                use_root_height_obs=use_root_height_obs,
            )
            frame_count = int(obs["state"].shape[0])
            take = min(rows_per_motion, frame_count)
            if take <= 0:
                continue
            indices = torch.linspace(0, frame_count - 1, take, device=obs["state"].device).round().long()
            selected = {key: value[indices].detach().cpu() for key, value in obs.items()}
            for key in observations:
                observations[key].append(selected[key])

            root_pos = ref["ref_body_pos"][indices, 0]
            root_quat_wxyz = torch.roll(ref["ref_body_rots"][indices, 0], shifts=1, dims=-1)
            qpos_rows.append(torch.cat((root_pos, root_quat_wxyz, ref["dof_pos"][indices]), dim=-1).cpu())
            qvel_rows.append(
                torch.cat(
                    (
                        ref["ref_body_vels"][indices, 0],
                        ref["ref_body_angular_vels"][indices, 0],
                        ref["ref_dof_vel"][indices],
                    ),
                    dim=-1,
                ).cpu()
            )

            state = selected["state"]
            object_start = 2 * int(num_dof) + 6
            present = ref["object_present"][indices].reshape(-1).detach().cpu()
            contact = state[:, object_start + 18 : object_start + 20]
            aux_rows["carry_bilateral_contact"].append(torch.all(contact > 0.5, dim=-1).float() * present)
            ground_height = float(env.carry_box_cfg.get("ground_center_height", 0.185))
            lift_target = float(env.carry_box_cfg.get("lift_height", 0.35))
            lift = torch.clamp((ref["object_pos"][indices, 2].cpu() - ground_height) / lift_target, 0.0, 1.0)
            aux_rows["carry_lift"].append(lift * present)
            upright = _carry_box_upright(
                ref["object_quat"][indices],
                env.box_geom_quat_xyzw,
            ).detach().cpu()
            aux_rows["carry_upright"].append(upright * present)
            aux_rows["carry_robot_upright"].append(
                torch.clamp(-state[:, 2 * int(num_dof) + 2], 0.0, 1.0) * present
            )
            aux_rows["carry_drop"].append(torch.zeros_like(present))

    data = {
        "next": {
            "observation": {key: torch.cat(values, dim=0) for key, values in observations.items()},
            "qpos": torch.cat(qpos_rows, dim=0),
            "qvel": torch.cat(qvel_rows, dim=0),
        },
        "aux_rewards": {key: torch.cat(values, dim=0) for key, values in aux_rows.items()},
    }
    return _ExpertInferenceDataset(data)


def _sample_inference_batch(dataset: Any, num_samples: int) -> dict[str, Any]:
    if isinstance(dataset, TrajectoryDictBufferMultiDim):
        for key in ("aux_rewards",):
            if key not in dataset.output_key_t:
                dataset.output_key_t.append(key)
        for key in ("qpos", "qvel"):
            if key not in dataset.output_key_tp1:
                dataset.output_key_tp1.append(key)
    sample_count = min(int(num_samples), int(dataset.size()))
    if sample_count <= 0:
        raise ValueError("Replay buffer contains no transitions")
    return dataset.sample(sample_count)


def _select_rows(tree: Any, mask: torch.Tensor) -> Any:
    return tree_map(lambda value: value[mask], tree)


def _inference_stats(reward: torch.Tensor) -> dict[str, float | int]:
    values = reward.reshape(-1)
    return {
        "count": int(values.numel()),
        "positive_count": int((values > 0.0).sum().item()),
        "high_count": int((values >= 0.75).sum().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
    }


def infer_phase_latents(
    model: Any,
    dataset: Any,
    *,
    num_samples: int,
    num_dof: int,
    target_speed: float,
    approach_topk: int,
    box_heading_sigma_deg: float,
    turn_yaw_rate: float,
    transport_speed: float = 0.8,
    transport_slow_speed: float = 0.08,
    transport_slowdown_distance: float = 0.8,
    align_speed: float = 0.12,
    align_topk: int = 1_000,
    lower_topk: int = 100,
    transport_topk: int = 30_000,
    place_topk: int = 30_000,
) -> tuple[
    torch.Tensor,
    dict[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    data = _sample_inference_batch(dataset, num_samples)
    next_obs = get_next("observation", data)
    state = torch.as_tensor(next_obs["state"])
    qpos = get_next("qpos", data)
    qvel = get_next("qvel", data)
    topk = min(int(approach_topk), int(state.shape[0]))
    if topk <= 0:
        raise ValueError("approach_topk must be positive")
    approach_reward, local_velocity = compute_approach_speed_reward(
        qpos,
        qvel,
        state,
        num_dof=num_dof,
        target_speed=target_speed,
        box_heading_sigma_deg=box_heading_sigma_deg,
    )
    approach_indices = torch.topk(approach_reward[:, 0], k=topk, largest=True).indices
    approach_obs = tree_map(lambda value: value[approach_indices], next_obs)
    selected_approach_reward = approach_reward[approach_indices]
    z_approach = model.reward_wr_inference(approach_obs, selected_approach_reward).reshape(1, -1)

    z_turn: dict[int, torch.Tensor] = {}
    turn_stats: dict[str, Any] = {}
    for direction in (-1, 1):
        target_yaw_rate = float(direction) * float(turn_yaw_rate)
        turn_reward, turn_components = compute_turn_reward(
            qpos,
            qvel,
            state,
            num_dof=num_dof,
            target_yaw_rate=target_yaw_rate,
        )
        turn_indices = torch.topk(turn_reward[:, 0], k=topk, largest=True).indices
        turn_obs = tree_map(lambda value: value[turn_indices], next_obs)
        selected_turn_reward = turn_reward[turn_indices]
        z_turn[direction] = model.reward_wr_inference(turn_obs, selected_turn_reward).reshape(1, -1)
        key = "left" if direction > 0 else "right"
        turn_stats[key] = {
            "reward": _inference_stats(selected_turn_reward),
            "selected_yaw_rate_mean": float(turn_components["root_yaw_rate"][turn_indices].mean().item()),
            "selected_planar_speed_mean": float(turn_components["planar_speed"][turn_indices].mean().item()),
        }

    carry_mask = state[:, -2] > 0.5
    if not bool(carry_mask.any()):
        raise ValueError("Sampled replay transitions contain no Carry states")
    carry_obs = _select_rows(next_obs, carry_mask)
    carry_aux = _select_rows(data["aux_rewards"], carry_mask)
    carry_state = state[carry_mask]
    lift_reward = compute_carry_lift_upright_reward(carry_aux).reshape(-1, 1)
    z_lift = model.reward_wr_inference(carry_obs, lift_reward).reshape(1, -1)

    transport_reward, transport_components = compute_transport_reward(
        carry_aux,
        carry_state,
        num_dof=num_dof,
        target_speed=transport_speed,
    )
    transport_k = min(int(transport_topk), int(transport_reward.shape[0]))
    if transport_k <= 0:
        raise ValueError("transport_topk must be positive")
    transport_indices = torch.topk(transport_reward[:, 0], k=transport_k, largest=True).indices
    transport_obs = tree_map(lambda value: value[transport_indices], carry_obs)
    selected_transport_reward = transport_reward[transport_indices]
    z_transport = model.reward_wr_inference(transport_obs, selected_transport_reward).reshape(1, -1)

    slow_reward, slow_components = compute_transport_reward(
        carry_aux,
        carry_state,
        num_dof=num_dof,
        target_speed=transport_slow_speed,
        forward_sigma=0.10,
        lateral_sigma=0.12,
        goal_distance_sigma=transport_slowdown_distance,
    )
    slow_indices = torch.topk(slow_reward[:, 0], k=transport_k, largest=True).indices
    slow_obs = tree_map(lambda value: value[slow_indices], carry_obs)
    selected_slow_reward = slow_reward[slow_indices]
    z_transport_slow = model.reward_wr_inference(slow_obs, selected_slow_reward).reshape(1, -1)

    align_reward, align_components = compute_transport_reward(
        carry_aux,
        carry_state,
        num_dof=num_dof,
        target_speed=align_speed,
        forward_sigma=0.08,
        lateral_sigma=0.10,
        goal_distance_sigma=0.35,
    )
    align_k = min(int(align_topk), int(align_reward.shape[0]))
    if align_k <= 0:
        raise ValueError("align_topk must be positive")
    align_indices = torch.topk(align_reward[:, 0], k=align_k, largest=True).indices
    align_obs = tree_map(lambda value: value[align_indices], carry_obs)
    selected_align_reward = align_reward[align_indices]
    z_align = model.reward_wr_inference(align_obs, selected_align_reward).reshape(1, -1)

    lower_reward, lower_components = compute_lower_reward(
        carry_aux,
        carry_state,
        num_dof=num_dof,
    )
    lower_k = min(int(lower_topk), int(lower_reward.shape[0]))
    if lower_k <= 0:
        raise ValueError("lower_topk must be positive")
    lower_indices = torch.topk(lower_reward[:, 0], k=lower_k, largest=True).indices
    lower_obs = tree_map(lambda value: value[lower_indices], carry_obs)
    selected_lower_reward = lower_reward[lower_indices]
    z_lower = model.reward_wr_inference(lower_obs, selected_lower_reward).reshape(1, -1)

    z_transport_turn: dict[int, torch.Tensor] = {}
    transport_turn_stats: dict[str, Any] = {}
    for direction in (-1, 1):
        turn_reward, turn_components = compute_transport_reward(
            carry_aux,
            carry_state,
            num_dof=num_dof,
            target_speed=transport_speed,
            target_yaw_rate=float(direction) * float(turn_yaw_rate),
            yaw_rate_sigma=0.25,
        )
        turn_indices = torch.topk(turn_reward[:, 0], k=transport_k, largest=True).indices
        turn_obs = tree_map(lambda value: value[turn_indices], carry_obs)
        selected_turn_reward = turn_reward[turn_indices]
        z_transport_turn[direction] = model.reward_wr_inference(turn_obs, selected_turn_reward).reshape(1, -1)
        key = "left" if direction > 0 else "right"
        transport_turn_stats[key] = {
            "reward": _inference_stats(selected_turn_reward),
            "selected_yaw_rate_mean": float(turn_components["root_yaw_rate"][turn_indices].mean().item()),
            "selected_toward_speed_mean": float(turn_components["toward_speed"][turn_indices].mean().item()),
        }

    place_reward, place_components = compute_place_reward(
        carry_aux,
        carry_state,
        num_dof=num_dof,
    )
    place_k = min(int(place_topk), int(place_reward.shape[0]))
    if place_k <= 0:
        raise ValueError("place_topk must be positive")
    place_indices = torch.topk(place_reward[:, 0], k=place_k, largest=True).indices
    place_obs = tree_map(lambda value: value[place_indices], carry_obs)
    selected_place_reward = place_reward[place_indices]
    z_place = model.reward_wr_inference(place_obs, selected_place_reward).reshape(1, -1)

    static_reward, static_components = compute_static_carry_reward(
        carry_aux,
        carry_state,
        num_dof=num_dof,
    )
    z_hold = model.reward_wr_inference(carry_obs, static_reward).reshape(1, -1)

    stats = {
        "sample_count": int(state.shape[0]),
        "carry_count": int(carry_mask.sum().item()),
        "approach_reward": _inference_stats(selected_approach_reward),
        "approach_selected_forward_speed_mean": float(local_velocity[approach_indices, 0].mean().item()),
        "turn_rewards": turn_stats,
        "lift_reward": _inference_stats(lift_reward),
        "transport_reward": _inference_stats(selected_transport_reward),
        "transport_selected_toward_speed_mean": float(
            transport_components["toward_speed"][transport_indices].mean().item()
        ),
        "transport_selected_desired_speed_mean": float(
            transport_components["desired_speed"][transport_indices].mean().item()
        ),
        "transport_selected_target_distance_mean": float(
            transport_components["target_distance"][transport_indices].mean().item()
        ),
        "transport_slow_reward": _inference_stats(selected_slow_reward),
        "transport_slow_selected_toward_speed_mean": float(
            slow_components["toward_speed"][slow_indices].mean().item()
        ),
        "transport_slow_selected_target_distance_mean": float(
            slow_components["target_distance"][slow_indices].mean().item()
        ),
        "align_reward": _inference_stats(selected_align_reward),
        "align_selected_toward_speed_mean": float(
            align_components["toward_speed"][align_indices].mean().item()
        ),
        "align_selected_target_distance_mean": float(
            align_components["target_distance"][align_indices].mean().item()
        ),
        "lower_reward": _inference_stats(selected_lower_reward),
        "lower_selected_vertical_speed_mean": float(
            lower_components["vertical_speed"][lower_indices].mean().item()
        ),
        "lower_selected_target_distance_mean": float(
            lower_components["target_distance"][lower_indices].mean().item()
        ),
        "transport_turn_rewards": transport_turn_stats,
        "place_reward": _inference_stats(selected_place_reward),
        "place_selected_target_distance_mean": float(
            place_components["target_distance"][place_indices].mean().item()
        ),
        "place_selected_released_mean": float(place_components["released"][place_indices].mean().item()),
        "hold_reward": _inference_stats(static_reward),
        "hold_selected_stationary_mean": float(static_components["stationary"].mean().item()),
    }
    return (
        z_approach,
        z_turn,
        z_lift,
        z_transport,
        z_transport_slow,
        z_align,
        z_transport_turn,
        z_lower,
        z_place,
        z_hold,
        stats,
    )


def _nearest_approach_latent(
    z_bank: Mapping[float, torch.Tensor],
    base_quat: torch.Tensor,
    box_direction_world: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    bearing_deg = _box_bearing_deg(base_quat, box_direction_world)

    def circular_error(angle: float) -> float:
        return abs((bearing_deg - float(angle) + 180.0) % 360.0 - 180.0)

    selected_angle = min(z_bank, key=circular_error)
    return z_bank[selected_angle], float(selected_angle), bearing_deg


def _box_bearing_deg(base_quat: torch.Tensor, box_direction_world: torch.Tensor) -> float:
    direction_3d = torch.cat((box_direction_world.reshape(1, 2), box_direction_world.new_zeros(1, 1)), dim=-1)
    direction_local = quat_rotate_inverse(base_quat.reshape(1, 4), direction_3d, w_last=True)[0, :2]
    return float(torch.rad2deg(torch.atan2(direction_local[1], direction_local[0])).item())


def _smooth_place_blend(place_step: int, settle_steps: int, lower_steps: int) -> float:
    """Ease from stable holding to release only after the settling interval."""

    if min(place_step, settle_steps) < 0 or lower_steps <= 0:
        raise ValueError("Place steps must be non-negative and lower_steps must be positive")
    progress = float(np.clip((int(place_step) - int(settle_steps)) / int(lower_steps), 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def _transport_turn_weight(
    *,
    goal_distance: float,
    bearing_deg: float,
    turn_boost_distance: float,
    place_distance: float,
    heading_blend_full_deg: float,
    far_max_turn_blend: float,
    near_max_turn_blend: float,
) -> float:
    """Fade from bearing-based steering to a turn-only latent near the goal."""

    if turn_boost_distance <= place_distance:
        raise ValueError("turn_boost_distance must exceed place_distance")
    if heading_blend_full_deg <= 0.0:
        raise ValueError("heading_blend_full_deg must be positive")
    if not 0.0 <= far_max_turn_blend <= near_max_turn_blend <= 1.0:
        raise ValueError("turn blend limits must satisfy 0 <= far <= near <= 1")

    near = float(np.clip(
        (float(turn_boost_distance) - float(goal_distance))
        / (float(turn_boost_distance) - float(place_distance)),
        0.0,
        1.0,
    ))
    near = near * near * (3.0 - 2.0 * near)
    angle_weight = float(np.clip(abs(float(bearing_deg)) / float(heading_blend_full_deg), 0.0, 1.0))
    far_turn_weight = angle_weight * float(far_max_turn_blend)
    return (1.0 - near) * far_turn_weight + near * float(near_max_turn_blend)


def _box_recovery_phase(horizontal_distance: float, approach_switch_distance: float) -> str:
    """Choose whether a failed placement first needs locomotion or re-grasping."""

    return "approach" if float(horizontal_distance) > float(approach_switch_distance) else "lift"


def _initial_states_at_distance(
    initial_states: dict[str, torch.Tensor],
    distance: float,
) -> dict[str, torch.Tensor]:
    if distance <= 0.0:
        raise ValueError("Initial root-to-box distance must be positive")
    result = {key: value.clone() if torch.is_tensor(value) else value for key, value in initial_states.items()}
    root_xy = result["root_states"][:, :2]
    object_xy = result["object_states"][:, :2]
    direction = root_xy - object_xy
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if bool((norm < 1e-6).any()):
        raise ValueError("Robot root and box centers share the same XY position")
    result["root_states"][:, :2] = object_xy + direction / norm * float(distance)
    result["root_states"][:, 7:13] = 0.0
    result["dof_states"][:, :, 1] = 0.0
    result["object_states"][:, 7:13] = 0.0
    return result


def _with_object_goal_offset(
    initial_states: dict[str, torch.Tensor],
    *,
    offset_x: float,
    offset_y: float,
    goal_height: float,
    goal_x: float | None = None,
    goal_y: float | None = None,
) -> dict[str, torch.Tensor]:
    """Set a box goal from either world coordinates or a world-frame offset."""

    if "object_states" not in initial_states or "target_object_pos" not in initial_states:
        raise ValueError("Carry goal offsets require object and target-object states")
    result = {key: value.clone() if torch.is_tensor(value) else value for key, value in initial_states.items()}
    if (goal_x is None) != (goal_y is None):
        raise ValueError("goal_x and goal_y must be provided together")
    if goal_x is None:
        goal_xy = result["object_states"][:, :2] + result["object_states"].new_tensor(
            [float(offset_x), float(offset_y)]
        )
    else:
        goal_xy = result["object_states"].new_tensor([float(goal_x), float(goal_y)]).reshape(1, 2)
        goal_xy = goal_xy.expand(result["target_object_pos"].shape[0], -1)
    result["target_object_pos"][:, :2] = goal_xy
    result["target_object_pos"][:, 2] = float(goal_height)
    return result


def _resolve_goal_center_height(
    carry_box_cfg: Mapping[str, Any],
    *,
    goal_height: float | None,
    goal_ground_z: float | None,
) -> tuple[float, float]:
    """Resolve the box-center height and the corresponding ground-plane height.

    The rollout target stores the *box center* position, while users usually
    specify a placement goal on the ground plane.  Keeping these two heights
    explicit prevents a goal such as ``(1, 0, 0)`` from putting the box center
    at z=0 and therefore sinking half of the box through the floor.
    """

    if goal_height is not None and goal_ground_z is not None:
        raise ValueError("--goal-height and --goal-z are mutually exclusive")
    half_size = carry_box_cfg.get("half_size", [0.2356, 0.2294, 0.1850])
    if len(half_size) < 3 or float(half_size[2]) <= 0.0:
        raise ValueError(f"Carry box half_size must provide a positive Z half-size, got {half_size!r}")
    half_height = float(half_size[2])
    configured_center = float(carry_box_cfg.get("ground_center_height", half_height))
    if goal_ground_z is not None:
        return float(goal_ground_z) + half_height, float(goal_ground_z)
    if goal_height is not None:
        return float(goal_height), float(goal_height) - half_height
    return configured_center, configured_center - half_height


def _phase_rollout_metrics(env: Any, info: Mapping[str, Any], phase: str, target_speed: float) -> dict[str, Any]:
    metrics = _rollout_metrics(env, info)
    root_xy = env.robot_root_states[0, :2]
    object_xy = env.object_pos[0, :2]
    direction = object_xy - root_xy
    distance = torch.linalg.vector_norm(direction)
    direction = direction / torch.clamp(distance, min=1e-6)
    toward_speed = torch.sum(env.robot_root_states[0, 7:9] * direction)
    box_bearing = _box_bearing_deg(env.base_quat[0], object_xy - root_xy)
    goal_direction = env.target_object_pos[0, :2] - object_xy
    goal_distance = torch.linalg.vector_norm(goal_direction)
    goal_unit_direction = goal_direction / torch.clamp(goal_distance, min=1e-6)
    goalward_object_speed = torch.sum(env.object_lin_vel[0, :2] * goal_unit_direction)
    goal_bearing = _box_bearing_deg(env.base_quat[0], goal_direction)
    metrics.update(
        {
            "phase": phase,
            "horizontal_distance": float(distance.item()),
            "approach_speed": float(toward_speed.item()),
            "approach_speed_error": float(abs(toward_speed.item() - target_speed)),
            "box_bearing_deg": box_bearing,
            "goal_distance": float(goal_distance.item()),
            "goal_bearing_deg": goal_bearing,
            "transport_goalward_speed": float(goalward_object_speed.item()),
            "object_height": float(env.object_pos[0, 2].item()),
            "object_speed": float(torch.linalg.vector_norm(env.object_lin_vel[0]).item()),
        }
    )
    return metrics


def _phase_summary(history: list[dict[str, Any]], dt: float, heading_threshold_deg: float) -> dict[str, Any]:
    approach_rows = [row for row in history if row["phase"] == "approach"]
    hold_rows = [row for row in history if row["phase"] == "hold"]
    transport_rows = [row for row in history if row["phase"] in ("transport", "align")]
    place_rows = [row for row in history if row["phase"] in ("place", "complete")]
    first_lift = next((index for index, row in enumerate(history) if float(row["lift"]) >= 0.5), None)
    first_hold = next((index for index, row in enumerate(history) if row["phase"] == "hold"), None)
    return {
        "approach_steps": len(approach_rows),
        "approach_duration_s": float(len(approach_rows) * dt),
        "approach_speed_mean": None
        if not approach_rows
        else float(np.mean([float(row["approach_speed"]) for row in approach_rows])),
        "approach_speed_mae": None
        if not approach_rows
        else float(np.mean([float(row["approach_speed_error"]) for row in approach_rows])),
        "approach_heading_error_mean_deg": None
        if not approach_rows
        else float(np.mean([abs(float(row["box_bearing_deg"])) for row in approach_rows])),
        "approach_heading_aligned_rate": 0.0
        if not approach_rows
        else float(
            np.mean(
                [abs(float(row["box_bearing_deg"])) <= float(heading_threshold_deg) for row in approach_rows]
            )
        ),
        "first_lift_s": None if first_lift is None else float((first_lift + 1) * dt),
        "first_hold_s": None if first_hold is None else float((first_hold + 1) * dt),
        "hold_rate": float(len(hold_rows) / len(history)),
        "transport_duration_s": float(len(transport_rows) * dt),
        "transport_goalward_speed_mean": None
        if not transport_rows
        else float(np.mean([float(row["transport_goalward_speed"]) for row in transport_rows])),
        "transport_goalward_speed_p90": None
        if not transport_rows
        else float(np.percentile([float(row["transport_goalward_speed"]) for row in transport_rows], 90)),
        "place_duration_s": float(len(place_rows) * dt),
        "final_goal_distance": None if not history else float(history[-1]["goal_distance"]),
        "completed": any(row["phase"] == "complete" for row in history),
        "place_attempts": 0 if not history else int(max(row.get("place_attempt", 0) for row in history)),
        "recovery_count": 0 if not history else int(max(row.get("recovery_count", 0) for row in history)),
        "phase_counts": {
            phase: sum(row["phase"] == phase for row in history)
            for phase in ("approach", "lift", "transport", "align", "place", "complete", "hold")
        },
    }


@torch.no_grad()
def run_phased_inference(args: argparse.Namespace) -> None:
    torch.manual_seed(int(args.seed))
    model_folder = args.model_folder.expanduser().resolve()
    model = load_model_from_checkpoint_dir(
        model_folder / "checkpoint",
        device=checkpoint_load_device(args.device),
    )
    model.to(args.device)
    model.eval()
    robot_config = _resolve_tracking_robot_config(args.robot_config, None)
    robot_training = load_robot_training_spec(robot_config)
    num_dof = len(robot_training.robot.control_joint_names)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()

    wrapped_env = None
    env = None
    env_cfg = None
    use_root_height_obs = False
    if args.inference_source == "expert":
        env_cfg, use_root_height_obs = load_mjlab_env_cfg(
            model_folder,
            data_path=None,
            robot_config=robot_config,
            device=args.device,
            headless=True,
            disable_dr=True,
            disable_obs_noise=True,
            max_episode_length_s=max(float(args.episode_seconds) + 5.0, 100.0),
        )
        if args.expert_data_paths:
            env_cfg = env_cfg.model_copy(
                update={
                    "lafan_tail_path": [str(path.expanduser().resolve()) for path in args.expert_data_paths],
                    "data_mix_weights": [1.0 / len(args.expert_data_paths)] * len(args.expert_data_paths),
                }
            )
        wrapped_env, _ = env_cfg.build(num_envs=1)
        env = wrapped_env._env
        dataset = _expert_inference_dataset(
            env,
            num_samples=int(args.num_samples),
            num_dof=num_dof,
            use_root_height_obs=use_root_height_obs,
        )
        loaded_buffer_path = "expert-motion-cache"
    else:
        dataset, loaded_buffer_path = _load_replay_buffer(
            model_folder,
            buffer_rank=int(args.buffer_rank),
            buffer_path=args.buffer_path,
        )
    (
        z_approach,
        z_turn,
        inferred_z_lift,
        z_transport,
        z_transport_slow,
        z_align,
        z_transport_turn,
        z_lower,
        z_place,
        z_hold,
        inference_stats,
    ) = infer_phase_latents(
        model,
        dataset,
        num_samples=int(args.num_samples),
        num_dof=num_dof,
        target_speed=float(args.approach_speed),
        approach_topk=int(args.approach_topk),
        box_heading_sigma_deg=float(args.box_heading_sigma_deg),
        turn_yaw_rate=float(args.turn_yaw_rate),
        transport_speed=float(args.transport_speed),
        transport_slow_speed=float(args.transport_slow_speed),
        transport_slowdown_distance=float(args.transport_slowdown_distance),
        align_speed=float(args.align_speed),
        align_topk=int(args.align_topk),
        lower_topk=int(args.lower_topk),
        transport_topk=int(args.transport_topk),
        place_topk=int(args.place_topk),
    )
    if args.lift_z_path is None:
        z_lift = inferred_z_lift
        z_lift_path = None
    else:
        z_lift_path = args.lift_z_path.expanduser().resolve()
        z_lift = torch.as_tensor(joblib.load(z_lift_path), dtype=torch.float32, device=args.device).reshape(1, -1)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(z_approach.detach().cpu().numpy(), output_dir / "z_approach.pkl")
    joblib.dump({direction: z.detach().cpu().numpy() for direction, z in z_turn.items()}, output_dir / "z_turn.pkl")
    joblib.dump(z_lift.detach().cpu().numpy(), output_dir / "z_lift.pkl")
    joblib.dump(z_transport.detach().cpu().numpy(), output_dir / "z_transport.pkl")
    joblib.dump(z_transport_slow.detach().cpu().numpy(), output_dir / "z_transport_slow.pkl")
    joblib.dump(z_align.detach().cpu().numpy(), output_dir / "z_align.pkl")
    joblib.dump(z_lower.detach().cpu().numpy(), output_dir / "z_lower.pkl")
    joblib.dump(
        {direction: z.detach().cpu().numpy() for direction, z in z_transport_turn.items()},
        output_dir / "z_transport_turn.pkl",
    )
    joblib.dump(z_place.detach().cpu().numpy(), output_dir / "z_place.pkl")
    joblib.dump(z_hold.detach().cpu().numpy(), output_dir / "z_hold.pkl")
    print(f"[INFO] phase inference stats={inference_stats}")
    inference_metadata = {
        "model_folder": str(model_folder),
        "inference_source": args.inference_source,
        "source": str(loaded_buffer_path),
        "num_dof": num_dof,
        "stats": inference_stats,
    }
    (output_dir / "inference_stats.json").write_text(
        json.dumps(inference_metadata, indent=2, sort_keys=True) + "\n"
    )
    if args.latents_only:
        if wrapped_env is not None:
            wrapped_env.close()
        return

    if env is None:
        env_cfg, use_root_height_obs = load_mjlab_env_cfg(
            model_folder,
            data_path=args.data_path,
            robot_config=robot_config,
            device=args.device,
            headless=args.headless,
            disable_dr=args.disable_dr,
            disable_obs_noise=args.disable_obs_noise,
            max_episode_length_s=max(float(args.episode_seconds) + 5.0, 100.0),
        )
        wrapped_env, _ = env_cfg.build(num_envs=1)
        env = wrapped_env._env
        env._motion_lib.load_all_motions()
    env.is_evaluating = True
    renderer = (
        MujocoQposRenderer(
            robot_xml,
            render_size=args.render_size,
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
            expected_qpos_size=7 + num_dof + 7,
            box_half_size=tuple(float(value) for value in env.carry_box_cfg["half_size"]),
            box_geom_pos=tuple(float(value) for value in env.carry_box_cfg["geom_pos"]),
            box_geom_quat_wxyz=tuple(float(value) for value in env.carry_box_cfg["geom_quat_wxyz"]),
            target_marker_radius=float(args.place_success_distance),
        )
        if args.save_mp4
        else None
    )

    try:
        _backward_obs, obs_dict = get_backward_observation(
            env,
            int(args.initial_motion_id),
            use_root_height_obs=use_root_height_obs,
        )
        num_frames = int(obs_dict["dof_pos"].shape[0])
        initial_obs = _select_frame(obs_dict, int(args.initial_frame), num_frames)
        base_initial_states = _target_states_from_obs(initial_obs, device=args.device, num_dof=num_dof)
        goal_height, goal_ground_z = _resolve_goal_center_height(
            env.carry_box_cfg,
            goal_height=args.goal_height,
            goal_ground_z=args.goal_z,
        )
        episode_steps = max(1, round(float(args.episode_seconds) / float(env.dt)))
        final_window = max(1, round(float(args.final_window_seconds) / float(env.dt)))
        place_settle_steps = max(0, round(float(args.place_settle_seconds) / float(env.dt)))
        place_lower_steps = max(1, round(float(args.place_lower_seconds) / float(env.dt)))
        place_release_steps = max(1, round(float(args.place_release_seconds) / float(env.dt)))
        place_retry_steps = max(1, round(float(args.place_retry_delay_seconds) / float(env.dt)))
        place_max_attempt_steps = max(1, round(float(args.place_max_attempt_seconds) / float(env.dt)))
        summaries: dict[str, Any] = {}

        if args.goal_offset_distances is None:
            goal_cases = [(None, float(args.goal_offset_x), float(args.goal_offset_y))]
        else:
            direction = np.asarray([float(args.goal_offset_x), float(args.goal_offset_y)], dtype=np.float64)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-9:
                raise ValueError("--goal-offset-x/y must define a direction with --goal-offset-distances")
            direction /= direction_norm
            goal_cases = [
                (float(goal_distance), float(direction[0] * goal_distance), float(direction[1] * goal_distance))
                for goal_distance in args.goal_offset_distances
            ]

        rollout_cases = [
            (float(distance), goal_distance, goal_offset_x, goal_offset_y)
            for distance in args.initial_distances
            for goal_distance, goal_offset_x, goal_offset_y in goal_cases
        ]
        for distance, requested_goal_distance, goal_offset_x, goal_offset_y in rollout_cases:
            goal_states = _with_object_goal_offset(
                base_initial_states,
                offset_x=goal_offset_x,
                offset_y=goal_offset_y,
                goal_height=goal_height,
                goal_x=args.goal_x,
                goal_y=args.goal_y,
            )
            target_states = _initial_states_at_distance(goal_states, distance)
            observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)
            if renderer is not None:
                renderer.set_target_marker(env.target_object_pos[0].detach().cpu().numpy())
            phase = "approach"
            lift_ready_steps = 0
            hold_lost_steps = 0
            place_enter_steps = 0
            place_drift_steps = 0
            placed_steps = 0
            place_phase_steps = 0
            place_failure_steps = 0
            completion_lost_steps = 0
            release_started_step: int | None = None
            place_attempt = 0
            recovery_count = 0
            frames: list[np.ndarray] = []
            history: list[dict[str, Any]] = []
            use_env_render = True
            for step in range(episode_steps):
                horizontal_distance = torch.linalg.vector_norm(env.object_pos[0, :2] - env.robot_root_states[0, :2])
                if phase == "approach" and float(horizontal_distance.item()) <= float(args.approach_switch_distance):
                    phase = "lift"

                if phase == "approach":
                    policy_observation = (
                        _masked_object_observation(observation)
                        if args.mask_approach_object_context
                        else observation
                    )
                    box_direction = env.object_pos[0, :2] - env.robot_root_states[0, :2]
                    box_bearing = _box_bearing_deg(env.base_quat[0], box_direction)
                    turn_weight = min(
                        abs(box_bearing) / float(args.heading_blend_full_deg),
                        1.0,
                    ) * float(args.max_turn_blend)
                    turn_direction = 1 if box_bearing >= 0.0 else -1
                    z = model.project_z(
                        (1.0 - turn_weight) * z_approach + turn_weight * z_turn[turn_direction]
                    )
                    approach_mode = "forward" if turn_weight < 1e-3 else ("forward_left" if turn_direction > 0 else "forward_right")
                elif phase == "lift":
                    policy_observation = observation
                    z = z_lift
                    approach_mode = None
                    box_bearing = None
                elif phase == "transport":
                    policy_observation = observation
                    goal_direction = env.target_object_pos[0, :2] - env.object_pos[0, :2]
                    goal_distance = float(torch.linalg.vector_norm(goal_direction).item())
                    goal_bearing = _box_bearing_deg(env.base_quat[0], goal_direction)
                    turn_weight = _transport_turn_weight(
                        goal_distance=goal_distance,
                        bearing_deg=goal_bearing,
                        turn_boost_distance=float(args.transport_turn_boost_distance),
                        place_distance=float(args.place_align_distance),
                        heading_blend_full_deg=float(args.transport_heading_blend_full_deg),
                        far_max_turn_blend=float(args.transport_max_turn_blend),
                        near_max_turn_blend=float(args.transport_near_max_turn_blend),
                    )
                    turn_direction = 1 if goal_bearing >= 0.0 else -1
                    z = model.project_z(
                        (1.0 - turn_weight) * z_transport
                        + turn_weight * z_transport_turn[turn_direction]
                    )
                    approach_mode = (
                        "transport_forward"
                        if turn_weight < 1e-3
                        else ("transport_left" if turn_direction > 0 else "transport_right")
                    )
                    box_bearing = None
                elif phase == "place":
                    policy_observation = _goal_lookahead_observation(
                        observation,
                        float(args.align_goal_lookahead),
                    )
                    if release_started_step is None:
                        place_blend = _smooth_place_blend(
                            place_phase_steps,
                            place_settle_steps,
                            place_lower_steps,
                        )
                        z = model.project_z((1.0 - place_blend) * z_align + place_blend * z_lower)
                        approach_mode = (
                            "place_settle" if place_blend <= 0.0 else f"place_lower_{place_blend:.2f}"
                        )
                    else:
                        release_blend = _smooth_place_blend(
                            place_phase_steps - release_started_step,
                            0,
                            place_release_steps,
                        )
                        z = model.project_z((1.0 - release_blend) * z_lower + release_blend * z_place)
                        approach_mode = f"place_release_{release_blend:.2f}"
                    box_bearing = None
                elif phase == "complete":
                    policy_observation = observation
                    z = z_place
                    approach_mode = "placed_verify"
                    box_bearing = None
                else:
                    policy_observation = observation
                    z = z_hold
                    approach_mode = None
                    box_bearing = None

                action = model.act(policy_observation, z, mean=True)
                observation, _reward, _terminated, _truncated, info = wrapped_env.step(action, to_numpy=False)
                metrics = _phase_rollout_metrics(env, info, phase, float(args.approach_speed))
                metrics["approach_mode"] = approach_mode
                metrics["box_bearing"] = box_bearing
                metrics["place_attempt"] = place_attempt
                metrics["recovery_count"] = recovery_count
                history.append(metrics)

                if phase == "lift":
                    ready = (
                        bool(metrics["bilateral_contact"])
                        and float(metrics["lift"]) >= float(args.hold_switch_lift)
                        and float(metrics["robot_upright"]) >= float(args.hold_switch_upright)
                    )
                    lift_ready_steps = lift_ready_steps + 1 if ready else 0
                    if lift_ready_steps >= int(args.phase_hold_steps):
                        phase = "transport"
                        hold_lost_steps = 0
                        place_enter_steps = 0
                elif phase == "transport":
                    lost = not bool(metrics["bilateral_contact"]) or float(metrics["lift"]) < float(args.hold_release_lift)
                    hold_lost_steps = hold_lost_steps + 1 if lost else 0
                    inside_place = (
                        not lost
                        and float(metrics["goal_distance"]) <= float(args.place_align_distance)
                    )
                    place_enter_steps = place_enter_steps + 1 if inside_place else 0
                    if hold_lost_steps >= int(args.phase_hold_steps):
                        phase = (
                            "approach"
                            if float(metrics["horizontal_distance"]) > float(args.approach_switch_distance)
                            else "lift"
                        )
                        lift_ready_steps = 0
                        place_enter_steps = 0
                    elif place_enter_steps >= int(args.place_enter_steps):
                        phase = "place"
                        place_attempt += 1
                        place_phase_steps = 0
                        place_failure_steps = 0
                        release_started_step = None
                        place_enter_steps = 0
                        place_drift_steps = 0
                        placed_steps = 0
                elif phase == "place":
                    place_phase_steps += 1
                    ground_error = abs(float(metrics["object_height"]) - goal_height)
                    drifted_while_holding = (
                        release_started_step is None
                        and bool(metrics["bilateral_contact"])
                        and float(metrics["goal_distance"]) > float(args.place_drift_distance)
                    )
                    place_drift_steps = place_drift_steps + 1 if drifted_while_holding else 0
                    if place_drift_steps >= int(args.place_drift_steps):
                        phase = "transport"
                        place_enter_steps = 0
                        place_drift_steps = 0
                        place_phase_steps = 0
                        place_failure_steps = 0
                        release_started_step = None
                        placed_steps = 0
                        continue_place_checks = False
                    else:
                        continue_place_checks = True
                    if (
                        continue_place_checks
                        and release_started_step is None
                        and bool(metrics["bilateral_contact"])
                        and ground_error <= float(args.place_release_height)
                    ):
                        release_started_step = place_phase_steps
                    placed = (
                        float(metrics["goal_distance"]) <= float(args.place_success_distance)
                        and ground_error <= float(args.place_success_height)
                        and float(metrics["object_speed"]) <= float(args.place_success_speed)
                        and float(metrics["box_upright"]) >= float(args.place_success_upright)
                        and not bool(metrics["bilateral_contact"])
                    )
                    placed_steps = placed_steps + 1 if placed and continue_place_checks else 0
                    if continue_place_checks and placed_steps >= int(args.place_success_steps):
                        phase = "complete"
                        completion_lost_steps = 0
                    elif continue_place_checks:
                        released = not bool(metrics["bilateral_contact"])
                        premature_release = released and ground_error > float(args.place_release_height)
                        settled_off_target = (
                            released
                            and ground_error <= float(args.place_success_height)
                            and float(metrics["object_speed"]) <= float(args.place_success_speed)
                            and float(metrics["goal_distance"]) > float(args.place_success_distance)
                        )
                        timed_out = place_phase_steps >= place_max_attempt_steps
                        drifted = float(metrics["goal_distance"]) > float(args.place_abort_distance)
                        failed = premature_release or settled_off_target or timed_out or drifted
                        place_failure_steps = place_failure_steps + 1 if failed else 0
                    if phase == "place" and place_failure_steps >= place_retry_steps:
                        phase = _box_recovery_phase(
                            float(metrics["horizontal_distance"]),
                            float(args.approach_switch_distance),
                        )
                        recovery_count += 1
                        lift_ready_steps = 0
                        hold_lost_steps = 0
                        place_enter_steps = 0
                        place_drift_steps = 0
                        placed_steps = 0
                        place_phase_steps = 0
                        place_failure_steps = 0
                        release_started_step = None
                elif phase == "complete":
                    ground_error = abs(float(metrics["object_height"]) - goal_height)
                    still_complete = (
                        float(metrics["goal_distance"]) <= float(args.place_success_distance)
                        and ground_error <= float(args.place_success_height)
                        and float(metrics["object_speed"]) <= float(args.place_success_speed)
                        and float(metrics["box_upright"]) >= float(args.place_success_upright)
                    )
                    completion_lost_steps = 0 if still_complete else completion_lost_steps + 1
                    if completion_lost_steps >= place_retry_steps:
                        phase = _box_recovery_phase(
                            float(metrics["horizontal_distance"]),
                            float(args.approach_switch_distance),
                        )
                        recovery_count += 1
                        lift_ready_steps = 0
                        hold_lost_steps = 0
                        place_enter_steps = 0
                        place_drift_steps = 0
                        placed_steps = 0
                        completion_lost_steps = 0
                elif phase == "hold":
                    # Retained for compatibility with previous summaries and latent dumps.
                    pass

                if renderer is not None:
                    frame, use_env_render = render_policy_frame(wrapped_env, renderer, use_env_render=use_env_render)
                    frames.append(frame)
                if step == 0 or step + 1 == episode_steps or (step + 1) % 100 == 0:
                    print(
                        f"[INFO] distance={distance:.2f}m step={step + 1}/{episode_steps} "
                        f"phase={phase} mode={approach_mode} box_distance={metrics['horizontal_distance']:.3f} "
                        f"bearing={metrics['box_bearing_deg']:.1f}deg "
                        f"goal_distance={metrics['goal_distance']:.3f}m",
                        flush=True,
                    )

            goal_suffix = (
                f"goal_distance_{requested_goal_distance:.1f}m"
                if requested_goal_distance is not None
                else ("absolute_goal" if args.goal_x is not None else "configured_goal")
            )
            name = f"carry_transport_place_initial_{distance:.1f}m_{goal_suffix}"
            video_path = output_dir / f"{name}.mp4"
            if renderer is not None:
                media.write_video(str(video_path), frames, fps=int(args.fps))
                print(f"[INFO] saved {video_path}")
            summaries[f"initial_{distance:.1f}m_{goal_suffix}"] = {
                "video": str(video_path),
                "initial_distance": distance,
                "goal_offset": [goal_offset_x, goal_offset_y],
                "requested_goal_distance": requested_goal_distance,
                **_summarize_rollout(history, final_window, float(env.dt)),
                **_phase_summary(history, float(env.dt), float(args.heading_aligned_deg)),
            }

        result = {
            "model_folder": str(model_folder),
            "buffer_path": str(loaded_buffer_path),
            "lift_z_path": None if z_lift_path is None else str(z_lift_path),
            "goal_offset": [float(args.goal_offset_x), float(args.goal_offset_y)],
            "goal_world_xy": None if args.goal_x is None else [float(args.goal_x), float(args.goal_y)],
            "goal_world_xyz": None
            if args.goal_x is None
            else [float(args.goal_x), float(args.goal_y), float(goal_height)],
            "goal_ground_xyz": None
            if args.goal_x is None
            else [float(args.goal_x), float(args.goal_y), float(goal_ground_z)],
            "goal_height": goal_height,
            "target_approach_speed": float(args.approach_speed),
            "heading_aligned_deg": float(args.heading_aligned_deg),
            "heading_blend_full_deg": float(args.heading_blend_full_deg),
            "max_turn_blend": float(args.max_turn_blend),
            "box_heading_sigma_deg": float(args.box_heading_sigma_deg),
            "turn_yaw_rate": float(args.turn_yaw_rate),
            "approach_switch_distance": float(args.approach_switch_distance),
            "transport_speed": float(args.transport_speed),
            "transport_slow_speed": float(args.transport_slow_speed),
            "transport_slowdown_distance": float(args.transport_slowdown_distance),
            "transport_turn_boost_distance": float(args.transport_turn_boost_distance),
            "transport_heading_blend_full_deg": float(args.transport_heading_blend_full_deg),
            "transport_max_turn_blend": float(args.transport_max_turn_blend),
            "transport_near_max_turn_blend": float(args.transport_near_max_turn_blend),
            "place_align_distance": float(args.place_align_distance),
            "place_enter_steps": int(args.place_enter_steps),
            "recovery_align_distance": float(args.recovery_align_distance),
            "place_drift_distance": float(args.place_drift_distance),
            "place_drift_steps": int(args.place_drift_steps),
            "align_speed": float(args.align_speed),
            "align_max_turn_blend": float(args.align_max_turn_blend),
            "align_goal_lookahead": float(args.align_goal_lookahead),
            "place_switch_distance": float(args.place_switch_distance),
            "place_success_distance": float(args.place_success_distance),
            "place_settle_seconds": float(args.place_settle_seconds),
            "place_lower_seconds": float(args.place_lower_seconds),
            "place_release_seconds": float(args.place_release_seconds),
            "place_retry_delay_seconds": float(args.place_retry_delay_seconds),
            "place_max_attempt_seconds": float(args.place_max_attempt_seconds),
            "place_release_height": float(args.place_release_height),
            "episode_seconds": float(args.episode_seconds),
            "inference_stats": inference_stats,
            "distances": summaries,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[INFO] saved summary {summary_path}")
    finally:
        if renderer is not None:
            renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument(
        "--lift-z-path",
        type=Path,
        default=None,
        help="Optional compatible lift latent. By default it is inferred from the selected replay buffer.",
    )
    parser.add_argument("--buffer-path", type=Path, default=None)
    parser.add_argument("--buffer-rank", type=int, default=0)
    parser.add_argument(
        "--inference-source",
        choices=("replay", "expert"),
        default="replay",
        help="Infer phase latents from a replay buffer or directly from expert motion caches.",
    )
    parser.add_argument(
        "--expert-data-paths",
        type=Path,
        nargs="+",
        default=None,
        help="Motion caches used by --inference-source expert; normally LAFAN followed by Carry.",
    )
    parser.add_argument("--num-samples", type=int, default=150_000)
    parser.add_argument("--approach-topk", type=int, default=30_000)
    parser.add_argument("--transport-topk", type=int, default=30_000)
    parser.add_argument("--place-topk", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--data-manifest", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--rebuild-motion-cache", action="store_true")
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--initial-motion-id", type=int, default=67)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--initial-distances", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0])
    parser.add_argument(
        "--goal-offset-x",
        type=float,
        default=1.0,
        help="Relative world-frame goal X offset from the initial box position (ignored with --goal-x/--goal-y).",
    )
    parser.add_argument(
        "--goal-offset-y",
        type=float,
        default=0.0,
        help="Relative world-frame goal Y offset from the initial box position (ignored with --goal-x/--goal-y).",
    )
    parser.add_argument(
        "--goal-offset-distances",
        type=float,
        nargs="+",
        default=None,
        help="Batch goal distances along the --goal-offset-x/y direction.",
    )
    parser.add_argument("--goal-x", type=float, default=None, help="Optional absolute world-frame box goal X.")
    parser.add_argument("--goal-y", type=float, default=None, help="Optional absolute world-frame box goal Y.")
    parser.add_argument(
        "--goal-z",
        type=float,
        default=None,
        help="Optional absolute world-frame ground-plane goal Z; the box-center target is goal-z + box half-height.",
    )
    parser.add_argument(
        "--goal-height",
        type=float,
        default=None,
        help="Optional absolute box-center goal height; mutually exclusive with --goal-z.",
    )
    parser.add_argument("--approach-speed", type=float, default=0.8)
    parser.add_argument("--box-heading-sigma-deg", type=float, default=15.0)
    parser.add_argument("--heading-aligned-deg", type=float, default=15.0)
    parser.add_argument("--heading-blend-full-deg", type=float, default=60.0)
    parser.add_argument(
        "--max-turn-blend",
        type=float,
        default=0.90,
        help="Maximum approach turn-latent weight; 0.90 always retains at least one-tenth forward latent.",
    )
    parser.add_argument("--turn-yaw-rate", type=float, default=0.8)
    parser.add_argument("--transport-speed", type=float, default=0.8)
    parser.add_argument("--transport-slow-speed", type=float, default=0.08)
    parser.add_argument("--transport-slowdown-distance", type=float, default=0.8)
    parser.add_argument("--align-speed", type=float, default=0.12)
    parser.add_argument("--align-topk", type=int, default=1_000)
    parser.add_argument("--lower-topk", type=int, default=100)
    parser.add_argument("--align-max-turn-blend", type=float, default=0.15)
    parser.add_argument(
        "--align-goal-lookahead",
        type=float,
        default=0.0,
        help="Extra observed XY goal distance during placement; disabled by default.",
    )
    parser.add_argument("--transport-heading-blend-full-deg", type=float, default=45.0)
    parser.add_argument("--transport-max-turn-blend", type=float, default=0.45)
    parser.add_argument(
        "--transport-near-max-turn-blend",
        type=float,
        default=0.8,
        help="Maximum near-goal Transport turn weight; 0.8 retains at least one-fifth forward latent.",
    )
    parser.add_argument("--transport-turn-boost-distance", type=float, default=1.2)
    parser.add_argument("--approach-switch-distance", type=float, default=0.75)
    parser.add_argument("--hold-switch-lift", type=float, default=0.5)
    parser.add_argument("--hold-release-lift", type=float, default=0.25)
    parser.add_argument("--hold-switch-upright", type=float, default=0.85)
    parser.add_argument("--phase-hold-steps", type=int, default=5)
    parser.add_argument("--place-align-distance", type=float, default=0.25)
    parser.add_argument("--place-enter-steps", type=int, default=5)
    parser.add_argument("--recovery-align-distance", type=float, default=0.25)
    parser.add_argument("--place-drift-distance", type=float, default=0.35)
    parser.add_argument("--place-drift-steps", type=int, default=10)
    parser.add_argument("--place-switch-distance", type=float, default=0.20)
    parser.add_argument("--place-abort-distance", type=float, default=0.80)
    parser.add_argument("--place-success-distance", type=float, default=0.20)
    parser.add_argument("--place-success-height", type=float, default=0.05)
    parser.add_argument("--place-success-speed", type=float, default=0.15)
    parser.add_argument("--place-success-upright", type=float, default=0.85)
    parser.add_argument("--place-success-steps", type=int, default=25)
    parser.add_argument("--place-settle-seconds", type=float, default=0.2)
    parser.add_argument("--place-lower-seconds", type=float, default=1.0)
    parser.add_argument("--place-release-seconds", type=float, default=2.0)
    parser.add_argument("--place-retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--place-max-attempt-seconds", type=float, default=10.0)
    parser.add_argument(
        "--place-release-height",
        type=float,
        default=0.08,
        help="Maximum box-center height error allowed when bilateral contact is released.",
    )
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--final-window-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    add_bool_arg(parser, "--disable-dr", True, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", True, "Disable observation noise.")
    add_bool_arg(
        parser,
        "--mask-approach-object-context",
        False,
        "Mask object context during approach for legacy policies without object-aware approach training.",
    )
    add_bool_arg(parser, "--save-mp4", True, "Render phased Carry videos.")
    add_bool_arg(parser, "--latents-only", False, "Only infer and save phase latents; skip rollout evaluation.")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=4.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    if args.inference_source == "expert" and not args.expert_data_paths:
        parser.error("--expert-data-paths is required with --inference-source expert")
    if args.heading_blend_full_deg <= 0.0:
        parser.error("--heading-blend-full-deg must be positive")
    if not 0.0 <= args.max_turn_blend < 1.0:
        parser.error("--max-turn-blend must be in [0, 1) so forward motion is always retained")
    if args.transport_heading_blend_full_deg <= 0.0:
        parser.error("--transport-heading-blend-full-deg must be positive")
    if not 0.0 <= args.transport_max_turn_blend < 1.0:
        parser.error("--transport-max-turn-blend must be in [0, 1)")
    if not args.transport_max_turn_blend <= args.transport_near_max_turn_blend <= 1.0:
        parser.error("--transport-near-max-turn-blend must be in [transport-max-turn-blend, 1]")
    if args.transport_speed <= 0.0:
        parser.error("--transport-speed must be positive")
    if args.transport_slow_speed <= 0.0:
        parser.error("--transport-slow-speed must be positive")
    if args.transport_slow_speed >= args.transport_speed:
        parser.error("--transport-slow-speed must be less than --transport-speed")
    if args.transport_turn_boost_distance <= args.place_align_distance:
        parser.error("--transport-turn-boost-distance must exceed --place-align-distance")
    if args.align_speed <= 0.0 or args.align_topk <= 0 or args.lower_topk <= 0:
        parser.error("--align-speed, --align-topk, and --lower-topk must be positive")
    if not 0.0 <= args.align_max_turn_blend < 1.0:
        parser.error("--align-max-turn-blend must be in [0, 1)")
    if args.align_goal_lookahead < 0.0:
        parser.error("--align-goal-lookahead must be non-negative")
    if args.place_success_distance > args.place_switch_distance:
        parser.error("--place-success-distance cannot exceed --place-switch-distance")
    if args.place_align_distance <= args.place_switch_distance:
        parser.error("--place-align-distance must exceed --place-switch-distance")
    if args.recovery_align_distance < args.place_align_distance:
        parser.error("--recovery-align-distance must be at least --place-align-distance")
    if not args.place_align_distance < args.place_drift_distance < args.place_abort_distance:
        parser.error("place hysteresis must satisfy align < drift < abort")
    if args.place_enter_steps <= 0 or args.place_drift_steps <= 0:
        parser.error("--place-enter-steps and --place-drift-steps must be positive")
    if args.place_abort_distance <= args.place_align_distance:
        parser.error("--place-abort-distance must exceed --place-align-distance")
    if args.place_abort_distance <= args.recovery_align_distance:
        parser.error("--place-abort-distance must exceed --recovery-align-distance")
    if args.place_abort_distance <= args.place_switch_distance:
        parser.error("--place-abort-distance must exceed --place-switch-distance")
    if not 0.0 < args.place_success_upright <= 1.0:
        parser.error("--place-success-upright must be in (0, 1]")
    if min(
        args.place_settle_seconds,
        args.place_lower_seconds,
        args.place_release_seconds,
        args.place_retry_delay_seconds,
    ) <= 0.0:
        parser.error("place settle/lower/release/retry durations must be positive")
    if args.place_max_attempt_seconds <= args.place_settle_seconds + args.place_lower_seconds + args.place_release_seconds:
        parser.error("--place-max-attempt-seconds must exceed settle + lower + release durations")
    if args.place_release_height <= args.place_success_height:
        parser.error("--place-release-height must exceed --place-success-height")
    if (args.goal_x is None) != (args.goal_y is None):
        parser.error("--goal-x and --goal-y must be provided together")
    if args.goal_height is not None and args.goal_z is not None:
        parser.error("--goal-height and --goal-z are mutually exclusive")
    if args.goal_offset_distances is not None:
        if args.goal_x is not None:
            parser.error("--goal-offset-distances cannot be combined with --goal-x/--goal-y")
        if any(distance <= 0.0 for distance in args.goal_offset_distances):
            parser.error("--goal-offset-distances values must be positive")

    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        if args.dataset is None:
            parser.error("--dataset is required with --data-manifest")
        args.data_path = Path(
            prepare_manifest_dataset_path(
                args.data_manifest,
                args.dataset,
                split="inference",
                rebuild_cache=bool(args.rebuild_motion_cache),
            )
        )
        manifest_robot_config = prepare_manifest_robot_config_path(args.data_manifest)
    args.robot_config = _resolve_tracking_robot_config(args.robot_config, manifest_robot_config)
    return args


def main() -> None:
    run_phased_inference(parse_args())


if __name__ == "__main__":
    main()
