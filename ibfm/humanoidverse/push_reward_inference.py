"""Infer ground-pushing skills from balanced UFO replay samples.

The exported latent bank is consumed by ``carry_phased_sim2sim`` with
``--controller-mode push``.  Existing Carry latents are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from humanoidverse.carry_phased_reward_inference import _expert_inference_dataset
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg
from humanoidverse.tracking_inference import DEFAULT_ROBOT_CONFIG
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _aux_vector(aux_rewards: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in aux_rewards:
        raise KeyError(f"Replay buffer is missing Carry auxiliary reward {name!r}")
    return torch.as_tensor(aux_rewards[name], dtype=torch.float32).reshape(-1)


def _push_components(
    aux_rewards: Mapping[str, torch.Tensor],
    state: torch.Tensor,
    *,
    num_dof: int,
) -> dict[str, torch.Tensor]:
    state = torch.as_tensor(state, dtype=torch.float32)
    object_start = 2 * int(num_dof) + 6
    if state.ndim != 2 or state.shape[-1] != object_start + 34:
        raise ValueError(
            f"Ground-push inference requires the non-ablated 34D object context, got {tuple(state.shape)}"
        )

    object_xy = state[:, object_start : object_start + 2]
    object_velocity = state[:, object_start + 9 : object_start + 12]
    object_ang_velocity = state[:, object_start + 12 : object_start + 15]
    target_xy = state[:, object_start + 15 : object_start + 17]
    contact = state[:, object_start + 18 : object_start + 20]
    target_distance = torch.linalg.vector_norm(target_xy, dim=-1)
    target_direction = target_xy / torch.clamp(target_distance.unsqueeze(-1), min=1e-6)
    lateral_direction = torch.stack((-target_direction[:, 1], target_direction[:, 0]), dim=-1)

    return {
        "object_present": state[:, object_start + 32],
        "contact_any": torch.max(contact, dim=-1).values,
        "contact_bilateral": torch.min(contact, dim=-1).values,
        "lift": _aux_vector(aux_rewards, "carry_lift"),
        "robot_upright": _aux_vector(aux_rewards, "carry_robot_upright"),
        "box_upright": _aux_vector(aux_rewards, "carry_upright"),
        "drop": _aux_vector(aux_rewards, "carry_drop"),
        "target_distance": target_distance,
        "toward_speed": torch.sum(object_velocity[:, :2] * target_direction, dim=-1),
        "lateral_speed": torch.sum(object_velocity[:, :2] * lateral_direction, dim=-1),
        "vertical_speed": object_velocity[:, 2],
        "angular_speed": torch.linalg.vector_norm(object_ang_velocity, dim=-1),
        # Root -> box projected onto box -> goal. Positive means the root is
        # behind the box rather than between the box and its goal.
        "behind_projection": torch.sum(object_xy * target_direction, dim=-1),
        "behind_lateral_error": torch.abs(torch.sum(object_xy * lateral_direction, dim=-1)),
        "root_yaw_rate": state[:, 2 * int(num_dof) + 5],
    }


def compute_push_contact_reward(
    aux_rewards: Mapping[str, torch.Tensor],
    state: torch.Tensor,
    *,
    num_dof: int,
    ground_lift_sigma: float = 0.08,
    object_speed_sigma: float = 0.12,
    behind_distance: float = 0.25,
    behind_sigma: float = 0.08,
    lateral_alignment_sigma: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer stable hand contact while standing behind a grounded box."""

    if min(
        ground_lift_sigma,
        object_speed_sigma,
        behind_sigma,
        lateral_alignment_sigma,
    ) <= 0.0:
        raise ValueError("Push-contact reward sigmas must be positive")
    components = _push_components(aux_rewards, state, num_dof=num_dof)
    on_ground = torch.exp(-torch.square(components["lift"] / float(ground_lift_sigma)))
    upright = components["robot_upright"] * components["box_upright"]
    stationary = torch.exp(
        -torch.square(components["toward_speed"] / float(object_speed_sigma))
        -torch.square(components["lateral_speed"] / float(object_speed_sigma))
        -torch.square(components["vertical_speed"] / float(object_speed_sigma))
        -torch.square(components["angular_speed"] / 0.5)
    )
    behind = torch.sigmoid(
        (components["behind_projection"] - float(behind_distance)) / float(behind_sigma)
    )
    alignment = torch.exp(
        -torch.square(components["behind_lateral_error"] / float(lateral_alignment_sigma))
    )
    reward = (
        components["object_present"]
        * components["contact_any"]
        * on_ground
        * upright
        * stationary
        * behind
        * alignment
        * (1.0 - components["drop"])
    )
    return reward.unsqueeze(-1), {
        **components,
        "on_ground": on_ground,
        "upright": upright,
        "stationary": stationary,
        "behind": behind,
        "alignment": alignment,
    }


def compute_push_transport_reward(
    aux_rewards: Mapping[str, torch.Tensor],
    state: torch.Tensor,
    *,
    num_dof: int,
    target_speed: float = 0.2,
    target_yaw_rate: float = 0.0,
    forward_sigma: float = 0.10,
    lateral_sigma: float = 0.12,
    yaw_rate_sigma: float = 0.25,
    ground_lift_sigma: float = 0.08,
    behind_distance: float = 0.25,
    behind_sigma: float = 0.10,
    lateral_alignment_sigma: float = 0.30,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer upright, target-directed pushing while the box remains grounded."""

    if target_speed <= 0.0 or min(
        forward_sigma,
        lateral_sigma,
        yaw_rate_sigma,
        ground_lift_sigma,
        behind_sigma,
        lateral_alignment_sigma,
    ) <= 0.0:
        raise ValueError("Push-transport speed and sigmas must be positive")
    components = _push_components(aux_rewards, state, num_dof=num_dof)
    speed_score = torch.exp(
        -torch.square((components["toward_speed"] - float(target_speed)) / float(forward_sigma))
        -torch.square(components["lateral_speed"] / float(lateral_sigma))
    )
    vertical_stable = torch.exp(-torch.square(components["vertical_speed"] / 0.10))
    angular_stable = torch.exp(-torch.square(components["angular_speed"] / 0.50))
    yaw_score = torch.exp(
        -torch.square((components["root_yaw_rate"] - float(target_yaw_rate)) / float(yaw_rate_sigma))
    )
    on_ground = torch.exp(-torch.square(components["lift"] / float(ground_lift_sigma)))
    upright = components["robot_upright"] * components["box_upright"]
    behind = torch.sigmoid(
        (components["behind_projection"] - float(behind_distance)) / float(behind_sigma)
    )
    alignment = torch.exp(
        -torch.square(components["behind_lateral_error"] / float(lateral_alignment_sigma))
    )
    # A small contact floor keeps useful near-contact expert-like states from
    # receiving exactly zero while still strongly preferring physical contact.
    contact_score = 0.20 + 0.80 * components["contact_bilateral"]
    reward = (
        components["object_present"]
        * contact_score
        * on_ground
        * upright
        * speed_score
        * vertical_stable
        * angular_stable
        * yaw_score
        * behind
        * alignment
        * (1.0 - components["drop"])
    )
    return reward.unsqueeze(-1), {
        **components,
        "speed_score": speed_score,
        "vertical_stable": vertical_stable,
        "angular_stable": angular_stable,
        "yaw_score": yaw_score,
        "on_ground": on_ground,
        "upright": upright,
        "behind": behind,
        "alignment": alignment,
    }


def compute_push_stop_reward(
    aux_rewards: Mapping[str, torch.Tensor],
    state: torch.Tensor,
    *,
    num_dof: int,
    goal_distance_sigma: float = 0.15,
    object_speed_sigma: float = 0.08,
    object_ang_speed_sigma: float = 0.30,
    ground_lift_sigma: float = 0.08,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer an upright grounded box that is stationary at its goal."""

    if min(goal_distance_sigma, object_speed_sigma, object_ang_speed_sigma, ground_lift_sigma) <= 0.0:
        raise ValueError("Push-stop reward sigmas must be positive")
    components = _push_components(aux_rewards, state, num_dof=num_dof)
    on_ground = torch.exp(-torch.square(components["lift"] / float(ground_lift_sigma)))
    near_goal = torch.exp(-torch.square(components["target_distance"] / float(goal_distance_sigma)))
    stationary = torch.exp(
        -torch.square(components["toward_speed"] / float(object_speed_sigma))
        -torch.square(components["lateral_speed"] / float(object_speed_sigma))
        -torch.square(components["vertical_speed"] / float(object_speed_sigma))
        -torch.square(components["angular_speed"] / float(object_ang_speed_sigma))
    )
    reward = (
        components["object_present"]
        * components["box_upright"]
        * components["robot_upright"]
        * on_ground
        * near_goal
        * stationary
        * (1.0 - components["drop"])
    )
    return reward.unsqueeze(-1), {
        **components,
        "on_ground": on_ground,
        "near_goal": near_goal,
        "stationary": stationary,
    }


def _select_latent(
    model: Any,
    observation: Mapping[str, torch.Tensor],
    reward: torch.Tensor,
    mask: torch.Tensor,
    rank_ids: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = _rank_balanced_topk(mask, reward[:, 0], rank_ids, int(topk))
    selected_obs = tree_map(lambda value: value[indices], observation)
    selected_reward = reward[indices]
    latent = model.reward_wr_inference(selected_obs, selected_reward).reshape(1, -1)
    return latent.detach().cpu(), indices


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
        "reward_max": float(reward[indices].max().item()),
    }
    for name in (
        "target_distance",
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
        "root_yaw_rate",
    ):
        result[f"{name}_mean"] = float(components[name][indices].mean().item())
    return result


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    robot = load_robot_training_spec(args.robot_config)
    num_dof = len(robot.robot.control_joint_names)
    wrapped_env = None
    if args.inference_source == "expert":
        env_cfg, use_root_height_obs = load_mjlab_env_cfg(
            args.model_folder,
            data_path=None,
            robot_config=args.robot_config,
            device=args.device,
            headless=True,
            disable_dr=True,
            disable_obs_noise=True,
            max_episode_length_s=100.0,
        )
        env_cfg = env_cfg.model_copy(
            update={
                "lafan_tail_path": [str(path.expanduser().resolve()) for path in args.expert_data_paths],
                "data_mix_weights": [1.0 / len(args.expert_data_paths)] * len(args.expert_data_paths),
            }
        )
        wrapped_env, _ = env_cfg.build(num_envs=1)
        dataset = _expert_inference_dataset(
            wrapped_env._env,
            num_samples=args.num_samples,
            num_dof=num_dof,
            use_root_height_obs=use_root_height_obs,
        )
        data = dataset.data
        data["rank"] = torch.zeros(dataset.size(), dtype=torch.long)
    else:
        data = sample_replay_ranks(args.buffer_paths, sample_count=args.num_samples, seed=args.seed)
    observation = data["next"]["observation"]
    state = torch.as_tensor(observation["state"], dtype=torch.float32)
    aux = data["aux_rewards"]
    rank_ids = torch.as_tensor(data["rank"], dtype=torch.long)

    expected_state_dim = 2 * num_dof + 40
    if state.shape[-1] != expected_state_dim:
        raise ValueError(f"Expected non-ablated Carry state {expected_state_dim}D, got {state.shape[-1]}D")

    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint", device=checkpoint_load_device(args.device), strict=False
    ).to(args.device).eval()

    contact_reward, contact_components = compute_push_contact_reward(aux, state, num_dof=num_dof)
    transport_reward, transport_components = compute_push_transport_reward(
        aux, state, num_dof=num_dof, target_speed=args.target_speed
    )
    stop_reward, stop_components = compute_push_stop_reward(aux, state, num_dof=num_dof)

    common = (
        (transport_components["object_present"] > 0.5)
        & (transport_components["lift"] <= args.max_lift)
        & (transport_components["robot_upright"] >= args.upright_min)
        & (transport_components["box_upright"] >= args.upright_min)
        & (transport_components["drop"] < 0.5)
        & (torch.abs(transport_components["vertical_speed"]) <= args.vertical_speed_max)
        & (transport_components["angular_speed"] <= args.angular_speed_max)
    )
    contact_mask = (
        common
        & (
            (contact_components["contact_bilateral"] > 0.5)
            if args.require_bilateral_contact
            else (contact_components["contact_any"] > 0.5)
        )
        & (torch.abs(contact_components["toward_speed"]) <= args.contact_speed_max)
        & (torch.abs(contact_components["lateral_speed"]) <= args.contact_speed_max)
        & (contact_components["behind_projection"] >= args.behind_projection_min)
        & (contact_components["behind_lateral_error"] <= args.behind_lateral_max)
    )
    transport_mask = (
        common
        & (
            (transport_components["contact_bilateral"] > 0.5)
            if args.require_bilateral_contact
            else (transport_components["contact_any"] > 0.5)
        )
        & (transport_components["target_distance"] >= args.target_distance_min)
        & (transport_components["toward_speed"] > 0.0)
        & (torch.abs(transport_components["lateral_speed"]) <= args.lateral_speed_max)
        & (transport_components["behind_projection"] >= args.behind_projection_min)
        & (transport_components["behind_lateral_error"] <= args.behind_lateral_max)
    )
    stop_mask = (
        common
        & (stop_components["target_distance"] <= args.stop_goal_distance_max)
        & (torch.abs(stop_components["toward_speed"]) <= args.stop_speed_max)
        & (torch.abs(stop_components["lateral_speed"]) <= args.stop_speed_max)
    )

    z_contact, contact_indices = _select_latent(
        model, observation, contact_reward, contact_mask, rank_ids, topk=args.contact_topk
    )
    z_transport, transport_indices = _select_latent(
        model, observation, transport_reward, transport_mask, rank_ids, topk=args.transport_topk
    )
    z_stop, stop_indices = _select_latent(
        model, observation, stop_reward, stop_mask, rank_ids, topk=args.stop_topk
    )

    z_turn: dict[int, np.ndarray] = {}
    turn_stats: dict[str, Any] = {}
    for direction in (-1, 1):
        reward, components = compute_push_transport_reward(
            aux,
            state,
            num_dof=num_dof,
            target_speed=args.target_speed,
            target_yaw_rate=direction * args.turn_yaw_rate,
        )
        turn_mask = transport_mask & (components["root_yaw_rate"] * direction > 0.0)
        latent, indices = _select_latent(
            model, observation, reward, turn_mask, rank_ids, topk=args.turn_topk
        )
        z_turn[direction] = latent.numpy()
        turn_stats["left" if direction > 0 else "right"] = _selection_stats(
            indices, reward, components, rank_ids
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("z_approach.pkl", "z_turn.pkl"):
        source = args.baseline_latent_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing baseline locomotion latent: {source}")
        shutil.copy2(source, args.output_dir / name)
    joblib.dump(z_contact.numpy(), args.output_dir / "z_push_contact.pkl")
    joblib.dump(z_transport.numpy(), args.output_dir / "z_push_transport.pkl")
    joblib.dump(z_turn, args.output_dir / "z_push_turn.pkl")
    joblib.dump(z_stop.numpy(), args.output_dir / "z_push_stop.pkl")

    stats = {
        "model_folder": str(args.model_folder),
        "inference_source": args.inference_source,
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "expert_data_paths": [str(path) for path in args.expert_data_paths],
        "sample_count": int(state.shape[0]),
        "target_speed": float(args.target_speed),
        "candidate_counts": {
            "contact": int(contact_mask.sum().item()),
            "transport": int(transport_mask.sum().item()),
            "stop": int(stop_mask.sum().item()),
        },
        "contact": _selection_stats(contact_indices, contact_reward, contact_components, rank_ids),
        "transport": _selection_stats(
            transport_indices, transport_reward, transport_components, rank_ids
        ),
        "turn": turn_stats,
        "stop": _selection_stats(stop_indices, stop_reward, stop_components, rank_ids),
    }
    (args.output_dir / "push_inference.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[push-inference] saved latent bank to {args.output_dir}", flush=True)
    print(json.dumps(stats["candidate_counts"], sort_keys=True), flush=True)
    if wrapped_env is not None:
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--baseline-latent-dir", type=Path, required=True)
    parser.add_argument("--inference-source", choices=("replay", "expert"), default="replay")
    parser.add_argument("--buffer-paths", type=Path, nargs="*", default=[])
    parser.add_argument("--expert-data-paths", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-speed", type=float, default=0.2)
    parser.add_argument("--turn-yaw-rate", type=float, default=0.3)
    parser.add_argument("--contact-topk", type=int, default=200)
    parser.add_argument("--transport-topk", type=int, default=100)
    parser.add_argument("--turn-topk", type=int, default=50)
    parser.add_argument("--stop-topk", type=int, default=200)
    parser.add_argument("--max-lift", type=float, default=0.08)
    parser.add_argument("--upright-min", type=float, default=0.85)
    parser.add_argument("--vertical-speed-max", type=float, default=0.15)
    parser.add_argument("--angular-speed-max", type=float, default=0.75)
    parser.add_argument("--lateral-speed-max", type=float, default=0.20)
    parser.add_argument("--contact-speed-max", type=float, default=0.15)
    parser.add_argument("--target-distance-min", type=float, default=0.50)
    parser.add_argument("--behind-projection-min", type=float, default=0.10)
    parser.add_argument("--behind-lateral-max", type=float, default=0.50)
    parser.add_argument("--stop-goal-distance-max", type=float, default=0.50)
    parser.add_argument("--stop-speed-max", type=float, default=0.20)
    parser.add_argument(
        "--require-bilateral-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require both physical hand contacts for push-transport candidates.",
    )
    args = parser.parse_args()
    positive = (
        args.num_samples,
        args.target_speed,
        args.turn_yaw_rate,
        args.contact_topk,
        args.transport_topk,
        args.turn_topk,
        args.stop_topk,
    )
    if min(positive) <= 0:
        parser.error("sample counts, target speeds, yaw rate, and top-k values must be positive")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("--device must be cpu or cuda[:index]")
    if args.inference_source == "replay" and not args.buffer_paths:
        parser.error("--buffer-paths is required with --inference-source replay")
    if args.inference_source == "expert" and not args.expert_data_paths:
        parser.error("--expert-data-paths is required with --inference-source expert")
    return args


if __name__ == "__main__":
    run(parse_args())
