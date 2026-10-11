"""Infer a continuous Carry-to-place latent from replay transitions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
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


def compute_direct_place_reward(
    state: torch.Tensor,
    *,
    num_dof: int,
    distance_sigma: float = 0.5,
    current_state: torch.Tensor | None = None,
    progress_sigma: float = 0.01,
    progress_weight: float = 0.75,
    progress_fade_distance: float | None = None,
    aux_rewards: Mapping[str, torch.Tensor] | None = None,
    terminal_gate_distance: float | None = None,
    terminal_height_sigma: float = 0.05,
    terminal_linear_speed_sigma: float = 0.10,
    terminal_angular_speed_sigma: float = 0.25,
    terminal_quality_scale: float = 0.5,
    lift_height: float = 0.35,
    slowdown_start_distance: float | None = None,
    lower_start_distance: float = 0.30,
    release_distance: float = 0.20,
    far_target_speed: float = 0.50,
    forward_speed_sigma: float = 0.15,
    lateral_speed_sigma: float = 0.10,
    lowering_vertical_speed: float = -0.08,
    lowering_vertical_speed_sigma: float = 0.08,
    lowering_horizontal_speed_sigma: float = 0.12,
    unsafe_release_scale: float = 0.50,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Blend goal progress with stable released placement in one continuous reward."""

    if float(distance_sigma) <= 0.0 or float(progress_sigma) <= 0.0:
        raise ValueError("distance_sigma and progress_sigma must be positive")
    if not 0.0 <= float(progress_weight) <= 1.0:
        raise ValueError("progress_weight must be in [0, 1]")
    if progress_fade_distance is not None and float(progress_fade_distance) <= 0.0:
        raise ValueError("progress_fade_distance must be positive when set")
    profile_scales = (
        far_target_speed,
        forward_speed_sigma,
        lateral_speed_sigma,
        lowering_vertical_speed_sigma,
        lowering_horizontal_speed_sigma,
    )
    if slowdown_start_distance is not None and (
        not float(slowdown_start_distance) > float(lower_start_distance) > float(release_distance) > 0.0
        or any(float(value) <= 0.0 for value in profile_scales)
        or float(lowering_vertical_speed) >= 0.0
        or float(unsafe_release_scale) < 0.0
    ):
        raise ValueError(
            "placement profile requires slowdown > lower > release > 0, positive scales, "
            "a negative lowering speed, and a non-negative unsafe-release scale"
        )
    terminal_scales = (
        terminal_height_sigma,
        terminal_linear_speed_sigma,
        terminal_angular_speed_sigma,
        lift_height,
    )
    if terminal_gate_distance is not None and (
        float(terminal_gate_distance) <= 0.0
        or float(terminal_quality_scale) <= 0.0
        or any(float(value) <= 0.0 for value in terminal_scales)
    ):
        raise ValueError("terminal gate and terminal reward scales must be positive")
    state = torch.as_tensor(state, dtype=torch.float32)
    if state.ndim != 2:
        raise ValueError(f"Expected state [N, D], got {tuple(state.shape)}")
    object_start = 2 * int(num_dof) + 6
    context_dim = _state_object_context_dim(state, num_dof=num_dof)
    target_delta = state[:, object_start + 15 : object_start + 18]
    target_distance = torch.linalg.vector_norm(target_delta, dim=-1)
    target_xy = target_delta[:, :2]
    horizontal_distance = torch.linalg.vector_norm(target_xy, dim=-1)
    object_present = state[:, object_start + context_dim - 2]
    proximity = torch.exp(-target_distance / float(distance_sigma))
    if current_state is None:
        progress = torch.zeros_like(target_distance)
    else:
        current_state = torch.as_tensor(current_state, dtype=torch.float32)
        if current_state.shape != state.shape:
            raise ValueError(
                f"current_state must match next state shape {tuple(state.shape)}, "
                f"got {tuple(current_state.shape)}"
            )
        current_target_delta = current_state[:, object_start + 15 : object_start + 18]
        current_distance = torch.linalg.vector_norm(current_target_delta, dim=-1)
        progress = current_distance - target_distance
    if current_state is None:
        horizontal_progress = torch.zeros_like(horizontal_distance)
    else:
        current_horizontal_distance = torch.linalg.vector_norm(
            current_state[:, object_start + 15 : object_start + 17], dim=-1
        )
        horizontal_progress = current_horizontal_distance - horizontal_distance
    progress_score = torch.tanh(progress / float(progress_sigma))
    if progress_fade_distance is None:
        progress_gate = torch.ones_like(target_distance)
    else:
        progress_gate = 1.0 - torch.exp(
            -target_distance / float(progress_fade_distance)
        )
    effective_progress_weight = float(progress_weight) * progress_gate
    transport_reward = (
        effective_progress_weight * progress_score
        + (1.0 - effective_progress_weight) * proximity
    )
    terminal_gate = torch.zeros_like(target_distance)
    terminal_quality = torch.zeros_like(target_distance)
    grounded = torch.zeros_like(target_distance)
    stationary = torch.zeros_like(target_distance)
    released = torch.zeros_like(target_distance)
    upright = torch.zeros_like(target_distance)
    if (terminal_gate_distance is not None or slowdown_start_distance is not None) and aux_rewards is None:
        raise ValueError("aux_rewards are required when terminal or placement-profile reward is enabled")

    def aux(name: str) -> torch.Tensor:
        if aux_rewards is None or name not in aux_rewards:
            raise KeyError(f"Replay buffer is missing Carry auxiliary reward {name!r}")
        return torch.as_tensor(aux_rewards[name], dtype=torch.float32).reshape(-1)

    if terminal_gate_distance is not None:
        object_lin_vel = state[:, object_start + 9 : object_start + 12]
        object_ang_vel = state[:, object_start + 12 : object_start + 15]
        hand_contact = state[:, object_start + 18 : object_start + 20].clamp(0.0, 1.0)
        normalized_height_sigma = float(terminal_height_sigma) / float(lift_height)
        grounded = torch.exp(
            -torch.square(aux("carry_lift") / normalized_height_sigma)
        )
        stationary = torch.exp(
            -torch.square(
                torch.linalg.vector_norm(object_lin_vel, dim=-1)
                / float(terminal_linear_speed_sigma)
            )
            -torch.square(
                torch.linalg.vector_norm(object_ang_vel, dim=-1)
                / float(terminal_angular_speed_sigma)
            )
        )
        released = (1.0 - hand_contact[:, 0]) * (1.0 - hand_contact[:, 1])
        upright = aux("carry_robot_upright") * aux("carry_upright")
        terminal_quality = (
            grounded
            * stationary
            * upright
            * (0.10 + 0.90 * released)
        )
        terminal_gate = torch.exp(
            -torch.square(target_distance / float(terminal_gate_distance))
        )
    reward = (
        (1.0 - terminal_gate) * transport_reward
        + terminal_gate * float(terminal_quality_scale) * terminal_quality
    )
    desired_goalward_speed = torch.zeros_like(horizontal_distance)
    goalward_speed = torch.zeros_like(horizontal_distance)
    lateral_speed = torch.zeros_like(horizontal_distance)
    navigation_score = torch.zeros_like(horizontal_distance)
    lower_gate = torch.zeros_like(horizontal_distance)
    lowering_quality = torch.zeros_like(horizontal_distance)
    unsafe_release = torch.zeros_like(horizontal_distance)
    if slowdown_start_distance is not None:
        object_lin_vel = state[:, object_start + 9 : object_start + 12]
        object_ang_vel = state[:, object_start + 12 : object_start + 15]
        hand_contact = state[:, object_start + 18 : object_start + 20].clamp(0.0, 1.0)
        bilateral_contact = hand_contact[:, 0] * hand_contact[:, 1]
        released = (1.0 - hand_contact[:, 0]) * (1.0 - hand_contact[:, 1])
        direction = target_xy / torch.clamp(horizontal_distance.unsqueeze(-1), min=1e-6)
        lateral_direction = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
        goalward_speed = torch.sum(object_lin_vel[:, :2] * direction, dim=-1)
        lateral_speed = torch.sum(object_lin_vel[:, :2] * lateral_direction, dim=-1)

        speed_fraction = torch.clamp(
            (horizontal_distance - float(release_distance))
            / (float(slowdown_start_distance) - float(release_distance)),
            min=0.0,
            max=1.0,
        )
        speed_fraction = speed_fraction * speed_fraction * (3.0 - 2.0 * speed_fraction)
        desired_goalward_speed = float(far_target_speed) * speed_fraction
        velocity_tracking = torch.exp(
            -torch.square(
                (goalward_speed - desired_goalward_speed) / float(forward_speed_sigma)
            )
            -torch.square(lateral_speed / float(lateral_speed_sigma))
        )
        progress_quality = 0.5 * (
            1.0 + torch.tanh(horizontal_progress / float(progress_sigma))
        )
        upright = aux("carry_robot_upright") * aux("carry_upright")
        lift = torch.clamp(aux("carry_lift"), min=0.0, max=1.0)
        carry_quality = bilateral_contact * upright * lift
        navigation_score = carry_quality * (
            0.75 * velocity_tracking + 0.25 * progress_quality
        )

        lower_exponent = torch.clamp(
            (horizontal_distance - float(lower_start_distance))
            / max(0.25 * (float(slowdown_start_distance) - float(lower_start_distance)), 1e-6),
            min=-60.0,
            max=60.0,
        )
        lower_gate = 1.0 / (1.0 + torch.exp(lower_exponent))
        horizontal_speed = torch.linalg.vector_norm(object_lin_vel[:, :2], dim=-1)
        angular_speed = torch.linalg.vector_norm(object_ang_vel, dim=-1)
        lowering_motion = torch.exp(
            -torch.square(
                (object_lin_vel[:, 2] - float(lowering_vertical_speed))
                / float(lowering_vertical_speed_sigma)
            )
            -torch.square(horizontal_speed / float(lowering_horizontal_speed_sigma))
            -torch.square(angular_speed / float(terminal_angular_speed_sigma))
        )
        grounded = torch.exp(
            -torch.square(aux("carry_lift") / (float(terminal_height_sigma) / float(lift_height)))
        )
        airborne = 1.0 - grounded
        goal_accuracy = torch.exp(
            -torch.square(horizontal_distance / float(release_distance))
        )
        lowering_quality = bilateral_contact * upright * airborne * lowering_motion
        stationary = torch.exp(
            -torch.square(
                torch.linalg.vector_norm(object_lin_vel, dim=-1)
                / float(terminal_linear_speed_sigma)
            )
            -torch.square(angular_speed / float(terminal_angular_speed_sigma))
        )
        terminal_quality = grounded * stationary * upright * released * goal_accuracy
        placement_quality = airborne * lowering_quality + grounded * terminal_quality
        unsafe_release = released * airborne
        reward = (
            (1.0 - lower_gate) * navigation_score
            + lower_gate * placement_quality
            - float(unsafe_release_scale) * lower_gate * unsafe_release
        )
    return reward.unsqueeze(-1), {
        "target_distance": target_distance,
        "horizontal_distance": horizontal_distance,
        "object_present": object_present,
        "progress": progress,
        "horizontal_progress": horizontal_progress,
        "progress_score": progress_score,
        "proximity": proximity,
        "progress_gate": progress_gate,
        "effective_progress_weight": effective_progress_weight,
        "transport_reward": transport_reward,
        "terminal_gate": terminal_gate,
        "terminal_quality": terminal_quality,
        "grounded": grounded,
        "stationary": stationary,
        "released": released,
        "upright": upright,
        "desired_goalward_speed": desired_goalward_speed,
        "goalward_speed": goalward_speed,
        "lateral_speed": lateral_speed,
        "navigation_score": navigation_score,
        "lower_gate": lower_gate,
        "lowering_quality": lowering_quality,
        "unsafe_release": unsafe_release,
    }


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    result = torch.quantile(value.float(), torch.tensor(levels, dtype=torch.float32))
    return {f"q{int(level * 100):02d}": float(item) for level, item in zip(levels, result)}


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
    observation = replay["next"]["observation"]
    state = observation["state"]
    expected_state_dim = 2 * num_dof + 40
    if int(state.shape[-1]) != expected_state_dim:
        raise ValueError(
            f"Expected the 98D non-ablated G1 Carry state, got {int(state.shape[-1])}D"
        )

    reward, components = compute_direct_place_reward(
        state,
        num_dof=num_dof,
        distance_sigma=float(args.distance_sigma),
        current_state=replay["current_state"],
        progress_sigma=float(args.progress_sigma),
        progress_weight=float(args.progress_weight),
        progress_fade_distance=args.progress_fade_distance,
        aux_rewards=replay["aux_rewards"],
        terminal_gate_distance=args.terminal_gate_distance,
        terminal_height_sigma=float(args.terminal_height_sigma),
        terminal_linear_speed_sigma=float(args.terminal_linear_speed_sigma),
        terminal_angular_speed_sigma=float(args.terminal_angular_speed_sigma),
        terminal_quality_scale=float(args.terminal_quality_scale),
        lift_height=float(args.lift_height),
        slowdown_start_distance=args.slowdown_start_distance,
        lower_start_distance=float(args.lower_start_distance),
        release_distance=float(args.release_distance),
        far_target_speed=float(args.far_target_speed),
        forward_speed_sigma=float(args.forward_speed_sigma),
        lateral_speed_sigma=float(args.lateral_speed_sigma),
        lowering_vertical_speed=float(args.lowering_vertical_speed),
        lowering_vertical_speed_sigma=float(args.lowering_vertical_speed_sigma),
        lowering_horizontal_speed_sigma=float(args.lowering_horizontal_speed_sigma),
        unsafe_release_scale=float(args.unsafe_release_scale),
    )
    carry_mask = components["object_present"] > 0.5
    if args.candidate_max_distance is not None:
        carry_mask &= components["horizontal_distance"] <= float(args.candidate_max_distance)
    carry_count = int(carry_mask.sum().item())
    if carry_count < int(args.min_carry_samples):
        raise ValueError(
            f"Only {carry_count} object-present replay states were sampled; "
            f"need at least {args.min_carry_samples}"
        )

    carry_observation = tree_map(lambda value: value[carry_mask], observation)
    carry_reward = reward[carry_mask]
    load_device = checkpoint_load_device(args.device)
    model = load_model_from_checkpoint_dir(
        args.model_folder / "checkpoint",
        device=load_device,
    ).to(args.device).eval()
    latent = model.reward_wr_inference(carry_observation, carry_reward).reshape(1, -1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = args.output_dir / "z_direct_place.pkl"
    joblib.dump(latent.detach().cpu().numpy(), latent_path)
    distance = components["horizontal_distance"][carry_mask]
    rank_ids = replay["rank"]
    component_stats: dict[str, object] = {}
    if args.slowdown_start_distance is not None:
        component_specs = {
            "slow": (
                components["navigation_score"],
                components["horizontal_distance"] >= float(args.lower_start_distance),
            ),
            "lower": (
                components["lowering_quality"],
                components["horizontal_distance"]
                <= float(args.lower_start_distance) + float(args.component_distance_margin),
            ),
            "release": (
                components["terminal_quality"],
                components["horizontal_distance"]
                <= float(args.release_distance) + float(args.component_distance_margin),
            ),
        }
        for name, (score, stage_mask) in component_specs.items():
            valid = carry_mask & stage_mask & (score > 0.0)
            indices = _rank_balanced_topk(
                valid,
                score,
                rank_ids,
                int(args.component_topk),
            )
            component_observation = tree_map(lambda value: value[indices], observation)
            component_reward = score[indices].reshape(-1, 1)
            component_latent = model.reward_wr_inference(
                component_observation, component_reward
            ).reshape(1, -1)
            component_path = args.output_dir / f"z_direct_place_{name}.pkl"
            joblib.dump(component_latent.detach().cpu().numpy(), component_path)
            component_stats[name] = {
                "candidate_count": int(valid.sum().item()),
                "selected_count": int(indices.numel()),
                "selected_per_rank": _rank_counts(
                    torch.isin(torch.arange(rank_ids.numel()), indices), rank_ids
                ),
                "selected_reward": _quantiles(component_reward[:, 0]),
                "latent_path": str(component_path),
                "latent_norm": float(torch.linalg.vector_norm(component_latent).item()),
            }
    metadata: dict[str, object] = {
        "model_folder": str(args.model_folder),
        "buffer_paths": [str(path) for path in args.buffer_paths],
        "sample_count": int(rank_ids.numel()),
        "carry_sample_count": carry_count,
        "carry_samples_per_rank": _rank_counts(carry_mask, rank_ids),
        "reward": (
            "continuous placement profile: distance-scheduled goalward speed + lateral "
            "velocity suppression, smoothly gated bilateral-contact lowering, then "
            "grounded stationary upright both-hands-released goal accuracy"
            if args.slowdown_start_distance is not None
            else (
                "(1-terminal_gate)*transport_reward + "
                "terminal_gate*terminal_quality_scale*terminal_quality"
            )
        ),
        "distance_sigma_m": float(args.distance_sigma),
        "progress_sigma_m_per_step": float(args.progress_sigma),
        "progress_weight": float(args.progress_weight),
        "progress_fade_distance_m": args.progress_fade_distance,
        "terminal_gate_distance_m": args.terminal_gate_distance,
        "terminal_height_sigma_m": float(args.terminal_height_sigma),
        "terminal_linear_speed_sigma_mps": float(args.terminal_linear_speed_sigma),
        "terminal_angular_speed_sigma_radps": float(args.terminal_angular_speed_sigma),
        "terminal_quality_scale": float(args.terminal_quality_scale),
        "lift_height_m": float(args.lift_height),
        "slowdown_start_distance_m": args.slowdown_start_distance,
        "lower_start_distance_m": float(args.lower_start_distance),
        "release_distance_m": float(args.release_distance),
        "far_target_speed_mps": float(args.far_target_speed),
        "forward_speed_sigma_mps": float(args.forward_speed_sigma),
        "lateral_speed_sigma_mps": float(args.lateral_speed_sigma),
        "lowering_vertical_speed_mps": float(args.lowering_vertical_speed),
        "lowering_vertical_speed_sigma_mps": float(args.lowering_vertical_speed_sigma),
        "lowering_horizontal_speed_sigma_mps": float(args.lowering_horizontal_speed_sigma),
        "unsafe_release_scale": float(args.unsafe_release_scale),
        "candidate_max_horizontal_distance_m": args.candidate_max_distance,
        "candidate_filter": (
            "object_present > 0.5 and horizontal_distance <= candidate_max_distance"
            if args.candidate_max_distance is not None
            else "object_present > 0.5"
        ),
        "target_horizontal_distance_m": _quantiles(distance),
        "reward_quantiles": _quantiles(carry_reward[:, 0]),
        "distance_progress_m_per_step": _quantiles(components["progress"][carry_mask]),
        "effective_progress_weight": _quantiles(
            components["effective_progress_weight"][carry_mask]
        ),
        "terminal_gate": _quantiles(components["terminal_gate"][carry_mask]),
        "terminal_quality": _quantiles(components["terminal_quality"][carry_mask]),
        "desired_goalward_speed": _quantiles(
            components["desired_goalward_speed"][carry_mask]
        ),
        "goalward_speed": _quantiles(components["goalward_speed"][carry_mask]),
        "lateral_speed": _quantiles(components["lateral_speed"][carry_mask]),
        "navigation_score": _quantiles(components["navigation_score"][carry_mask]),
        "lower_gate": _quantiles(components["lower_gate"][carry_mask]),
        "lowering_quality": _quantiles(components["lowering_quality"][carry_mask]),
        "unsafe_release": _quantiles(components["unsafe_release"][carry_mask]),
        "component_topk": int(args.component_topk),
        "component_distance_margin_m": float(args.component_distance_margin),
        "component_latents": component_stats,
        "latent_path": str(latent_path),
        "latent_norm": float(torch.linalg.vector_norm(latent).item()),
    }
    metadata_path = args.output_dir / "direct_place_inference.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] saved direct-place latent to {latent_path}", flush=True)
    print(f"[INFO] saved inference metadata to {metadata_path}", flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--buffer-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=800_000)
    parser.add_argument("--min-carry-samples", type=int, default=10_000)
    parser.add_argument("--distance-sigma", type=float, default=0.5)
    parser.add_argument("--progress-sigma", type=float, default=0.01)
    parser.add_argument("--progress-weight", type=float, default=0.75)
    parser.add_argument(
        "--progress-fade-distance",
        type=float,
        default=None,
        help=(
            "Optional distance scale that smoothly removes the progress term near the goal; "
            "this changes reward inference only and does not create a runtime phase."
        ),
    )
    parser.add_argument(
        "--terminal-gate-distance",
        type=float,
        default=0.30,
        help=(
            "3D goal-distance scale for smoothly blending the terminal placement quality "
            "into the same reward; set 0 to disable."
        ),
    )
    parser.add_argument("--terminal-height-sigma", type=float, default=0.05)
    parser.add_argument("--terminal-linear-speed-sigma", type=float, default=0.10)
    parser.add_argument("--terminal-angular-speed-sigma", type=float, default=0.25)
    parser.add_argument("--terminal-quality-scale", type=float, default=0.5)
    parser.add_argument("--lift-height", type=float, default=0.35)
    parser.add_argument(
        "--slowdown-start-distance",
        type=float,
        default=None,
        help="Enable the continuous placement profile and begin slowing at this XY distance.",
    )
    parser.add_argument("--lower-start-distance", type=float, default=0.30)
    parser.add_argument("--release-distance", type=float, default=0.20)
    parser.add_argument("--far-target-speed", type=float, default=0.50)
    parser.add_argument("--forward-speed-sigma", type=float, default=0.15)
    parser.add_argument("--lateral-speed-sigma", type=float, default=0.10)
    parser.add_argument("--lowering-vertical-speed", type=float, default=-0.08)
    parser.add_argument("--lowering-vertical-speed-sigma", type=float, default=0.08)
    parser.add_argument("--lowering-horizontal-speed-sigma", type=float, default=0.12)
    parser.add_argument("--unsafe-release-scale", type=float, default=0.50)
    parser.add_argument(
        "--candidate-max-distance",
        type=float,
        default=None,
        help="Optional maximum replay XY goal distance used to infer this near-goal latent.",
    )
    parser.add_argument("--component-topk", type=int, default=1000)
    parser.add_argument("--component-distance-margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.min_carry_samples <= 0:
        parser.error("--num-samples and --min-carry-samples must be positive")
    if args.distance_sigma <= 0.0 or args.progress_sigma <= 0.0:
        parser.error("--distance-sigma and --progress-sigma must be positive")
    if not 0.0 <= args.progress_weight <= 1.0:
        parser.error("--progress-weight must be in [0, 1]")
    if args.progress_fade_distance is not None and args.progress_fade_distance <= 0.0:
        parser.error("--progress-fade-distance must be positive")
    if args.terminal_gate_distance == 0.0:
        args.terminal_gate_distance = None
    elif args.terminal_gate_distance < 0.0:
        parser.error("--terminal-gate-distance must be non-negative")
    if min(
        args.terminal_height_sigma,
        args.terminal_linear_speed_sigma,
        args.terminal_angular_speed_sigma,
        args.terminal_quality_scale,
        args.lift_height,
    ) <= 0.0:
        parser.error("terminal reward scales and --lift-height must be positive")
    if args.slowdown_start_distance is not None and not (
        args.slowdown_start_distance > args.lower_start_distance > args.release_distance > 0.0
    ):
        parser.error("placement distances must satisfy slowdown > lower > release > 0")
    if min(
        args.far_target_speed,
        args.forward_speed_sigma,
        args.lateral_speed_sigma,
        args.lowering_vertical_speed_sigma,
        args.lowering_horizontal_speed_sigma,
    ) <= 0.0:
        parser.error("placement speed targets and sigmas must be positive")
    if args.lowering_vertical_speed >= 0.0:
        parser.error("--lowering-vertical-speed must be negative")
    if args.unsafe_release_scale < 0.0:
        parser.error("--unsafe-release-scale must be non-negative")
    if args.candidate_max_distance is not None and args.candidate_max_distance <= 0.0:
        parser.error("--candidate-max-distance must be positive")
    if args.component_topk <= 0 or args.component_distance_margin < 0.0:
        parser.error("--component-topk must be positive and its distance margin non-negative")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("--device must be cpu or cuda[:index]")
    return args


if __name__ == "__main__":
    run(parse_args())
