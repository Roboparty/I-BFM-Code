from __future__ import annotations

from argparse import Namespace

import torch

from humanoidverse.push_reward_inference import (
    compute_push_contact_reward,
    compute_push_stop_reward,
    compute_push_transport_reward,
)
from humanoidverse.carry_phased_sim2sim import PushPhaseController, _push_turn_weight


def _aux(count: int) -> dict[str, torch.Tensor]:
    return {
        "carry_bilateral_contact": torch.ones(count, 1),
        "carry_lift": torch.zeros(count, 1),
        "carry_robot_upright": torch.ones(count, 1),
        "carry_upright": torch.ones(count, 1),
        "carry_drop": torch.zeros(count, 1),
    }


def _state(count: int) -> torch.Tensor:
    state = torch.zeros(count, 98)
    state[:, 60] = -1.0
    state[:, 64] = 0.55  # box is in front of root
    state[:, 79] = 1.0   # target is in front of box
    state[:, 82:84] = 1.0
    state[:, 96:98] = 1.0
    return state


def _controller() -> PushPhaseController:
    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        push_standoff_distance=0.55,
        push_staging_tolerance=0.20,
        push_behind_min=0.30,
        push_lateral_max=0.30,
        push_reapproach_distance=0.90,
        push_phase_hold_steps=5,
        push_contact_steps=5,
        push_require_bilateral_contact=True,
        push_lost_contact_steps=10,
        push_max_lift=0.15,
        push_upright_min=0.80,
        push_stop_enter_distance=0.25,
        push_stop_enter_steps=5,
        push_stop_exit_distance=0.35,
        push_stop_exit_steps=10,
        push_success_distance=0.20,
        push_success_speed=0.08,
        push_success_steps=25,
    )
    return PushPhaseController(args, dt=0.02)


def _metrics(**overrides: float | bool) -> dict[str, float | bool]:
    values: dict[str, float | bool] = {
        "horizontal_distance": 0.55,
        "goal_distance": 1.0,
        "object_height": 0.185,
        "object_speed": 0.0,
        "lift": 0.0,
        "root_height": 0.8,
        "pelvis_up_z": 1.0,
        "robot_upright": 1.0,
        "box_upright": 1.0,
        "any_contact": False,
        "bilateral_contact": False,
        "push_staging_distance": 0.1,
        "push_behind_projection": 0.55,
        "push_lateral_error": 0.0,
    }
    values.update(overrides)
    return values


def test_push_transport_reward_prefers_grounded_goalward_contact() -> None:
    state = _state(4)
    aux = _aux(4)
    state[:, 73] = torch.tensor([0.2, -0.2, 0.2, 0.2])
    state[2, 82:84] = 0.0
    aux["carry_lift"][3] = 1.0

    reward, components = compute_push_transport_reward(aux, state, num_dof=29, target_speed=0.2)

    assert reward[0].item() > 0.8
    assert reward[1].item() < 1e-5
    assert reward[2].item() < reward[0].item()
    assert reward[3].item() < 1e-10
    assert components["toward_speed"][0].item() == torch.tensor(0.2).item()


def test_push_contact_reward_prefers_stationary_contact_behind_box() -> None:
    state = _state(3)
    aux = _aux(3)
    state[1, 64] = -0.55
    state[2, 73] = 0.5

    reward, _components = compute_push_contact_reward(aux, state, num_dof=29)

    assert reward[0].item() > 0.9
    assert reward[1].item() < 0.05
    assert reward[2].item() < 1e-6


def test_push_stop_reward_prefers_stationary_grounded_goal() -> None:
    state = _state(3)
    aux = _aux(3)
    state[:, 79] = torch.tensor([0.05, 0.5, 0.05])
    state[2, 73] = 0.4

    reward, _components = compute_push_stop_reward(aux, state, num_dof=29)

    assert reward[0].item() > reward[1].item()
    assert reward[0].item() > reward[2].item()


def test_push_controller_requires_staging_then_contact() -> None:
    controller = _controller()
    for _ in range(5):
        controller.update(_metrics(), goal_height=0.185)
    assert controller.phase == "push_contact"

    for _ in range(4):
        controller.update(_metrics(any_contact=True, bilateral_contact=True), goal_height=0.185)
    assert controller.phase == "push_contact"
    controller.update(_metrics(any_contact=True, bilateral_contact=True), goal_height=0.185)
    assert controller.phase == "push_transport"


def test_push_turn_boost_stays_zero_when_heading_is_aligned() -> None:
    aligned = _push_turn_weight(
        goal_distance=0.25,
        bearing_deg=0.0,
        turn_boost_distance=0.8,
        stop_distance=0.25,
        heading_blend_full_deg=45.0,
        far_max_turn_blend=0.45,
        near_max_turn_blend=0.75,
    )
    turning = _push_turn_weight(
        goal_distance=0.25,
        bearing_deg=45.0,
        turn_boost_distance=0.8,
        stop_distance=0.25,
        heading_blend_full_deg=45.0,
        far_max_turn_blend=0.45,
        near_max_turn_blend=0.75,
    )

    assert aligned == 0.0
    assert turning == 0.75


def test_push_controller_stop_hysteresis_and_completion() -> None:
    controller = _controller()
    controller.phase = "push_transport"
    inside = _metrics(any_contact=True, bilateral_contact=True, goal_distance=0.24)
    for _ in range(5):
        controller.update(inside, goal_height=0.185)
    assert controller.phase == "push_stop"

    for _ in range(9):
        controller.update(_metrics(goal_distance=0.36), goal_height=0.185)
    controller.update(_metrics(goal_distance=0.34), goal_height=0.185)
    assert controller.phase == "push_stop"

    success = _metrics(goal_distance=0.10, object_speed=0.01)
    for _ in range(25):
        controller.update(success, goal_height=0.185)
    assert controller.phase == "complete"
