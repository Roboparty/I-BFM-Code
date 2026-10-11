from __future__ import annotations

from argparse import Namespace

import torch

from humanoidverse.carry_phased_sim2sim import DirectPushController
from humanoidverse.minimal_push_reward_inference import compute_minimal_push_reward


def _state(count: int) -> torch.Tensor:
    state = torch.zeros(count, 98)
    state[:, 79] = 1.0  # box-to-goal direction in the root frame
    state[:, 96:98] = 1.0
    return state


def _aux(count: int) -> dict[str, torch.Tensor]:
    return {
        "carry_lift": torch.zeros(count, 1),
        "carry_robot_upright": torch.ones(count, 1),
        "carry_upright": torch.ones(count, 1),
        "carry_drop": torch.zeros(count, 1),
    }


def test_minimal_reward_prefers_low_box_at_target_speed() -> None:
    state = _state(3)
    aux = _aux(3)
    state[:, 73] = torch.tensor([0.25, 0.0, 0.25])
    aux["carry_lift"][2] = 0.5

    reward, components = compute_minimal_push_reward(aux, state, num_dof=29)

    assert reward[0].item() > 0.99
    assert reward[1].item() < 1e-3
    assert reward[2].item() < 1e-10
    assert abs(components["box_center_height"][0].item() - 0.186028293) < 1e-6


def test_minimal_reward_does_not_depend_on_contact_or_upright() -> None:
    state = _state(2)
    aux = _aux(2)
    state[:, 73] = 0.25
    state[1, 82:84] = 0.0
    aux["carry_robot_upright"][1] = 0.0
    aux["carry_upright"][1] = 0.0
    aux["carry_drop"][1] = 1.0

    reward, _components = compute_minimal_push_reward(aux, state, num_dof=29)

    assert torch.allclose(reward[0], reward[1])


def test_direct_controller_never_changes_phase_or_masks_observation() -> None:
    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
    )
    controller = DirectPushController(args, dt=0.02)
    observation = {"state": torch.ones(1, 98)}
    direct = torch.randn(1, 256)

    policy_observation, latent, mode = controller.latent(
        None,
        observation,
        {},
        base_rotation=torch.eye(3).numpy(),
        box_direction=torch.ones(2).numpy(),
        goal_direction=torch.ones(2).numpy(),
        z={"push_direct": direct},
    )
    for _ in range(100):
        controller.update({}, goal_height=0.186)

    assert policy_observation is observation
    assert latent is direct
    assert mode == "push_direct"
    assert controller.phase == "push_direct"
    assert controller.phase_steps == 100
