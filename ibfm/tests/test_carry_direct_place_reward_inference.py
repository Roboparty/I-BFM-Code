from argparse import Namespace

import numpy as np
import torch

from humanoidverse.carry_direct_place_reward_inference import compute_direct_place_reward
from humanoidverse.carry_phased_sim2sim import (
    CarryDirectPlaceController,
    DirectPlaceController,
    _summarize_speed_rollout,
)


def _state(distances: list[float]) -> torch.Tensor:
    state = torch.zeros(len(distances), 98)
    state[:, 79] = torch.tensor(distances)
    state[:, 96:98] = 1.0
    return state


def _summary_row(**overrides: float | bool | str) -> dict[str, float | bool | str]:
    row: dict[str, float | bool | str] = {
        "phase": "place_direct",
        "horizontal_distance": 0.3,
        "goal_distance": 0.1,
        "object_height": 0.9,
        "object_speed": 0.1,
        "object_vertical_speed": -0.1,
        "root_height": 0.75,
        "lift": 1.0,
        "any_contact": True,
        "bilateral_contact": True,
        "transport_goalward_speed": 0.0,
    }
    row.update(overrides)
    return row


def test_direct_place_summary_requires_crouch_and_safe_airborne_contact() -> None:
    history = [
        _summary_row(),
        _summary_row(object_height=0.30, root_height=0.45),
        _summary_row(object_height=0.23, root_height=0.40, bilateral_contact=False),
        _summary_row(
            phase="complete",
            object_height=0.186,
            object_speed=0.0,
            object_vertical_speed=0.0,
            bilateral_contact=False,
        ),
    ]

    summary = _summarize_speed_rollout(
        history, final_phase="complete", placement_ground_height=0.186
    )

    assert summary["direct_place_controlled"] is True
    assert summary["direct_place_airborne_bilateral_contact_rate"] == 1.0
    assert summary["direct_place_min_root_height"] == 0.40
    assert summary["direct_place_unsafe_high_release"] is False


def test_direct_place_summary_rejects_fast_high_contact_loss() -> None:
    history = [
        _summary_row(),
        _summary_row(
            object_height=0.50,
            object_vertical_speed=-1.0,
            root_height=0.45,
            any_contact=False,
            bilateral_contact=False,
        ),
        _summary_row(
            phase="complete",
            object_height=0.186,
            object_speed=0.0,
            object_vertical_speed=0.0,
            root_height=0.75,
            any_contact=False,
            bilateral_contact=False,
        ),
    ]

    summary = _summarize_speed_rollout(
        history, final_phase="complete", placement_ground_height=0.186
    )

    assert summary["direct_place_controlled"] is False
    assert summary["direct_place_unsafe_high_release"] is True
    assert summary["direct_place_first_high_simultaneous_contact_loss_height"] == 0.50


def test_direct_place_reward_only_depends_monotonically_on_goal_distance() -> None:
    state = _state([0.0, 0.5, 1.0, 1.5])
    reward, components = compute_direct_place_reward(
        state,
        num_dof=29,
        distance_sigma=0.5,
        progress_weight=0.0,
    )

    assert torch.all(reward[:-1] > reward[1:])
    assert torch.allclose(components["target_distance"], torch.tensor([0.0, 0.5, 1.0, 1.5]))
    assert torch.allclose(reward[:, 0], torch.exp(-components["target_distance"] / 0.5))


def test_direct_place_reward_prefers_progress_over_stationary_state() -> None:
    current = _state([1.0, 1.0, 1.0])
    next_state = _state([0.99, 1.0, 1.01])
    reward, components = compute_direct_place_reward(
        next_state,
        current_state=current,
        num_dof=29,
        distance_sigma=0.5,
        progress_sigma=0.01,
        progress_weight=0.75,
    )

    assert reward[0] > reward[1] > reward[2]
    assert torch.allclose(components["progress"], torch.tensor([0.01, 0.0, -0.01]), atol=1e-6)


def test_direct_place_progress_fades_smoothly_near_goal() -> None:
    current = _state([0.01, 1.01])
    next_state = _state([0.0, 1.0])
    _reward, components = compute_direct_place_reward(
        next_state,
        current_state=current,
        num_dof=29,
        progress_weight=0.75,
        progress_fade_distance=0.35,
    )

    assert components["effective_progress_weight"][0].item() == 0.0
    assert components["effective_progress_weight"][1].item() > 0.70


def test_integrated_terminal_reward_prefers_grounded_stable_release() -> None:
    current = _state([0.11, 0.11, 0.11])
    next_state = _state([0.10, 0.10, 0.10])
    next_state[1, 73] = 0.3
    next_state[2, 82:84] = 1.0
    aux = {
        "carry_lift": torch.tensor([[0.0], [0.0], [0.5]]),
        "carry_robot_upright": torch.ones(3, 1),
        "carry_upright": torch.ones(3, 1),
    }

    reward, components = compute_direct_place_reward(
        next_state,
        current_state=current,
        aux_rewards=aux,
        num_dof=29,
        terminal_gate_distance=0.30,
        terminal_quality_scale=0.5,
    )

    assert reward[0] > reward[1] > reward[2]
    assert components["released"][0].item() == 1.0
    assert components["released"][2].item() == 0.0
    assert components["terminal_gate"][0].item() > 0.8


def test_continuous_profile_slows_goalward_speed_and_penalizes_lateral_motion() -> None:
    current = _state([0.71, 0.46, 0.31, 0.21])
    state = _state([0.70, 0.45, 0.30, 0.20])
    state[:, 82:84] = 1.0
    aux = {
        "carry_lift": torch.ones(4, 1),
        "carry_robot_upright": torch.ones(4, 1),
        "carry_upright": torch.ones(4, 1),
    }
    _reward, components = compute_direct_place_reward(
        state,
        current_state=current,
        aux_rewards=aux,
        num_dof=29,
        slowdown_start_distance=0.60,
        lower_start_distance=0.30,
        release_distance=0.20,
    )

    desired = components["desired_goalward_speed"]
    assert desired[0].item() == 0.5
    assert desired[0] > desired[1] > desired[2] > desired[3]

    aligned = _state([0.45, 0.45])
    aligned[:, 82:84] = 1.0
    aligned[:, 73] = components["desired_goalward_speed"][1]
    aligned[1, 73] = 0.0
    aligned[1, 74] = components["desired_goalward_speed"][1]
    reward, _components = compute_direct_place_reward(
        aligned,
        current_state=_state([0.46, 0.46]),
        aux_rewards={key: value[:2] for key, value in aux.items()},
        num_dof=29,
        slowdown_start_distance=0.60,
        lower_start_distance=0.30,
        release_distance=0.20,
    )
    assert reward[0] > reward[1]


def test_continuous_profile_requires_holding_until_grounded() -> None:
    state = _state([0.20, 0.20, 0.20])
    state[0, 82:84] = 1.0
    state[:, 75] = -0.08
    aux = {
        "carry_lift": torch.tensor([[0.5], [0.5], [0.0]]),
        "carry_robot_upright": torch.ones(3, 1),
        "carry_upright": torch.ones(3, 1),
    }
    reward, components = compute_direct_place_reward(
        state,
        current_state=_state([0.21, 0.21, 0.21]),
        aux_rewards=aux,
        num_dof=29,
        slowdown_start_distance=0.60,
        lower_start_distance=0.30,
        release_distance=0.20,
    )

    assert reward[0] > reward[1]
    assert reward[2] > reward[1]
    assert components["unsafe_release"][1].item() > 0.9


def test_direct_place_controller_never_switches_or_blends_latents() -> None:
    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        place_success_distance=0.20,
        direct_place_terminal_blend_distance=0.35,
        direct_place_terminal_blend_center=None,
        direct_place_terminal_blend_width=0.03,
        direct_place_terminal_speed_sigma=None,
        direct_place_terminal_max_blend=1.0,
        direct_place_success_distance=0.30,
        place_success_height=0.05,
        place_success_speed=0.15,
        place_success_upright=0.85,
        place_success_steps=25,
    )
    controller = DirectPlaceController(args, dt=0.02)
    observation = {"state": torch.randn(1, 98)}
    direct = torch.randn(1, 256)

    returned_observation, latent, mode = controller.latent(
        object(),
        observation,
        {"goal_distance": 1.0},
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z={"direct_place": direct},
    )
    placed_metrics = {
        "goal_distance": 0.25,
        "object_height": 0.185,
        "object_speed": 0.0,
        "box_upright": 1.0,
        "any_contact": False,
    }
    for _ in range(25):
        controller.update({**placed_metrics, "any_contact": True}, goal_height=0.185)
    assert controller.phase == "place_direct"
    for _ in range(25):
        controller.update(placed_metrics, goal_height=0.185)

    assert returned_observation is observation
    assert latent is direct
    assert mode == "place_direct"
    assert controller.phase == "complete"
    assert controller.phase_steps == 0
    assert controller.placed_steps == 25


def test_carry_direct_place_controller_switches_after_transport_gate() -> None:
    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        fall_root_height=0.30,
        fall_tilted_root_height=0.55,
        fall_upright_threshold=0.30,
        fall_detect_steps=10,
        phase_hold_steps=2,
        hold_release_lift=0.25,
        place_align_distance=0.60,
        place_enter_steps=2,
    )
    controller = CarryDirectPlaceController(args, dt=0.02)
    controller.phase = "transport"
    metrics = {
        "root_height": 0.80,
        "pelvis_up_z": 1.0,
        "bilateral_contact": True,
        "lift": 1.0,
        "goal_distance": 0.30,
    }

    controller.update(metrics, goal_height=0.185)
    assert controller.phase == "transport"
    controller.update(metrics, goal_height=0.185)
    assert controller.phase == "place_direct"
    assert controller.place_attempt == 1

    direct = torch.randn(1, 4)
    observation = {"state": torch.zeros(1, 98)}
    returned_observation, latent, mode = controller.latent(
        object(),
        observation,
        {"goal_distance": 0.30},
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z={"direct_place": direct},
    )
    assert returned_observation is observation
    assert latent is direct
    assert mode == "place_direct"


def test_direct_place_controller_blends_terminal_latent_continuously() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        place_success_distance=0.20,
        direct_place_terminal_blend_distance=0.35,
        direct_place_terminal_blend_center=None,
        direct_place_terminal_blend_width=0.03,
        direct_place_terminal_speed_sigma=None,
        direct_place_terminal_max_blend=1.0,
    )
    controller = DirectPlaceController(args, dt=0.02)
    transport = torch.zeros(1, 4)
    terminal = torch.ones(1, 4)

    _obs, far_latent, _mode = controller.latent(
        Policy(),
        {"state": torch.zeros(1, 98)},
        {"goal_distance": 1.0},
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z={"direct_place": transport, "direct_place_terminal": terminal},
    )
    _obs, near_latent, _mode = controller.latent(
        Policy(),
        {"state": torch.zeros(1, 98)},
        {"goal_distance": 0.1},
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z={"direct_place": transport, "direct_place_terminal": terminal},
    )

    assert torch.all(near_latent > far_latent)
    assert controller.phase == "place_direct"


def test_direct_place_sigmoid_blend_is_quiet_far_and_strong_near() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        place_success_distance=0.20,
        direct_place_terminal_blend_distance=0.35,
        direct_place_terminal_blend_center=0.25,
        direct_place_terminal_blend_width=0.03,
        direct_place_terminal_speed_sigma=None,
        direct_place_terminal_max_blend=1.0,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {"direct_place": torch.zeros(1, 1), "direct_place_terminal": torch.ones(1, 1)}

    for distance, expected in ((0.35, 0.034), (0.25, 0.5), (0.20, 0.841)):
        _obs, latent, _mode = controller.latent(
            Policy(),
            {"state": torch.zeros(1, 98)},
            {"goal_distance": distance},
            base_rotation=np.eye(3),
            box_direction=np.ones(2),
            goal_direction=np.ones(2),
            z=z,
        )
        assert abs(latent.item() - expected) < 0.01


def test_direct_place_smoothstep_blend_decelerates_from_point_six_to_point_three() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_terminal_blend_center=None,
        direct_place_terminal_blend_distance=0.35,
        direct_place_terminal_blend_width=0.03,
        direct_place_terminal_speed_sigma=None,
        direct_place_terminal_max_blend=1.0,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {"direct_place": torch.zeros(1, 1), "direct_place_terminal": torch.ones(1, 1)}
    common = dict(
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )
    values = []
    for distance in (0.60, 0.45, 0.30):
        _obs, latent, _mode = controller.latent(
            Policy(), {"state": torch.zeros(1, 98)}, {"goal_distance": distance}, **common
        )
        values.append(latent.item())
    assert np.allclose(values, [0.0, 0.5, 1.0])


def test_direct_place_profile_blends_slow_lower_and_release_continuously() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_profile_release_distance=0.20,
        direct_place_profile_lower_full_distance=0.25,
        direct_place_release_lift_start=0.20,
        direct_place_release_lift_end=0.05,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {
        "direct_place": torch.zeros(1, 1),
        "direct_place_slow": torch.ones(1, 1),
        "direct_place_lower": torch.full((1, 1), 2.0),
        "direct_place_release": torch.full((1, 1), 3.0),
    }
    common = dict(
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )
    values = []
    for distance, lift in ((0.60, 1.0), (0.45, 1.0), (0.25, 1.0), (0.20, 0.0)):
        _obs, latent, _mode = controller.latent(
            Policy(),
            {"state": torch.zeros(1, 98)},
            {"goal_distance": distance, "lift": lift},
            **common,
        )
        values.append(latent.item())
    assert np.allclose(values, [0.0, 0.5, 2.0, 3.0])


def test_direct_place_profile_uses_hold_guard_for_fast_descent() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_profile_lower_full_distance=0.25,
        direct_place_release_lift_start=0.20,
        direct_place_release_lift_end=0.05,
        direct_place_descent_guard=True,
        direct_place_descent_guard_start_speed=0.10,
        direct_place_descent_guard_full_speed=0.25,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {
        "direct_place": torch.zeros(1, 1),
        "direct_place_slow": torch.ones(1, 1),
        "direct_place_lower": torch.full((1, 1), 2.0),
        "direct_place_release": torch.full((1, 1), 3.0),
        "direct_place_guard": torch.full((1, 1), 4.0),
    }

    _obs, latent, mode = controller.latent(
        Policy(),
        {"state": torch.zeros(1, 98)},
        {
            "goal_distance": 0.25,
            "lift": 1.0,
            "object_vertical_speed": -0.30,
            "bilateral_contact": True,
        },
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )

    assert latent.item() == 4.0
    assert "g1.00" in mode


def test_direct_place_can_disable_descent_guard_without_disabling_contact_recovery() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_profile_lower_full_distance=0.25,
        direct_place_release_lift_start=0.20,
        direct_place_release_lift_end=0.05,
        direct_place_descent_guard=False,
        direct_place_descent_guard_start_speed=0.10,
        direct_place_descent_guard_full_speed=0.25,
        direct_place_contact_guard_steps=2,
        direct_place_contact_guard_lift=0.10,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {
        "direct_place": torch.zeros(1, 1),
        "direct_place_slow": torch.ones(1, 1),
        "direct_place_lower": torch.full((1, 1), 2.0),
        "direct_place_release": torch.full((1, 1), 3.0),
        "direct_place_guard": torch.full((1, 1), 4.0),
        "direct_place_recover": torch.full((1, 1), 5.0),
    }
    common = dict(
        observation={"state": torch.zeros(1, 98)},
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )
    fast = {
        "goal_distance": 0.25,
        "lift": 1.0,
        "object_vertical_speed": -0.50,
        "bilateral_contact": True,
    }

    _obs, unguarded, mode = controller.latent(Policy(), metrics=fast, **common)
    controller.latent(Policy(), metrics=fast, **common)
    lost = {**fast, "bilateral_contact": False}
    controller.latent(Policy(), metrics=lost, **common)
    _obs, recovered, recovered_mode = controller.latent(
        Policy(), metrics=lost, **common
    )

    assert unguarded.item() == 2.0
    assert "g0.00" in mode
    assert recovered.item() == 5.0
    assert "rec1" in recovered_mode


def test_direct_place_contact_recovery_ignores_reset_transient_then_recovers() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_profile_lower_full_distance=0.25,
        direct_place_release_lift_start=0.20,
        direct_place_release_lift_end=0.05,
        direct_place_descent_guard_start_speed=0.10,
        direct_place_descent_guard_full_speed=0.25,
        direct_place_contact_guard_steps=2,
        direct_place_contact_guard_lift=0.10,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {
        "direct_place": torch.zeros(1, 1),
        "direct_place_slow": torch.ones(1, 1),
        "direct_place_lower": torch.full((1, 1), 2.0),
        "direct_place_release": torch.full((1, 1), 3.0),
        "direct_place_guard": torch.full((1, 1), 4.0),
        "direct_place_recover": torch.full((1, 1), 5.0),
    }
    common = dict(
        observation={"state": torch.zeros(1, 98)},
        metrics={
            "goal_distance": 0.25,
            "lift": 1.0,
            "object_vertical_speed": 0.0,
            "bilateral_contact": False,
        },
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )

    _obs, reset_latent, _mode = controller.latent(Policy(), **common)
    common["metrics"] = {**common["metrics"], "bilateral_contact": True}
    controller.latent(Policy(), **common)
    common["metrics"] = {**common["metrics"], "bilateral_contact": False}
    _obs, first_miss, _mode = controller.latent(Policy(), **common)
    _obs, recovered, mode = controller.latent(Policy(), **common)

    assert reset_latent.item() == 2.0
    assert first_miss.item() == 2.0
    assert recovered.item() == 5.0
    assert "rec1" in mode


def test_direct_place_profile_latches_lower_intent_near_goal() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_profile_release_distance=0.20,
        direct_place_profile_lower_full_distance=0.25,
        direct_place_release_lift_start=0.20,
        direct_place_release_lift_end=0.05,
        direct_place_profile_commit_seconds=0.04,
        direct_place_profile_commit_distance=0.30,
        direct_place_profile_commit_speed=0.20,
        direct_place_profile_commit_reset_distance=0.60,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {
        "direct_place": torch.zeros(1, 1),
        "direct_place_slow": torch.ones(1, 1),
        "direct_place_lower": torch.full((1, 1), 2.0),
        "direct_place_release": torch.full((1, 1), 3.0),
    }
    common = dict(
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )
    for _ in range(2):
        _obs, latent, _mode = controller.latent(
            Policy(),
            {"state": torch.zeros(1, 98)},
            {"goal_distance": 0.29, "object_speed": 0.05, "lift": 1.0},
            **common,
        )
    assert np.isclose(controller.direct_place_lower_commitment, 1.0)
    assert latent.item() == 2.0

    _obs, latent, _mode = controller.latent(
        Policy(),
        {"state": torch.zeros(1, 98)},
        {"goal_distance": 0.40, "object_speed": 0.05, "lift": 1.0},
        **common,
    )
    assert controller.direct_place_lower_commitment == 1.0
    assert latent.item() == 2.0


def test_direct_place_descent_guard_uses_guard_latent_when_enabled() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_profile_lower_full_distance=0.25,
        direct_place_release_lift_start=0.20,
        direct_place_release_lift_end=0.05,
        direct_place_descent_guard=True,
        direct_place_descent_guard_start_speed=0.10,
        direct_place_descent_guard_full_speed=0.25,
        direct_place_descent_guard_max_blend=1.0,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {
        "direct_place": torch.zeros(1, 1),
        "direct_place_slow": torch.ones(1, 1),
        "direct_place_lower": torch.full((1, 1), 2.0),
        "direct_place_release": torch.full((1, 1), 3.0),
        "direct_place_guard": torch.full((1, 1), 4.0),
    }

    common = dict(
        observation={"state": torch.zeros(1, 98)},
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )
    fast_metrics = {
        "goal_distance": 0.45,
        "lift": 1.0,
        "object_vertical_speed": -0.30,
        "bilateral_contact": True,
    }
    _obs, latent, mode = controller.latent(
        Policy(), metrics=fast_metrics, **common
    )

    assert latent.item() == 4.0
    assert "g1.00" in mode


def test_direct_place_descent_guard_releases_below_start_speed() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_profile_lower_full_distance=0.25,
        direct_place_release_lift_start=0.20,
        direct_place_release_lift_end=0.05,
        direct_place_descent_guard=True,
        direct_place_descent_guard_start_speed=0.10,
        direct_place_descent_guard_full_speed=0.25,
        direct_place_descent_guard_max_blend=1.0,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {
        "direct_place": torch.zeros(1, 1),
        "direct_place_slow": torch.ones(1, 1),
        "direct_place_lower": torch.full((1, 1), 2.0),
        "direct_place_release": torch.full((1, 1), 3.0),
        "direct_place_guard": torch.full((1, 1), 4.0),
    }
    common = dict(
        observation={"state": torch.zeros(1, 98)},
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )
    metrics = {
        "goal_distance": 0.25,
        "lift": 1.0,
        "object_vertical_speed": -0.30,
        "bilateral_contact": True,
    }
    _obs, entered, entered_mode = controller.latent(
        Policy(), metrics=metrics, **common
    )
    _obs, held, held_mode = controller.latent(
        Policy(), metrics={**metrics, "object_vertical_speed": -0.08}, **common
    )
    _obs, resumed, resumed_mode = controller.latent(
        Policy(), metrics={**metrics, "object_vertical_speed": -0.04}, **common
    )

    assert entered.item() == 4.0
    assert held.item() == 2.0
    assert resumed.item() == 2.0
    assert "g1.00" in entered_mode
    assert "g0.00" in held_mode and "g0.00" in resumed_mode


def test_direct_place_far_profile_keeps_guard_latent() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_profile_lower_full_distance=0.25,
        direct_place_release_lift_start=0.20,
        direct_place_release_lift_end=0.05,
        direct_place_profile_commit_seconds=0.10,
        direct_place_profile_far_commit_seconds=0.04,
        direct_place_profile_far_commit_distance=1.25,
        direct_place_profile_commit_distance=0.30,
        direct_place_profile_commit_speed=0.20,
        direct_place_profile_commit_reset_distance=0.60,
        direct_place_descent_guard=True,
        direct_place_descent_guard_start_speed=0.10,
        direct_place_descent_guard_full_speed=0.25,
        direct_place_descent_guard_max_blend=1.0,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {
        "direct_place": torch.zeros(1, 1),
        "direct_place_slow": torch.ones(1, 1),
        "direct_place_lower": torch.full((1, 1), 2.0),
        "direct_place_release": torch.full((1, 1), 3.0),
        "direct_place_guard": torch.full((1, 1), 4.0),
    }
    common = dict(
        observation={"state": torch.zeros(1, 98)},
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )
    controller.latent(
        Policy(),
        metrics={
            "goal_distance": 1.5,
            "object_speed": 0.0,
            "lift": 1.0,
            "object_vertical_speed": 0.0,
            "bilateral_contact": True,
        },
        **common,
    )
    _obs, latent, mode = controller.latent(
        Policy(),
        metrics={
            "goal_distance": 0.25,
            "object_speed": 0.0,
            "lift": 1.0,
            "object_vertical_speed": -0.30,
            "bilateral_contact": True,
        },
        **common,
    )

    assert latent.item() == 4.0
    assert "g1.00" in mode


def test_direct_place_profile_uses_faster_commitment_for_far_initial_goal() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_profile_lower_full_distance=0.25,
        direct_place_release_lift_start=0.20,
        direct_place_release_lift_end=0.05,
        direct_place_profile_commit_seconds=0.10,
        direct_place_profile_far_commit_seconds=0.04,
        direct_place_profile_far_commit_distance=1.25,
        direct_place_profile_commit_distance=0.30,
        direct_place_profile_commit_speed=0.20,
        direct_place_profile_commit_reset_distance=0.60,
    )
    z = {
        "direct_place": torch.zeros(1, 1),
        "direct_place_slow": torch.ones(1, 1),
        "direct_place_lower": torch.full((1, 1), 2.0),
        "direct_place_release": torch.full((1, 1), 3.0),
    }
    common = dict(
        observation={"state": torch.zeros(1, 98)},
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )

    far = DirectPlaceController(args, dt=0.02)
    near = DirectPlaceController(args, dt=0.02)
    far.latent(Policy(), metrics={"goal_distance": 1.5, "lift": 1.0}, **common)
    near.latent(Policy(), metrics={"goal_distance": 1.0, "lift": 1.0}, **common)
    near_goal = {
        "goal_distance": 0.29,
        "object_speed": 0.05,
        "lift": 1.0,
    }
    for _ in range(2):
        far.latent(Policy(), metrics=near_goal, **common)
        near.latent(Policy(), metrics=near_goal, **common)

    assert far.direct_place_lower_commitment == 1.0
    assert np.isclose(near.direct_place_lower_commitment, 0.4)


def test_direct_place_turn_fades_out_before_lowering() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        direct_place_turn_fade_start_distance=0.80,
        direct_place_turn_fade_end_distance=0.60,
        direct_place_turn_heading_full_deg=45.0,
        direct_place_turn_max_blend=0.40,
        direct_place_terminal_blend_start_distance=0.60,
        direct_place_terminal_blend_end_distance=0.30,
        direct_place_profile_release_distance=0.20,
        direct_place_profile_lower_full_distance=0.25,
        direct_place_release_lift_start=0.20,
        direct_place_release_lift_end=0.05,
        direct_place_profile_commit_seconds=0.0,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {
        "direct_place": torch.zeros(1, 1),
        "direct_place_turn": {-1: torch.ones(1, 1), 1: torch.ones(1, 1)},
        "direct_place_slow": torch.zeros(1, 1),
        "direct_place_lower": torch.zeros(1, 1),
        "direct_place_release": torch.zeros(1, 1),
    }
    common = dict(
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.asarray((0.0, 1.0)),
        z=z,
    )
    _obs, far, _mode = controller.latent(
        Policy(), {"state": torch.zeros(1, 98)},
        {"goal_distance": 1.0, "object_speed": 0.0, "lift": 1.0}, **common,
    )
    _obs, near, _mode = controller.latent(
        Policy(), {"state": torch.zeros(1, 98)},
        {"goal_distance": 0.60, "object_speed": 0.0, "lift": 1.0}, **common,
    )
    assert abs(far.item() - 0.40) < 1e-6
    assert near.item() == 0.0


def test_direct_place_speed_gate_suppresses_release_while_box_moves() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    args = Namespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        place_success_distance=0.20,
        direct_place_terminal_blend_distance=0.35,
        direct_place_terminal_blend_center=0.30,
        direct_place_terminal_blend_width=0.03,
        direct_place_terminal_speed_sigma=0.10,
        direct_place_terminal_max_blend=1.0,
    )
    controller = DirectPlaceController(args, dt=0.02)
    z = {"direct_place": torch.zeros(1, 1), "direct_place_terminal": torch.ones(1, 1)}
    common = dict(
        base_rotation=np.eye(3),
        box_direction=np.ones(2),
        goal_direction=np.ones(2),
        z=z,
    )
    _obs, moving, _mode = controller.latent(
        Policy(), {"state": torch.zeros(1, 98)}, {"goal_distance": 0.30, "object_speed": 0.2}, **common
    )
    _obs, stopped, _mode = controller.latent(
        Policy(), {"state": torch.zeros(1, 98)}, {"goal_distance": 0.30, "object_speed": 0.0}, **common
    )

    assert moving.item() < 0.02
    assert abs(stopped.item() - 0.5) < 1e-6
