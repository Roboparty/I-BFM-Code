from __future__ import annotations

import types
import unittest

import torch

from humanoidverse.goal_recovery_inference import (
    _apply_initial_distance,
    _apply_initial_pose,
    _clip_observed_object_distance,
)
from humanoidverse.tracking_inference import (
    _apply_lie_down_initialization,
    _downward_force_active,
    _target_states_from_obs,
    _tracking_z,
)


def _tracking_obs(*, object_present: float | None) -> dict[str, torch.Tensor]:
    obs = {
        "ref_body_pos": torch.zeros(2, 1, 3),
        "ref_body_rots": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * 2),
        "ref_body_vels": torch.zeros(2, 1, 3),
        "ref_body_angular_vels": torch.zeros(2, 1, 3),
        "dof_pos": torch.zeros(2, 3),
        "ref_dof_vel": torch.zeros(2, 3),
        "object_pos": torch.tensor([[0.0, 0.0, -10.0]] * 2),
        "object_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2),
        "object_lin_vel": torch.zeros(2, 3),
        "object_ang_vel": torch.zeros(2, 3),
        "object_target_pos": torch.tensor([0.0, 0.0, -10.0]),
        "object_target_quat": torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }
    if object_present is not None:
        obs["object_present"] = torch.full((2, 1), object_present)
    return obs


class TrackingTargetStatesTest(unittest.TestCase):
    def test_preserves_absent_object_flag(self) -> None:
        target = _target_states_from_obs(_tracking_obs(object_present=0.0), "cpu", num_dof=3)

        torch.testing.assert_close(target["object_present"], torch.zeros(1, 1))
        torch.testing.assert_close(target["object_states"][0, :3], torch.tensor([0.0, 0.0, -10.0]))

    def test_preserves_present_object_flag(self) -> None:
        target = _target_states_from_obs(_tracking_obs(object_present=1.0), "cpu", num_dof=3)

        torch.testing.assert_close(target["object_present"], torch.ones(1, 1))

    def test_rejects_object_schema_without_presence_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "object_present"):
            _target_states_from_obs(_tracking_obs(object_present=None), "cpu", num_dof=3)

    def test_lie_down_initialization_rotates_root_without_mutating_input(self) -> None:
        target = _target_states_from_obs(_tracking_obs(object_present=0.0), "cpu", num_dof=3)
        original = target["root_states"].clone()

        left = _apply_lie_down_initialization(target, side="left")
        right = _apply_lie_down_initialization(target, side="right")

        torch.testing.assert_close(target["root_states"], original)
        torch.testing.assert_close(left["root_states"][:, 2], torch.tensor([0.5]))
        torch.testing.assert_close(right["root_states"][:, 2], torch.tensor([0.5]))
        expected = 2**-0.5
        torch.testing.assert_close(
            left["root_states"][:, 3:7],
            torch.tensor([[expected, 0.0, 0.0, expected]]),
        )
        torch.testing.assert_close(
            right["root_states"][:, 3:7],
            torch.tensor([[-expected, 0.0, 0.0, expected]]),
        )

    def test_lie_down_initialization_rejects_unknown_side(self) -> None:
        target = _target_states_from_obs(_tracking_obs(object_present=0.0), "cpu", num_dof=3)

        with self.assertRaisesRegex(ValueError, "lie-down side"):
            _apply_lie_down_initialization(target, side="back")

    def test_goal_initial_pose_zeros_robot_joint_and_object_velocities(self) -> None:
        obs = _tracking_obs(object_present=1.0)
        obs["ref_body_vels"][:] = 1.0
        obs["ref_body_angular_vels"][:] = 2.0
        obs["ref_dof_vel"][:] = 3.0
        obs["object_lin_vel"][:] = 4.0
        obs["object_ang_vel"][:] = 5.0
        target = _target_states_from_obs(obs, "cpu", num_dof=3)

        result = _apply_initial_pose(target, "standing", lying_root_height=0.5)

        torch.testing.assert_close(result["root_states"][:, 7:13], torch.zeros(1, 6))
        torch.testing.assert_close(result["dof_states"][:, :, 1], torch.zeros(1, 3))
        torch.testing.assert_close(result["object_states"][:, 7:13], torch.zeros(1, 6))
        torch.testing.assert_close(target["object_states"][:, 7:10], torch.full((1, 3), 4.0))
        torch.testing.assert_close(target["object_states"][:, 10:13], torch.full((1, 3), 5.0))

    def test_goal_initial_distance_moves_only_robot_xy_and_zeros_velocities(self) -> None:
        target = _target_states_from_obs(_tracking_obs(object_present=1.0), "cpu", num_dof=3)
        target["root_states"][0, :2] = torch.tensor([0.0, 0.0])
        target["object_states"][0, :2] = torch.tensor([0.5, 0.0])
        target["root_states"][0, 7:13] = 2.0
        target["object_states"][0, 7:13] = 3.0
        original = {key: value.clone() for key, value in target.items()}

        result = _apply_initial_distance(target, 5.0)

        self.assertAlmostEqual(
            torch.linalg.vector_norm(result["root_states"][0, :2] - result["object_states"][0, :2]).item(),
            5.0,
        )
        torch.testing.assert_close(result["root_states"][0, 7:13], torch.zeros(6))
        torch.testing.assert_close(result["object_states"][0, 7:13], torch.zeros(6))
        for key in target:
            torch.testing.assert_close(target[key], original[key])


class GoalRecoveryObservationClippingTest(unittest.TestCase):
    @staticmethod
    def _observation() -> dict[str, torch.Tensor]:
        num_dof = 3
        robot_state = torch.zeros(1, 2 * num_dof + 6)
        object_context = torch.zeros(1, 34)
        object_context[0, :3] = torch.tensor([3.0, 4.0, 0.7])
        object_context[0, 20:26] = torch.tensor([0.5, 0.2, 1.0, 0.5, -0.2, 1.0])
        object_context[0, 26:32] = (
            object_context[0, 20:26].reshape(2, 3) - object_context[0, :3]
        ).reshape(-1)
        state = torch.cat((robot_state, object_context), dim=-1)
        return {
            "state": state,
            "privileged_state": state.clone(),
            "last_action": torch.zeros(1, num_dof),
        }

    def test_clips_planar_distance_and_preserves_bearing_height(self) -> None:
        observation = self._observation()

        clipped = _clip_observed_object_distance(observation, 2.0)

        start = 12
        expected_object_pos = torch.tensor([[1.2, 1.6, 0.7]])
        torch.testing.assert_close(clipped["state"][:, start : start + 3], expected_object_pos)
        torch.testing.assert_close(clipped["privileged_state"][:, start : start + 3], expected_object_pos)
        expected_hand_minus_box = (
            clipped["state"][:, start + 20 : start + 26].reshape(1, 2, 3)
            - expected_object_pos.unsqueeze(-2)
        ).reshape(1, 6)
        torch.testing.assert_close(clipped["state"][:, start + 26 : start + 32], expected_hand_minus_box)

    def test_does_not_modify_input_or_clip_nearby_object(self) -> None:
        observation = self._observation()
        original = {key: value.clone() for key, value in observation.items()}

        clipped = _clip_observed_object_distance(observation, 10.0)

        for key in original:
            torch.testing.assert_close(observation[key], original[key])
            torch.testing.assert_close(clipped[key], original[key])
        self.assertIsNot(clipped["state"], observation["state"])

    def test_none_returns_original_observation(self) -> None:
        observation = self._observation()

        self.assertIs(_clip_observed_object_distance(observation, None), observation)


class _TrackingModel:
    def __init__(self, seq_length: int, *, expert_z_mode: str = "mean", expert_z_horizon: int | None = None):
        self.cfg = types.SimpleNamespace(seq_length=seq_length)
        self.cfg.train = types.SimpleNamespace(
            expert_z_mode=expert_z_mode,
            expert_z_horizon=seq_length if expert_z_horizon is None else expert_z_horizon,
        )

    def backward_map(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.clone()

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        return z


class TrackingLatentTest(unittest.TestCase):
    def test_defaults_to_model_sequence_length(self) -> None:
        obs = torch.arange(5, dtype=torch.float32).unsqueeze(-1)

        z = _tracking_z(_TrackingModel(seq_length=3), obs)

        torch.testing.assert_close(z[:, 0], torch.tensor([1.0, 2.0, 3.0, 3.5, 4.0]))

    def test_explicit_sequence_length_overrides_model(self) -> None:
        obs = torch.arange(4, dtype=torch.float32).unsqueeze(-1)

        z = _tracking_z(_TrackingModel(seq_length=8), obs, seq_length=1)

        torch.testing.assert_close(z, obs)

    def test_rejects_non_positive_sequence_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _tracking_z(_TrackingModel(seq_length=8), torch.zeros(2, 1), seq_length=0)

    def test_direct_mode_uses_future_goal_and_holds_final_latent(self) -> None:
        obs = torch.arange(6, dtype=torch.float32).unsqueeze(-1)

        z = _tracking_z(
            _TrackingModel(seq_length=8, expert_z_mode="direct", expert_z_horizon=3),
            obs,
        )

        torch.testing.assert_close(z[:, 0], torch.tensor([2.0, 3.0, 4.0, 5.0, 5.0, 5.0]))


class ExternalForceScheduleTest(unittest.TestCase):
    def test_force_window_uses_simulation_time(self) -> None:
        active = [_downward_force_active(step, dt=0.02, start_s=0.04, duration_s=0.04) for step in range(5)]

        self.assertEqual(active, [False, False, True, True, False])

    def test_zero_duration_disables_force(self) -> None:
        self.assertFalse(_downward_force_active(10, dt=0.02, start_s=0.0, duration_s=0.0))

    def test_invalid_force_timing_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _downward_force_active(0, dt=0.02, start_s=-1.0, duration_s=0.1)


if __name__ == "__main__":
    unittest.main()
