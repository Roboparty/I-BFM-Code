from __future__ import annotations

import unittest

import torch

from humanoidverse.carry_speed_latent_sweep import (
    _boxward_approach_kinematics,
    _even_counts,
    _global_topk,
    _rank_balanced_topk,
    _speed_closeness_score,
    _speed_name,
    infer_speed_candidates,
)


class _FakeRewardInferenceModel:
    def reward_wr_inference(self, observation, reward):
        state = observation["state"]
        weights = reward[:, 0] / torch.clamp(reward[:, 0].sum(), min=1e-6)
        return torch.sum(state[:, :4] * weights.unsqueeze(-1), dim=0, keepdim=True)


class CarrySpeedLatentSweepTest(unittest.TestCase):
    def test_speed_name_preserves_fractional_calibration_values(self) -> None:
        self.assertEqual(_speed_name(0.3), "speed_0p3")
        self.assertEqual(_speed_name(0.35), "speed_0p35")

    def test_even_counts_preserves_total(self) -> None:
        self.assertEqual(_even_counts(10, 3), [4, 3, 3])

    def test_rank_balanced_topk_uses_each_rank(self) -> None:
        mask = torch.ones(8, dtype=torch.bool)
        rank = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        score = torch.arange(8, dtype=torch.float32)

        selected = _rank_balanced_topk(mask, score, rank, topk=4)

        self.assertEqual(int((rank[selected] == 0).sum()), 2)
        self.assertEqual(int((rank[selected] == 1).sum()), 2)

    def test_speed_closeness_score_and_global_topk_depend_only_on_absolute_error(self) -> None:
        speed = torch.tensor([0.20, 0.49, 0.50, 0.52, 0.90])
        score = _speed_closeness_score(speed, target_speed=0.5, sigma=0.05)

        selected = _global_topk(torch.ones(5, dtype=torch.bool), score, topk=3)

        self.assertEqual(selected.tolist(), [2, 1, 3])
        self.assertGreater(float(score[1]), float(score[3]))
        self.assertGreater(float(score[3]), float(score[0]))

    def test_boxward_approach_speed_uses_object_direction(self) -> None:
        state = torch.zeros(2, 98)
        state[:, 64:66] = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        velocity = torch.tensor([[0.5, 0.1, 0.0], [0.1, 0.5, 0.0]])

        result = _boxward_approach_kinematics(state, velocity, num_dof=29)

        torch.testing.assert_close(result["toward_speed"], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(result["lateral_speed"], torch.tensor([0.1, -0.1]))
        torch.testing.assert_close(result["distance"], torch.tensor([1.0, 2.0]))

    def test_controlled_inference_separates_lafan_approach_and_carry_transport(self) -> None:
        count = 16
        state = torch.zeros(count, 98)
        state[:, 60] = -1.0
        state[:, 79] = 1.0
        qpos = torch.zeros(count, 36)
        qpos[:, 3] = 1.0
        qvel = torch.zeros(count, 35)
        rank = torch.arange(count) % 2

        # LAFAN rows: root moves forward at the requested Approach speed.
        qvel[:8, 0] = 0.3
        # Carry rows: the lifted box moves toward its target at Transport speed.
        state[8:, -2:] = 1.0
        state[8:, 73] = 0.3
        aux = {
            "carry_bilateral_contact": torch.zeros(count, 1),
            "carry_lift": torch.zeros(count, 1),
            "carry_robot_upright": torch.zeros(count, 1),
            "carry_upright": torch.zeros(count, 1),
            "carry_drop": torch.zeros(count, 1),
        }
        for key in ("carry_bilateral_contact", "carry_lift", "carry_robot_upright", "carry_upright"):
            aux[key][8:] = 1.0
        observation = {
            "state": state,
            "privileged_state": torch.zeros(count, 497),
            "last_action": torch.zeros(count, 29),
        }
        data = {
            "next": {"observation": observation, "qpos": qpos, "qvel": qvel},
            "aux_rewards": aux,
            "rank": rank,
        }

        candidates, stats = infer_speed_candidates(
            _FakeRewardInferenceModel(),
            data,
            num_dof=29,
            speed=0.3,
            topks=[4],
            speed_widths=[0.05],
            approach_upright_min=0.9,
            transport_upright_min=0.85,
            transport_lift_min=0.5,
            lateral_speed_max=0.15,
            vertical_speed_max=0.2,
            yaw_rate_max=0.3,
            transport_target_distance_min=0.5,
        )

        self.assertIn(4, candidates)
        self.assertEqual(stats["approach_candidate_count"], 8)
        self.assertEqual(stats["transport_candidate_count"], 8)
        self.assertAlmostEqual(stats["topk"]["4"]["approach_speed_mean"], 0.3, places=5)
        self.assertAlmostEqual(stats["topk"]["4"]["transport_speed_mean"], 0.3, places=5)

    def test_object_aware_approach_uses_precontact_carry_rows(self) -> None:
        count = 16
        state = torch.zeros(count, 98)
        state[:, 60] = -1.0
        state[:, 79] = 1.0
        state[:, -2:] = 1.0
        state[:8, 64] = 1.0
        state[8:, 73] = 0.3
        state[8:, 82:84] = 1.0
        qpos = torch.zeros(count, 36)
        qpos[:, 3] = 1.0
        qvel = torch.zeros(count, 35)
        qvel[:8, 0] = 0.3
        rank = torch.arange(count) % 2
        aux = {
            "carry_bilateral_contact": torch.zeros(count, 1),
            "carry_lift": torch.zeros(count, 1),
            "carry_robot_upright": torch.ones(count, 1),
            "carry_upright": torch.ones(count, 1),
            "carry_drop": torch.zeros(count, 1),
        }
        aux["carry_bilateral_contact"][8:] = 1.0
        aux["carry_lift"][8:] = 1.0
        observation = {
            "state": state,
            "privileged_state": torch.zeros(count, 497),
            "last_action": torch.zeros(count, 29),
        }
        data = {
            "next": {"observation": observation, "qpos": qpos, "qvel": qvel},
            "aux_rewards": aux,
            "rank": rank,
        }

        candidates, stats = infer_speed_candidates(
            _FakeRewardInferenceModel(),
            data,
            num_dof=29,
            speed=0.3,
            topks=[4],
            speed_widths=[0.05],
            approach_upright_min=0.9,
            transport_upright_min=0.85,
            transport_lift_min=0.5,
            lateral_speed_max=0.15,
            vertical_speed_max=0.2,
            yaw_rate_max=0.3,
            transport_target_distance_min=0.5,
            approach_source="carry-precontact",
        )

        self.assertIn(4, candidates)
        self.assertEqual(stats["approach_source"], "carry-precontact")
        self.assertEqual(stats["approach_speed_semantics"], "robot_boxward")
        self.assertEqual(stats["approach_candidate_count"], 8)
        self.assertAlmostEqual(stats["topk"]["4"]["approach_speed_mean"], 0.3, places=5)

    def test_approach_foot_stability_filters_unstable_precontact_rows(self) -> None:
        count = 12
        state = torch.zeros(count, 98)
        state[:, 60] = -1.0
        state[:, 79] = 1.0
        state[:, -2:] = 1.0
        state[:, 64] = 1.0
        state[8:, 73] = 0.5
        state[8:, 82:84] = 1.0
        qpos = torch.zeros(count, 36)
        qpos[:, 3] = 1.0
        qvel = torch.zeros(count, 35)
        qvel[:, 0] = 0.5
        aux = {
            "carry_bilateral_contact": torch.zeros(count, 1),
            "carry_lift": torch.zeros(count, 1),
            "carry_robot_upright": torch.ones(count, 1),
            "carry_upright": torch.ones(count, 1),
            "carry_drop": torch.zeros(count, 1),
            "penalty_ankle_roll": torch.zeros(count, 1),
            "penalty_feet_ori": torch.zeros(count, 1),
            "penalty_slippage": torch.zeros(count, 1),
            "penalty_undesired_contact": torch.zeros(count, 1),
        }
        aux["penalty_slippage"][4:] = 0.2
        aux["penalty_slippage"][8:] = 0.0
        aux["carry_bilateral_contact"][8:] = 1.0
        aux["carry_lift"][8:] = 1.0
        observation = {
            "state": state,
            "privileged_state": torch.zeros(count, 497),
            "last_action": torch.zeros(count, 29),
        }
        data = {
            "next": {"observation": observation, "qpos": qpos, "qvel": qvel},
            "aux_rewards": aux,
            "rank": torch.zeros(count, dtype=torch.long),
        }

        _candidates, stats = infer_speed_candidates(
            _FakeRewardInferenceModel(),
            data,
            num_dof=29,
            speed=0.5,
            topks=[4],
            speed_widths=[0.05],
            approach_upright_min=0.9,
            transport_upright_min=0.85,
            transport_lift_min=0.5,
            lateral_speed_max=0.15,
            vertical_speed_max=0.2,
            yaw_rate_max=0.3,
            transport_target_distance_min=0.5,
            approach_source="carry-precontact",
            approach_foot_stability=True,
            approach_slippage_max=0.1,
        )

        self.assertTrue(stats["approach_foot_stability_enabled"])
        self.assertEqual(stats["approach_candidate_count"], 4)
        self.assertEqual(stats["topk"]["4"]["approach_slippage_mean"], 0.0)

    def test_approach_motion_quality_filters_unstable_pose_rows(self) -> None:
        count = 12
        state = torch.zeros(count, 98)
        state[:, 60] = -1.0
        state[:, 79] = 1.0
        state[:, -2:] = 1.0
        state[:, 64] = 1.0
        state[8:, 73] = 0.3
        state[8:, 82:84] = 1.0
        qpos = torch.zeros(count, 36)
        qpos[:, 2] = 0.75
        qpos[:, 3] = 1.0
        qvel = torch.zeros(count, 35)
        qvel[:, 0] = 0.3
        aux = {
            "carry_bilateral_contact": torch.zeros(count, 1),
            "carry_lift": torch.zeros(count, 1),
            "carry_robot_upright": torch.ones(count, 1),
            "carry_upright": torch.ones(count, 1),
            "carry_drop": torch.zeros(count, 1),
            "penalty_action_rate": torch.zeros(count, 1),
            "penalty_torques": torch.full((count, 1), 1000.0),
            "carry_body_mpjpe": torch.full((count, 1), 0.10),
            "carry_root_rot_err": torch.full((count, 1), 10.0),
            "feet_heading_alignment": torch.full((count, 1), 0.10),
        }
        aux["penalty_action_rate"][4:8] = 20.0
        aux["carry_bilateral_contact"][8:] = 1.0
        aux["carry_lift"][8:] = 1.0
        observation = {
            "state": state,
            "privileged_state": torch.zeros(count, 497),
            "last_action": torch.zeros(count, 29),
        }
        data = {
            "next": {"observation": observation, "qpos": qpos, "qvel": qvel},
            "aux_rewards": aux,
            "rank": torch.zeros(count, dtype=torch.long),
        }

        _candidates, stats = infer_speed_candidates(
            _FakeRewardInferenceModel(),
            data,
            num_dof=29,
            speed=0.3,
            topks=[4],
            speed_widths=[0.05],
            approach_upright_min=0.9,
            transport_upright_min=0.85,
            transport_lift_min=0.5,
            lateral_speed_max=0.15,
            vertical_speed_max=0.2,
            yaw_rate_max=0.3,
            transport_target_distance_min=0.5,
            approach_source="carry-precontact",
            approach_motion_quality=True,
            approach_action_rate_max=15.0,
            speed_closeness_only=True,
        )

        self.assertTrue(stats["approach_motion_quality_enabled"])
        self.assertEqual(stats["approach_candidate_count"], 4)
        self.assertEqual(stats["topk"]["4"]["approach_action_rate_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
