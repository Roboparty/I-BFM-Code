from __future__ import annotations

import unittest

import torch

from humanoidverse.carry_dance_height_inference import compute_box_height_threshold_reward


class CarryDanceHeightInferenceTest(unittest.TestCase):
    def test_height_reward_is_strict_binary_threshold(self) -> None:
        ground = 0.186028293
        lift_height = 0.35
        threshold = (0.5 - ground) / lift_height
        aux = {
            "carry_lift": torch.tensor(
                [[threshold - 1e-4], [threshold], [threshold + 1e-4]], dtype=torch.float32
            )
        }

        reward = compute_box_height_threshold_reward(
            aux,
            threshold_world_height=0.5,
            ground_center_height=ground,
            lift_height=lift_height,
        )

        torch.testing.assert_close(reward, torch.tensor([[0.0], [0.0], [1.0]]))

    def test_height_reward_does_not_require_other_carry_signals(self) -> None:
        reward = compute_box_height_threshold_reward(
            {"carry_lift": torch.tensor([[0.0], [1.0]])}
        )

        torch.testing.assert_close(reward, torch.tensor([[0.0], [1.0]]))

    def test_height_reward_rejects_threshold_beyond_clipped_signal(self) -> None:
        with self.assertRaisesRegex(ValueError, "un-clipped carry_lift range"):
            compute_box_height_threshold_reward(
                {"carry_lift": torch.ones(1, 1)},
                threshold_world_height=0.6,
            )


if __name__ == "__main__":
    unittest.main()
