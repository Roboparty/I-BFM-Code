import math
import unittest

import numpy as np

from rl_policy.carry_runtime import CarryMocapFrame, project_z
from rl_policy.carry_solve_close_166559744_runtime import (
    IntactLogmap166559744SolveCloseController,
)


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.asarray(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        dtype=np.float64,
    )


def _latents() -> dict[str, object]:
    result: dict[str, object] = {}
    for index, name in enumerate(
        ("approach", "lift", "transport", "transport_slow", "align", "lower", "place")
    ):
        value = np.zeros(256, dtype=np.float32)
        value[index] = 16.0
        result[name] = value
    result["turn"] = {-1: result["approach"].copy(), 1: result["approach"].copy()}
    result["transport_turn"] = {
        -1: result["transport"].copy(),
        1: result["transport"].copy(),
    }
    result["get_up"] = np.ones((1, 256), dtype=np.float32)
    return result


def _pivot_assets():
    candidates = {}
    normal = {}
    names = {}
    for direction, start in ((-1, 30), (1, 40)):
        candidates[direction] = []
        for index in range(5):
            value = np.zeros(256, dtype=np.float32)
            value[start + index] = 16.0
            candidates[direction].append(value)
        value = np.zeros(256, dtype=np.float32)
        value[start - 1] = 16.0
        normal[direction] = value
        names[direction] = [f"candidate_{index}" for index in range(5)]
    return candidates, normal, names


def _frame(*, base_yaw=0.0, box_yaw=0.0, box_height=0.5):
    return CarryMocapFrame(
        sequence=1,
        timestamp_ns=1,
        robot_root_position=np.asarray([0.0, 0.0, 0.8]),
        robot_root_quaternion_wxyz=_quat_from_yaw(base_yaw),
        object_position=np.asarray([0.0, 0.0, box_height]),
        object_quaternion_wxyz=_quat_from_yaw(box_yaw),
        object_linear_velocity_world=np.zeros(3),
        object_angular_velocity_world=np.zeros(3),
        hand_contact=np.ones(2),
        target_position_world=np.asarray([0.0, 0.6, 0.2]),
    )


def _observation():
    return {
        "state": np.zeros(98, dtype=np.float32),
        "last_action": np.zeros(29, dtype=np.float32),
        "history_actor": np.zeros(372, dtype=np.float32),
    }


def _metrics(goal_distance=0.6, box_height=0.5):
    return {
        "horizontal_distance": 0.4,
        "goal_distance": goal_distance,
        "object_height": box_height,
        "object_speed": 0.2,
        "lift": 1.0,
        "root_height": 0.8,
        "pelvis_up_z": 1.0,
        "robot_upright": 1.0,
        "box_upright": 1.0,
        "bilateral_contact": True,
    }


class SolveCloseControllerTest(unittest.TestCase):
    def _controller(self, **solve_overrides):
        phase = {
            "fall_recovery_enabled": True,
            "transport_heading_blend_full_deg": 45.0,
            "transport_max_turn_blend": 0.45,
            "transport_near_max_turn_blend": 0.80,
            "transport_turn_boost_distance": 1.20,
            "place_align_distance": 0.40,
            "place_enter_steps": 5,
        }
        solve = {
            "near_align_distance": 0.80,
            "near_align_enter_bearing_deg": 30.0,
            "near_align_exit_bearing_deg": 15.0,
            "near_align_enter_steps": 3,
            "near_align_exit_steps": 5,
            "near_align_blend_seconds": 0.20,
            "pivot_response_seconds": 0.50,
            "pivot_early_wrong_seconds": 0.25,
            "pivot_max_seconds": 5.0,
            "pivot_cooldown_seconds": 0.50,
            "pivot_stall_window_seconds": 0.50,
        }
        solve.update(solve_overrides)
        candidates, normal, names = _pivot_assets()
        controller = IntactLogmap166559744SolveCloseController(
            phase, solve, _latents(), 0.02, candidates, normal, names
        )
        controller.phase = "transport"
        return controller, candidates, normal

    def test_near_large_bearing_uses_first_pivot_candidate(self):
        controller, candidates, _ = self._controller()
        latent = None
        mode = None
        for _ in range(12):
            _, latent, mode = controller.select(
                _observation(), _metrics(), _frame()
            )
        self.assertTrue(controller.near_align_active)
        self.assertAlmostEqual(controller.near_align_weight, 1.0)
        self.assertEqual(mode, "transport_align_left_c0")
        np.testing.assert_allclose(latent, project_z(candidates[1][0]), atol=1.0e-6)

    def test_alignment_blocks_distance_only_place_entry(self):
        controller, _, _ = self._controller(near_align_blend_seconds=0.0)
        controller.near_align_active = True
        controller.near_align_weight = 1.0
        for _ in range(10):
            controller.update(_metrics(goal_distance=0.2))
        self.assertEqual(controller.phase, "transport")
        self.assertEqual(controller.place_enter_steps, 0)

    def test_wrong_signed_yaw_switches_candidate(self):
        controller, _, _ = self._controller()
        for step in range(20):
            yaw = -0.004 * step
            controller.select(
                _observation(),
                _metrics(),
                _frame(base_yaw=yaw, box_yaw=yaw),
            )
        self.assertTrue(controller.near_align_active)
        self.assertGreater(controller.pivot_candidate_index, 0)

    def test_far_transport_uses_original_turn_latent(self):
        controller, _, normal = self._controller()
        metrics = _metrics(goal_distance=1.5)
        frame = _frame()
        _, latent, mode = controller.select(_observation(), metrics, frame)
        self.assertEqual(mode, "transport_left")
        turn_weight = 0.45
        expected = project_z(
            (1.0 - turn_weight) * controller.latents["transport"]
            + turn_weight * normal[1]
        )
        np.testing.assert_allclose(latent, expected, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
