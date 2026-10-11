from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import torch

from humanoidverse.agents.buffers.transition import DictBuffer
from humanoidverse.carry_phased_reward_inference import (
    _box_recovery_phase,
    _goal_lookahead_observation,
    _initial_states_at_distance,
    _masked_object_observation,
    _nearest_approach_latent,
    _resolve_goal_center_height,
    _smooth_place_blend,
    _transport_turn_weight,
    _with_object_goal_offset,
    compute_approach_speed_reward,
    compute_lower_reward,
    compute_place_reward,
    compute_static_carry_reward,
    compute_transport_reward,
    compute_turn_reward,
)
from humanoidverse.carry_phased_sim2sim import _summarize_speed_rollout
from humanoidverse.carry_reward_inference import (
    CARRY_REWARD_KEYS,
    _apply_standing_distance_offset,
    compute_carry_lift_upright_reward,
)
from humanoidverse.envs.carry_observations import OBJECT_CONTEXT_DIM


class CarryRewardInferenceTest(unittest.TestCase):
    def test_speed_rollout_summary_separates_approach_and_transport(self) -> None:
        base = {
            "horizontal_distance": 1.0,
            "goal_distance": 1.0,
            "object_height": 0.5,
            "object_speed": 0.3,
            "lift": 1.0,
            "robot_upright": 1.0,
            "box_upright": 1.0,
            "bilateral_contact": True,
            "approach_boxward_speed": 0.2,
            "transport_goalward_speed": 0.4,
        }
        history = [{"phase": "approach", **base}, {"phase": "transport", **base}]

        summary = _summarize_speed_rollout(history, final_phase="transport")

        self.assertAlmostEqual(summary["approach_boxward_speed_mean"], 0.2)
        self.assertAlmostEqual(summary["transport_goalward_speed_mean"], 0.4)
        self.assertEqual(summary["phase_counts"]["transport"], 1)

    def test_place_blend_settles_then_smoothly_releases(self) -> None:
        self.assertEqual(_smooth_place_blend(0, settle_steps=50, lower_steps=150), 0.0)
        self.assertEqual(_smooth_place_blend(50, settle_steps=50, lower_steps=150), 0.0)
        self.assertAlmostEqual(_smooth_place_blend(125, settle_steps=50, lower_steps=150), 0.5)
        self.assertEqual(_smooth_place_blend(200, settle_steps=50, lower_steps=150), 1.0)
        self.assertEqual(_smooth_place_blend(300, settle_steps=50, lower_steps=150), 1.0)

    def test_transport_turn_blend_increases_near_goal(self) -> None:
        kwargs = {
            "bearing_deg": 45.0,
            "turn_boost_distance": 0.8,
            "place_distance": 0.25,
            "heading_blend_full_deg": 45.0,
            "far_max_turn_blend": 0.45,
            "near_max_turn_blend": 1.0,
        }

        far = _transport_turn_weight(goal_distance=0.8, **kwargs)
        middle = _transport_turn_weight(goal_distance=0.525, **kwargs)
        near = _transport_turn_weight(goal_distance=0.25, **kwargs)
        aligned = _transport_turn_weight(goal_distance=0.25, **{**kwargs, "bearing_deg": 0.0})

        self.assertEqual(far, 0.45)
        self.assertAlmostEqual(middle, 0.725)
        self.assertEqual(near, 1.0)
        self.assertEqual(aligned, 1.0)

        aligned_middle = _transport_turn_weight(
            goal_distance=0.525,
            **{**kwargs, "bearing_deg": 0.0},
        )
        self.assertAlmostEqual(aligned_middle, 0.5)

    def test_failed_place_regrasps_near_box_and_approaches_far_box(self) -> None:
        self.assertEqual(_box_recovery_phase(0.5, approach_switch_distance=0.75), "lift")
        self.assertEqual(_box_recovery_phase(1.0, approach_switch_distance=0.75), "approach")

    def test_goal_lookahead_changes_only_target_delta(self) -> None:
        state = torch.zeros(1, 98)
        state[:, 79:82] = torch.tensor([0.3, 0.4, -0.5])
        observation = {"state": state, "last_action": torch.ones(1, 29)}

        shifted = _goal_lookahead_observation(observation, 0.2)

        torch.testing.assert_close(shifted["state"][:, 79:82], torch.tensor([[0.42, 0.56, -0.5]]))
        torch.testing.assert_close(shifted["state"][:, :79], state[:, :79])
        torch.testing.assert_close(shifted["state"][:, 82:], state[:, 82:])
        torch.testing.assert_close(shifted["last_action"], observation["last_action"])
        torch.testing.assert_close(observation["state"][:, 79:82], torch.tensor([[0.3, 0.4, -0.5]]))

    def test_approach_reward_prefers_target_forward_speed(self) -> None:
        qpos = torch.zeros(3, 7)
        qpos[:, 3] = 1.0  # MuJoCo identity quaternion in wxyz order.
        qvel = torch.zeros(3, 6)
        qvel[:, 0] = torch.tensor([0.5, 0.0, 1.0])
        state = torch.zeros(3, 98)
        state[:, 60] = -1.0

        reward, local_velocity = compute_approach_speed_reward(qpos, qvel, state, num_dof=29)

        self.assertGreater(reward[0].item(), reward[1].item())
        self.assertGreater(reward[0].item(), reward[2].item())
        self.assertAlmostEqual(local_velocity[0, 0].item(), 0.5)

    def test_approach_reward_supports_lateral_direction(self) -> None:
        qpos = torch.zeros(2, 7)
        qpos[:, 3] = 1.0
        qvel = torch.zeros(2, 6)
        qvel[0, 1] = 0.5
        qvel[1, 0] = 0.5
        state = torch.zeros(2, 98)
        state[:, 60] = -1.0

        reward, _velocity = compute_approach_speed_reward(
            qpos,
            qvel,
            state,
            num_dof=29,
            target_angle_deg=90.0,
        )

        self.assertGreater(reward[0].item(), 0.99)
        self.assertLess(reward[1].item(), 0.01)

    def test_approach_heading_reward_prefers_box_in_front(self) -> None:
        qpos = torch.zeros(2, 7)
        qpos[:, 3] = 1.0
        qvel = torch.zeros(2, 6)
        qvel[:, 0] = 0.5
        state = torch.zeros(2, 98)
        state[:, 60] = -1.0
        state[:, -2] = 1.0
        state[0, 64:66] = torch.tensor([1.0, 0.0])
        state[1, 64:66] = torch.tensor([0.0, 1.0])

        reward, _velocity = compute_approach_speed_reward(
            qpos,
            qvel,
            state,
            num_dof=29,
            box_heading_sigma_deg=15.0,
        )

        self.assertGreater(reward[0].item(), 0.99)
        self.assertLess(reward[1].item(), 1e-6)

    def test_turn_reward_prefers_requested_yaw_rate(self) -> None:
        qpos = torch.zeros(3, 7)
        qpos[:, 3] = 1.0
        qvel = torch.zeros(3, 6)
        state = torch.zeros(3, 98)
        state[:, 60] = -1.0
        state[:, 63] = torch.tensor([0.5, -0.5, 0.0])

        reward, components = compute_turn_reward(
            qpos,
            qvel,
            state,
            num_dof=29,
            target_yaw_rate=0.5,
        )

        self.assertGreater(reward[0].item(), 0.99)
        self.assertLess(reward[1].item(), reward[2].item())
        self.assertAlmostEqual(components["root_yaw_rate"][0].item(), 0.5)

    def test_nearest_approach_latent_uses_local_box_bearing(self) -> None:
        z_bank = {-90.0: torch.tensor([[-1.0]]), 0.0: torch.tensor([[0.0]]), 90.0: torch.tensor([[1.0]])}
        identity_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0])

        z, selected_angle, bearing = _nearest_approach_latent(
            z_bank,
            identity_xyzw,
            torch.tensor([0.0, 2.0]),
        )

        self.assertEqual(selected_angle, 90.0)
        self.assertAlmostEqual(bearing, 90.0)
        torch.testing.assert_close(z, torch.tensor([[1.0]]))

    def test_static_reward_penalizes_velocity(self) -> None:
        aux = {key: torch.ones(2, 1) for key in CARRY_REWARD_KEYS}
        aux["carry_drop"] = torch.zeros(2, 1)
        state = torch.zeros(2, 98)
        state[:, 60] = -1.0
        state[:, -2:] = 1.0
        state[1, 29:58] = 2.0
        state[1, 73:76] = 1.0

        reward, components = compute_static_carry_reward(aux, state, num_dof=29)

        self.assertGreater(reward[0].item(), 0.99)
        self.assertLess(reward[1].item(), 1e-6)
        self.assertGreater(components["object_lin_speed"][1].item(), 0.9)

    def test_transport_reward_requires_carry_and_goalward_velocity(self) -> None:
        aux = {key: torch.ones(3, 1) for key in CARRY_REWARD_KEYS}
        aux["carry_drop"] = torch.zeros(3, 1)
        aux["carry_bilateral_contact"][2] = 0.0
        state = torch.zeros(3, 98)
        state[:, 60] = -1.0
        state[:, -2:] = 1.0
        state[:, 79] = 1.0
        state[0, 73] = 0.35
        state[1, 73] = -0.35
        state[2, 73] = 0.35

        reward, components = compute_transport_reward(aux, state, num_dof=29, target_speed=0.35)

        self.assertGreater(reward[0].item(), 0.99)
        self.assertLess(reward[1].item(), 1e-4)
        self.assertEqual(reward[2].item(), 0.0)
        self.assertAlmostEqual(components["toward_speed"][0].item(), 0.35, places=5)

    def test_transport_reward_defaults_to_0p8_goalward_speed(self) -> None:
        aux = {key: torch.ones(1, 1) for key in CARRY_REWARD_KEYS}
        aux["carry_drop"] = torch.zeros(1, 1)
        state = torch.zeros(1, 98)
        state[:, 60] = -1.0
        state[:, -2:] = 1.0
        state[:, 79] = 1.0
        state[:, 73] = 0.8

        reward, components = compute_transport_reward(aux, state, num_dof=29)

        self.assertGreater(reward[0].item(), 0.99)
        self.assertAlmostEqual(components["desired_speed"][0].item(), 0.8, places=5)

    def test_transport_slow_reward_prefers_near_goal_samples(self) -> None:
        aux = {key: torch.ones(2, 1) for key in CARRY_REWARD_KEYS}
        aux["carry_drop"] = torch.zeros(2, 1)
        state = torch.zeros(2, 98)
        state[:, 60] = -1.0
        state[:, -2:] = 1.0
        state[:, 79] = torch.tensor([0.2, 1.6])
        state[:, 73] = 0.08

        reward, components = compute_transport_reward(
            aux,
            state,
            num_dof=29,
            target_speed=0.08,
            goal_distance_sigma=0.8,
        )

        torch.testing.assert_close(components["desired_speed"], torch.tensor([0.08, 0.08]))
        self.assertGreater(reward[0].item(), reward[1].item() * 40.0)

    def test_place_reward_prefers_released_stationary_goal_state(self) -> None:
        aux = {key: torch.ones(3, 1) for key in CARRY_REWARD_KEYS}
        aux["carry_drop"] = torch.zeros(3, 1)
        aux["carry_lift"] = torch.zeros(3, 1)
        aux["carry_lift"][2] = 1.0
        aux["carry_bilateral_contact"] = torch.tensor([[0.0], [1.0], [0.0]])
        state = torch.zeros(3, 98)
        state[:, 60] = -1.0
        state[:, -2:] = 1.0
        state[2, 79] = 1.0

        reward, _components = compute_place_reward(aux, state, num_dof=29)

        self.assertGreater(reward[0].item(), 0.99)
        self.assertLess(reward[1].item(), reward[0].item())
        self.assertLess(reward[2].item(), 1e-4)

    def test_place_reward_keeps_dense_signal_before_box_touches_ground(self) -> None:
        aux = {key: torch.ones(1, 1) for key in CARRY_REWARD_KEYS}
        aux["carry_drop"] = torch.zeros(1, 1)
        state = torch.zeros(1, 98)
        state[:, 60] = -1.0
        state[:, -2:] = 1.0

        reward, components = compute_place_reward(aux, state, num_dof=29)

        self.assertGreater(reward.item(), 0.19)
        self.assertLess(reward.item(), 0.25)
        self.assertLess(components["placement_quality"].item(), 1e-6)

    def test_lower_reward_prefers_contact_controlled_downward_motion(self) -> None:
        aux = {key: torch.ones(3, 1) for key in CARRY_REWARD_KEYS}
        aux["carry_drop"] = torch.zeros(3, 1)
        aux["carry_bilateral_contact"][2] = 0.0
        state = torch.zeros(3, 98)
        state[:, 60] = -1.0
        state[:, -2:] = 1.0
        state[:, 79] = 0.1
        state[:, 75] = torch.tensor([-0.08, 0.2, -0.08])

        reward, components = compute_lower_reward(aux, state, num_dof=29)

        self.assertGreater(reward[0].item(), 0.8)
        self.assertLess(reward[1].item(), 1e-4)
        self.assertEqual(reward[2].item(), 0.0)
        self.assertAlmostEqual(components["vertical_speed"][0].item(), -0.08, places=5)

    def test_object_mask_preserves_robot_state(self) -> None:
        state = torch.arange(98, dtype=torch.float32).unsqueeze(0)
        observation = {"state": state, "last_action": torch.ones(1, 29)}

        masked = _masked_object_observation(observation)

        torch.testing.assert_close(masked["state"][:, :-OBJECT_CONTEXT_DIM], state[:, :-OBJECT_CONTEXT_DIM])
        torch.testing.assert_close(
            masked["state"][:, -OBJECT_CONTEXT_DIM:], torch.zeros(1, OBJECT_CONTEXT_DIM)
        )
        torch.testing.assert_close(masked["last_action"], observation["last_action"])
        self.assertFalse(masked["state"].data_ptr() == state.data_ptr())

    def test_absolute_initial_distance_zeros_velocities(self) -> None:
        states = {
            "root_states": torch.tensor([[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]),
            "object_states": torch.tensor([[0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]),
            "dof_states": torch.ones(1, 2, 2),
        }

        moved = _initial_states_at_distance(states, 3.0)

        distance = torch.linalg.vector_norm(moved["root_states"][0, :2] - moved["object_states"][0, :2])
        self.assertAlmostEqual(distance.item(), 3.0)
        torch.testing.assert_close(moved["root_states"][0, 7:13], torch.zeros(6))
        torch.testing.assert_close(moved["object_states"][0, 7:13], torch.zeros(6))
        torch.testing.assert_close(moved["dof_states"][0, :, 1], torch.zeros(2))

    def test_goal_offset_only_changes_target_object_position(self) -> None:
        states = {
            "root_states": torch.zeros(1, 13),
            "object_states": torch.tensor([[2.0, 3.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            "target_object_pos": torch.tensor([[9.0, 9.0, 9.0]]),
        }

        moved = _with_object_goal_offset(states, offset_x=1.5, offset_y=-0.5, goal_height=0.185)

        torch.testing.assert_close(moved["target_object_pos"], torch.tensor([[3.5, 2.5, 0.185]]))
        torch.testing.assert_close(moved["object_states"], states["object_states"])
        torch.testing.assert_close(states["target_object_pos"], torch.tensor([[9.0, 9.0, 9.0]]))

    def test_absolute_goal_overrides_relative_offset(self) -> None:
        states = {
            "object_states": torch.tensor([[2.0, 3.0, 0.2, 0.0, 0.0, 0.0, 1.0]]),
            "target_object_pos": torch.zeros(1, 3),
        }

        moved = _with_object_goal_offset(
            states,
            offset_x=100.0,
            offset_y=100.0,
            goal_height=0.185,
            goal_x=-1.0,
            goal_y=4.0,
        )

        torch.testing.assert_close(moved["target_object_pos"], torch.tensor([[-1.0, 4.0, 0.185]]))

    def test_goal_ground_z_maps_to_box_center_height(self) -> None:
        center, ground = _resolve_goal_center_height(
            {"half_size": [0.175, 0.175, 0.185]},
            goal_height=None,
            goal_ground_z=0.0,
        )
        self.assertAlmostEqual(center, 0.185)
        self.assertAlmostEqual(ground, 0.0)

    def test_standing_distance_offset_moves_only_robot_xy(self) -> None:
        states = {
            "root_states": torch.tensor([[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            "object_states": torch.tensor([[0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            "dof_states": torch.ones(1, 2, 2),
        }

        moved, distance = _apply_standing_distance_offset(states, 0.25)

        self.assertAlmostEqual(distance, 0.75)
        torch.testing.assert_close(moved["root_states"][0, :2], torch.tensor([-0.25, 0.0]))
        torch.testing.assert_close(moved["root_states"][0, 2:], states["root_states"][0, 2:])
        torch.testing.assert_close(moved["object_states"], states["object_states"])
        torch.testing.assert_close(states["root_states"][0, :2], torch.tensor([0.0, 0.0]))

    def test_buffer_load_device_overrides_saved_config(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "config.json").write_text(
                '{"capacity": 1, "device": "cuda", "nested_key_separator": "-", '
                '"_idx": 0, "_is_full": true, "__target__": '
                '"humanoidverse.agents.buffers.transition.DictBuffer"}'
            )
            with h5py.File(path / "buffer.hdf5", "w") as handle:
                handle.create_dataset("value", data=[[1.0]])

            buffer = DictBuffer.load(path, device="cpu")

            self.assertEqual(buffer.device, "cpu")
            self.assertEqual(buffer.storage["value"].device.type, "cpu")

    def test_reward_matches_configured_formula_and_clamps_drop(self) -> None:
        aux = {
            "carry_bilateral_contact": torch.tensor([[1.0], [1.0], [0.0]]),
            "carry_lift": torch.tensor([[1.0], [0.5], [1.0]]),
            "carry_robot_upright": torch.ones(3, 1),
            "carry_upright": torch.ones(3, 1),
            "carry_drop": torch.tensor([[0.0], [1.0], [0.0]]),
        }
        reward = compute_carry_lift_upright_reward(aux)
        torch.testing.assert_close(reward, torch.tensor([[1.0], [0.375], [0.0]]))

    def test_reward_requires_every_carry_signal(self) -> None:
        with self.assertRaisesRegex(KeyError, "carry_drop"):
            compute_carry_lift_upright_reward(
                {
                    "carry_bilateral_contact": torch.ones(1, 1),
                    "carry_lift": torch.ones(1, 1),
                    "carry_robot_upright": torch.ones(1, 1),
                    "carry_upright": torch.ones(1, 1),
                }
            )


if __name__ == "__main__":
    unittest.main()
