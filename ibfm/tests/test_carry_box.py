from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from humanoidverse.agents.evaluations.humanoidverse_mjlab import _add_env_origin, _initial_carry_target
from humanoidverse.agents.envs.humanoidverse_mjlab import (
    _action_target_scale,
    _carry_aware_motion_start_times,
    _carry_box_upright,
    _carry_training_reset,
    _apply_object_mocap_pose_error,
    _isolate_absent_box_states,
    _lie_down_eligible,
    _reset_actions_for_object_presence,
    _should_reset_after_step,
    _validated_action_scale_overrides,
)
from humanoidverse.agents.envs.expert_motion_loader import apply_source_mixed_expert_priorities
from humanoidverse.agents.presets import build_agent_preset
from humanoidverse.envs.carry_observations import (
    CARRY_HAND_SPATIAL_DIM,
    OBJECT_PRESENT_INDEX,
    OBJECT_CONTEXT_DIM,
    build_object_context,
    compute_hand_capsule_box_surface_distance,
    compute_hand_contact_points,
    compute_hand_spatial_observation,
    compute_carry_object_observation,
    empty_object_context,
    infer_carry_phase_contact,
    infer_hand_box_contact,
    infer_hand_box_contact_from_distance,
    select_reference_hand_box_contact,
    update_hand_box_contact_hysteresis,
)
from humanoidverse.train import build_ufo_mjlab_config
from humanoidverse.tracking_inference import _target_states_from_obs


class CarryBoxTest(unittest.TestCase):
    def test_add_env_origin_offsets_positions_without_mutating_input(self) -> None:
        state = torch.tensor([0.5, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        original = state.clone()
        world_state = _add_env_origin(state, torch.tensor([10.0, 20.0, 0.0]))

        torch.testing.assert_close(world_state[:3], torch.tensor([10.5, 19.8, 0.3]))
        torch.testing.assert_close(world_state[3:], state[3:])
        torch.testing.assert_close(state, original)

    def test_initial_carry_target_accepts_static_and_time_varying_data(self) -> None:
        static_target = torch.tensor([1.0, 2.0, 3.0])
        time_varying_target = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        torch.testing.assert_close(_initial_carry_target(static_target, 3, "target"), static_target)
        torch.testing.assert_close(_initial_carry_target(time_varying_target, 3, "target"), time_varying_target[0])

    def test_box_geometry_matches_dataset_oriented_bounds(self) -> None:
        config = OmegaConf.load(Path("configs/tasks/carry_box.yaml"))
        expected_full_size = torch.tensor([0.3350532, 0.3377126, 0.3600858])
        torch.testing.assert_close(torch.tensor(config.half_size) * 2.0, expected_full_size)
        self.assertEqual(len(config.geom_pos), 3)
        self.assertEqual(len(config.geom_quat_wxyz), 4)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(torch.tensor(config.geom_quat_wxyz))), 1.0, places=5)
        self.assertEqual(float(config.disturbance_velocity), 0.0)
        self.assertEqual(list(config.absent_position), [100.0, 100.0, 0.186028293])
        self.assertEqual(float(config.mass), 1.25)
        self.assertEqual(list(config.mass_scale_range), [0.8, 1.2])
        self.assertTrue(bool(config.object_mocap_pose_randomization.enabled))
        self.assertEqual(float(config.object_mocap_pose_randomization.position_error_xyz_m), 0.02)
        self.assertEqual(float(config.object_mocap_pose_randomization.rotation_error_rpy_deg), 2.0)
        self.assertEqual(
            [float(config.mass) * float(scale) for scale in config.mass_scale_range],
            [1.0, 1.5],
        )

    def test_object_mocap_pose_error_changes_estimate_without_changing_inputs(self) -> None:
        position = torch.tensor([[1.0, 2.0, 0.5]])
        quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        linear_velocity = torch.tensor([[0.1, 0.2, 0.3]])
        angular_velocity = torch.tensor([[0.0, 0.0, 1.0]])
        original_position = position.clone()
        original_quaternion = quaternion.clone()

        observed = _apply_object_mocap_pose_error(
            position,
            quaternion,
            linear_velocity,
            angular_velocity,
            position_error_local=torch.tensor([[0.02, 0.0, 0.0]]),
            quaternion_error_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        )

        torch.testing.assert_close(observed[0], torch.tensor([[1.02, 2.0, 0.5]]))
        torch.testing.assert_close(observed[1], quaternion)
        torch.testing.assert_close(observed[2], torch.tensor([[0.1, 0.22, 0.3]]))
        torch.testing.assert_close(observed[3], angular_velocity)
        torch.testing.assert_close(position, original_position)
        torch.testing.assert_close(quaternion, original_quaternion)

    def test_object_mocap_pose_error_is_expressed_in_object_frame(self) -> None:
        half_sqrt = 2.0**-0.5
        object_quaternion = torch.tensor([[0.0, 0.0, half_sqrt, half_sqrt]])
        quaternion_error = torch.tensor([[half_sqrt, 0.0, 0.0, half_sqrt]])

        observed = _apply_object_mocap_pose_error(
            object_pos=torch.zeros(1, 3),
            object_quat_xyzw=object_quaternion,
            object_lin_vel=torch.zeros(1, 3),
            object_ang_vel=torch.zeros(1, 3),
            position_error_local=torch.tensor([[0.02, 0.0, 0.0]]),
            quaternion_error_xyzw=quaternion_error,
        )

        torch.testing.assert_close(observed[0], torch.tensor([[0.0, 0.02, 0.0]]), atol=1.0e-6, rtol=0.0)
        expected_quaternion = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        torch.testing.assert_close(observed[1], expected_quaternion, atol=1.0e-6, rtol=0.0)

    def test_default_carry_config_enables_complete_observation(self) -> None:
        config = OmegaConf.load(Path("configs/tasks/carry_box.yaml"))

        self.assertTrue(bool(config.observe_target_delta))
        self.assertTrue(bool(config.observe_object_present))
        self.assertTrue(bool(config.observe_task_id))

    def test_mixed_start_rules_only_constrain_carry_samples(self) -> None:
        sampled = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adjusted = _carry_aware_motion_start_times(
            sampled,
            object_present=torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
            object_is_lifted=torch.tensor([True, False, False, False]),
            random_draw=torch.tensor([0.1, 0.1, 0.9, 0.1]),
            random_start_probability=0.5,
        )

        torch.testing.assert_close(adjusted, torch.tensor([1.0, 0.0, 3.0, 4.0]))

    def test_absent_box_is_parked_without_changing_carry_box(self) -> None:
        object_state = torch.zeros(2, 13)
        object_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        object_state[0, :3] = torch.tensor([0.5, 0.0, 0.185])
        object_state[1, :3] = torch.tensor([0.0, 0.0, -10.0])
        result = _isolate_absent_box_states(
            object_state,
            object_present=torch.tensor([[1.0], [0.0]]),
            env_origins=torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 0.0]]),
            absent_position=[100.0, 100.0, 0.185],
        )

        torch.testing.assert_close(result[0], object_state[0])
        torch.testing.assert_close(result[1, :3], torch.tensor([110.0, 120.0, 0.185]))
        torch.testing.assert_close(result[1, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
        torch.testing.assert_close(result[1, 7:13], torch.zeros(6))

    def test_reset_actions_keep_original_lafan_zero_semantics(self) -> None:
        reference_actions = torch.tensor([[0.2, -0.3], [0.4, 0.5]])
        actions = _reset_actions_for_object_presence(reference_actions, torch.tensor([[1.0], [0.0]]))
        torch.testing.assert_close(actions, torch.tensor([[0.2, -0.3], [0.0, 0.0]]))

    def test_evaluation_does_not_replace_terminal_state_when_auto_reset_is_disabled(self) -> None:
        reset = torch.tensor([False, True])

        self.assertFalse(_should_reset_after_step(reset, is_evaluating=True, auto_reset=False))
        self.assertTrue(_should_reset_after_step(reset, is_evaluating=True, auto_reset=True))
        self.assertTrue(_should_reset_after_step(reset, is_evaluating=False, auto_reset=False))

    def test_carry_fall_and_drop_do_not_end_training_episode(self) -> None:
        base_reset = torch.tensor([False, True, False, False])
        success = torch.tensor([False, False, True, False])

        training_reset = _carry_training_reset(
            base_reset,
            success,
            is_evaluating=False,
        )

        # Index 3 represents a fallen/dropped Carry episode. Those conditions are
        # metrics only and must not reset the environment before recovery.
        torch.testing.assert_close(training_reset, torch.tensor([False, True, True, False]))
        torch.testing.assert_close(
            _carry_training_reset(base_reset, success, is_evaluating=True),
            base_reset,
        )

    def test_lie_down_initialization_only_applies_to_lafan_in_mixed_env(self) -> None:
        object_present = torch.tensor([[0.0], [1.0], [0.0], [1.0]])

        eligible = _lie_down_eligible(object_present, carry_box_enabled=True)

        torch.testing.assert_close(eligible, torch.tensor([True, False, True, False]))
        torch.testing.assert_close(
            _lie_down_eligible(object_present, carry_box_enabled=False),
            torch.ones(4, dtype=torch.bool),
        )

    def test_joint_manifest_uses_lafan_to_carry_eight_to_two_ratio(self) -> None:
        manifest = OmegaConf.load(Path("configs/data/g1_carry_lafan_mix.yaml"))
        weights = {dataset.name: float(dataset.weight) for dataset in manifest.datasets}

        self.assertEqual(weights, {"g1_carry_box": 0.2, "g1_lafan": 0.8})

    def test_latest_joint_manifest_uses_grounded_carry_without_approach(self) -> None:
        manifest = OmegaConf.load(Path("configs/data/g1_carry_omniretarget_no_foot_lafan_mix.yaml"))
        datasets = {dataset.name: dataset for dataset in manifest.datasets}

        self.assertEqual(
            {name: float(dataset.weight) for name, dataset in datasets.items()},
            {
                "g1_carry_box_omniretarget_no_foot": 0.5,
                "g1_lafan": 0.5,
            },
        )
        carry = datasets["g1_carry_box_omniretarget_no_foot"]
        self.assertIn(
            "g1_carry_box_omniretarget_no_foot_train_near10s_grounded_mujoco_hysteresis_4p5_6cm_ufo.pkl",
            str(carry.train_path),
        )
        self.assertIn(
            "g1_carry_box_omniretarget_no_foot_full_grounded_mujoco_hysteresis_4p5_6cm_ufo.pkl",
            str(carry.inference_path),
        )

    def test_expert_priorities_keep_manifest_source_mass(self) -> None:
        expert_buffer = SimpleNamespace(priorities=torch.full((4,), 0.25))
        motion_lib = SimpleNamespace(_sampling_prob=torch.tensor([0.4, 0.4, 0.1, 0.1]))

        apply_source_mixed_expert_priorities(expert_buffer, motion_lib)

        torch.testing.assert_close(expert_buffer.priorities, motion_lib._sampling_prob)
        self.assertAlmostEqual(float(expert_buffer.priorities[:2].sum()), 0.8)
        self.assertAlmostEqual(float(expert_buffer.priorities[2:].sum()), 0.2)

    def test_carry_and_motion_share_waist_scales(self) -> None:
        carry_config = OmegaConf.to_container(OmegaConf.load(Path("configs/tasks/carry_box.yaml")), resolve=True)
        hv_config = OmegaConf.create(
            {
                "robot": {
                    "dof_names": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_knee_joint"],
                    "dof_effort_limit_list": [88.0, 50.0, 50.0, 100.0],
                    "control": {
                        "action_scale": 0.25,
                        "action_rescale": True,
                        "stiffness": {"waist_yaw": 40.17924, "waist_roll": 28.50125, "waist_pitch": 40.0, "knee": 100.0},
                    },
                }
            }
        )
        robot_training = {
            "action_scale_overrides": {
                "waist_yaw_joint": 0.0334827,
                "waist_roll_joint": 0.023751,
                "waist_pitch_joint": 0.08,
            }
        }
        overrides = _validated_action_scale_overrides(hv_config.robot.dof_names, carry_config, robot_training)

        carry_scale = _action_target_scale(hv_config, overrides)
        motion_scale = _action_target_scale(hv_config, robot_training["action_scale_overrides"])

        torch.testing.assert_close(carry_scale, torch.tensor([88.0 / 1200.0, 1.0 / 24.0, 0.1, 0.25]), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(motion_scale, carry_scale)
        self.assertEqual(
            overrides,
            {"waist_yaw_joint": 0.0334827, "waist_roll_joint": 0.023751, "waist_pitch_joint": 0.08},
        )

    def test_box_upright_uses_marker_to_geometry_rotation(self) -> None:
        config = OmegaConf.load(Path("configs/tasks/carry_box.yaml"))
        marker_quat_xyzw = torch.tensor([[0.97845894, 0.19036694, 0.04372535, -0.06683361]])
        geom_quat_wxyz = torch.tensor(config.geom_quat_wxyz)
        geom_quat_xyzw = torch.cat((geom_quat_wxyz[1:], geom_quat_wxyz[:1]))

        upright = _carry_box_upright(marker_quat_xyzw, geom_quat_xyzw)

        self.assertGreater(float(upright.item()), 0.99)

    def test_object_observation_is_root_local_and_18d(self) -> None:
        base_pos = torch.tensor([[1.0, 2.0, 0.5]])
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        observation = compute_carry_object_observation(
            base_pos,
            identity,
            torch.tensor([[1.5, 2.0, 0.7]]),
            identity,
            torch.tensor([[0.1, 0.0, 0.0]]),
            torch.zeros(1, 3),
            torch.tensor([[2.0, 2.0, 0.2]]),
        )
        self.assertEqual(observation.shape, (1, 18))
        torch.testing.assert_close(observation[0, :3], torch.tensor([0.5, 0.0, 0.2]))
        torch.testing.assert_close(observation[0, -3:], torch.tensor([0.5, 0.0, -0.5]))

    def test_object_context_masks_motion_and_keeps_fixed_schema(self) -> None:
        object_observation = torch.arange(36, dtype=torch.float32).reshape(2, 18)
        hand_contact = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        hand_spatial = torch.arange(24, dtype=torch.float32).reshape(2, CARRY_HAND_SPATIAL_DIM)
        context = build_object_context(
            object_observation,
            torch.tensor([[1.0], [0.0]]),
            hand_contact,
            hand_spatial,
        )

        self.assertEqual(context.shape, (2, OBJECT_CONTEXT_DIM))
        torch.testing.assert_close(context[0, :18], object_observation[0])
        torch.testing.assert_close(context[0, 18:20], hand_contact[0])
        torch.testing.assert_close(context[0, 20:OBJECT_PRESENT_INDEX], hand_spatial[0])
        torch.testing.assert_close(context[0, OBJECT_PRESENT_INDEX:], torch.ones(2))
        torch.testing.assert_close(context[1], torch.zeros(OBJECT_CONTEXT_DIM))
        self.assertEqual(empty_object_context(torch.ones(3, 7)).shape, (3, OBJECT_CONTEXT_DIM))

        legacy = build_object_context(
            object_observation,
            torch.tensor([[1.0], [0.0]]),
            hand_contact,
            hand_spatial,
            include_hand_spatial=False,
        )
        self.assertEqual(legacy.shape, (2, 22))
        torch.testing.assert_close(legacy[0, -2:], torch.ones(2))
        self.assertEqual(
            empty_object_context(torch.ones(3, 7), include_hand_spatial=False).shape,
            (3, 22),
        )

    def test_hand_spatial_observation_is_root_local_and_relative_to_box(self) -> None:
        half_turn = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        hand_pos = torch.tensor([[[2.0, 2.0, 0.5], [0.0, 2.0, 0.5]]])
        observation = compute_hand_spatial_observation(
            torch.tensor([[1.0, 2.0, 0.5]]),
            half_turn,
            hand_pos,
            torch.tensor([[1.0, 1.0, 0.5]]),
        )

        self.assertEqual(observation.shape, (1, CARRY_HAND_SPATIAL_DIM))
        torch.testing.assert_close(observation[0, :6], torch.tensor([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(observation[0, 6:], torch.tensor([-1.0, -1.0, 0.0, 1.0, -1.0, 0.0]), atol=1e-6, rtol=0)

    def test_reference_contact_prefers_stored_labels_with_per_motion_fallback(self) -> None:
        phase = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        stored = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        selected = select_reference_hand_box_contact(
            phase,
            stored,
            torch.tensor([[1.0], [0.0], [1.0]]),
        )

        torch.testing.assert_close(selected, torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]))

    def test_object_context_ablation_preserves_schema_and_zeros_only_selected_channels(self) -> None:
        object_observation = torch.arange(18, dtype=torch.float32).reshape(1, 18)
        present = torch.ones(1, 1)
        contact = torch.ones(1, 2)
        hand_spatial = torch.arange(CARRY_HAND_SPATIAL_DIM, dtype=torch.float32).reshape(1, -1)

        no_task_id = build_object_context(
            object_observation,
            present,
            contact,
            hand_spatial,
            observe_task_id=False,
        )
        no_target_delta = build_object_context(
            object_observation,
            present,
            contact,
            hand_spatial,
            observe_target_delta=False,
        )
        no_task_or_present = build_object_context(
            object_observation,
            present,
            contact,
            hand_spatial,
            observe_object_present=False,
            observe_task_id=False,
        )

        self.assertEqual(no_task_id.shape, (1, OBJECT_CONTEXT_DIM))
        self.assertEqual(no_target_delta.shape, (1, OBJECT_CONTEXT_DIM))
        torch.testing.assert_close(
            no_task_id[0, : OBJECT_PRESENT_INDEX + 1],
            build_object_context(object_observation, present, contact, hand_spatial)[0, : OBJECT_PRESENT_INDEX + 1],
        )
        self.assertEqual(float(no_task_id[0, OBJECT_PRESENT_INDEX + 1]), 0.0)
        torch.testing.assert_close(no_target_delta[0, :15], object_observation[0, :15])
        torch.testing.assert_close(no_target_delta[0, 15:18], torch.zeros(3))
        torch.testing.assert_close(no_target_delta[0, 18:20], contact[0])
        torch.testing.assert_close(no_target_delta[0, 20:OBJECT_PRESENT_INDEX], hand_spatial[0])
        torch.testing.assert_close(no_target_delta[0, OBJECT_PRESENT_INDEX:], torch.ones(2))
        torch.testing.assert_close(
            no_task_or_present[0, :OBJECT_PRESENT_INDEX],
            torch.cat((object_observation[0], contact[0], hand_spatial[0])),
        )
        torch.testing.assert_close(no_task_or_present[0, OBJECT_PRESENT_INDEX:], torch.zeros(2))

    def test_reference_hand_contact_uses_surface_distance_hysteresis(self) -> None:
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)
        hand_pos = torch.tensor(
            [
                [[1.01, 0.0, 0.0], [1.03, 0.0, 0.0]],
                [[1.03, 0.0, 0.0], [1.01, 0.0, 0.0]],
                [[1.05, 0.0, 0.0], [1.03, 0.0, 0.0]],
            ]
        )
        contact = infer_hand_box_contact(
            hand_pos,
            torch.zeros(3, 3),
            identity,
            torch.ones(3),
            torch.zeros(3),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            enter_distance=0.02,
            exit_distance=0.04,
        )

        torch.testing.assert_close(
            contact,
            torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        )

    def test_reference_hand_contact_accounts_for_hand_collision_radius(self) -> None:
        contact = infer_hand_box_contact(
            torch.tensor([[[1.05, 0.0, 0.0], [1.06, 0.0, 0.0]]]),
            torch.zeros(1, 3),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.ones(3),
            torch.zeros(3),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            hand_contact_radius=0.035,
            enter_distance=0.02,
            exit_distance=0.04,
        )

        torch.testing.assert_close(contact, torch.tensor([[1.0, 0.0]]))

    def test_reference_hand_contact_respects_box_geom_pose(self) -> None:
        half_turn_z = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
        contact = infer_hand_box_contact(
            torch.tensor([[[1.0, 0.505, 0.0], [1.0, -0.55, 0.0]]]),
            torch.zeros(1, 3),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([0.5, 0.25, 0.5]),
            torch.tensor([1.0, 0.0, 0.0]),
            half_turn_z,
            enter_distance=0.02,
            exit_distance=0.04,
        )

        torch.testing.assert_close(contact, torch.tensor([[1.0, 0.0]]))

    def test_hand_contact_points_apply_body_rotation(self) -> None:
        half_turn_z = torch.tensor([[[0.0, 0.0, 2**-0.5, 2**-0.5]]]).repeat(1, 2, 1)
        points = compute_hand_contact_points(
            torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]]),
            half_turn_z,
            torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        )

        torch.testing.assert_close(
            points,
            torch.tensor([[[1.0, 2.1, 3.0], [0.0, 0.2, 0.0]]]),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_hand_capsule_box_distance_uses_full_segment(self) -> None:
        identity_hands = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
        distance = compute_hand_capsule_box_surface_distance(
            torch.zeros(1, 2, 3),
            identity_hands,
            torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                ]
            ),
            torch.tensor([[1.5, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([0.25, 0.25, 0.25]),
            torch.zeros(3),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            hand_contact_radius=0.10,
        )

        torch.testing.assert_close(distance, torch.tensor([[0.15, 1.15]]), atol=1e-6, rtol=1e-6)

    def test_hand_capsule_box_distance_uses_closest_capsule_per_hand(self) -> None:
        identity_hands = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
        segments = torch.tensor(
            [
                [[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]],
            ]
        )
        distance = compute_hand_capsule_box_surface_distance(
            torch.zeros(1, 2, 3),
            identity_hands,
            segments,
            torch.tensor([[1.5, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([0.25, 0.25, 0.25]),
            torch.zeros(3),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            hand_contact_radius=0.10,
        )

        torch.testing.assert_close(distance, torch.tensor([[0.15, 0.15]]), atol=1e-6, rtol=1e-6)

    def test_hand_capsule_box_distance_respects_body_and_box_rotation(self) -> None:
        half_turn_z = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
        distance = compute_hand_capsule_box_surface_distance(
            torch.tensor([[[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]),
            half_turn_z.reshape(1, 1, 4).repeat(1, 2, 1),
            torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ]
            ),
            torch.tensor([[1.0, 1.5, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([0.25, 0.50, 0.25]),
            torch.zeros(3),
            half_turn_z,
            hand_contact_radius=0.10,
        )

        torch.testing.assert_close(distance[0, 0], torch.tensor(0.15), atol=1e-6, rtol=1e-6)
        self.assertGreater(float(distance[0, 1]), 2.0)

    def test_capsule_distance_hysteresis_has_offline_and_online_parity(self) -> None:
        distance = torch.tensor([[0.05, 0.01], [0.03, 0.03], [0.01, 0.05]])
        offline = infer_hand_box_contact_from_distance(distance, enter_distance=0.02, exit_distance=0.04)
        online = torch.zeros(2, dtype=torch.bool)
        online_frames = []
        for frame_distance in distance:
            online = update_hand_box_contact_hysteresis(
                frame_distance,
                online,
                enter_distance=0.02,
                exit_distance=0.04,
            )
            online_frames.append(online)

        torch.testing.assert_close(torch.stack(online_frames).to(offline.dtype), offline)

    def test_carry_phase_contact_uses_box_bottom_clearance_without_hysteresis(self) -> None:
        object_pos = torch.tensor([[0.0, 0.0, 0.105], [0.0, 0.0, 0.1051], [0.0, 0.0, 0.10]])
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)
        contact = infer_carry_phase_contact(
            object_pos,
            identity,
            torch.ones(3, 1),
            torch.tensor([0.1, 0.1, 0.1]),
            torch.zeros(3),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            ground_surface_height=0.0,
            lift_clearance=0.005,
        )

        torch.testing.assert_close(contact, torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]))

    def test_carry_phase_contact_uses_oriented_box_lowest_point(self) -> None:
        quarter_turn_y = torch.tensor([0.0, 2**-0.5, 0.0, 2**-0.5])
        contact = infer_carry_phase_contact(
            torch.tensor([[0.0, 0.0, 0.16], [0.0, 0.0, 0.21]]),
            quarter_turn_y.repeat(2, 1),
            torch.ones(2, 1),
            torch.tensor([0.2, 0.1, 0.1]),
            torch.zeros(3),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            ground_surface_height=0.0,
            lift_clearance=0.005,
        )

        torch.testing.assert_close(contact, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))

    def test_carry_phase_contact_masks_object_free_samples(self) -> None:
        contact = infer_carry_phase_contact(
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.zeros(1, 1),
            torch.tensor([0.1, 0.1, 0.1]),
            torch.zeros(3),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )

        torch.testing.assert_close(contact, torch.zeros(1, 2))

    def test_fb_carry_preset_keeps_original_ufo_aux_rewards(self) -> None:
        config = build_agent_preset(
            agent="fb",
            device="cpu",
            compile=False,
            update_z_every_step=100,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
            wandb_project="test",
            carry_box=True,
        )["agent_cfg"]
        expected_safety_scales = {
            "penalty_action_rate": -0.1,
            "penalty_feet_ori": -0.4,
            "penalty_ankle_roll": -4.0,
            "limits_dof_pos": -10.0,
            "penalty_slippage": -4.0,
            "penalty_undesired_contact": -1.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        }
        for reward_name, scale in expected_safety_scales.items():
            self.assertEqual(config.aux_rewards_scaling[reward_name], scale)
        self.assertEqual(set(config.aux_rewards), set(expected_safety_scales))
        self.assertFalse(any(name.startswith("carry_") for name in config.aux_rewards))
        self.assertFalse(any(name.startswith("carry_") for name in config.aux_rewards_scaling))
        self.assertEqual(config.train.reg_coeff, 0.05)
        self.assertEqual(config.train.reg_coeff_aux, 0.02)

        lafan_config = build_agent_preset(
            agent="fb",
            device="cpu",
            compile=False,
            update_z_every_step=100,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
            wandb_project="test",
            carry_box=False,
        )["agent_cfg"]
        self.assertEqual(lafan_config.aux_rewards_scaling["penalty_slippage"], -2.0)

    def test_carry_training_keeps_lafan_recovery_curriculum_without_pushes(self) -> None:
        config = build_ufo_mjlab_config(
            device="cpu",
            work_dir="runs/test_carry_config",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            task="carry_box",
        )
        self.assertIn("env.config.lie_down_init=True", config.env.hydra_overrides)
        self.assertIn("env.config.lie_down_init_prob=0.3", config.env.hydra_overrides)
        self.assertNotIn("env.config.lie_down_init=False", config.env.hydra_overrides)
        self.assertIn("domain_rand.push_robots=False", config.env.hydra_overrides)
        self.assertEqual(float(config.env.carry_box["disturbance_velocity"]), 0.0)

    def test_smoke_crosses_the_optimizer_path(self) -> None:
        config = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo-smoke-config",
            num_envs=16,
            num_env_steps=2048,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            disable_eval_prioritization=True,
            task="carry_box",
        )
        self.assertEqual(config.num_seed_steps, 0)
        self.assertLessEqual(config.update_agent_every, config.num_env_steps)
        self.assertEqual(config.num_agent_updates, 1)

    def test_carry_dr_profile_enables_requested_robot_randomization(self) -> None:
        config = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo-carry-dr-config",
            num_envs=16,
            num_env_steps=2048,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            disable_eval_prioritization=True,
            task="carry_box",
            carry_config="configs/tasks/carry_box_mass0p7_range0p5_1p0_pd_delay_rfi_footmu0p4_1p25.yaml",
        )
        dr = config.env.carry_box["domain_rand_overrides"]
        self.assertEqual(dr["kp_range"], [0.75, 1.25])
        self.assertEqual(dr["kd_range"], [0.75, 1.25])
        self.assertEqual(dr["ctrl_delay_step_range"], [0, 1])
        self.assertEqual(dr["rfi_lim"], 0.1)
        self.assertEqual(dr["friction_range"], [0.4, 1.25])
        self.assertEqual(dr["friction_geom_names"], "(left|right)_foot[1-7]_collision")
        self.assertEqual(dr["terrain_friction"], [0.4, 0.005, 0.0001])

    def test_tracking_target_states_include_carry_object(self) -> None:
        obs = {
            "ref_body_pos": torch.zeros(2, 1, 3),
            "ref_body_rots": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * 2),
            "ref_body_vels": torch.zeros(2, 1, 3),
            "ref_body_angular_vels": torch.zeros(2, 1, 3),
            "dof_pos": torch.zeros(2, 29),
            "ref_dof_vel": torch.zeros(2, 29),
            "object_pos": torch.tensor([[0.5, 0.0, 0.185], [0.6, 0.0, 0.2]]),
            "object_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2),
            "object_lin_vel": torch.zeros(2, 3),
            "object_ang_vel": torch.zeros(2, 3),
            "object_target_pos": torch.tensor([[1.5, 0.0, 0.185]] * 2),
            "object_target_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2),
            "object_present": torch.ones(2, 1),
        }

        target_states = _target_states_from_obs(obs, "cpu", num_dof=29)

        self.assertEqual(target_states["object_states"].shape, (1, 13))
        torch.testing.assert_close(target_states["object_states"][0, :3], obs["object_pos"][0])
        torch.testing.assert_close(target_states["target_object_pos"][0], obs["object_target_pos"][0])
        torch.testing.assert_close(target_states["object_present"], torch.ones(1, 1))

    def test_tracking_renderer_receives_box_geometry_pose(self) -> None:
        source = Path("humanoidverse/tracking_inference.py").read_text()
        self.assertIn('env.carry_box_cfg.get("geom_pos"', source)
        self.assertIn('env.carry_box_cfg.get("geom_quat_wxyz"', source)
        self.assertIn("box_geom_pos=box_geom_pos", source)
        self.assertIn("box_geom_quat_wxyz=box_geom_quat_wxyz", source)


if __name__ == "__main__":
    unittest.main()
