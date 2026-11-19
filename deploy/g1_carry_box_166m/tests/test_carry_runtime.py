import numpy as np
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import tempfile
import time
import json

import joblib
import zmq

from rl_policy.carry_runtime import (
    BestCarry140935168StrongController,
    CarryActorObservation,
    CarryDataUnavailable,
    CarryMocapFrame,
    CarryPhaseController,
    CarryRuntime,
    H20Reward736PhaseController,
    HandBoxContactEstimator,
    InitialCarryScene,
    NewIntactHybridDirectPlaceController,
    OBJECT_CONTEXT_DIM,
    ONNX_INPUT_DIM,
    MAX_LOCAL_SELF_DIM,
    PRIVILEGED_STATE_DIM,
    SPHERE_LOG_ONNX_INPUT_DIM,
    STATE_DIM,
    TrackingReferenceAnchor,
    build_carry_inference_trace,
    _segment_aabb_distance,
    compute_humanoid_observations_max_numpy,
    load_fixed_goal_latent,
    load_get_up_latent,
    load_tracking_latents,
    project_z,
)
from rl_policy.lumo_mocap import FrameExtrinsic, LuMoCarrySource, RigidBodySelector
from utils.math import quat_rotate_numpy


class FakeFK:
    def hand_points(self, root_position, root_quaternion_wxyz, joint_position):
        return np.asarray([[0.5, 0.2, 0.8], [0.5, -0.2, 0.8]])


def frame(distance=0.7):
    return CarryMocapFrame.from_mapping({
        "sequence": 1,
        "timestamp_ns": 1,
        "robot_root": {"position": [0, 0, 0.8], "quaternion_wxyz": [1, 0, 0, 0]},
        "object": {"position": [distance, 0, 0.186], "quaternion_wxyz": [1, 0, 0, 0],
                   "linear_velocity_world": [0, 0, 0], "angular_velocity_world": [0, 0, 0]},
        "contacts": {"left": False, "right": True},
        "target_position_world": [2, 1, 0.186],
    })


def phase_latents():
    z = np.ones(256, dtype=np.float32)
    result = {
        name: z.copy()
        for name in (
            "approach",
            "lift",
            "transport",
            "transport_slow",
            "align",
            "lower",
            "place",
        )
    }
    result["turn"] = {-1: z.copy(), 1: z.copy()}
    result["transport_turn"] = {-1: z.copy(), 1: z.copy()}
    return result


def hybrid_phase_latents():
    result = phase_latents()
    for index, name in enumerate(
        (
            "hold",
            "direct_place",
            "direct_place_slow",
            "direct_place_lower",
            "direct_place_release",
            "direct_place_recover",
            "direct_place_stand",
        ),
        start=1,
    ):
        value = np.zeros(256, dtype=np.float32)
        value[index] = 16.0
        result[name] = value
    result["get_up"] = np.ones((1, 256), dtype=np.float32)
    return result


class CarryRuntimeTest(unittest.TestCase):
    def test_inference_trace_preserves_exact_1252d_actor_schema(self):
        actor = np.arange(499, dtype=np.float32)
        privileged = np.arange(497, dtype=np.float32) + 1000
        latent = np.arange(256, dtype=np.float32) + 2000
        actor_input = np.concatenate((actor, privileged, latent))

        trace = build_carry_inference_trace(
            actor, privileged, latent, actor_input
        )

        self.assertEqual(trace["actor_input"].shape, (1252,))
        self.assertEqual(trace["state"].shape, (98,))
        self.assertEqual(trace["last_action"].shape, (29,))
        self.assertEqual(trace["history_actor"].shape, (372,))
        self.assertEqual(trace["privileged_state"].shape, (497,))
        self.assertEqual(trace["latent"].shape, (256,))
        np.testing.assert_array_equal(
            np.concatenate(
                (
                    trace["state"],
                    trace["last_action"],
                    trace["history_actor"],
                    trace["privileged_state"],
                    trace["latent"],
                )
            ),
            trace["actor_input"],
        )

    @staticmethod
    def _deployment_calibration_runtime():
        runtime = CarryRuntime.__new__(CarryRuntime)
        runtime.deployment_calibration_enabled = True
        runtime.deployment_contact_calibration_enabled = True
        runtime.controller_profile = "intact_logmap150544384_strong"
        runtime.calibration_stage = "pelvis"
        runtime.calibration_capture_active = False
        runtime.calibration_contact_label = None
        runtime.calibration_result_path = None
        runtime.saved_calibration_source_path = None
        runtime.saved_calibration_detected = False
        runtime.saved_calibration_last_error = None
        runtime.calibration_cfg = SimpleNamespace(
            expected_standing_pelvis_height=0.80,
            min_pelvis_samples=4,
            min_box_height_samples=4,
            min_contact_samples_per_label=4,
            max_samples_per_bucket=100,
            max_default_joint_error=0.15,
            min_pelvis_upright=0.90,
            max_pelvis_height_std=0.015,
            max_pelvis_height_correction=0.25,
            min_box_upright=0.90,
            max_box_speed=0.10,
            max_box_height_std=0.010,
            max_box_height_correction=0.15,
            contact_quantile=0.90,
            no_contact_quantile=0.10,
            contact_min_separation=0.010,
            contact_exit_margin=0.015,
            contact_max_enter_distance=0.15,
        )
        runtime.calibration_samples = {
            "pelvis_height": [],
            "pelvis_upright": [],
            "box_height": [],
            "contact": [[[], []], [[], []]],
        }
        runtime.calibration_results = {}
        runtime.support_pose_joint_indices = np.arange(15, dtype=np.int64)
        controller_cfg = SimpleNamespace(
            fall_root_height=0.30,
            fall_tilted_root_height=0.50,
            get_up_root_height=0.65,
            ground_center_height=0.18,
        )
        runtime.controller = SimpleNamespace(cfg=controller_cfg)
        runtime._calibration_base_root_thresholds = {
            "fall_root_height": 0.30,
            "fall_tilted_root_height": 0.50,
            "get_up_root_height": 0.65,
        }
        runtime.subscriber = SimpleNamespace(
            target_position_world=np.asarray([1.5, 0.0, 0.18], dtype=np.float64)
        )
        runtime.contact_estimator = HandBoxContactEstimator(
            {
                "half_size": [0.175, 0.175, 0.175],
                "capsule_radius": 0.035,
                "enter_distance": 0.045,
                "exit_distance": 0.060,
                "use_hand_rigid_origins": True,
                "hand_origin_enter_distance": 0.045,
                "hand_origin_exit_distance": 0.060,
            }
        )
        return runtime

    def test_fk_only_calibration_completes_after_box_height(self):
        runtime = self._deployment_calibration_runtime()
        runtime.deployment_contact_calibration_enabled = False
        runtime.contact_estimator.use_hand_rigid_origins = False
        runtime.calibration_stage = "box_height"
        runtime.calibration_samples["box_height"] = [0.19] * 4

        accepted, message = runtime.advance_deployment_calibration(
            default_pose_active=True
        )

        self.assertTrue(accepted)
        self.assertTrue(runtime.deployment_calibration_complete)
        self.assertIn("FK capsule", message)
        self.assertEqual(
            runtime.calibration_results["contact"]["mode"], "fk_capsule_fixed"
        )
        self.assertFalse(runtime.calibration_results["contact"]["calibrated"])
        self.assertAlmostEqual(
            runtime.calibration_results["contact"]["left"]["enter_distance_m"],
            0.045,
        )

    def test_fk_only_saved_calibration_does_not_require_contact_result(self):
        runtime = self._deployment_calibration_runtime()
        runtime.deployment_contact_calibration_enabled = False
        runtime.contact_estimator.use_hand_rigid_origins = False
        payload = {
            "schema_version": 1,
            "complete": True,
            "controller_profile": runtime.controller_profile,
            "results": {
                "pelvis": {
                    "standing_height_m": 0.82,
                    "height_correction_m": 0.02,
                },
                "box_height": {"ground_center_height_m": 0.19},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            runtime.calibration_result_path = path
            accepted, message = runtime.activate_saved_deployment_calibration()

        self.assertTrue(accepted, message)
        self.assertTrue(runtime.deployment_calibration_complete)

    def test_pelvis_calibration_ignores_arm_joint_residual(self):
        runtime = self._deployment_calibration_runtime()
        mocap = frame()
        default = np.zeros(29, dtype=np.float64)
        joint = default.copy()
        joint[18] = 0.25
        metrics = {
            "root_height": 0.82,
            "pelvis_up_z": 0.99,
            "box_upright": 1.0,
            "object_speed": 0.0,
        }

        accepted, _ = runtime.start_deployment_calibration_capture(
            default_pose_active=True
        )
        self.assertTrue(accepted)
        runtime.observe_deployment_calibration(
            mocap, metrics, joint, default, default_pose_active=True
        )
        self.assertEqual(len(runtime.calibration_samples["pelvis_height"]), 1)

        joint[0] = 0.20
        runtime.observe_deployment_calibration(
            mocap, metrics, joint, default, default_pose_active=True
        )
        self.assertEqual(len(runtime.calibration_samples["pelvis_height"]), 1)

    def test_deployment_calibration_updates_thresholds_and_requires_rigid_hands(self):
        runtime = self._deployment_calibration_runtime()
        mocap = frame()
        joint = np.zeros(29, dtype=np.float64)
        pelvis_metrics = {
            "root_height": 0.82,
            "pelvis_up_z": 0.99,
            "box_upright": 1.0,
            "object_speed": 0.0,
        }
        runtime.observe_deployment_calibration(
            mocap, pelvis_metrics, joint, joint, default_pose_active=True
        )
        self.assertEqual(runtime.calibration_samples["pelvis_height"], [])
        accepted, _ = runtime.start_deployment_calibration_capture(
            default_pose_active=True
        )
        self.assertTrue(accepted)
        for _ in range(4):
            runtime.observe_deployment_calibration(
                mocap, pelvis_metrics, joint, joint, default_pose_active=True
            )
        finalized = runtime.maybe_finalize_deployment_calibration_capture(
            default_pose_active=True
        )
        self.assertIsNotNone(finalized)
        accepted, _ = finalized
        self.assertTrue(accepted)
        self.assertEqual(runtime.calibration_stage, "box_height")
        self.assertAlmostEqual(runtime.controller.cfg.fall_root_height, 0.32)
        self.assertAlmostEqual(runtime.controller.cfg.get_up_root_height, 0.67)

        box_frame = CarryMocapFrame(
            **{**mocap.__dict__, "object_position": np.asarray([0.7, 0.0, 0.19])}
        )
        accepted, _ = runtime.start_deployment_calibration_capture(
            default_pose_active=True
        )
        self.assertTrue(accepted)
        for _ in range(4):
            runtime.observe_deployment_calibration(
                box_frame, pelvis_metrics, joint, joint, default_pose_active=True
            )
        finalized = runtime.maybe_finalize_deployment_calibration_capture(
            default_pose_active=True
        )
        self.assertIsNotNone(finalized)
        accepted, _ = finalized
        self.assertTrue(accepted)
        self.assertEqual(runtime.calibration_stage, "contact")
        self.assertAlmostEqual(runtime.controller.cfg.ground_center_height, 0.19)
        self.assertAlmostEqual(runtime.subscriber.target_position_world[2], 0.19)

        # FK fallback samples are deliberately excluded from rigid-origin calibration.
        runtime.contact_estimator.last_geometry_source[:] = "fk_capsule"
        runtime.contact_estimator.last_hand_origin_surface_distance[:] = 0.08
        runtime.set_calibration_contact_label(False)
        runtime.observe_deployment_calibration(
            box_frame, pelvis_metrics, joint, joint, default_pose_active=False
        )
        self.assertEqual(runtime.calibration_samples["contact"][0][0], [])
        self.assertIsNone(runtime.calibration_contact_label)

        for label, distance in ((False, 0.08), (True, 0.02)):
            for _ in range(4):
                runtime.set_calibration_contact_label(label)
                runtime.contact_estimator.last_geometry_source[:] = "rigid_origin"
                runtime.contact_estimator.last_hand_origin_surface_distance[:] = distance
                runtime.observe_deployment_calibration(
                    box_frame, pelvis_metrics, joint, joint, default_pose_active=False
                )
                self.assertIsNone(runtime.calibration_contact_label)
            runtime.observe_deployment_calibration(
                box_frame, pelvis_metrics, joint, joint, default_pose_active=False
            )
            self.assertEqual(
                len(runtime.calibration_samples["contact"][0][int(label)]), 4
            )
        accepted, _ = runtime.advance_deployment_calibration(
            default_pose_active=False
        )
        self.assertTrue(accepted)
        self.assertTrue(runtime.deployment_calibration_complete)
        np.testing.assert_allclose(
            runtime.contact_estimator.hand_origin_enter_distance, [0.05, 0.05]
        )
        np.testing.assert_allclose(
            runtime.contact_estimator.hand_origin_exit_distance, [0.065, 0.065]
        )
        snapshot = runtime.deployment_calibration_snapshot()
        self.assertEqual(snapshot["sample_counts"]["contact"]["left"]["contact"], 4)

    def test_saved_deployment_calibration_is_validated_before_application(self):
        runtime = self._deployment_calibration_runtime()
        payload = {
            "schema_version": 1,
            "complete": True,
            "controller_profile": runtime.controller_profile,
            "results": {
                "pelvis": {
                    "standing_height_m": 0.82,
                    "height_correction_m": 0.02,
                },
                "box_height": {"ground_center_height_m": 0.19},
                "contact": {
                    "left": {"enter_distance_m": 0.04, "exit_distance_m": 0.05},
                    "right": {"enter_distance_m": 0.045, "exit_distance_m": 0.055},
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            runtime.calibration_result_path = path

            accepted, _ = runtime.activate_saved_deployment_calibration()

            self.assertTrue(accepted)
            self.assertTrue(runtime.deployment_calibration_complete)
            self.assertAlmostEqual(runtime.controller.cfg.fall_root_height, 0.32)
            self.assertAlmostEqual(runtime.controller.cfg.get_up_root_height, 0.67)
            self.assertAlmostEqual(runtime.controller.cfg.ground_center_height, 0.19)
            np.testing.assert_allclose(
                runtime.contact_estimator.hand_origin_enter_distance,
                [0.04, 0.045],
            )
            np.testing.assert_allclose(
                runtime.contact_estimator.hand_origin_exit_distance,
                [0.05, 0.055],
            )

            rejected = self._deployment_calibration_runtime()
            rejected.calibration_result_path = path
            payload["controller_profile"] = "wrong_profile"
            path.write_text(json.dumps(payload), encoding="utf-8")
            accepted, message = rejected.activate_saved_deployment_calibration()
            self.assertFalse(accepted)
            self.assertIn("profile mismatch", message)
            self.assertEqual(rejected.calibration_stage, "pelvis")

    def test_sphere_log_input_dimensions_and_max_local_self_identity(self):
        self.assertEqual(MAX_LOCAL_SELF_DIM, 463)
        self.assertEqual(PRIVILEGED_STATE_DIM, 497)
        self.assertEqual(SPHERE_LOG_ONNX_INPUT_DIM, 1252)

        positions = np.zeros((31, 3), dtype=np.float64)
        positions[:, 0] = np.arange(31, dtype=np.float64) * 0.1
        positions[0, 2] = 0.8
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (31, 1))
        linear_velocity = np.tile([1.0, 2.0, 3.0], (31, 1))
        angular_velocity = np.tile([4.0, 5.0, 6.0], (31, 1))
        observation = compute_humanoid_observations_max_numpy(
            positions, quaternions, linear_velocity, angular_velocity
        )

        self.assertEqual(observation.shape, (463,))
        self.assertAlmostEqual(float(observation[0]), 0.8)
        np.testing.assert_allclose(observation[1:4], [0.1, 0.0, -0.8])
        np.testing.assert_allclose(observation[91:97], [1, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(observation[277:280], [1, 2, 3])
        np.testing.assert_allclose(observation[370:373], [4, 5, 6])

    def test_tracking_latents_are_validated_and_projected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zs.pkl"
            joblib.dump(np.stack((np.ones(256), np.arange(1, 257))), path)
            latents = load_tracking_latents(path)
            self.assertEqual(latents.shape, (2, 256))
            np.testing.assert_allclose(np.linalg.norm(latents, axis=1), [16.0, 16.0])

            joblib.dump(np.ones((2, 255)), path)
            with self.assertRaisesRegex(ValueError, "shape"):
                load_tracking_latents(path)

    def test_get_up_latent_accepts_one_fixed_reward_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "z_get_up.pkl"
            joblib.dump(np.arange(1, 257, dtype=np.float32), path)
            latent = load_get_up_latent(path)
            self.assertEqual(latent.shape, (1, 256))
            self.assertAlmostEqual(float(np.linalg.norm(latent[0])), 16.0, places=5)

    def test_tracking_anchor_aligns_xy_yaw_but_preserves_floor_z(self):
        anchor = TrackingReferenceAnchor(
            {
                "root_position": [1.0, 2.0, 0.8],
                "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "object_position": [1.5, 2.0, 0.2],
                "object_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "target_position": [3.0, 2.0, 0.2],
                "max_object_position_error": 0.01,
                "max_object_orientation_error_deg": 1.0,
            }
        )
        yaw_90 = np.asarray([2**-0.5, 0.0, 0.0, 2**-0.5])
        live = CarryMocapFrame(
            sequence=1,
            timestamp_ns=1,
            robot_root_position=np.asarray([10.0, 20.0, 0.9]),
            robot_root_quaternion_wxyz=yaw_90,
            object_position=np.asarray([10.0, 20.5, 0.2]),
            object_quaternion_wxyz=yaw_90,
            object_linear_velocity_world=np.zeros(3),
            object_angular_velocity_world=np.zeros(3),
            hand_contact=np.zeros(2),
            target_position_world=np.zeros(3),
        )
        aligned = anchor.align(live)
        np.testing.assert_allclose(aligned.target_position_world, [10.0, 22.0, 0.2], atol=1e-7)

        anchor.reset()
        bad = CarryMocapFrame(**{**live.__dict__, "object_position": np.asarray([11.0, 20.5, 0.2])})
        with self.assertRaisesRegex(RuntimeError, "box position mismatch"):
            anchor.align(bad)

    def test_tracking_anchor_can_disable_orientation_rejection(self):
        anchor = TrackingReferenceAnchor(
            {
                "root_position": [0.0, 0.0, 0.8],
                "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "object_position": [0.5, 0.0, 0.2],
                "object_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "target_position": [1.0, 0.0, 0.2],
                "max_object_position_error": 0.01,
                "max_object_orientation_error_deg": None,
            }
        )
        live = CarryMocapFrame(
            sequence=1,
            timestamp_ns=1,
            robot_root_position=np.asarray([0.0, 0.0, 0.8]),
            robot_root_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            object_position=np.asarray([0.5, 0.0, 0.2]),
            object_quaternion_wxyz=np.asarray([0.0, 1.0, 0.0, 0.0]),
            object_linear_velocity_world=np.zeros(3),
            object_angular_velocity_world=np.zeros(3),
            hand_contact=np.zeros(2),
            target_position_world=np.zeros(3),
        )
        aligned = anchor.align(live)
        np.testing.assert_allclose(aligned.target_position_world, [1.0, 0.0, 0.2])

    def test_fixed_goal_latent_is_one_projected_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "z_goal.pkl"
            joblib.dump(np.arange(1, 257, dtype=np.float32).reshape(1, 256), path)
            latent = load_fixed_goal_latent(path)
            self.assertEqual(latent.shape, (256,))
            self.assertTrue(np.all(np.isfinite(latent)))
            self.assertAlmostEqual(float(np.linalg.norm(latent)), 16.0, places=5)

            joblib.dump({"invalid": np.ones(256, dtype=np.float32)}, path)
            with self.assertRaisesRegex(TypeError, "one vector"):
                load_fixed_goal_latent(path)

    def test_mocap_packet_and_actor_schema(self):
        observer = CarryActorObservation(np.zeros(29), FakeFK())
        obs = observer.build(frame(), np.zeros(29), np.zeros(29), [1, 0, 0, 0], np.zeros(3), np.zeros(29))
        self.assertEqual(obs["state"].shape, (STATE_DIM,))
        self.assertEqual(obs["history_actor"].shape, (372,))
        np.testing.assert_allclose(obs["state"][-16:-14], [0.0, 1.0])
        np.testing.assert_allclose(obs["state"][-14:-8], [0.5, 0.2, 0.0, 0.5, -0.2, 0.0])
        np.testing.assert_allclose(obs["state"][-2:], [1.0, 1.0])
        second = observer.build(frame(), np.ones(29), np.zeros(29), [1, 0, 0, 0], np.zeros(3), np.ones(29))
        self.assertEqual(np.count_nonzero(second["history_actor"]), 1)
        third = observer.build(frame(), np.ones(29), np.zeros(29), [1, 0, 0, 0], np.zeros(3), np.ones(29))
        self.assertGreater(np.count_nonzero(third["history_actor"]), 0)

    def test_approach_masks_all_object_context(self):
        z = np.ones(256, dtype=np.float32)
        latents = {name: z.copy() for name in ("approach", "lift", "transport", "transport_slow", "align", "lower", "place")}
        latents["turn"] = {-1: z.copy(), 1: z.copy()}
        latents["transport_turn"] = {-1: z.copy(), 1: z.copy()}
        controller = CarryPhaseController({}, latents, 0.02)
        observation = {"state": np.ones(98), "last_action": np.zeros(29), "history_actor": np.zeros(372)}
        actor, latent, _ = controller.select(observation, controller.metrics(frame()), frame())
        self.assertEqual(actor.shape[0] + latent.shape[0], ONNX_INPUT_DIM)
        self.assertTrue(np.all(actor[STATE_DIM - OBJECT_CONTEXT_DIM:STATE_DIM] == 0.0))
        self.assertAlmostEqual(float(np.linalg.norm(project_z(z))), 16.0, places=5)

    def test_fall_recovery_uses_mocap_pelvis_pose_and_distance(self):
        z = np.ones(256, dtype=np.float32)
        latents = {name: z.copy() for name in ("approach", "lift", "transport", "transport_slow", "align", "lower", "place")}
        latents["turn"] = {-1: z.copy(), 1: z.copy()}
        latents["transport_turn"] = {-1: z.copy(), 1: z.copy()}
        latents["get_up"] = np.tile(z, (20, 1))
        controller = CarryPhaseController({"fall_recovery_enabled": True}, latents, 0.02)
        far_fallen = controller.metrics(CarryMocapFrame(**{
            **frame(distance=2.0).__dict__,
            "robot_root_position": np.asarray([0.0, 0.0, 0.2]),
            "robot_root_quaternion_wxyz": np.asarray([2**-0.5, 2**-0.5, 0.0, 0.0]),
        }))

        for _ in range(9):
            controller.update(far_fallen)
        self.assertEqual(controller.phase, "approach")
        controller.update(far_fallen)
        self.assertEqual(controller.phase, "get_up")
        self.assertEqual(controller.fall_event_count, 1)

        controller.reset()
        near_fallen = dict(far_fallen, horizontal_distance=1.5)
        for _ in range(10):
            controller.update(near_fallen)
        self.assertEqual(controller.phase, "lift")

    def test_get_up_requires_stable_mocap_recovery(self):
        z = np.ones(256, dtype=np.float32)
        latents = {name: z.copy() for name in ("approach", "lift", "transport", "transport_slow", "align", "lower", "place")}
        latents["turn"] = {-1: z.copy(), 1: z.copy()}
        latents["transport_turn"] = {-1: z.copy(), 1: z.copy()}
        latents["get_up"] = np.tile(z, (20, 1))
        controller = CarryPhaseController({"fall_recovery_enabled": True}, latents, 0.02)
        controller.phase = "get_up"
        recovered = {
            "horizontal_distance": 2.0,
            "root_height": 0.70,
            "pelvis_up_z": 0.90,
        }
        for _ in range(24):
            controller.update(recovered)
        self.assertEqual(controller.phase, "get_up")
        controller.update(recovered)
        self.assertEqual(controller.phase, "approach")

    def test_bestcarry_strong_locks_get_up_even_when_box_is_near(self):
        latents = phase_latents()
        latents["get_up"] = np.ones((1, 256), dtype=np.float32)
        controller = BestCarry140935168StrongController(
            {"fall_recovery_enabled": True, "fall_detect_steps": 2},
            latents,
            0.02,
        )
        fallen = {
            "horizontal_distance": 0.2,
            "goal_distance": 1.0,
            "object_height": controller.cfg.ground_center_height,
            "object_speed": 0.0,
            "lift": 0.0,
            "root_height": 0.2,
            "pelvis_up_z": 0.0,
            "robot_upright": 0.0,
            "box_upright": 1.0,
            "bilateral_contact": False,
        }
        controller.update(fallen)
        controller.update(fallen)
        self.assertEqual(controller.phase, "get_up")
        self.assertEqual(controller.fall_event_count, 1)

        # Near-box distance must not bypass get-up before stable recovery.
        for _ in range(10):
            controller.update(fallen)
        self.assertEqual(controller.phase, "get_up")

        recovered = dict(fallen, root_height=0.8, pelvis_up_z=1.0)
        for _ in range(controller.cfg.get_up_stable_steps):
            controller.update(recovered)
        self.assertEqual(controller.phase, "lift")

    def test_bestcarry_strong_recovers_to_approach_when_box_is_far(self):
        latents = phase_latents()
        latents["get_up"] = np.ones((1, 256), dtype=np.float32)
        controller = BestCarry140935168StrongController(
            {"fall_recovery_enabled": True, "get_up_stable_steps": 2},
            latents,
            0.02,
        )
        controller.phase = "get_up"
        recovered = {
            "horizontal_distance": 1.2,
            "root_height": 0.8,
            "pelvis_up_z": 1.0,
        }
        controller.update(recovered)
        controller.update(recovered)
        self.assertEqual(controller.phase, "approach")

    def test_bestcarry_strong_uses_separate_lift_ready_debounce(self):
        latents = phase_latents()
        latents["get_up"] = np.ones((1, 256), dtype=np.float32)
        controller = BestCarry140935168StrongController(
            {
                "fall_recovery_enabled": True,
                "lift_ready_steps": 20,
                "phase_hold_steps": 5,
            },
            latents,
            0.02,
        )
        controller.phase = "lift"
        ready = {
            "horizontal_distance": 0.4,
            "goal_distance": 1.0,
            "object_height": 0.6,
            "object_speed": 0.0,
            "lift": 1.0,
            "root_height": 0.8,
            "pelvis_up_z": 1.0,
            "robot_upright": 1.0,
            "box_upright": 1.0,
            "bilateral_contact": True,
        }
        for _ in range(19):
            controller.update(ready)
        self.assertEqual(controller.phase, "lift")
        controller.update(ready)
        self.assertEqual(controller.phase, "transport")

        lost = dict(ready, bilateral_contact=False)
        for _ in range(4):
            controller.update(lost)
        self.assertEqual(controller.phase, "transport")
        controller.update(lost)
        self.assertEqual(controller.phase, "lift")

    def test_normal_place_crouch_is_not_misclassified_as_fallen(self):
        z = np.ones(256, dtype=np.float32)
        latents = {name: z.copy() for name in ("approach", "lift", "transport", "transport_slow", "align", "lower", "place")}
        latents["turn"] = {-1: z.copy(), 1: z.copy()}
        latents["transport_turn"] = {-1: z.copy(), 1: z.copy()}
        latents["get_up"] = z.reshape(1, -1)
        controller = CarryPhaseController({"fall_recovery_enabled": True}, latents, 0.02)
        controller.phase = "place"
        crouched = {
            "horizontal_distance": 0.45,
            "goal_distance": 0.15,
            "object_height": 0.70,
            "object_speed": 0.0,
            "lift": 1.0,
            "root_height": 0.471,
            "pelvis_up_z": 0.450,
            "robot_upright": 0.450,
            "box_upright": 1.0,
            "bilateral_contact": True,
        }

        for _ in range(20):
            controller.update(crouched)

        self.assertEqual(controller.phase, "place")
        self.assertEqual(controller.fall_detect_steps, 0)

    def test_h20_profile_uses_one_meter_switch_and_debounced_place_entry(self):
        controller = H20Reward736PhaseController(
            {
                "approach_switch_distance": 1.0,
                "place_align_distance": 0.25,
                "place_enter_steps": 5,
            },
            phase_latents(),
            0.02,
        )
        far = controller.metrics(frame(distance=1.2))
        controller.update(far)
        self.assertEqual(controller.phase, "approach")

        near = dict(far, horizontal_distance=0.99)
        controller.update(near)
        self.assertEqual(controller.phase, "lift")

        controller.phase = "transport"
        transport = dict(
            near,
            bilateral_contact=True,
            lift=1.0,
            robot_upright=1.0,
            goal_distance=0.20,
        )
        for _ in range(4):
            controller.update(transport)
        self.assertEqual(controller.phase, "transport")
        controller.update(transport)
        self.assertEqual(controller.phase, "place")

    def test_legacy_controller_still_enters_place_without_h20_debounce(self):
        controller = CarryPhaseController({}, phase_latents(), 0.02)
        controller.phase = "transport"
        metrics = controller.metrics(frame(distance=0.4))
        metrics.update(bilateral_contact=True, lift=1.0, goal_distance=0.30)
        controller.update(metrics)
        self.assertEqual(controller.phase, "place")

    def test_h20_profile_retries_instead_of_aborting_on_a_high_release(self):
        controller = H20Reward736PhaseController(
            {
                "ground_center_height": 0.186,
                "place_release_height": 0.08,
                "place_retry_delay_seconds": 1.0,
            },
            phase_latents(),
            0.02,
        )
        controller.phase = "place"
        high_release = {
            "horizontal_distance": 0.4,
            "goal_distance": 0.1,
            "object_height": 0.6,
            "object_speed": 0.5,
            "lift": 1.0,
            "root_height": 0.75,
            "pelvis_up_z": 1.0,
            "robot_upright": 1.0,
            "box_upright": 1.0,
            "bilateral_contact": False,
        }
        for _ in range(49):
            controller.update(high_release)
        self.assertEqual(controller.phase, "place")
        controller.update(high_release)
        self.assertEqual(controller.phase, "lift")

    def test_newintact_hybrid_enters_direct_place_without_handoff(self):
        controller = NewIntactHybridDirectPlaceController(
            {
                "approach_switch_distance": 0.70,
                "place_align_distance": 0.60,
                "place_enter_steps": 5,
            },
            hybrid_phase_latents(),
            0.02,
        )
        far = controller.metrics(frame(distance=0.71))
        controller.update(far)
        self.assertEqual(controller.phase, "approach")
        controller.update(dict(far, horizontal_distance=0.69))
        self.assertEqual(controller.phase, "lift")

        controller.phase = "transport"
        transport = dict(
            far,
            bilateral_contact=True,
            any_contact=True,
            lift=1.0,
            robot_upright=1.0,
            goal_distance=0.50,
        )
        for _ in range(5):
            controller.update(transport)
        self.assertEqual(controller.phase, "place_direct")

    def test_newintact_lift_contact_timeout_is_configured_in_seconds(self):
        controller = NewIntactHybridDirectPlaceController(
            {"lift_contact_timeout_seconds": 1.50},
            hybrid_phase_latents(),
            0.02,
        )
        self.assertEqual(controller.lift_contact_timeout_steps, 75)
        controller.phase = "lift"
        missing_contact = controller.metrics(
            frame(distance=0.4), np.asarray([0.10, 0.10])
        )
        missing_contact.update(root_height=0.8, pelvis_up_z=1.0)
        for _ in range(74):
            controller.update(missing_contact)
        self.assertEqual(controller.phase, "lift")
        controller.update(missing_contact)
        self.assertEqual(controller.phase, "approach")
        self.assertTrue(controller.approach_reacquire_active)

    def test_newintact_live_initial_pose_reorients_then_uses_distance_switch(self):
        controller = NewIntactHybridDirectPlaceController(
            {
                "initial_approach_from_live_pose": True,
                "approach_switch_distance": 0.70,
                "approach_turn_exit_deg": 5.0,
                "approach_pickup_max_bearing_deg": 8.0,
                "approach_reacquire_min_seconds": 0.30,
                "approach_reacquire_ready_steps": 5,
            },
            hybrid_phase_latents(),
            0.02,
        )
        self.assertTrue(controller.approach_reorient)
        self.assertFalse(controller.approach_reacquire_active)

        side_frame = CarryMocapFrame(
            **{
                **frame(distance=0.4).__dict__,
                "object_position": np.asarray([0.0, -0.4, 0.186]),
            }
        )
        side_metrics = controller.metrics(side_frame, np.asarray([0.10, 0.10]))
        observation = {
            "state": np.zeros(98),
            "last_action": np.zeros(29),
            "history_actor": np.zeros(372),
        }
        _, _, mode = controller.select(observation, side_metrics, side_frame)
        self.assertIn("approach_reorient", mode)
        for _ in range(controller.approach_reacquire_min_steps + 10):
            controller.update(side_metrics)
        self.assertEqual(controller.phase, "approach")

        aligned_frame = frame(distance=0.5)
        aligned_metrics = controller.metrics(
            aligned_frame, np.asarray([0.10, 0.12])
        )
        aligned_metrics.update(
            root_height=0.8,
            pelvis_up_z=1.0,
            approach_boxward_speed=0.0,
        )
        controller.select(observation, aligned_metrics, aligned_frame)
        self.assertFalse(controller.approach_reorient)
        controller.update(aligned_metrics)
        self.assertEqual(controller.phase, "lift")

    def test_newintact_direct_place_latches_release_at_requested_lift(self):
        latents = hybrid_phase_latents()
        controller = NewIntactHybridDirectPlaceController(
            {
                "direct_place_release_lift_start": 0.183,
                "direct_place_release_latch_lift": 0.06,
                "direct_place_release_latch_distance": 0.38,
            },
            latents,
            0.02,
        )
        controller.phase = "place_direct"
        metrics = controller.metrics(frame(distance=0.4))
        metrics.update(
            goal_distance=0.20,
            object_speed=0.0,
            object_vertical_speed=0.0,
            lift=0.05,
            box_upright=1.0,
            bilateral_contact=True,
            any_contact=True,
        )
        observation = {
            "state": np.zeros(98),
            "last_action": np.zeros(29),
            "history_actor": np.zeros(372),
        }
        _, latent, mode = controller.select(observation, metrics, frame())
        self.assertTrue(controller.direct_place_release_latched)
        self.assertIn("rl1", mode)
        np.testing.assert_allclose(latent, latents["direct_place_release"])

    def test_newintact_high_drop_retries_without_terminating_controller(self):
        controller = NewIntactHybridDirectPlaceController(
            {"place_retry_delay_seconds": 1.0}, hybrid_phase_latents(), 0.02
        )
        controller.phase = "place_direct"
        high_drop = {
            "horizontal_distance": 0.5,
            "goal_distance": 0.6,
            "object_height": 0.6,
            "object_speed": 0.5,
            "object_vertical_speed": -0.4,
            "lift": 1.0,
            "root_height": 0.8,
            "pelvis_up_z": 1.0,
            "robot_upright": 1.0,
            "box_upright": 1.0,
            "bilateral_contact": False,
            "any_contact": False,
        }
        for _ in range(100):
            controller.update(high_drop)
        self.assertEqual(controller.phase, "place_direct")

        settled_off_target = dict(
            high_drop,
            object_height=controller.cfg.ground_center_height,
            object_speed=0.0,
            object_vertical_speed=0.0,
        )
        for _ in range(50):
            controller.update(settled_off_target)
        self.assertEqual(controller.phase, "approach")
        self.assertTrue(controller.approach_reacquire_active)

        pregrasp = dict(
            settled_off_target,
            horizontal_distance=0.5,
            left_hand_box_gap=0.10,
            right_hand_box_gap=0.12,
            root_height=0.8,
            pelvis_up_z=1.0,
            approach_boxward_speed=0.0,
        )
        for _ in range(controller.approach_reacquire_min_steps - 1):
            controller.update(pregrasp)
        self.assertEqual(controller.phase, "approach")
        for _ in range(controller.cfg.approach_reacquire_ready_steps):
            controller.update(pregrasp)
        self.assertEqual(controller.phase, "lift")

    def test_newintact_get_up_always_finishes_before_reacquire(self):
        controller = NewIntactHybridDirectPlaceController(
            {"fall_recovery_enabled": True, "fall_detect_steps": 2},
            hybrid_phase_latents(),
            0.02,
        )
        fallen = {
            "horizontal_distance": 0.2,
            "goal_distance": 1.0,
            "object_height": controller.cfg.ground_center_height,
            "object_speed": 0.0,
            "object_vertical_speed": 0.0,
            "lift": 0.0,
            "root_height": 0.2,
            "pelvis_up_z": 0.0,
            "robot_upright": 0.0,
            "box_upright": 1.0,
            "bilateral_contact": False,
            "any_contact": False,
            "left_hand_box_gap": 0.1,
            "right_hand_box_gap": 0.1,
            "approach_boxward_speed": 0.0,
        }
        controller.update(fallen)
        controller.update(fallen)
        self.assertEqual(controller.phase, "get_up")

        recovered = dict(fallen, root_height=0.8, pelvis_up_z=1.0)
        for _ in range(controller.cfg.get_up_stable_steps):
            controller.update(recovered)
        self.assertEqual(controller.phase, "approach")
        self.assertTrue(controller.approach_reacquire_active)

    def test_initial_scene_validates_alignment_and_freezes_forward_target(self):
        geom_quat = np.asarray([0.07016202, 0.97153219, 0.22227911, 0.04236142])
        object_reference_quat = geom_quat * np.asarray([1.0, -1.0, -1.0, -1.0])
        scene = InitialCarryScene(
            {
                "expected_box_distance": 1.2,
                "box_distance_tolerance": 0.1,
                "max_box_bearing_deg": 8.0,
                "expected_box_center_height": 0.186,
                "box_center_height_tolerance": 0.03,
                "max_box_yaw_error_deg": 10.0,
                "target_forward_distance": 1.5,
            },
            geom_quat,
        )
        aligned_frame = CarryMocapFrame(
            **{
                **frame(distance=1.2).__dict__,
                "object_quaternion_wxyz": object_reference_quat,
            }
        )
        accepted = scene.apply(aligned_frame)
        np.testing.assert_allclose(accepted.target_position_world, [2.7, 0.0, 0.186])

        moved = CarryMocapFrame(
            **{
                **aligned_frame.__dict__,
                "object_position": np.asarray([1.3, 0.2, 0.186]),
            }
        )
        frozen = scene.apply(moved)
        np.testing.assert_allclose(frozen.target_position_world, [2.7, 0.0, 0.186])

        scene.reset()
        misaligned = CarryMocapFrame(
            **{
                **aligned_frame.__dict__,
                "object_position": np.asarray([0.0, 1.2, 0.186]),
            }
        )
        with self.assertRaisesRegex(CarryDataUnavailable, "initial Carry scene rejected"):
            scene.apply(misaligned)

    def test_initial_scene_freezes_configured_world_target(self):
        geom_quat = np.asarray([0.07016202, 0.97153219, 0.22227911, 0.04236142])
        object_reference_quat = geom_quat * np.asarray([1.0, -1.0, -1.0, -1.0])
        scene = InitialCarryScene(
            {
                "expected_box_distance": 1.2,
                "box_distance_tolerance": 0.1,
                "max_box_bearing_deg": 8.0,
                "expected_box_center_height": 0.186,
                "box_center_height_tolerance": 0.03,
                "max_box_yaw_error_deg": 10.0,
                "target_mode": "configured_world",
            },
            geom_quat,
        )
        target = np.asarray([1.4995, -0.0029, 0.1920])
        aligned_frame = CarryMocapFrame(
            **{
                **frame(distance=1.2).__dict__,
                "object_quaternion_wxyz": object_reference_quat,
                "target_position_world": target,
            }
        )
        accepted = scene.apply(aligned_frame)
        np.testing.assert_allclose(accepted.target_position_world, target)

        changed_upstream = CarryMocapFrame(
            **{
                **aligned_frame.__dict__,
                "target_position_world": np.asarray([9.0, 9.0, 9.0]),
            }
        )
        frozen = scene.apply(changed_upstream)
        np.testing.assert_allclose(frozen.target_position_world, target)

    def test_initial_scene_checks_physical_geom_pose_but_keeps_reference_pose(self):
        geom_position = np.asarray([0.00102584, -0.00160677, 0.00637628])
        geom_quat = np.asarray([0.07016202, 0.97153219, 0.22227911, 0.04236142])
        reference_quat = geom_quat * np.asarray([1.0, -1.0, -1.0, -1.0])
        physical_center = np.asarray([1.2, 0.0, 0.186028293])
        reference_position = physical_center - quat_rotate_numpy(
            reference_quat.reshape(1, 4), geom_position.reshape(1, 3)
        )[0]
        scene = InitialCarryScene(
            {
                "box_pose_semantics": "physical_geom",
                "expected_box_distance": 1.2,
                "box_distance_tolerance": 0.01,
                "max_box_bearing_deg": 1.0,
                "expected_box_center_height": 0.186028293,
                "box_center_height_tolerance": 0.001,
                "max_box_yaw_error_deg": 1.0,
                "target_mode": "configured_world",
            },
            geom_quat,
            geom_position,
        )
        reference_frame = CarryMocapFrame(
            **{
                **frame(distance=1.2).__dict__,
                "object_position": reference_position,
                "object_quaternion_wxyz": reference_quat,
            }
        )
        accepted = scene.apply(reference_frame)
        np.testing.assert_allclose(accepted.object_position, reference_position)
        self.assertGreater(abs(accepted.object_position[2] - physical_center[2]), 0.001)

    def test_rejects_invalid_quaternion(self):
        payload = {
            "sequence": 1, "timestamp_ns": 1,
            "robot_root": {"position": [0, 0, 0], "quaternion_wxyz": [0, 0, 0, 0]},
            "object": {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0],
                       "linear_velocity_world": [0, 0, 0], "angular_velocity_world": [0, 0, 0]},
            "contacts": {"left": False, "right": False}, "target_position_world": [0, 0, 0],
        }
        with self.assertRaisesRegex(ValueError, "zero norm"):
            CarryMocapFrame.from_mapping(payload)

    def test_oriented_box_contact_hysteresis(self):
        estimator = HandBoxContactEstimator({
            "half_size": [0.2, 0.2, 0.2],
            "enter_distance": 0.045,
            "exit_distance": 0.060,
            "capsule_radius": 1.0e-9,
        })
        mocap = frame(distance=0.0)
        mocap = CarryMocapFrame(
            **{**mocap.__dict__, "object_position": np.zeros(3), "hand_contact": np.zeros(2)}
        )
        np.testing.assert_allclose(
            estimator.update(mocap, [[[0.24, 0.0, 0.0]] * 2, [[0.25, 0.0, 0.0]] * 2]),
            [1.0, 0.0],
        )
        np.testing.assert_allclose(
            estimator.update(mocap, [[[0.255, 0.0, 0.0]] * 2, [[0.23, 0.0, 0.0]] * 2]),
            [1.0, 1.0],
        )
        np.testing.assert_allclose(
            estimator.update(mocap, [[[0.27, 0.0, 0.0]] * 2, [[0.265, 0.0, 0.0]] * 2]),
            [0.0, 0.0],
        )

    def test_hand_rigid_origins_override_fk_per_hand_and_fall_back(self):
        estimator = HandBoxContactEstimator({
            "half_size": [0.2, 0.2, 0.2],
            "enter_distance": 0.045,
            "exit_distance": 0.060,
            "capsule_radius": 1.0e-9,
            "use_hand_rigid_origins": True,
            "hand_origin_enter_distance": 0.045,
            "hand_origin_exit_distance": 0.060,
        })
        base = frame(distance=0.0)
        marker_left = CarryMocapFrame(
            **{
                **base.__dict__,
                "object_position": np.zeros(3),
                "hand_contact": np.zeros(2),
                "hand_rigid_origin_positions_world": np.asarray(
                    [[0.22, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                "hand_rigid_origin_present": np.asarray([1.0, 0.0]),
            }
        )
        np.testing.assert_allclose(
            estimator.update(marker_left, [[[0.30, 0.0, 0.0]] * 2] * 2),
            [1.0, 0.0],
        )
        self.assertEqual(estimator.last_geometry_source.tolist(), ["rigid_origin", "fk_capsule"])
        np.testing.assert_allclose(estimator.last_surface_distance, [0.02, 0.10])

        marker_right = CarryMocapFrame(
            **{
                **marker_left.__dict__,
                "hand_rigid_origin_positions_world": np.asarray(
                    [[0.0, 0.0, 0.0], [0.22, 0.0, 0.0]]
                ),
                "hand_rigid_origin_present": np.asarray([0.0, 1.0]),
            }
        )
        np.testing.assert_allclose(
            estimator.update(marker_right, [[[0.23, 0.0, 0.0]] * 2] * 2),
            [1.0, 1.0],
        )
        self.assertEqual(estimator.last_geometry_source.tolist(), ["fk_capsule", "rigid_origin"])

        no_markers = CarryMocapFrame(
            **{
                **marker_right.__dict__,
                "hand_rigid_origin_present": np.zeros(2),
            }
        )
        np.testing.assert_allclose(
            estimator.update(no_markers, [[[0.27, 0.0, 0.0]] * 2] * 2),
            [0.0, 0.0],
        )
        self.assertEqual(estimator.last_geometry_source.tolist(), ["fk_capsule", "fk_capsule"])

    def test_hand_rigid_origin_far_from_fk_is_rejected(self):
        estimator = HandBoxContactEstimator({
            "half_size": [0.2, 0.2, 0.2],
            "enter_distance": 0.045,
            "exit_distance": 0.060,
            "capsule_radius": 1.0e-9,
            "use_hand_rigid_origins": True,
            "hand_origin_max_fk_distance": 0.05,
        })
        base = frame(distance=0.0)
        mocap = CarryMocapFrame(
            **{
                **base.__dict__,
                "object_position": np.zeros(3),
                "hand_contact": np.zeros(2),
                "hand_rigid_origin_positions_world": np.asarray(
                    [[0.22, 0.0, 0.0], [0.22, 0.0, 0.0]]
                ),
                "hand_rigid_origin_present": np.ones(2),
            }
        )
        np.testing.assert_allclose(
            estimator.update(mocap, [[[0.30, 0.0, 0.0]] * 2] * 2),
            [0.0, 0.0],
        )
        self.assertEqual(estimator.last_geometry_source.tolist(), ["fk_capsule", "fk_capsule"])
        np.testing.assert_allclose(estimator.last_hand_origin_fk_distance, [0.08, 0.08])

    def test_segment_box_distance_uses_full_capsule_axis(self):
        half_size = np.asarray([0.2, 0.2, 0.2])
        self.assertEqual(
            _segment_aabb_distance(np.asarray([[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]), half_size),
            0.0,
        )
        self.assertAlmostEqual(
            _segment_aabb_distance(np.asarray([[0.3, -0.4, 0.0], [0.3, 0.4, 0.0]]), half_size),
            0.1,
        )

    def test_lumo_rigid_pose_conversion(self):
        def rigid(rigid_id, name, xyz, linear):
            return SimpleNamespace(
                Id=rigid_id,
                Name=name,
                IsTrack=True,
                QualityGrade=1,
                X=xyz[0], Y=xyz[1], Z=xyz[2],
                qw=1.0, qx=0.0, qy=0.0, qz=0.0,
                speeds=SimpleNamespace(
                    XfSpeed=linear[0], YfSpeed=linear[1], ZfSpeed=linear[2]
                ),
                palstance=SimpleNamespace(
                    fXPalstance=0.0, fYPalstance=0.0, fZPalstance=0.0
                ),
            )

        source = object.__new__(LuMoCarrySource)
        source.root_selector = RigidBodySelector(name="G1_WAIST")
        source.object_selector = RigidBodySelector(name="CARRY_BOX")
        source.root_extrinsic = FrameExtrinsic.from_mapping(
            {"position": [0.0, 0.0, -0.1]}, "root"
        )
        source.object_extrinsic = FrameExtrinsic.from_mapping(
            {"position": [0.0, 0.0, 0.1]}, "object"
        )
        source.world_translation = np.asarray([1.0, 2.0, 0.0])
        source.world_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
        source.position_scale = 0.001
        source.velocity_scale = 0.001
        source.angular_velocity_scale = 1.0
        source.angular_velocity_frame = "world"
        source.enforce_quality_grade = True
        source.max_quality_grade = 2
        source.velocity_alpha = 1.0
        source.target_position_world = np.asarray([3.0, 2.0, 0.186])
        source.filtered_root_angular_velocity = None
        source.filtered_linear_velocity = None
        source.filtered_angular_velocity = None
        sdk_frame = SimpleNamespace(
            FrameId=12,
            TimeStamp=345,
            rigidBodys=[
                rigid(1, "G1_WAIST", [0.0, 0.0, 800.0], [0.0, 0.0, 0.0]),
                rigid(2, "CARRY_BOX", [500.0, 0.0, 200.0], [100.0, 0.0, 0.0]),
            ],
        )
        payload = source.frame_to_payload(sdk_frame)
        np.testing.assert_allclose(payload["robot_root"]["position"], [1.0, 2.0, 0.7])
        np.testing.assert_allclose(payload["robot_root"]["linear_velocity_world"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(payload["robot_root"]["angular_velocity_world"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(payload["object"]["position"], [1.5, 2.0, 0.3])
        np.testing.assert_allclose(payload["object"]["linear_velocity_world"], [0.1, 0.0, 0.0])
        self.assertEqual(payload["lumo"]["object_quality_grade"], 1)

    def test_mocap_actor_inertial_measurement_replaces_onboard_imu(self):
        runtime = CarryRuntime.__new__(CarryRuntime)
        runtime.actor_inertial_source = "mocap"
        mocap = CarryMocapFrame(
            **{
                **frame().__dict__,
                "robot_root_quaternion_wxyz": np.asarray(
                    [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
                ),
                "robot_root_angular_velocity_world": np.asarray([1.0, 0.0, 0.0]),
            }
        )
        onboard = SimpleNamespace(
            root_quat_b=np.asarray([0.0, 1.0, 0.0, 0.0]),
            root_ang_vel_b=np.asarray([9.0, 8.0, 7.0]),
        )

        quaternion, omega_body = runtime.actor_inertial_measurement(mocap, onboard)

        np.testing.assert_allclose(quaternion, mocap.robot_root_quaternion_wxyz)
        np.testing.assert_allclose(omega_body, [0.0, -1.0, 0.0], atol=1.0e-7)

    def test_mocap_actor_inertial_measurement_fails_closed_without_angular_velocity(self):
        runtime = CarryRuntime.__new__(CarryRuntime)
        runtime.actor_inertial_source = "mocap"
        with self.assertRaisesRegex(
            CarryDataUnavailable, "requires fresh robot-root angular velocity"
        ):
            runtime.actor_inertial_measurement(
                frame(),
                SimpleNamespace(
                    root_quat_b=np.asarray([1.0, 0.0, 0.0, 0.0]),
                    root_ang_vel_b=np.zeros(3),
                ),
            )

    def test_lumo_hand_origins_are_optional_and_ignore_orientation(self):
        config = {
            "root_rigid_body": {"name": "G1_WAIST"},
            "object_rigid_body": {"name": "CARRY_BOX"},
            "left_hand_rigid_body": {"name": "LEFT_HAND"},
            "right_hand_rigid_body": {"name": "RIGHT_HAND"},
            "position_scale": 0.001,
            "velocity_scale": 0.001,
            "velocity_filter_alpha": 1.0,
            "max_quality_grade": 2,
            "target_position_world": [1.0, 0.0, 0.175],
            "require_calibrated": True,
            "calibration": {
                "confirmed": True,
                "lumo_world_to_ufo_world": {"translation": [1.0, 2.0, 0.0]},
            },
        }
        source = LuMoCarrySource(config, 100.0, connect=False)

        def rigid(rigid_id, name, xyz, *, tracked=True, quality=1, quaternion=(1, 0, 0, 0)):
            return SimpleNamespace(
                Id=rigid_id,
                Name=name,
                IsTrack=tracked,
                QualityGrade=quality,
                X=xyz[0],
                Y=xyz[1],
                Z=xyz[2],
                qw=quaternion[0],
                qx=quaternion[1],
                qy=quaternion[2],
                qz=quaternion[3],
                speeds=SimpleNamespace(XfSpeed=0.0, YfSpeed=0.0, ZfSpeed=0.0),
                palstance=SimpleNamespace(
                    fXPalstance=0.0, fYPalstance=0.0, fZPalstance=0.0
                ),
            )

        sdk_frame = SimpleNamespace(
            FrameId=1,
            TimeStamp=2,
            rigidBodys=[
                rigid(1, "G1_WAIST", [0.0, 0.0, 800.0]),
                rigid(2, "CARRY_BOX", [500.0, 0.0, 175.0]),
                # A zero quaternion is deliberate: origin-only hand input must
                # not parse or depend on rigid orientation.
                rigid(3, "LEFT_HAND", [300.0, 100.0, 900.0], quaternion=(0, 0, 0, 0)),
                rigid(4, "RIGHT_HAND", [300.0, -100.0, 900.0], tracked=False),
            ],
        )
        payload = source.frame_to_payload(sdk_frame)
        left = payload["hand_rigid_origins"]["left"]
        right = payload["hand_rigid_origins"]["right"]
        self.assertTrue(left["present"])
        np.testing.assert_allclose(left["position"], [1.3, 2.1, 0.9])
        self.assertFalse(right["present"])
        self.assertEqual(right["reason"], "not tracked")

        parsed = CarryMocapFrame.from_mapping(payload)
        np.testing.assert_allclose(parsed.hand_rigid_origin_present, [1.0, 0.0])
        np.testing.assert_allclose(
            parsed.hand_rigid_origin_positions_world[0], [1.3, 2.1, 0.9]
        )

    def test_lumo_physical_box_pose_round_trips_through_training_geom(self):
        from rl_policy.carry_runtime import quat_mul, quat_rotate_numpy

        geom_quat = np.asarray([0.07016202, 0.97153219, 0.22227911, 0.04236142])
        geom_pos = np.asarray([0.00102584, -0.00160677, 0.00637628])
        inverse_geom_quat = geom_quat * np.asarray([1.0, -1.0, -1.0, -1.0])
        reference_offset = np.asarray([-0.00054322, -0.00285849, 0.00598539])
        config = {
            "root_rigid_body": {"name": "G1_WAIST"},
            "object_rigid_body": {"name": "CARRY_BOX"},
            "position_scale": 0.001,
            "velocity_scale": 0.001,
            "angular_velocity_scale": 1.0,
            "angular_velocity_frame": "world",
            "velocity_filter_alpha": 1.0,
            "max_quality_grade": 2,
            "target_position_world": [1.0, 0.0, 0.175],
            "target_quaternion_world": [1.0, 0.0, 0.0, 0.0],
            "target_pose_semantics": "physical_geom",
            "require_calibrated": True,
            "calibration": {
                "confirmed": True,
                "lumo_world_to_ufo_world": {},
                "root_rigid_to_training_root": {},
                "object_rigid_to_training_object": {
                    "position": reference_offset.tolist(),
                    "quaternion_wxyz": inverse_geom_quat.tolist(),
                },
            },
        }
        source = LuMoCarrySource(config, 100.0, connect=False)

        def rigid(rigid_id, name, xyz):
            return SimpleNamespace(
                Id=rigid_id,
                Name=name,
                IsTrack=True,
                QualityGrade=1,
                X=xyz[0],
                Y=xyz[1],
                Z=xyz[2],
                qw=1.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                speeds=SimpleNamespace(XfSpeed=0.0, YfSpeed=0.0, ZfSpeed=0.0),
                palstance=SimpleNamespace(
                    fXPalstance=0.0, fYPalstance=0.0, fZPalstance=0.0
                ),
            )

        sdk_frame = SimpleNamespace(
            FrameId=1,
            TimeStamp=2,
            rigidBodys=[
                rigid(1, "G1_WAIST", [0.0, 0.0, 800.0]),
                rigid(2, "CARRY_BOX", [500.0, 0.0, 175.0]),
            ],
        )
        payload = source.frame_to_payload(sdk_frame)
        reference_position = np.asarray(payload["object"]["position"])
        reference_quaternion = np.asarray(payload["object"]["quaternion_wxyz"])
        reconstructed_position = reference_position + quat_rotate_numpy(
            reference_quaternion.reshape(1, 4), geom_pos.reshape(1, 3)
        )[0]
        reconstructed_quaternion = quat_mul(
            reference_quaternion.reshape(1, 4), geom_quat.reshape(1, 4)
        )[0]
        np.testing.assert_allclose(reconstructed_position, [0.5, 0.0, 0.175], atol=1e-8)
        np.testing.assert_allclose(
            reconstructed_quaternion / np.linalg.norm(reconstructed_quaternion),
            [1.0, 0.0, 0.0, 0.0],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            payload["target_position_world"],
            np.asarray([1.0, 0.0, 0.175]) + reference_offset,
            atol=1e-8,
        )

        ground_config = {
            **config,
            "target_position_world": [1.5, 0.0, 0.0],
            "target_pose_semantics": "physical_ground",
            "target_box_center_height": 0.175,
        }
        ground_source = LuMoCarrySource(ground_config, 100.0, connect=False)
        np.testing.assert_allclose(
            ground_source.target_position_world,
            np.asarray([1.5, 0.0, 0.175]) + reference_offset,
            atol=1e-8,
        )

    def test_lumo_object_velocity_can_be_derived_from_position_delta(self):
        config = {
            "root_rigid_body": {"name": "G1_WAIST"},
            "object_rigid_body": {"name": "CARRY_BOX"},
            "position_scale": 0.001,
            "velocity_scale": 0.001,
            "velocity_filter_alpha": 1.0,
            "root_linear_velocity_source": "position_delta",
            "object_linear_velocity_source": "position_delta",
            "position_delta_min_dt_seconds": 0.001,
            "position_delta_max_dt_seconds": 0.25,
            "position_delta_max_speed_mps": 5.0,
            "target_position_world": [1.5, 0.0, 0.175],
            "require_calibrated": True,
            "calibration": {"confirmed": True},
        }
        source = LuMoCarrySource(config, 200.0, connect=False)

        def rigid(rigid_id, name, x_mm, sdk_speed_mm_s):
            return SimpleNamespace(
                Id=rigid_id,
                Name=name,
                IsTrack=True,
                QualityGrade=1,
                X=x_mm,
                Y=0.0,
                Z=800.0 if name == "G1_WAIST" else 175.0,
                qw=1.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                speeds=SimpleNamespace(
                    XfSpeed=sdk_speed_mm_s, YfSpeed=0.0, ZfSpeed=0.0
                ),
                palstance=SimpleNamespace(
                    fXPalstance=0.0, fYPalstance=0.0, fZPalstance=0.0
                ),
            )

        def frame(frame_id, object_x_mm):
            return SimpleNamespace(
                FrameId=frame_id,
                TimeStamp=frame_id,
                rigidBodys=[
                    rigid(1, "G1_WAIST", 0.0, 0.0),
                    # Deliberately wrong SDK speed: 10 mm/s = 0.01 m/s.
                    rigid(2, "CARRY_BOX", object_x_mm, 10.0),
                ],
            )

        first = source.frame_to_payload(frame(1, 500.0), receipt_monotonic=10.0)
        second = source.frame_to_payload(frame(2, 536.0), receipt_monotonic=10.02)
        after_gap = source.frame_to_payload(frame(3, 900.0), receipt_monotonic=10.50)

        np.testing.assert_allclose(first["object"]["linear_velocity_world"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(second["object"]["linear_velocity_world"], [1.8, 0.0, 0.0])
        np.testing.assert_allclose(
            second["lumo"]["object_linear_velocity_sdk_world"], [0.01, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            second["lumo"]["object_linear_velocity_position_delta_world"],
            [1.8, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            after_gap["object"]["linear_velocity_world"], [0.0, 0.0, 0.0]
        )

    def test_lumo_poll_differentiates_only_latest_published_frame(self):
        config = {
            "root_rigid_body": {"name": "G1_WAIST"},
            "object_rigid_body": {"name": "CARRY_BOX"},
            "position_scale": 0.001,
            "velocity_scale": 0.001,
            "velocity_filter_alpha": 1.0,
            "root_linear_velocity_source": "position_delta",
            "object_linear_velocity_source": "position_delta",
            "position_delta_min_dt_seconds": 0.001,
            "position_delta_max_dt_seconds": 0.25,
            "position_delta_max_speed_mps": 5.0,
            "target_position_world": [1.5, 0.0, 0.175],
            "require_calibrated": True,
            "calibration": {"confirmed": True},
        }
        source = LuMoCarrySource(config, 200.0, connect=False)

        def rigid(rigid_id, name, x_mm):
            return SimpleNamespace(
                Id=rigid_id,
                Name=name,
                IsTrack=True,
                QualityGrade=1,
                X=x_mm,
                Y=0.0,
                Z=800.0 if name == "G1_WAIST" else 175.0,
                qw=1.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                speeds=SimpleNamespace(XfSpeed=0.0, YfSpeed=0.0, ZfSpeed=0.0),
                palstance=SimpleNamespace(
                    fXPalstance=0.0, fYPalstance=0.0, fZPalstance=0.0
                ),
            )

        def make_frame(frame_id, root_x_mm, object_x_mm):
            return SimpleNamespace(
                FrameId=frame_id,
                TimeStamp=frame_id,
                rigidBodys=[
                    rigid(1, "G1_WAIST", root_x_mm),
                    rigid(2, "CARRY_BOX", object_x_mm),
                ],
            )

        initial = make_frame(1, 0.0, 500.0)
        source.latest_payload = source.frame_to_payload(
            initial, receipt_monotonic=10.0
        )
        source.last_frame_id = 1
        source.last_receive_monotonic = 10.0

        parsed_messages = []

        class FakeFrame:
            def ParseFromString(self, message):
                parsed_messages.append(message)
                frame_id, root_x_mm, object_x_mm = (
                    float(value) for value in message.decode().split(",")
                )
                parsed = make_frame(int(frame_id), root_x_mm, object_x_mm)
                self.FrameId = parsed.FrameId
                self.TimeStamp = parsed.TimeStamp
                self.rigidBodys = parsed.rigidBodys

        class FakeSocket:
            def __init__(self):
                self.messages = [b"2,10,518", b"3,20,536"]

            def recv(self, flags=0):
                if self.messages:
                    return self.messages.pop(0)
                raise zmq.Again()

        source.proto = SimpleNamespace(Frame=FakeFrame)
        source.socket = FakeSocket()
        with patch(
            "rl_policy.lumo_mocap.time.monotonic",
            side_effect=[10.02, 10.021],
        ):
            payload = source.poll()

        self.assertEqual(parsed_messages, [b"3,20,536"])
        self.assertEqual(payload["sequence"], 3)
        np.testing.assert_allclose(
            payload["robot_root"]["linear_velocity_world"], [1.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            payload["object"]["linear_velocity_world"], [1.8, 0.0, 0.0]
        )
        controller = NewIntactHybridDirectPlaceController(
            {}, hybrid_phase_latents(), 0.02
        )
        metrics = controller.metrics(CarryMocapFrame.from_mapping(payload))
        self.assertAlmostEqual(metrics["approach_boxward_speed"], 1.0)
        self.assertGreater(
            metrics["approach_boxward_speed"],
            controller.cfg.approach_reacquire_max_boxward_speed,
        )

    def test_lumo_ignores_quality_when_disabled_but_retains_stream_timeout(self):
        config = {
            "root_rigid_body": {"name": "G1_WAIST"},
            "object_rigid_body": {"name": "CARRY_BOX"},
            "enforce_quality_grade": False,
            "max_quality_grade": 2,
            "target_position_world": [1.0, 0.0, 0.175],
            "require_calibrated": True,
            "calibration": {"confirmed": True},
        }
        source = LuMoCarrySource(config, 200.0, connect=False)

        def rigid(rigid_id, name, quality):
            return SimpleNamespace(
                Id=rigid_id,
                Name=name,
                IsTrack=True,
                QualityGrade=quality,
                X=0.0,
                Y=0.0,
                Z=800.0 if name == "G1_WAIST" else 175.0,
                qw=1.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                speeds=SimpleNamespace(XfSpeed=0.0, YfSpeed=0.0, ZfSpeed=0.0),
                palstance=SimpleNamespace(
                    fXPalstance=0.0, fYPalstance=0.0, fZPalstance=0.0
                ),
            )

        class FakeFrame:
            def ParseFromString(self, message):
                frame_id, quality = (int(value) for value in message.decode().split(","))
                self.FrameId = frame_id
                self.TimeStamp = frame_id
                self.rigidBodys = [
                    rigid(1, "G1_WAIST", quality),
                    rigid(2, "CARRY_BOX", 1),
                ]

        class FakeSocket:
            def __init__(self):
                self.messages = []

            def recv(self, flags=0):
                if self.messages:
                    return self.messages.pop(0)
                raise zmq.Again()

        source.proto = SimpleNamespace(Frame=FakeFrame)
        source.socket = FakeSocket()
        source.socket.messages.append(b"1,1")
        valid = source.poll()
        self.assertEqual(valid["sequence"], 1)

        source.socket.messages.append(b"2,3")
        quality_three = source.poll()
        self.assertEqual(quality_three["sequence"], 2)
        self.assertEqual(quality_three["lumo"]["root_quality_grade"], 3)
        self.assertIsNone(source.last_rejected_reason)

        source.last_receive_monotonic = time.monotonic() - 0.201
        with self.assertRaisesRegex(RuntimeError, "valid frame is stale"):
            source.poll()

    def test_lumo_rejects_same_root_and_object_rigid(self):
        source = object.__new__(LuMoCarrySource)
        source.root_selector = RigidBodySelector(name="SAME")
        source.object_selector = RigidBodySelector(name="SAME")
        sdk_frame = SimpleNamespace(
            rigidBodys=[SimpleNamespace(Id=1, Name="SAME", IsTrack=True)]
        )
        with self.assertRaisesRegex(RuntimeError, "same LuMo rigid body"):
            source.frame_to_payload(sdk_frame)


if __name__ == "__main__":
    unittest.main()
