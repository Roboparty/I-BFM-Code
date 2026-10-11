from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from humanoidverse.carry_goal_sim2sim import (
    BOX_BODY_NAME,
    BOX_GEOM_NAME,
    BOX_JOINT_NAME,
    ActorObservation,
    _actuator_indices,
    _build_model,
    _frame_or_static,
    _joint_addresses,
    _load_box_config,
    _match_joint_parameter,
    _patch_mode15_joint_physics,
    _physical_hand_contacts,
)
from humanoidverse.carry_phased_sim2sim import (
    CapsuleBoxContactHysteresis,
    CarryDirectPlaceController,
    PhaseController,
    _align_initial_box_yaw_with_robot,
)
from humanoidverse.get_up_reward_inference import compute_get_up_reward
from humanoidverse.utils.robot_spec import load_robot_training_spec

ROBOT_CONFIG = Path("configs/robots/g1_29dof.yaml")
BOX_CONFIG = Path("configs/tasks/carry_box.yaml")


def _phase_controller() -> PhaseController:
    args = SimpleNamespace(
        place_settle_seconds=0.2,
        place_lower_seconds=1.0,
        place_release_seconds=2.0,
        place_retry_delay_seconds=1.0,
        place_max_attempt_seconds=10.0,
        approach_switch_distance=0.75,
        hold_switch_lift=0.5,
        hold_release_lift=0.25,
        hold_switch_upright=0.85,
        phase_hold_steps=5,
        place_align_distance=0.25,
        place_enter_steps=5,
        place_drift_distance=0.35,
        place_drift_steps=10,
        place_release_height=0.08,
        place_success_distance=0.20,
        place_success_height=0.05,
        place_success_speed=0.15,
        place_success_upright=0.85,
        place_success_steps=25,
        direct_place_success_distance=0.30,
        place_abort_distance=0.80,
        get_up_distance=1.5,
        fall_upright_threshold=0.20,
        fall_root_height=0.30,
        fall_tilted_root_height=0.50,
        fall_detect_steps=10,
        get_up_upright_threshold=0.85,
        get_up_root_height=0.65,
        get_up_stable_steps=25,
    )
    return PhaseController(args, dt=0.02)


def _phase_metrics(goal_distance: float) -> dict[str, float | bool]:
    return {
        "horizontal_distance": 0.5,
        "goal_distance": goal_distance,
        "object_height": 0.5,
        "object_speed": 0.0,
        "lift": 1.0,
        "root_height": 0.8,
        "pelvis_up_z": 1.0,
        "robot_upright": 1.0,
        "box_upright": 1.0,
        "bilateral_contact": True,
    }


def _carry_direct_controller() -> CarryDirectPlaceController:
    return CarryDirectPlaceController(_phase_controller().args, dt=0.02)


def _direct_place_metrics(
    goal_distance: float,
    *,
    horizontal_distance: float,
) -> dict[str, float | bool]:
    return _phase_metrics(goal_distance) | {
        "horizontal_distance": horizontal_distance,
        "object_height": 0.185,
        "object_speed": 0.0,
        "lift": 0.0,
        "bilateral_contact": False,
        "any_contact": False,
    }


def test_carry_direct_complete_displacement_resumes_lift_and_clears_release() -> None:
    controller = _carry_direct_controller()
    controller.phase = "complete"
    controller.direct_place_release_latched = True
    displaced = _direct_place_metrics(1.0, horizontal_distance=0.5)

    for _ in range(controller.place_retry_steps - 1):
        controller.update(displaced, goal_height=0.185)
    assert controller.phase == "complete"

    controller.update(displaced, goal_height=0.185)

    assert controller.phase == "lift"
    assert not controller.direct_place_release_latched
    assert controller.recovery_count == 1


def test_carry_direct_complete_far_displacement_resumes_approach() -> None:
    controller = _carry_direct_controller()
    controller.phase = "complete"
    controller.direct_place_release_latched = True
    displaced = _direct_place_metrics(1.0, horizontal_distance=1.5)

    for _ in range(controller.place_retry_steps):
        controller.update(displaced, goal_height=0.185)

    assert controller.phase == "approach"
    assert not controller.direct_place_release_latched
    assert controller.recovery_count == 1


def test_carry_direct_retries_released_box_that_settled_off_target() -> None:
    controller = _carry_direct_controller()
    controller.phase = "place_direct"
    controller.direct_place_release_latched = True
    missed = _direct_place_metrics(0.5, horizontal_distance=0.5)

    for _ in range(controller.place_retry_steps):
        controller.update(missed, goal_height=0.185)

    assert controller.phase == "lift"
    assert not controller.direct_place_release_latched
    assert controller.recovery_count == 1


def test_carry_direct_place_fall_enters_get_up_and_clears_release() -> None:
    controller = _carry_direct_controller()
    controller.phase = "place_direct"
    controller.direct_place_release_latched = True
    fallen = _direct_place_metrics(0.5, horizontal_distance=0.5) | {
        "root_height": 0.2,
        "pelvis_up_z": 0.0,
        "robot_upright": 0.0,
    }

    for _ in range(controller.args.fall_detect_steps - 1):
        controller.update(fallen, goal_height=0.185)
    assert controller.phase == "place_direct"

    controller.update(fallen, goal_height=0.185)

    assert controller.phase == "get_up"
    assert controller.fall_event_count == 1
    assert not controller.direct_place_release_latched


def test_approach_turn_uses_hysteresis_and_slew_before_returning_to_straight() -> None:
    class Policy:
        @staticmethod
        def project_z(latent: torch.Tensor) -> torch.Tensor:
            return latent

    controller = _phase_controller()
    controller.args.heading_blend_full_deg = 60.0
    controller.args.max_turn_blend = 0.90
    controller.args.approach_turn_enter_deg = 12.0
    controller.args.approach_turn_exit_deg = 5.0
    controller.args.approach_turn_blend_seconds = 0.04
    controller.args.mask_approach_object_context = False
    observation = {"state": torch.zeros(1, 98)}
    z = {
        "approach": torch.zeros(1, 1),
        "turn": {1: torch.ones(1, 1), -1: -torch.ones(1, 1)},
    }

    def command(bearing_deg: float) -> tuple[float, str]:
        radians = np.deg2rad(bearing_deg)
        _obs, latent, mode = controller.latent(
            Policy(),
            observation,
            {},
            base_rotation=np.eye(3),
            box_direction=np.asarray([np.cos(radians), np.sin(radians)]),
            goal_direction=np.ones(2),
            z=z,
        )
        return float(latent.item()), mode

    first, first_mode = command(20.0)
    full, full_mode = command(18.0)
    exit_first, exit_mode = command(3.0)
    straight, straight_mode = command(-3.0)
    below_enter, below_enter_mode = command(-10.0)

    assert np.isclose(first, 0.45)
    assert np.isclose(full, 0.90)
    assert first_mode == full_mode == "forward_left"
    assert np.isclose(exit_first, 0.45)
    assert exit_mode == "forward_left"
    assert straight == 0.0
    assert straight_mode == below_enter_mode == "forward"
    assert below_enter == 0.0


def test_sim2sim_model_has_box_floor_and_all_actuators():
    training = load_robot_training_spec(ROBOT_CONFIG)
    model = _build_model(Path(training.robot.xml_path), _load_box_config(BOX_CONFIG), render_size=64)
    joint_names = list(training.robot.control_joint_names)

    qpos, qvel = _joint_addresses(model, joint_names)
    actuators = _actuator_indices(model, joint_names)
    _patch_mode15_joint_physics(model, joint_names)

    assert model.nq == training.robot.nq + 7
    assert model.nv == training.robot.nv + 6
    assert qpos.shape == qvel.shape == actuators.shape == (29,)
    assert model.body(BOX_BODY_NAME).id > 0
    assert model.geom(BOX_GEOM_NAME).id >= 0
    assert np.all(model.dof_armature[qvel] > 0.0)
    assert np.allclose(model.dof_frictionloss[qvel], 0.01)


def test_initial_box_yaw_can_align_with_robot():
    training = load_robot_training_spec(ROBOT_CONFIG)
    model = _build_model(Path(training.robot.xml_path), _load_box_config(BOX_CONFIG), render_size=64)
    data = mujoco.MjData(model)
    yaw = -0.75
    data.qpos[3:7] = np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    box_joint_id = model.joint(BOX_JOINT_NAME).id
    box_qpos_address = int(model.jnt_qposadr[box_joint_id])
    box_qvel_address = int(model.jnt_dofadr[box_joint_id])
    data.qpos[box_qpos_address + 3 : box_qpos_address + 7] = np.asarray(
        [0.5, 0.5, 0.5, 0.5]
    )
    data.qvel[box_qvel_address : box_qvel_address + 6] = 1.0

    _align_initial_box_yaw_with_robot(
        model,
        data,
        qpos_address=box_qpos_address,
        qvel_address=box_qvel_address,
        enabled=True,
    )

    expected_rotation = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    box_geom_id = model.geom(BOX_GEOM_NAME).id
    np.testing.assert_allclose(
        data.geom_xmat[box_geom_id].reshape(3, 3),
        expected_rotation,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(data.qvel[box_qvel_address : box_qvel_address + 6], 0.0)


def test_actor_observation_matches_checkpoint_schema():
    observer = ActorObservation(
        device=torch.device("cpu"),
        default_dof_pos=np.zeros(29),
        target_object_pos=np.asarray([1.0, 0.0, 0.5]),
        hand_contact_offsets=np.zeros((2, 3)),
        include_hand_spatial=True,
    )
    observation = observer.build(
        dof_pos=np.zeros(29),
        dof_vel=np.zeros(29),
        base_pos=np.asarray([0.0, 0.0, 0.8]),
        base_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        base_ang_vel_world=np.zeros(3),
        object_pos=np.asarray([0.5, 0.0, 0.185]),
        object_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        object_lin_vel=np.zeros(3),
        object_ang_vel=np.zeros(3),
        hand_body_pos=np.asarray([[0.4, 0.2, 0.5], [0.4, -0.2, 0.5]]),
        hand_body_quat_wxyz=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)),
        hand_contact=np.asarray([1.0, 0.0]),
        current_action=np.zeros(29),
    )

    assert observation["state"].shape == (1, 98)
    assert observation["last_action"].shape == (1, 29)
    assert observation["history_actor"].shape == (1, 372)
    assert torch.equal(observation["state"][0, 82:84], torch.tensor([1.0, 0.0]))
    assert torch.equal(observation["state"][0, -2:], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(
        observation["state"][0, 84:90],
        torch.tensor([0.4, 0.2, -0.3, 0.4, -0.2, -0.3]),
    )
    assert torch.count_nonzero(observation["history_actor"]) == 0


def test_contact_reader_uses_current_mujoco_contacts():
    training = load_robot_training_spec(ROBOT_CONFIG)
    model = _build_model(Path(training.robot.xml_path), _load_box_config(BOX_CONFIG), render_size=64)
    data = mujoco.MjData(model)
    result = _physical_hand_contacts(
        model,
        data,
        box_geom_id=model.geom(BOX_GEOM_NAME).id,
        hand_body_ids=tuple(model.body(name).id for name in training.robot.hands),
    )
    assert np.array_equal(result, np.zeros(2, dtype=np.float32))


def test_capsule_contact_hysteresis_reads_current_mujoco_geometry():
    training = load_robot_training_spec(ROBOT_CONFIG)
    box_config = _load_box_config(BOX_CONFIG)
    model = _build_model(Path(training.robot.xml_path), box_config, render_size=64)
    data = mujoco.MjData(model)
    box_joint_id = model.joint(BOX_JOINT_NAME).id
    box_qpos_address = int(model.jnt_qposadr[box_joint_id])
    data.qpos[:3] = [0.0, 0.0, 0.8]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[box_qpos_address : box_qpos_address + 3] = [1.0, 0.0, 0.185]
    data.qpos[box_qpos_address + 3 : box_qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    estimator = CapsuleBoxContactHysteresis(
        box_config,
        hand_body_ids=tuple(model.body(name).id for name in training.robot.hands),
        box_qpos_address=box_qpos_address,
        enter_distance=0.045,
        exit_distance=0.060,
    )
    contact, surface_distance = estimator.update(data)
    assert contact.shape == surface_distance.shape == (2,)
    assert np.all(np.isfinite(surface_distance))
    assert np.array_equal(contact, np.zeros(2, dtype=np.float32))


def test_place_entry_requires_consecutive_steps_inside_threshold():
    controller = _phase_controller()
    controller.phase = "transport"

    for _ in range(4):
        controller.update(_phase_metrics(0.24), goal_height=0.185)
    controller.update(_phase_metrics(0.26), goal_height=0.185)
    for _ in range(4):
        controller.update(_phase_metrics(0.24), goal_height=0.185)

    assert controller.phase == "transport"
    assert controller.place_enter_steps == 4
    controller.update(_phase_metrics(0.24), goal_height=0.185)
    assert controller.phase == "place"
    assert controller.place_enter_steps == 0


def test_fallen_near_box_must_finish_get_up_before_lift():
    controller = _phase_controller()
    fallen = _phase_metrics(1.0)
    fallen.update(root_height=0.2, pelvis_up_z=0.0, horizontal_distance=0.5)

    for _ in range(controller.args.fall_detect_steps):
        controller.update(fallen, goal_height=0.185)

    assert controller.phase == "get_up"
    for _ in range(controller.args.get_up_stable_steps + 5):
        controller.update(fallen, goal_height=0.185)
    assert controller.phase == "get_up"


def test_recovered_get_up_uses_approach_distance_for_next_phase():
    near = _phase_controller()
    near.phase = "get_up"
    near_metrics = _phase_metrics(1.0)
    near_metrics["horizontal_distance"] = 0.5
    for _ in range(near.args.get_up_stable_steps):
        near.update(near_metrics, goal_height=0.185)
    assert near.phase == "lift"

    far = _phase_controller()
    far.phase = "get_up"
    far_metrics = _phase_metrics(1.0)
    far_metrics["horizontal_distance"] = 1.0
    for _ in range(far.args.get_up_stable_steps):
        far.update(far_metrics, goal_height=0.185)
    assert far.phase == "approach"


def test_place_exit_requires_consecutive_pre_release_drift():
    controller = _phase_controller()
    controller.phase = "place"

    for _ in range(9):
        controller.update(_phase_metrics(0.36), goal_height=0.185)
    controller.update(_phase_metrics(0.34), goal_height=0.185)
    for _ in range(9):
        controller.update(_phase_metrics(0.36), goal_height=0.185)

    assert controller.phase == "place"
    assert controller.place_drift_steps == 9
    controller.update(_phase_metrics(0.36), goal_height=0.185)
    assert controller.phase == "transport"
    assert controller.place_drift_steps == 0


def test_place_drift_does_not_cancel_release_in_progress():
    controller = _phase_controller()
    controller.phase = "place"
    controller.release_started_step = 0

    for _ in range(20):
        controller.update(_phase_metrics(0.36), goal_height=0.185)

    assert controller.phase == "place"
    assert controller.place_drift_steps == 0


def test_complete_latent_blends_directly_from_place_to_stand():
    controller = _phase_controller()
    controller.phase = "complete"
    controller.phase_steps = controller.place_release_steps // 2

    class IdentityPolicy:
        @staticmethod
        def project_z(value):
            return value

    observation = {"state": torch.zeros(1, 1)}
    z = {
        "place": torch.zeros(1, 4),
        "stand": torch.ones(1, 4),
        "get_up": torch.full((1, 4), 1000.0),
    }
    _, midpoint, mode = controller.latent(
        IdentityPolicy(),
        observation,
        {},
        base_rotation=np.eye(3),
        box_direction=np.zeros(2),
        goal_direction=np.zeros(2),
        z=z,
    )

    assert mode == "placed_verify"
    assert torch.all(midpoint > 0.0)
    assert torch.all(midpoint < 1.0)

    controller.phase_steps = controller.place_release_steps
    _, final, _ = controller.latent(
        IdentityPolicy(),
        observation,
        {},
        base_rotation=np.eye(3),
        box_direction=np.zeros(2),
        goal_direction=np.zeros(2),
        z=z,
    )
    torch.testing.assert_close(final, z["stand"])


def test_legacy_736_complete_keeps_place_latent():
    controller = _phase_controller()
    controller.args.legacy_736_flow = True
    controller.phase = "complete"
    controller.phase_steps = controller.place_release_steps

    class IdentityPolicy:
        @staticmethod
        def project_z(value):
            return value

    observation = {"state": torch.zeros(1, 1)}
    z = {
        "place": torch.zeros(1, 4),
        "stand": torch.ones(1, 4),
    }
    _, latent, mode = controller.latent(
        IdentityPolicy(),
        observation,
        {},
        base_rotation=np.eye(3),
        box_direction=np.zeros(2),
        goal_direction=np.zeros(2),
        z=z,
    )

    assert mode == "placed_verify"
    torch.testing.assert_close(latent, z["place"])


def test_legacy_736_place_completion_does_not_require_release_gate():
    controller = _phase_controller()
    controller.args.legacy_736_flow = True
    controller.phase = "place"
    settled = _phase_metrics(0.10) | {
        "object_height": 0.185,
        "lift": 0.0,
        "bilateral_contact": False,
    }

    for _ in range(controller.args.place_success_steps):
        controller.update(settled, goal_height=0.185)

    assert controller.phase == "complete"
    assert controller.release_started_step is None


def test_legacy_736_get_up_near_box_returns_directly_to_lift():
    controller = _phase_controller()
    controller.args.legacy_736_flow = True
    controller.phase = "get_up"

    controller.update(_phase_metrics(0.5), goal_height=0.185)

    assert controller.phase == "lift"


def test_place_accepts_debounced_settled_ground_release_when_enabled():
    controller = _phase_controller()
    controller.args.accept_settled_ground_release = True
    controller.phase = "place"
    settled = _phase_metrics(0.10) | {
        "object_height": 0.185,
        "lift": 0.0,
        "bilateral_contact": False,
    }

    for _ in range(controller.args.place_success_steps - 1):
        controller.update(settled, goal_height=0.185)
    assert controller.phase == "place"
    assert controller.release_started_step is None

    controller.update(settled, goal_height=0.185)
    assert controller.phase == "complete"


def test_place_rejects_settled_ground_release_by_default_without_release_gate():
    controller = _phase_controller()
    controller.phase = "place"
    settled = _phase_metrics(0.10) | {
        "object_height": 0.185,
        "lift": 0.0,
        "bilateral_contact": False,
    }

    for _ in range(controller.args.place_success_steps):
        controller.update(settled, goal_height=0.185)

    assert controller.phase == "place"
    assert controller.placed_steps == 0


def test_fall_detection_uses_debounce_then_selects_far_get_up():
    controller = _phase_controller()
    fallen = _phase_metrics(1.0) | {
        "horizontal_distance": 2.0,
        "root_height": 0.2,
        "pelvis_up_z": 0.1,
        "robot_upright": 0.1,
    }

    for _ in range(9):
        controller.update(fallen, goal_height=0.185)
    assert controller.phase == "approach"

    controller.update(fallen, goal_height=0.185)
    assert controller.phase == "get_up"
    assert controller.fall_event_count == 1


def test_fallen_near_box_selects_get_up():
    controller = _phase_controller()
    fallen = _phase_metrics(1.0) | {
        "horizontal_distance": 1.49,
        "root_height": 0.2,
        "pelvis_up_z": 0.1,
        "robot_upright": 0.1,
    }

    for _ in range(10):
        controller.update(fallen, goal_height=0.185)

    assert controller.phase == "get_up"
    assert controller.fall_event_count == 1


def test_normal_place_crouch_is_not_misclassified_as_fallen():
    controller = _phase_controller()
    controller.phase = "place"
    crouched = _phase_metrics(0.15) | {
        "object_height": 0.70,
        "root_height": 0.471,
        "pelvis_up_z": 0.450,
        "robot_upright": 0.450,
    }

    for _ in range(20):
        controller.update(crouched, goal_height=0.185)

    assert controller.phase == "place"
    assert controller.fall_detect_steps == 0


def test_fall_detection_accepts_hard_low_or_tilted_and_low():
    hard_low = _phase_metrics(1.0) | {
        "horizontal_distance": 2.0,
        "root_height": 0.29,
        "pelvis_up_z": 1.0,
    }
    tilted_and_low = _phase_metrics(1.0) | {
        "horizontal_distance": 2.0,
        "root_height": 0.35,
        "pelvis_up_z": 0.10,
    }

    for metrics in (hard_low, tilted_and_low):
        controller = _phase_controller()
        for _ in range(10):
            controller.update(metrics, goal_height=0.185)
        assert controller.phase == "get_up"


def test_get_up_requires_stable_recovery_before_approach():
    controller = _phase_controller()
    controller.phase = "get_up"
    recovered = _phase_metrics(1.0) | {
        "horizontal_distance": 2.0,
        "root_height": 0.7,
        "pelvis_up_z": 0.9,
        "robot_upright": 0.9,
    }

    for _ in range(24):
        controller.update(recovered, goal_height=0.185)
    assert controller.phase == "get_up"

    controller.update(recovered, goal_height=0.185)
    assert controller.phase == "approach"
    assert controller.get_up_stable_steps == 0


def test_get_up_stays_active_while_fallen_near_box():
    controller = _phase_controller()
    controller.phase = "get_up"
    fallen_near = _phase_metrics(1.0) | {
        "horizontal_distance": 1.5,
        "root_height": 0.2,
        "pelvis_up_z": 0.1,
        "robot_upright": 0.1,
    }

    controller.update(fallen_near, goal_height=0.185)
    assert controller.phase == "get_up"


def test_get_up_reward_prefers_high_upright_stationary_root():
    qpos = torch.zeros((3, 36))
    qvel = torch.zeros((3, 35))
    state = torch.zeros((3, 98))
    qpos[:, 3] = 1.0
    qpos[:, 2] = torch.tensor([0.20, 0.80, 0.80])
    state[:, 60] = torch.tensor([-1.0, -1.0, 0.0])
    qvel[2, :3] = 2.0

    reward, components = compute_get_up_reward(qpos, qvel, state, num_dof=29)

    assert reward.shape == (3, 1)
    assert reward[1, 0] > 0.9
    assert reward[0, 0] < 0.01
    assert reward[2, 0] == 0.0
    torch.testing.assert_close(components["root_height"], torch.tensor([0.2, 0.8, 0.8]))


def test_joint_parameter_prefers_most_specific_match():
    values = {"pitch": 1.0, "waist_pitch": 40.0}
    assert _match_joint_parameter("waist_pitch_joint", values) == 40.0


def test_frame_or_static_supports_both_motion_schemas():
    assert np.array_equal(_frame_or_static(np.asarray([1.0, 2.0, 3.0]), 7, width=3), [1.0, 2.0, 3.0])
    framed = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.array_equal(_frame_or_static(framed, 1, width=3), [4.0, 5.0, 6.0])
