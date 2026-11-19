from collections import deque
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_policy.ufo_policy import UFODeployPolicy


class _FakeStateProcessor:
    def __init__(self, joint_pos):
        self.joint_pos = np.asarray(joint_pos, dtype=np.float64)


class _FakeCommandSender:
    def __init__(self):
        self.commands = []

    def send_command(self, cmd_q, cmd_dq, cmd_tau):
        self.commands.append(
            (
                np.asarray(cmd_q, dtype=np.float64).copy(),
                np.asarray(cmd_dq, dtype=np.float64).copy(),
                np.asarray(cmd_tau, dtype=np.float64).copy(),
            )
        )


class _FakeZmqSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    def recv(self, flags=0):
        if self.messages:
            return self.messages.pop(0)
        raise zmq.Again()


def _policy(num_dofs=3):
    policy = UFODeployPolicy.__new__(UFODeployPolicy)
    policy.num_dofs = num_dofs
    policy.num_actions = num_dofs
    policy.use_policy_action = True
    policy.start_motion = True
    policy.get_ready_state = True
    policy.task_type = "tracking"
    policy.t = 5
    policy.t_start = 0
    policy.t_stop = 0
    policy.last_action = np.ones(num_dofs, dtype=np.float32)
    policy.joint_pos_lower_limit = np.full(num_dofs, -10.0)
    policy.joint_pos_upper_limit = np.full(num_dofs, 10.0)
    policy.joint_velocity_limit = np.ones(num_dofs, dtype=np.float64)
    policy.q_target_slew_safety_factor = 0.5
    policy.rl_dt = 0.1
    policy.last_cmd_q = None
    policy.last_pico_buttons = {}
    policy.stop_latched = False
    policy.pico_enable_released_after_stop = True
    policy.stop_latch_enable_released_after_stop = True
    policy._last_safe_stop_warning = 0.0
    policy._last_invalid_z_warning = 0.0
    policy._last_z_timeout_warning = 0.0
    policy._last_stop_latch_warning = 0.0
    policy.state_processor = _FakeStateProcessor(np.zeros(num_dofs))
    policy.command_sender = _FakeCommandSender()
    return policy


def test_stale_realtime_z_enters_safe_stop():
    policy = _policy()
    policy.ctx_source = "zmq"
    policy.ctx_norm_ref = 16.0
    policy.ctx_latest = np.ones(256, dtype=np.float32)
    policy.ctx_window = deque([np.ones(256, dtype=np.float32)], maxlen=3)
    policy.ctx_zmq_timeout_s = 0.001
    policy.ctx_zmq_start_monotonic = time.monotonic() - 1.0
    policy.ctx_last_z_monotonic = time.monotonic() - 1.0

    policy._check_realtime_z_watchdog()

    assert not policy.use_policy_action
    assert not policy.start_motion
    assert not policy.get_ready_state
    assert policy.t == policy.t_stop
    assert len(policy.ctx_window) == 0
    assert policy.ctx_latest.shape == (256,)
    assert policy.ctx_latest[0] == 16.0


def test_nan_realtime_z_is_ignored():
    policy = _policy()
    policy.ctx_zmq = _FakeZmqSocket([np.full(256, np.nan, dtype=np.float32).tobytes()])
    policy.ctx_latest = np.arange(256, dtype=np.float32)
    policy.ctx_last_z_monotonic = None

    received = policy._poll_realtime_z()

    assert not received
    np.testing.assert_array_equal(policy.ctx_latest, np.arange(256, dtype=np.float32))
    assert policy.ctx_last_z_monotonic is None


def test_r2_latch_blocks_pico_enable_until_release_and_repress():
    policy = _policy()
    policy.enter_stop_latch("test R2")

    policy.handle_pico_buttons(
        {"right_key_one": True, "right_key_two": True, "left_key_one": False}
    )
    assert not policy.use_policy_action
    assert not policy.start_motion

    policy.handle_pico_buttons(
        {"right_key_one": False, "right_key_two": False, "left_key_one": False}
    )
    policy.handle_pico_buttons(
        {"right_key_one": True, "right_key_two": True, "left_key_one": False}
    )
    assert policy.use_policy_action
    assert policy.start_motion


def test_r2_latch_blocks_wireless_r1_enable():
    policy = _policy()
    policy.enter_stop_latch("test R2")

    policy.handle_joystick_button("R1")

    assert not policy.use_policy_action
    assert policy.stop_latched


class _FakeCarryCalibrationRuntime:
    def __init__(self):
        self.deployment_calibration_enabled = True
        self.calibration_stage = "pelvis"
        self.calibration_contact_label = None
        self.calibration_cfg = SimpleNamespace(max_default_joint_error=0.15)
        self.runtime_mode = "phased"
        self.controller_profile = "intact_logmap150544384_strong"
        self.calibration_result_path = None
        self.reset_count = 0
        self.saved_activation_count = 0
        self.capture_start_count = 0

    @property
    def deployment_calibration_complete(self):
        return self.calibration_stage == "complete"

    def clear_calibration_contact_label(self):
        self.calibration_contact_label = None

    def set_calibration_contact_label(self, contact):
        if self.calibration_stage != "contact":
            return False, "wrong stage"
        self.calibration_contact_label = int(bool(contact))
        return True, "label set"

    def advance_deployment_calibration(self, *, default_pose_active):
        if self.calibration_stage == "contact" and not default_pose_active:
            self.calibration_stage = "complete"
            self.calibration_contact_label = None
            return True, "complete"
        return False, "not ready"

    def activate_saved_deployment_calibration(self):
        self.saved_activation_count += 1
        self.calibration_stage = "complete"
        return True, "saved calibration applied"

    def start_deployment_calibration_capture(self, *, default_pose_active):
        if not default_pose_active:
            return False, "default pose required"
        self.capture_start_count += 1
        return True, "capture started"

    def deployment_calibration_snapshot(self):
        return {
            "enabled": True,
            "stage": self.calibration_stage,
            "complete": self.deployment_calibration_complete,
            "results": {},
        }

    @staticmethod
    def support_pose_joint_error(joint_position, default_joint_position):
        current = np.asarray(joint_position, dtype=np.float64).reshape(-1)
        default = np.asarray(default_joint_position, dtype=np.float64).reshape(-1)
        support_joint_count = min(15, current.size)
        return float(
            np.max(
                np.abs(
                    current[:support_joint_count] - default[:support_joint_count]
                )
            )
        )

    def reset(self):
        self.reset_count += 1


def test_carry_calibration_buttons_gate_r1_and_toggle_default_pose():
    policy = _policy()
    policy.task_type = "carry"
    policy.use_policy_action = False
    policy.get_ready_state = False
    policy.default_dof_angles = np.zeros(policy.num_dofs, dtype=np.float64)
    policy.init_count = 17
    policy.carry_runtime = _FakeCarryCalibrationRuntime()
    policy.data_recorder = None

    policy.handle_joystick_button("R1")
    assert not policy.use_policy_action

    policy.handle_joystick_button("A")
    assert policy.get_ready_state
    assert policy.init_count == 0

    policy.carry_runtime.calibration_stage = "contact"
    policy.handle_joystick_button("X")
    assert policy.carry_runtime.calibration_contact_label is None

    policy.handle_joystick_button("A")
    assert not policy.get_ready_state
    policy.handle_joystick_button("X")
    assert policy.carry_runtime.calibration_contact_label == 0
    policy.handle_joystick_button("Y")
    assert policy.carry_runtime.calibration_contact_label == 1

    policy.handle_joystick_button("B")
    assert policy.carry_runtime.deployment_calibration_complete
    assert not policy.use_policy_action

    policy.handle_joystick_button("A")
    assert policy.get_ready_state
    policy.handle_joystick_button("R1")
    assert policy.use_policy_action
    assert not policy.get_ready_state
    assert policy.carry_runtime.reset_count == 1


def test_carry_b_starts_capture_only_after_default_pose_is_enabled():
    policy = _policy()
    policy.task_type = "carry"
    policy.use_policy_action = False
    policy.get_ready_state = False
    policy.default_dof_angles = np.zeros(policy.num_dofs, dtype=np.float64)
    policy.carry_runtime = _FakeCarryCalibrationRuntime()

    policy.handle_joystick_button("B")
    assert policy.carry_runtime.capture_start_count == 0
    policy.handle_joystick_button("A")
    policy.handle_joystick_button("B")
    assert policy.carry_runtime.capture_start_count == 1
    assert policy.carry_runtime.calibration_stage == "pelvis"
    assert not policy.use_policy_action


def _wireless_message(**pressed):
    values = {
        name: False
        for name in ("A", "B", "X", "Y", "L1", "L2", "R1", "R2")
    }
    values.update(pressed)
    return SimpleNamespace(**values)


def test_wireless_l1_r1_applies_saved_calibration_and_starts_policy():
    policy = _policy(num_dofs=29)
    policy.task_type = "carry"
    policy.use_policy_action = False
    # Saved-calibration start must also support deliberate fallen/non-default
    # starts without requiring A first.
    policy.get_ready_state = False
    policy.default_dof_angles = np.zeros(policy.num_dofs, dtype=np.float64)
    policy.state_processor.joint_pos[0] = 0.20
    policy.state_processor.joint_pos[18] = 0.20
    policy.carry_runtime = _FakeCarryCalibrationRuntime()
    policy.last_wc_msg = _wireless_message()
    policy.robot = SimpleNamespace(
        read_wireless_controller=lambda: _wireless_message(L1=True, R1=True)
    )

    policy.process_joystick_input()

    assert policy.carry_runtime.saved_activation_count == 1
    assert policy.carry_runtime.deployment_calibration_complete
    assert policy.use_policy_action
    assert not policy.get_ready_state
    assert policy.carry_runtime.reset_count == 1


def test_carry_r1_has_no_joint_pose_error_gate():
    policy = _policy(num_dofs=29)
    policy.task_type = "carry"
    policy.use_policy_action = False
    policy.get_ready_state = True
    policy.default_dof_angles = np.zeros(policy.num_dofs, dtype=np.float64)
    policy.carry_runtime = _FakeCarryCalibrationRuntime()
    policy.carry_runtime.calibration_stage = "complete"
    policy.state_processor.joint_pos[18] = 0.20
    policy.state_processor.joint_pos[0] = 0.20

    policy.handle_joystick_button("R1")
    assert policy.use_policy_action


def test_q_target_slew_limit_uses_joint_velocity_limit():
    policy = _policy(num_dofs=2)
    policy.last_cmd_q = np.array([0.0, 0.0])
    policy.joint_velocity_limit = np.array([10.0, 2.0])
    policy.rl_dt = 0.1
    policy.q_target_slew_safety_factor = 0.5

    limited = policy._apply_q_target_slew_limit(np.array([10.0, -10.0]))

    np.testing.assert_allclose(limited, np.array([0.5, -0.1]))


def test_zero_q_target_slew_factor_disables_filter():
    policy = _policy(num_dofs=2)
    policy.last_cmd_q = np.array([0.0, 0.0])
    policy.joint_velocity_limit = np.array([10.0, 2.0])
    policy.rl_dt = 0.1
    policy.q_target_slew_safety_factor = 0.0

    requested = np.array([1.25, -0.75])
    limited = policy._apply_q_target_slew_limit(requested)

    np.testing.assert_allclose(limited, requested)
    assert limited is not requested


def test_g1_real_requires_explicit_real_robot_env():
    old = os.environ.pop("UFO_REAL_ROBOT_OK", None)
    try:
        try:
            UFODeployPolicy(
                robot_config={"ROBOT_TYPE": "g1_real"},
                policy_config={},
                exp_config={},
                model_path="",
            )
        except RuntimeError as exc:
            assert "UFO_REAL_ROBOT_OK=1" in str(exc)
        else:
            raise AssertionError("g1_real policy started without UFO_REAL_ROBOT_OK=1")
    finally:
        if old is not None:
            os.environ["UFO_REAL_ROBOT_OK"] = old


def test_h20_real_requires_experimental_ack_before_robot_initialization():
    old_real = os.environ.get("UFO_REAL_ROBOT_OK")
    old_h20 = os.environ.pop("UFO_H20_PAIR80M_EXPERIMENTAL_OK", None)
    os.environ["UFO_REAL_ROBOT_OK"] = "1"
    try:
        try:
            UFODeployPolicy(
                robot_config={"ROBOT_TYPE": "g1_real"},
                policy_config={},
                exp_config={
                    "type": "carry",
                    "phase_controller_profile": "h20_reward736",
                },
                model_path="",
            )
        except RuntimeError as exc:
            assert "UFO_H20_PAIR80M_EXPERIMENTAL_OK=1" in str(exc)
        else:
            raise AssertionError("H20 real policy started without experimental acknowledgement")
    finally:
        if old_real is None:
            os.environ.pop("UFO_REAL_ROBOT_OK", None)
        else:
            os.environ["UFO_REAL_ROBOT_OK"] = old_real
        if old_h20 is not None:
            os.environ["UFO_H20_PAIR80M_EXPERIMENTAL_OK"] = old_h20


def test_newintact_hybrid_requires_experimental_ack_before_robot_initialization():
    old_real = os.environ.get("UFO_REAL_ROBOT_OK")
    old_ack = os.environ.pop("UFO_NEWINTACT80M_HYBRID_EXPERIMENTAL_OK", None)
    os.environ["UFO_REAL_ROBOT_OK"] = "1"
    try:
        try:
            UFODeployPolicy(
                robot_config={"ROBOT_TYPE": "g1_real"},
                policy_config={},
                exp_config={
                    "type": "carry",
                    "phase_controller_profile": "newintact80m_hybrid_direct_place",
                },
                model_path="",
            )
        except RuntimeError as exc:
            assert "UFO_NEWINTACT80M_HYBRID_EXPERIMENTAL_OK=1" in str(exc)
        else:
            raise AssertionError(
                "newintact hybrid started without experimental acknowledgement"
            )
    finally:
        if old_real is None:
            os.environ.pop("UFO_REAL_ROBOT_OK", None)
        else:
            os.environ["UFO_REAL_ROBOT_OK"] = old_real
        if old_ack is not None:
            os.environ["UFO_NEWINTACT80M_HYBRID_EXPERIMENTAL_OK"] = old_ack


def test_bestcarry_strong_requires_experimental_ack_before_robot_initialization():
    old_real = os.environ.get("UFO_REAL_ROBOT_OK")
    old_ack = os.environ.pop(
        "UFO_BESTCARRY140935168_STRONG_EXPERIMENTAL_OK", None
    )
    os.environ["UFO_REAL_ROBOT_OK"] = "1"
    try:
        try:
            UFODeployPolicy(
                robot_config={"ROBOT_TYPE": "g1_real"},
                policy_config={},
                exp_config={
                    "type": "carry",
                    "phase_controller_profile": "bestcarry140935168_strong",
                },
                model_path="",
            )
        except RuntimeError as exc:
            assert "UFO_BESTCARRY140935168_STRONG_EXPERIMENTAL_OK=1" in str(exc)
        else:
            raise AssertionError(
                "best-carry strong started without experimental acknowledgement"
            )
    finally:
        if old_real is None:
            os.environ.pop("UFO_REAL_ROBOT_OK", None)
        else:
            os.environ["UFO_REAL_ROBOT_OK"] = old_real
        if old_ack is not None:
            os.environ["UFO_BESTCARRY140935168_STRONG_EXPERIMENTAL_OK"] = old_ack


def test_intact_logmap_gpu_requires_experimental_ack_before_robot_initialization():
    old_real = os.environ.get("UFO_REAL_ROBOT_OK")
    key = "UFO_INTACT_LOGMAP150544384_STRONG_GPU_EXPERIMENTAL_OK"
    old_ack = os.environ.pop(key, None)
    os.environ["UFO_REAL_ROBOT_OK"] = "1"
    try:
        try:
            UFODeployPolicy(
                robot_config={"ROBOT_TYPE": "g1_real"},
                policy_config={},
                exp_config={
                    "type": "carry",
                    "phase_controller_profile": "intact_logmap150544384_strong",
                },
                model_path="",
            )
        except RuntimeError as exc:
            assert f"{key}=1" in str(exc)
        else:
            raise AssertionError("INTACT-logmap GPU started without experimental acknowledgement")
    finally:
        if old_real is None:
            os.environ.pop("UFO_REAL_ROBOT_OK", None)
        else:
            os.environ["UFO_REAL_ROBOT_OK"] = old_real
        if old_ack is not None:
            os.environ[key] = old_ack


def test_lafan_tracking_requires_experimental_ack_before_robot_initialization():
    old_real = os.environ.get("UFO_REAL_ROBOT_OK")
    old_ack = os.environ.pop("UFO_LAFAN140935168_EXPERIMENTAL_OK", None)
    os.environ["UFO_REAL_ROBOT_OK"] = "1"
    try:
        try:
            UFODeployPolicy(
                robot_config={"ROBOT_TYPE": "g1_real"},
                policy_config={},
                exp_config={"type": "lafan_tracking"},
                model_path="",
            )
        except RuntimeError as exc:
            assert "UFO_LAFAN140935168_EXPERIMENTAL_OK=1" in str(exc)
        else:
            raise AssertionError(
                "LAFAN tracking started without experimental acknowledgement"
            )
    finally:
        if old_real is None:
            os.environ.pop("UFO_REAL_ROBOT_OK", None)
        else:
            os.environ["UFO_REAL_ROBOT_OK"] = old_real
        if old_ack is not None:
            os.environ["UFO_LAFAN140935168_EXPERIMENTAL_OK"] = old_ack


def test_lafan_tracking_requires_suspended_ack_before_robot_initialization():
    old_real = os.environ.get("UFO_REAL_ROBOT_OK")
    old_experimental = os.environ.get("UFO_LAFAN140935168_EXPERIMENTAL_OK")
    old_suspended = os.environ.pop("UFO_LAFAN_SUSPENDED_TEST_OK", None)
    os.environ["UFO_REAL_ROBOT_OK"] = "1"
    os.environ["UFO_LAFAN140935168_EXPERIMENTAL_OK"] = "1"
    try:
        try:
            UFODeployPolicy(
                robot_config={"ROBOT_TYPE": "g1_real"},
                policy_config={},
                exp_config={"type": "lafan_tracking", "suspended_only": True},
                model_path="",
            )
        except RuntimeError as exc:
            assert "UFO_LAFAN_SUSPENDED_TEST_OK=1" in str(exc)
        else:
            raise AssertionError("LAFAN tracking started without suspended acknowledgement")
    finally:
        if old_real is None:
            os.environ.pop("UFO_REAL_ROBOT_OK", None)
        else:
            os.environ["UFO_REAL_ROBOT_OK"] = old_real
        if old_experimental is None:
            os.environ.pop("UFO_LAFAN140935168_EXPERIMENTAL_OK", None)
        else:
            os.environ["UFO_LAFAN140935168_EXPERIMENTAL_OK"] = old_experimental
        if old_suspended is not None:
            os.environ["UFO_LAFAN_SUSPENDED_TEST_OK"] = old_suspended


if __name__ == "__main__":
    test_stale_realtime_z_enters_safe_stop()
    test_nan_realtime_z_is_ignored()
    test_r2_latch_blocks_pico_enable_until_release_and_repress()
    test_r2_latch_blocks_wireless_r1_enable()
    test_carry_calibration_buttons_gate_r1_and_toggle_default_pose()
    test_carry_b_starts_capture_only_after_default_pose_is_enabled()
    test_wireless_l1_r1_applies_saved_calibration_and_starts_policy()
    test_carry_r1_has_no_joint_pose_error_gate()
    test_q_target_slew_limit_uses_joint_velocity_limit()
    test_zero_q_target_slew_factor_disables_filter()
    test_g1_real_requires_explicit_real_robot_env()
    test_h20_real_requires_experimental_ack_before_robot_initialization()
    test_newintact_hybrid_requires_experimental_ack_before_robot_initialization()
    test_bestcarry_strong_requires_experimental_ack_before_robot_initialization()
    test_intact_logmap_gpu_requires_experimental_ack_before_robot_initialization()
    test_lafan_tracking_requires_experimental_ack_before_robot_initialization()
    test_lafan_tracking_requires_suspended_ack_before_robot_initialization()
    print("ufo_policy safety tests ok")
