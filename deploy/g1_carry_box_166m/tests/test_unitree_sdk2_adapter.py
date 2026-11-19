from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rl_policy.unitree_sdk2_adapter import (
    ControlMode,
    G1Command,
    G1_NUM_MOTOR,
    UnitreeSdk2G1Interface,
)


class FakeMotorCommand:
    def __init__(self) -> None:
        self.mode = 0
        self.q = self.dq = self.tau = self.kp = self.kd = 0.0


class FakeCrc:
    def Crc(self, _message):
        return 12345


def make_adapter() -> UnitreeSdk2G1Interface:
    adapter = object.__new__(UnitreeSdk2G1Interface)
    adapter._lock = threading.Lock()
    adapter._latest_low_state = SimpleNamespace(mode_machine=5)
    adapter._latest_low_state_monotonic = time.monotonic()
    adapter._control_mode = ControlMode.PR
    adapter._low_cmd_factory = lambda: SimpleNamespace(
        mode_pr=0,
        mode_machine=0,
        motor_cmd=[FakeMotorCommand() for _ in range(35)],
        crc=0,
    )
    adapter._crc = FakeCrc()
    adapter.damping_kd = 1.25
    adapter.low_state_timeout_s = 0.1
    return adapter


def _fake_motor_state(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        q=float(index),
        dq=float(index) + 0.1,
        ddq=float(index) + 0.2,
        tau_est=float(index) + 0.3,
        temperature=[index, index + 1],
        vol=float(index) + 40.0,
        motorstate=index + 100,
    )


class UnitreeSdk2AdapterTest(unittest.TestCase):
    def test_dual_low_command_gate_rejects_partial_authorization(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "authorization is incomplete"):
                UnitreeSdk2G1Interface("enp-test", {"ENABLE_LOW_LEVEL_COMMANDS": True})
        with patch.dict(os.environ, {"UFO_ALLOW_LOW_LEVEL_COMMANDS": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "authorization is incomplete"):
                UnitreeSdk2G1Interface("enp-test", {"ENABLE_LOW_LEVEL_COMMANDS": False})

    def test_wireless_button_mapping_matches_official_example(self):
        raw = bytearray(40)
        raw[2] = (1 << 0) | (1 << 4) | (1 << 5)
        raw[3] = (1 << 0) | (1 << 2)
        state = UnitreeSdk2G1Interface.decode_wireless_controller(raw)
        self.assertTrue(state.R1)
        self.assertTrue(state.R2)
        self.assertTrue(state.L2)
        self.assertTrue(state.A)
        self.assertTrue(state.X)
        self.assertFalse(state.B)

    def test_command_mapping_and_damping(self):
        adapter = make_adapter()
        values = np.arange(G1_NUM_MOTOR, dtype=np.float64)
        command = G1Command(
            q_target=values.tolist(),
            dq_target=(values + 1).tolist(),
            tau_ff=(values + 2).tolist(),
            kp=(values + 3).tolist(),
            kd=(values + 4).tolist(),
        )
        message = adapter._command_message(command, damping=False)
        self.assertEqual(message.mode_pr, 0)
        self.assertEqual(message.mode_machine, 5)
        self.assertEqual(message.motor_cmd[28].q, 28.0)
        self.assertEqual(message.motor_cmd[28].kd, 32.0)
        self.assertEqual(message.motor_cmd[29].mode, 0)
        self.assertEqual(message.crc, 12345)

        damping = adapter._command_message(command, damping=True)
        self.assertEqual(damping.motor_cmd[0].kp, 0.0)
        self.assertEqual(damping.motor_cmd[0].kd, 1.25)
        self.assertEqual(damping.motor_cmd[0].tau, 0.0)

    def test_zero_command_arrays_do_not_alias(self):
        command = UnitreeSdk2G1Interface.create_zero_command()
        command.q_target[0] = 1.0
        self.assertEqual(command.dq_target[0], 0.0)
        self.assertEqual(len(command.kp), G1_NUM_MOTOR)

    def test_low_state_exposes_motor_torque_and_health_diagnostics(self):
        adapter = make_adapter()
        adapter._latest_low_state = SimpleNamespace(
            imu_state=SimpleNamespace(
                quaternion=[1.0, 0.0, 0.0, 0.0],
                gyroscope=[0.0, 0.0, 0.0],
            ),
            motor_state=[_fake_motor_state(i) for i in range(G1_NUM_MOTOR)],
            tick=123,
            mode_machine=5,
        )

        state = adapter.read_low_state()

        self.assertEqual(state.motor.tau_est.shape, (G1_NUM_MOTOR,))
        self.assertEqual(state.motor.temperature.shape, (G1_NUM_MOTOR, 2))
        self.assertAlmostEqual(state.motor.ddq[7], 7.2)
        self.assertAlmostEqual(state.motor.tau_est[7], 7.3)
        self.assertAlmostEqual(state.motor.voltage[7], 47.0)
        self.assertEqual(state.motor.state_code[7], 107)


if __name__ == "__main__":
    unittest.main()
