"""Official Unitree SDK2 backend for workstation-hosted G1 control.

Importing this module is inert. DDS is initialized only when
``UnitreeSdk2G1Interface`` is constructed, and a LowCmd publisher is created
only after both the configuration and the environment explicitly allow it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


G1_NUM_MOTOR = 29
G1_LOW_CMD_MOTOR_SLOTS = 35


class ControlMode(IntEnum):
    PR = 0
    AB = 1


@dataclass
class G1Command:
    q_target: list[float]
    dq_target: list[float]
    tau_ff: list[float]
    kp: list[float]
    kd: list[float]


@dataclass(frozen=True)
class WirelessControllerState:
    L1: bool = False
    L2: bool = False
    R1: bool = False
    R2: bool = False
    A: bool = False
    B: bool = False
    X: bool = False
    Y: bool = False


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values, got {result.shape}")
    return result


class UnitreeSdk2G1Interface:
    """Adapter from official ``unitree_sdk2py`` DDS objects to UFO's API."""

    def __init__(self, network_interface: str, config: Mapping[str, Any]) -> None:
        if not network_interface:
            raise ValueError("A non-empty G1 DDS network interface is required")

        self.config = dict(config)
        self.network_interface = str(network_interface)
        self.domain_id = int(self.config.get("DOMAIN_ID", 0))
        self.low_state_topic = str(self.config.get("LOW_STATE_TOPIC", "rt/lowstate"))
        self.low_cmd_topic = str(self.config.get("LOW_CMD_TOPIC", "rt/lowcmd"))
        self.low_state_timeout_s = float(self.config.get("LOW_STATE_TIMEOUT_MS", 100.0)) / 1000.0
        self.command_timeout_s = float(self.config.get("LOW_COMMAND_TIMEOUT_MS", 100.0)) / 1000.0
        self.command_period_s = 1.0 / float(self.config.get("LOW_COMMAND_RATE_HZ", 500.0))
        self.damping_kd = float(self.config.get("LOW_COMMAND_DAMPING_KD", 1.0))
        if self.low_state_timeout_s <= 0.0 or self.command_timeout_s <= 0.0:
            raise ValueError("G1 state and command timeouts must be positive")
        if self.command_period_s <= 0.0 or self.damping_kd < 0.0:
            raise ValueError("G1 command rate must be positive and damping Kd non-negative")

        config_allows = bool(self.config.get("ENABLE_LOW_LEVEL_COMMANDS", False))
        env_allows = os.environ.get("UFO_ALLOW_LOW_LEVEL_COMMANDS") == "1"
        self.low_commands_enabled = config_allows and env_allows
        if config_allows != env_allows:
            raise RuntimeError(
                "Low-level command authorization is incomplete: set both "
                "ENABLE_LOW_LEVEL_COMMANDS=true in the robot config and "
                "UFO_ALLOW_LOW_LEVEL_COMMANDS=1 in the launch environment, or disable both."
            )
        self.release_motion_service = bool(self.config.get("RELEASE_MOTION_SERVICE", False))
        if self.release_motion_service and not self.low_commands_enabled:
            raise RuntimeError("RELEASE_MOTION_SERVICE requires low-level commands to be explicitly enabled")

        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as exc:
            raise ImportError(
                "Official unitree_sdk2py is unavailable. Install requirements/unitree_workstation.txt "
                "inside the Python 3.10 x86_64 environment."
            ) from exc

        ChannelFactoryInitialize(self.domain_id, self.network_interface)
        self._low_cmd_factory = unitree_hg_msg_dds__LowCmd_
        self._low_cmd_type = LowCmd_
        self._crc = CRC()
        self._lock = threading.Lock()
        self._latest_low_state: Any | None = None
        self._latest_low_state_monotonic: float | None = None
        self._control_mode = ControlMode.PR
        self._latest_command: G1Command | None = None
        self._latest_command_monotonic: float | None = None
        self._publisher_stop = threading.Event()
        self._publisher_thread: threading.Thread | None = None
        self._publisher_error: RuntimeError | None = None

        self._lowstate_subscriber = ChannelSubscriber(self.low_state_topic, LowState_)
        self._lowstate_subscriber.Init(self._on_low_state, 1)

        self._lowcmd_publisher = None
        if self.low_commands_enabled:
            self._prepare_low_level_mode()
            self._lowcmd_publisher = ChannelPublisher(self.low_cmd_topic, self._low_cmd_type)
            self._lowcmd_publisher.Init()

    def _prepare_low_level_mode(self) -> None:
        if not self.release_motion_service:
            return
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

        client = MotionSwitcherClient()
        client.SetTimeout(float(self.config.get("MOTION_SWITCHER_TIMEOUT_S", 5.0)))
        client.Init()
        attempts = int(self.config.get("MOTION_SWITCHER_RELEASE_ATTEMPTS", 5))
        for _ in range(max(attempts, 1)):
            status, result = client.CheckMode()
            if status != 0 or result is None:
                raise RuntimeError(f"MotionSwitcher CheckMode failed with status={status}")
            if not str(result.get("name", "")):
                return
            status, _ = client.ReleaseMode()
            if status != 0:
                raise RuntimeError(f"MotionSwitcher ReleaseMode failed with status={status}")
            time.sleep(0.2)
        raise RuntimeError("MotionSwitcher service remained active after bounded release attempts")

    def _on_low_state(self, message: Any) -> None:
        try:
            if int(self._crc.Crc(message)) != int(message.crc):
                return
        except Exception:
            return
        with self._lock:
            self._latest_low_state = message
            self._latest_low_state_monotonic = time.monotonic()

    def _current_low_state(self) -> Any | None:
        with self._lock:
            message = self._latest_low_state
            received = self._latest_low_state_monotonic
        if message is None or received is None:
            return None
        age = time.monotonic() - received
        if age > self.low_state_timeout_s:
            raise RuntimeError(f"G1 low state is stale ({age * 1000.0:.0f} ms)")
        return message

    def set_control_mode(self, mode: ControlMode | int) -> None:
        self._control_mode = ControlMode(int(mode))

    def read_low_state(self) -> Any | None:
        message = self._current_low_state()
        if message is None:
            return None
        return SimpleNamespace(
            imu=SimpleNamespace(
                quat=np.asarray(message.imu_state.quaternion, dtype=np.float64).copy(),
                omega=np.asarray(message.imu_state.gyroscope, dtype=np.float64).copy(),
            ),
            motor=SimpleNamespace(
                q=np.asarray([item.q for item in message.motor_state[:G1_NUM_MOTOR]], dtype=np.float64),
                dq=np.asarray([item.dq for item in message.motor_state[:G1_NUM_MOTOR]], dtype=np.float64),
                ddq=np.asarray(
                    [item.ddq for item in message.motor_state[:G1_NUM_MOTOR]],
                    dtype=np.float64,
                ),
                # Unitree names this field tau_est: it is the motor-side torque
                # estimate reported by LowState, not a load-cell measurement.
                tau_est=np.asarray(
                    [item.tau_est for item in message.motor_state[:G1_NUM_MOTOR]],
                    dtype=np.float64,
                ),
                temperature=np.asarray(
                    [item.temperature for item in message.motor_state[:G1_NUM_MOTOR]],
                    dtype=np.int64,
                ),
                voltage=np.asarray(
                    [item.vol for item in message.motor_state[:G1_NUM_MOTOR]],
                    dtype=np.float64,
                ),
                state_code=np.asarray(
                    [item.motorstate for item in message.motor_state[:G1_NUM_MOTOR]],
                    dtype=np.uint32,
                ),
            ),
            tick=int(message.tick),
            mode_machine=int(message.mode_machine),
        )

    @staticmethod
    def decode_wireless_controller(data: Any) -> WirelessControllerState:
        raw = bytes(data)
        if len(raw) < 4:
            return WirelessControllerState()
        first = raw[2]
        second = raw[3]
        return WirelessControllerState(
            R1=bool(first & (1 << 0)),
            L1=bool(first & (1 << 1)),
            R2=bool(first & (1 << 4)),
            L2=bool(first & (1 << 5)),
            A=bool(second & (1 << 0)),
            B=bool(second & (1 << 1)),
            X=bool(second & (1 << 2)),
            Y=bool(second & (1 << 3)),
        )

    def read_wireless_controller(self) -> WirelessControllerState:
        try:
            message = self._current_low_state()
        except RuntimeError:
            return WirelessControllerState()
        if message is None:
            return WirelessControllerState()
        return self.decode_wireless_controller(message.wireless_remote)

    @staticmethod
    def create_zero_command() -> G1Command:
        zeros = [0.0] * G1_NUM_MOTOR
        return G1Command(
            q_target=zeros.copy(),
            dq_target=zeros.copy(),
            tau_ff=zeros.copy(),
            kp=zeros.copy(),
            kd=zeros.copy(),
        )

    def write_low_command(self, command: G1Command) -> None:
        if not self.low_commands_enabled or self._lowcmd_publisher is None:
            raise RuntimeError("Low-level G1 command publication is disabled")
        if self._publisher_error is not None:
            raise self._publisher_error
        validated = G1Command(
            q_target=_finite_vector(command.q_target, G1_NUM_MOTOR, "q_target").tolist(),
            dq_target=_finite_vector(command.dq_target, G1_NUM_MOTOR, "dq_target").tolist(),
            tau_ff=_finite_vector(command.tau_ff, G1_NUM_MOTOR, "tau_ff").tolist(),
            kp=_finite_vector(command.kp, G1_NUM_MOTOR, "kp").tolist(),
            kd=_finite_vector(command.kd, G1_NUM_MOTOR, "kd").tolist(),
        )
        if np.any(np.asarray(validated.kp) < 0.0) or np.any(np.asarray(validated.kd) < 0.0):
            raise ValueError("G1 Kp/Kd values must be non-negative")
        if self._current_low_state() is None:
            raise RuntimeError("Cannot publish G1 command before receiving a valid low state")
        with self._lock:
            self._latest_command = validated
            self._latest_command_monotonic = time.monotonic()
        if self._publisher_thread is None:
            self._publisher_thread = threading.Thread(
                target=self._publisher_loop,
                name="g1-lowcmd-500hz",
                daemon=True,
            )
            self._publisher_thread.start()

    def _command_message(self, command: G1Command | None, *, damping: bool) -> Any:
        message = self._low_cmd_factory()
        with self._lock:
            low_state = self._latest_low_state
        if low_state is None:
            raise RuntimeError("No G1 low state is available for mode_machine")
        message.mode_pr = int(self._control_mode)
        message.mode_machine = int(low_state.mode_machine)
        for index in range(G1_LOW_CMD_MOTOR_SLOTS):
            motor = message.motor_cmd[index]
            motor.mode = 1 if index < G1_NUM_MOTOR else 0
            if index >= G1_NUM_MOTOR:
                motor.q = motor.dq = motor.tau = motor.kp = motor.kd = 0.0
            elif damping or command is None:
                motor.q = motor.dq = motor.tau = motor.kp = 0.0
                motor.kd = self.damping_kd
            else:
                motor.q = command.q_target[index]
                motor.dq = command.dq_target[index]
                motor.tau = command.tau_ff[index]
                motor.kp = command.kp[index]
                motor.kd = command.kd[index]
        message.crc = self._crc.Crc(message)
        return message

    def _publisher_loop(self) -> None:
        next_deadline = time.perf_counter()
        try:
            while not self._publisher_stop.is_set():
                with self._lock:
                    command = self._latest_command
                    updated = self._latest_command_monotonic
                    state_received = self._latest_low_state_monotonic
                now = time.monotonic()
                state_stale = state_received is None or now - state_received > self.low_state_timeout_s
                command_stale = updated is None or now - updated > self.command_timeout_s
                if not state_stale:
                    message = self._command_message(command, damping=command_stale)
                    if not bool(self._lowcmd_publisher.Write(message)):
                        raise RuntimeError("Official Unitree SDK2 LowCmd publisher write failed")
                next_deadline += self.command_period_s
                delay = next_deadline - time.perf_counter()
                if delay > 0.0:
                    self._publisher_stop.wait(delay)
                else:
                    next_deadline = time.perf_counter()
        except Exception as exc:
            self._publisher_error = RuntimeError(f"G1 LowCmd publisher stopped: {exc}")
            self._publisher_stop.set()

    def close(self) -> None:
        self._publisher_stop.set()
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=1.0)
        if self._lowcmd_publisher is not None:
            self._lowcmd_publisher.Close()
        if self._lowstate_subscriber is not None:
            self._lowstate_subscriber.Close()
