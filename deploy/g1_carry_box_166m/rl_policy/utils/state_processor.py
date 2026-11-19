import numpy as np
import zmq
import threading
import time


from utils.strings import unitree_joint_names
from loguru import logger
from typing import Dict
from utils.common import ZMQSubscriber, PORTS, LowStateMessage

class StateProcessor:
    """Listens to the unitree sdk channels and converts observation into isaac compatible order.
    Assumes the message in the channel follows the joint order of unitree_joint_names.
    """
    def __init__(self, robot_config, dest_joint_names):
        self.robot_type = robot_config["ROBOT_TYPE"]
        self.mocap_ip = robot_config.get("MOCAP_IP", "localhost")
        # Initialize state source
        if self.robot_type == "g1_real":
            self.robot = robot_config["robot"]
        else:
            supported_types = {"g1_29dof"}
            if self.robot_type not in supported_types:
                raise NotImplementedError(
                    f"Robot type {self.robot_type} is not supported by the UFO-Deploy release runtime"
                )

            self.low_state_port = robot_config.get(
                "LOW_STATE_PORT", PORTS.get("low_state", 55900)
            )
            state_host = robot_config.get("LOW_STATE_HOST", "127.0.0.1")
            state_endpoint = f"tcp://{state_host}:{self.low_state_port}"

            self.zmq_context = zmq.Context.instance()
            self.low_state_socket: zmq.Socket = self.zmq_context.socket(zmq.SUB)
            self.low_state_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.low_state_socket.setsockopt(zmq.CONFLATE, 1)
            self.low_state_socket.setsockopt(zmq.RCVTIMEO, 10)
            self.low_state_socket.connect(state_endpoint)
            self.latest_low_state: LowStateMessage | None = None

        # Initialize joint mapping
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [unitree_joint_names.index(name) for name in dest_joint_names]
        self.joint_names = dest_joint_names

        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)

        # create views of qpos and qvel
        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]

        self.root_quat_b = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]

        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]
        # Diagnostics carried alongside the observation.  NaN means that the
        # selected state backend did not publish the corresponding quantity.
        self.joint_acc = np.full(self.num_dof, np.nan, dtype=np.float64)
        self.joint_tau_est = np.full(self.num_dof, np.nan, dtype=np.float64)
        self.motor_temperature = np.full((self.num_dof, 2), -1, dtype=np.int64)
        self.motor_voltage = np.full(self.num_dof, np.nan, dtype=np.float64)
        self.motor_state_code = np.zeros(self.num_dof, dtype=np.uint32)
        self.low_state_tick: int | None = None
        self.low_state_mode_machine: int | None = None
        self.low_state_received_monotonic_ns: int | None = None

        self.mocap_subscribers: Dict[str, ZMQSubscriber] = {}  # Dictionary to store ZMQ subscribers
        self.mocap_threads = {}      # Dictionary to store subscriber threads
        self.mocap_data = {}         # Dictionary to store received mocap data
        self.mocap_data_lock = threading.Lock()  # Lock for thread-safe access

    def register_subscriber(self, object_name: str, port: int | None = None):
        if object_name in self.mocap_subscribers:
            return

        # init ZMQ subscriber
        port = PORTS.get(f"{object_name}_pose", port)
        subscriber = ZMQSubscriber(port)
        self.mocap_subscribers[object_name] = subscriber

        def _sub_thread(obj_name: str):
            while True:
                try:
                    pose_msg = self.mocap_subscribers[obj_name].receive_pose()
                    if pose_msg:
                        with self.mocap_data_lock:
                            self.mocap_data[f"{obj_name}_pos"] = pose_msg.position
                            self.mocap_data[f"{obj_name}_quat"] = pose_msg.quaternion
                except zmq.Again:
                    time.sleep(0.001)
                except Exception as e:
                    logger.warning(f"{obj_name} subscriber error: {e}")
                    time.sleep(0.01)

        # start subscriber thread
        th = threading.Thread(target=_sub_thread, args=(object_name,), daemon=True)
        th.start()
        self.mocap_threads[object_name] = th


    def get_mocap_data(self, key: str):
        """Thread-safe method to get mocap data"""
        with self.mocap_data_lock:
            return self.mocap_data.get(key, None)

    def _prepare_low_state(self):
        if hasattr(self, "low_state_socket"):
            self._receive_low_state()
            if not self.latest_low_state:
                return False

            low_state = self.latest_low_state
            self.root_quat_b[:] = low_state.quaternion
            self.root_ang_vel_b[:] = low_state.gyroscope

            source_joint_pos = low_state.joint_positions
            source_joint_vel = low_state.joint_velocities
            source_joint_tau = low_state.joint_torques
            for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
                self.joint_pos[dst_idx] = source_joint_pos[src_idx]
                self.joint_vel[dst_idx] = source_joint_vel[src_idx]
                self.joint_tau_est[dst_idx] = (
                    np.nan if source_joint_tau is None else source_joint_tau[src_idx]
                )

            self.low_state_tick = int(low_state.tick)
            self.low_state_mode_machine = None
            self.low_state_received_monotonic_ns = time.monotonic_ns()

            return True
        elif hasattr(self, "robot"):
            try:
                state = self.robot.read_low_state()
            except Exception as e:
                logger.warning(f"Failed to read G1 low state: {e}")
                return False

            if state is None:
                return False

            # IMU
            self.root_quat_b[:] = state.imu.quat  # [w, x, y, z]
            self.root_ang_vel_b[:] = state.imu.omega

            # Joints
            source_joint_acc = getattr(state.motor, "ddq", None)
            source_joint_tau = getattr(state.motor, "tau_est", None)
            source_temperature = getattr(state.motor, "temperature", None)
            source_voltage = getattr(state.motor, "voltage", None)
            source_state_code = getattr(state.motor, "state_code", None)
            for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
                self.joint_pos[dst_idx] = state.motor.q[src_idx]
                self.joint_vel[dst_idx] = state.motor.dq[src_idx]
                self.joint_acc[dst_idx] = (
                    np.nan if source_joint_acc is None else source_joint_acc[src_idx]
                )
                self.joint_tau_est[dst_idx] = (
                    np.nan if source_joint_tau is None else source_joint_tau[src_idx]
                )
                if source_temperature is not None:
                    self.motor_temperature[dst_idx] = source_temperature[src_idx]
                self.motor_voltage[dst_idx] = (
                    np.nan if source_voltage is None else source_voltage[src_idx]
                )
                self.motor_state_code[dst_idx] = (
                    0 if source_state_code is None else source_state_code[src_idx]
                )
            self.low_state_tick = int(state.tick)
            self.low_state_mode_machine = int(state.mode_machine)
            self.low_state_received_monotonic_ns = time.monotonic_ns()
            return True

    def _receive_low_state(self):
        """Fetch the most recent low state message from the ZMQ socket."""
        if not hasattr(self, "low_state_socket"):
            return

        while True:
            try:
                data = self.low_state_socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            try:
                self.latest_low_state = LowStateMessage.from_bytes(data)
            except Exception as exc:
                logger.warning(f"Failed to decode low state message: {exc}")
