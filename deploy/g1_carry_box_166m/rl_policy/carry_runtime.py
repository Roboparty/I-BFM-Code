"""Real-time observation and phase control for the object-aware G1 Carry policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Mapping

import joblib
from loguru import logger
import numpy as np

from utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
    quat_rotate_numpy,
    yaw_quat,
)


STATE_DIM = 98
HISTORY_DIM = 372
ACTOR_OBSERVATION_DIM = STATE_DIM + 29 + HISTORY_DIM
LATENT_DIM = 256
ONNX_INPUT_DIM = ACTOR_OBSERVATION_DIM + LATENT_DIM
OBJECT_CONTEXT_DIM = 34
MAX_LOCAL_SELF_DIM = 463
PRIVILEGED_STATE_DIM = MAX_LOCAL_SELF_DIM + OBJECT_CONTEXT_DIM
SPHERE_LOG_ONNX_INPUT_DIM = ACTOR_OBSERVATION_DIM + PRIVILEGED_STATE_DIM + LATENT_DIM


def build_carry_inference_trace(
    actor: np.ndarray,
    privileged_state: np.ndarray | None,
    latent: np.ndarray,
    actor_input: np.ndarray,
) -> dict[str, np.ndarray | None]:
    """Snapshot the exact actor-visible tensors with explicit schema slices."""
    actor_flat = np.asarray(actor, dtype=np.float32).reshape(-1)
    if actor_flat.shape != (ACTOR_OBSERVATION_DIM,):
        raise CarryDataUnavailable(
            f"invalid Carry actor observation shape: {actor_flat.shape}"
        )
    latent_flat = np.asarray(latent, dtype=np.float32).reshape(-1)
    if latent_flat.shape != (LATENT_DIM,):
        raise CarryDataUnavailable(f"invalid Carry latent shape: {latent_flat.shape}")
    privileged_flat = None
    if privileged_state is not None:
        privileged_flat = np.asarray(privileged_state, dtype=np.float32).reshape(-1)
        if privileged_flat.shape != (PRIVILEGED_STATE_DIM,):
            raise CarryDataUnavailable(
                f"invalid Carry privileged-state shape: {privileged_flat.shape}"
            )
    input_flat = np.asarray(actor_input, dtype=np.float32).reshape(-1)
    expected_input_dim = ACTOR_OBSERVATION_DIM + LATENT_DIM + (
        0 if privileged_flat is None else PRIVILEGED_STATE_DIM
    )
    if input_flat.shape != (expected_input_dim,):
        raise CarryDataUnavailable(
            f"invalid Carry actor-input trace shape: {input_flat.shape}"
        )
    return {
        "actor_input": input_flat.copy(),
        # Slice the post-controller actor, not the pre-selection observation,
        # so these arrays exactly concatenate to the actor-visible prefix.
        "state": actor_flat[:STATE_DIM].copy(),
        "last_action": actor_flat[STATE_DIM : STATE_DIM + 29].copy(),
        "history_actor": actor_flat[STATE_DIM + 29 :].copy(),
        "privileged_state": (
            None if privileged_flat is None else privileged_flat.copy()
        ),
        "latent": latent_flat.copy(),
    }


class CarryDataUnavailable(RuntimeError):
    """Raised when a safe Carry actor observation cannot be constructed."""


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values, got shape={result.shape}")
    return result


def _unit_quaternion_wxyz(value: Any, name: str) -> np.ndarray:
    quat = _vector(value, 4, name)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-6:
        raise ValueError(f"{name} has zero norm")
    return quat / norm


def _binary(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value) in (0.0, 1.0):
        return float(value)
    raise ValueError(f"{name} must be bool, 0, or 1")


@dataclass(frozen=True)
class CarryMocapFrame:
    sequence: int
    timestamp_ns: int
    robot_root_position: np.ndarray
    robot_root_quaternion_wxyz: np.ndarray
    object_position: np.ndarray
    object_quaternion_wxyz: np.ndarray
    object_linear_velocity_world: np.ndarray
    object_angular_velocity_world: np.ndarray
    hand_contact: np.ndarray
    target_position_world: np.ndarray
    object_present: float = 1.0
    task_id: float = 1.0
    robot_root_linear_velocity_world: np.ndarray | None = None
    robot_root_angular_velocity_world: np.ndarray | None = None
    hand_rigid_origin_positions_world: np.ndarray | None = None
    hand_rigid_origin_present: np.ndarray | None = None
    hand_rigid_origin_quality_grade: np.ndarray | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CarryMocapFrame":
        root = payload.get("robot_root")
        obj = payload.get("object")
        contacts = payload.get("contacts")
        if not isinstance(root, Mapping) or not isinstance(obj, Mapping) or not isinstance(contacts, Mapping):
            raise ValueError("payload requires robot_root, object, and contacts objects")
        contact = np.asarray(
            [_binary(contacts.get("left"), "contacts.left"), _binary(contacts.get("right"), "contacts.right")],
            dtype=np.float64,
        )
        object_present = _binary(payload.get("object_present", 1.0), "object_present")
        task_id = _binary(payload.get("task_id", 1.0), "task_id")
        hand_origin_positions = np.zeros((2, 3), dtype=np.float64)
        hand_origin_present = np.zeros(2, dtype=np.float64)
        hand_origin_quality = np.full(2, -1, dtype=np.int32)
        hand_origins = payload.get("hand_rigid_origins")
        if hand_origins is not None:
            if not isinstance(hand_origins, Mapping):
                raise ValueError("hand_rigid_origins must be a mapping")
            for index, side in enumerate(("left", "right")):
                entry = hand_origins.get(side)
                if entry is None:
                    continue
                if not isinstance(entry, Mapping):
                    raise ValueError(f"hand_rigid_origins.{side} must be a mapping")
                present = _binary(
                    entry.get("present", False),
                    f"hand_rigid_origins.{side}.present",
                )
                hand_origin_present[index] = present
                if entry.get("quality_grade") is not None:
                    hand_origin_quality[index] = int(entry["quality_grade"])
                if present > 0.5:
                    hand_origin_positions[index] = _vector(
                        entry.get("position"),
                        3,
                        f"hand_rigid_origins.{side}.position",
                    )
        sequence = int(payload["sequence"])
        timestamp_ns = int(payload["timestamp_ns"])
        if sequence < 0 or timestamp_ns <= 0:
            raise ValueError("sequence must be non-negative and timestamp_ns must be positive")
        return cls(
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            robot_root_position=_vector(root["position"], 3, "robot_root.position"),
            robot_root_quaternion_wxyz=_unit_quaternion_wxyz(
                root["quaternion_wxyz"], "robot_root.quaternion_wxyz"
            ),
            object_position=_vector(obj["position"], 3, "object.position"),
            object_quaternion_wxyz=_unit_quaternion_wxyz(
                obj["quaternion_wxyz"], "object.quaternion_wxyz"
            ),
            object_linear_velocity_world=_vector(
                obj["linear_velocity_world"], 3, "object.linear_velocity_world"
            ),
            object_angular_velocity_world=_vector(
                obj["angular_velocity_world"], 3, "object.angular_velocity_world"
            ),
            hand_contact=contact,
            target_position_world=_vector(payload["target_position_world"], 3, "target_position_world"),
            object_present=object_present,
            task_id=task_id,
            robot_root_linear_velocity_world=_vector(
                root.get("linear_velocity_world", [0.0, 0.0, 0.0]),
                3,
                "robot_root.linear_velocity_world",
            ),
            robot_root_angular_velocity_world=(
                None
                if root.get("angular_velocity_world") is None
                else _vector(
                    root["angular_velocity_world"],
                    3,
                    "robot_root.angular_velocity_world",
                )
            ),
            hand_rigid_origin_positions_world=hand_origin_positions,
            hand_rigid_origin_present=hand_origin_present,
            hand_rigid_origin_quality_grade=hand_origin_quality,
        )


class CarryMocapSubscriber:
    """Latest-value JSON subscriber with receipt-time and sequence watchdogs."""

    def __init__(self, address: str, timeout_ms: float) -> None:
        import zmq

        self.timeout_s = float(timeout_ms) / 1000.0
        if self.timeout_s <= 0.0:
            raise ValueError("mocap timeout must be positive")
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.RCVTIMEO, 0)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(str(address))
        self.latest: CarryMocapFrame | None = None
        self.last_receive_monotonic: float | None = None
        self.last_sequence: int | None = None

    def poll(self) -> CarryMocapFrame:
        import zmq

        while True:
            try:
                raw = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                payload = json.loads(raw.decode("utf-8"))
                frame = CarryMocapFrame.from_mapping(payload)
            except Exception as exc:
                raise CarryDataUnavailable(f"invalid mocap packet: {exc}") from exc
            if self.last_sequence is not None and frame.sequence <= self.last_sequence:
                raise CarryDataUnavailable(
                    f"non-increasing mocap sequence {frame.sequence} <= {self.last_sequence}"
                )
            self.latest = frame
            self.last_sequence = frame.sequence
            self.last_receive_monotonic = time.monotonic()

        if self.latest is None or self.last_receive_monotonic is None:
            raise CarryDataUnavailable("no valid mocap packet received")
        age = time.monotonic() - self.last_receive_monotonic
        if age > self.timeout_s:
            raise CarryDataUnavailable(f"mocap packet is stale ({age * 1000.0:.0f} ms)")
        return self.latest


class G1HandForwardKinematics:
    """Compute the wrist-yaw proxy points and collision capsules used in training."""

    def __init__(
        self,
        xml_path: str | Path,
        joint_names: list[str],
        offsets: np.ndarray,
        capsule_segments: np.ndarray | None = None,
        privileged_body_names: list[str] | None = None,
        privileged_torso_body_name: str = "torso_link",
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser().resolve()))
        self.data = mujoco.MjData(self.model)
        self.joint_qpos_addresses = np.asarray(
            [int(self.model.jnt_qposadr[self.model.joint(name).id]) for name in joint_names], dtype=np.int32
        )
        self.joint_qvel_addresses = np.asarray(
            [int(self.model.jnt_dofadr[self.model.joint(name).id]) for name in joint_names], dtype=np.int32
        )
        self.body_ids = (
            int(self.model.body("left_wrist_yaw_link").id),
            int(self.model.body("right_wrist_yaw_link").id),
        )
        self.offsets = np.asarray(offsets, dtype=np.float64).reshape(2, 3)
        self.capsule_segments = (
            np.repeat(self.offsets[:, None, :], 2, axis=1)
            if capsule_segments is None
            else np.asarray(capsule_segments, dtype=np.float64).reshape(2, 2, 3)
        )
        self.privileged_body_ids = (
            None
            if privileged_body_names is None
            else np.asarray(
                [int(self.model.body(name).id) for name in privileged_body_names],
                dtype=np.int32,
            )
        )
        self.privileged_torso_body_id = (
            None
            if privileged_body_names is None
            else int(self.model.body(privileged_torso_body_name).id)
        )
        if self.privileged_body_ids is not None:
            if self.privileged_body_ids.shape != (30,):
                raise ValueError(
                    "sphere-log privileged_body_names must contain the exact 30 training bodies, "
                    f"got {self.privileged_body_ids.shape[0]}"
                )
            if self.privileged_body_ids[0] != int(self.model.body("pelvis").id):
                raise ValueError("sphere-log privileged_body_names must start with pelvis")
            if self.privileged_torso_body_id not in self.privileged_body_ids:
                raise ValueError("sphere-log privileged torso must be present in privileged_body_names")

    def hand_geometry(
        self,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        joint_position: np.ndarray,
        *,
        root_linear_velocity_world: np.ndarray | None = None,
        root_angular_velocity_body: np.ndarray | None = None,
        joint_velocity: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:3] = root_position
        self.data.qpos[3:7] = root_quaternion_wxyz
        self.data.qpos[self.joint_qpos_addresses] = joint_position
        self.data.qvel[:] = 0.0
        if root_linear_velocity_world is not None:
            self.data.qvel[:3] = _vector(
                root_linear_velocity_world, 3, "root_linear_velocity_world"
            )
        if root_angular_velocity_body is not None:
            # MuJoCo free-joint rotational qvel is expressed in the child-body frame,
            # matching the Unitree IMU gyroscope convention.
            self.data.qvel[3:6] = _vector(
                root_angular_velocity_body, 3, "root_angular_velocity_body"
            )
        if joint_velocity is not None:
            self.data.qvel[self.joint_qvel_addresses] = _vector(
                joint_velocity, 29, "joint_velocity"
            )
        self.mujoco.mj_forward(self.model, self.data)
        points = np.empty((2, 3), dtype=np.float64)
        segments = np.empty((2, 2, 3), dtype=np.float64)
        for index, body_id in enumerate(self.body_ids):
            body_quat = self.data.xquat[body_id].reshape(1, 4)
            points[index] = self.data.xpos[body_id] + quat_rotate_numpy(
                body_quat, self.offsets[index].reshape(1, 3)
            )[0]
            segments[index] = self.data.xpos[body_id] + quat_rotate_numpy(
                np.repeat(body_quat, 2, axis=0), self.capsule_segments[index]
            )
        return points, segments

    def privileged_state(self, object_context: np.ndarray) -> np.ndarray:
        """Build the training-time 497D max-local-self Carry observation."""

        if self.privileged_body_ids is None or self.privileged_torso_body_id is None:
            raise RuntimeError("privileged FK bodies were not configured")
        body_pos = np.asarray(
            [self.data.xpos[body_id] for body_id in self.privileged_body_ids],
            dtype=np.float64,
        )
        body_quat = np.asarray(
            [self.data.xquat[body_id] for body_id in self.privileged_body_ids],
            dtype=np.float64,
        )
        velocities = []
        for body_id in self.privileged_body_ids:
            velocity = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_objectVelocity(
                self.model,
                self.data,
                self.mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                velocity,
                0,
            )
            velocities.append(velocity)
        velocities_array = np.asarray(velocities, dtype=np.float64)
        body_ang_vel = velocities_array[:, :3]
        body_vel = velocities_array[:, 3:]

        torso_matches = np.flatnonzero(
            self.privileged_body_ids == self.privileged_torso_body_id
        )
        if torso_matches.shape != (1,):
            raise RuntimeError("privileged torso body must occur exactly once")
        torso_index = int(torso_matches[0])
        head_offset_local = np.asarray((0.0, 0.0, 0.35), dtype=np.float64)
        head_offset_world = (
            self.data.xmat[self.privileged_torso_body_id].reshape(3, 3)
            @ head_offset_local
        )
        head_pos = body_pos[torso_index] + head_offset_world
        head_quat = body_quat[torso_index]
        head_ang_vel = body_ang_vel[torso_index]
        # Preserve the frozen validation implementation exactly, including its
        # local head-offset cross term.
        head_vel = body_vel[torso_index] + np.cross(
            head_ang_vel, head_offset_local
        )

        max_local_self = compute_humanoid_observations_max_numpy(
            np.concatenate((body_pos, head_pos[None]), axis=0),
            np.concatenate((body_quat, head_quat[None]), axis=0),
            np.concatenate((body_vel, head_vel[None]), axis=0),
            np.concatenate((body_ang_vel, head_ang_vel[None]), axis=0),
        )
        privileged = np.concatenate(
            (max_local_self, _vector(object_context, OBJECT_CONTEXT_DIM, "object_context"))
        ).astype(np.float32)
        if privileged.shape != (PRIVILEGED_STATE_DIM,) or not np.all(np.isfinite(privileged)):
            raise CarryDataUnavailable(
                f"invalid sphere-log privileged_state shape/value: {privileged.shape}"
            )
        return privileged

    def hand_points(
        self, root_position: np.ndarray, root_quaternion_wxyz: np.ndarray, joint_position: np.ndarray
    ) -> np.ndarray:
        return self.hand_geometry(root_position, root_quaternion_wxyz, joint_position)[0]


def compute_humanoid_observations_max_numpy(
    body_position_world: np.ndarray,
    body_quaternion_wxyz: np.ndarray,
    body_linear_velocity_world: np.ndarray,
    body_angular_velocity_world: np.ndarray,
) -> np.ndarray:
    """NumPy equivalent of training ``compute_humanoid_observations_max``."""

    positions = np.asarray(body_position_world, dtype=np.float64).reshape(31, 3)
    quaternions = np.asarray(body_quaternion_wxyz, dtype=np.float64).reshape(31, 4)
    linear_velocity = np.asarray(body_linear_velocity_world, dtype=np.float64).reshape(31, 3)
    angular_velocity = np.asarray(body_angular_velocity_world, dtype=np.float64).reshape(31, 3)
    if not all(
        np.all(np.isfinite(value))
        for value in (positions, quaternions, linear_velocity, angular_velocity)
    ):
        raise CarryDataUnavailable("non-finite body state in privileged observation")

    root_rotation = matrix_from_quat(quaternions[0].reshape(1, 4))[0]
    heading = math.atan2(float(root_rotation[1, 0]), float(root_rotation[0, 0]))
    cos_heading = math.cos(heading)
    sin_heading = math.sin(heading)
    heading_inverse = np.asarray(
        ((cos_heading, sin_heading, 0.0), (-sin_heading, cos_heading, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )

    local_position = (heading_inverse @ (positions - positions[0]).T).T[1:].reshape(-1)
    body_rotation = matrix_from_quat(quaternions)
    local_rotation = np.einsum("ij,bjk->bik", heading_inverse, body_rotation)
    local_rotation_tan_norm = np.concatenate(
        (local_rotation[:, :, 0], local_rotation[:, :, 2]), axis=-1
    ).reshape(-1)
    local_linear_velocity = (heading_inverse @ linear_velocity.T).T.reshape(-1)
    local_angular_velocity = (heading_inverse @ angular_velocity.T).T.reshape(-1)
    result = np.concatenate(
        (
            positions[0, 2:3],
            local_position,
            local_rotation_tan_norm,
            local_linear_velocity,
            local_angular_velocity,
        )
    ).astype(np.float32)
    if result.shape != (MAX_LOCAL_SELF_DIM,) or not np.all(np.isfinite(result)):
        raise CarryDataUnavailable(
            f"invalid max-local-self observation shape/value: {result.shape}"
        )
    return result


def _segment_aabb_distance(segment: np.ndarray, half_size: np.ndarray) -> float:
    """Exact Euclidean distance between a line segment and an axis-aligned box."""

    start, end = np.asarray(segment, dtype=np.float64).reshape(2, 3)
    half = np.asarray(half_size, dtype=np.float64).reshape(3)
    direction = end - start
    breaks = [0.0, 1.0]
    for axis in range(3):
        if abs(direction[axis]) < 1.0e-12:
            continue
        for boundary in (-half[axis], half[axis]):
            value = (boundary - start[axis]) / direction[axis]
            if 0.0 < value < 1.0:
                breaks.append(float(value))
    breaks = sorted(set(breaks))

    def squared_distance(parameter: float) -> float:
        point = start + parameter * direction
        outside = np.maximum(np.abs(point) - half, 0.0)
        return float(np.dot(outside, outside))

    best = min(squared_distance(value) for value in breaks)
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        middle = 0.5 * (lower + upper)
        point = start + middle * direction
        active = np.abs(point) > half
        if not bool(np.any(active)):
            return 0.0
        boundary = np.sign(point[active]) * half[active]
        residual = start[active] - boundary
        active_direction = direction[active]
        denominator = float(np.dot(active_direction, active_direction))
        if denominator > 1.0e-12:
            optimum = float(np.clip(-np.dot(active_direction, residual) / denominator, lower, upper))
            best = min(best, squared_distance(optimum))
    return math.sqrt(max(best, 0.0))


class HandBoxContactEstimator:
    """Estimate per-hand contact from FK capsules and oriented-box distance."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.half_size = _vector(config.get("half_size"), 3, "contact_estimator.half_size")
        if np.any(self.half_size <= 0.0):
            raise ValueError("contact_estimator.half_size must be positive")
        self.geom_position = _vector(
            config.get("geom_position", [0.0, 0.0, 0.0]), 3, "contact_estimator.geom_position"
        )
        self.geom_quaternion = _unit_quaternion_wxyz(
            config.get("geom_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
            "contact_estimator.geom_quaternion_wxyz",
        )
        self.enter_distance = float(config.get("enter_distance", 0.045))
        self.exit_distance = float(config.get("exit_distance", 0.060))
        self.capsule_radius = float(config.get("capsule_radius", 0.035))
        self.use_hand_rigid_origins = bool(
            config.get("use_hand_rigid_origins", False)
        )
        self.hand_origin_enter_distance = self._distance_pair(
            config.get("hand_origin_enter_distance", self.enter_distance),
            "hand_origin_enter_distance",
        )
        self.hand_origin_exit_distance = self._distance_pair(
            config.get("hand_origin_exit_distance", self.exit_distance),
            "hand_origin_exit_distance",
        )
        self.hand_origin_max_fk_distance = float(
            config.get("hand_origin_max_fk_distance", np.inf)
        )
        if not 0.0 <= self.enter_distance < self.exit_distance:
            raise ValueError("contact estimator requires 0 <= enter_distance < exit_distance")
        if not np.isfinite(self.capsule_radius) or self.capsule_radius <= 0.0:
            raise ValueError("contact_estimator.capsule_radius must be positive")
        if np.any(self.hand_origin_enter_distance < 0.0) or np.any(
            self.hand_origin_enter_distance >= self.hand_origin_exit_distance
        ):
            raise ValueError(
                "hand-origin contact requires 0 <= enter_distance < exit_distance"
            )
        if self.hand_origin_max_fk_distance <= 0.0 or np.isnan(
            self.hand_origin_max_fk_distance
        ):
            raise ValueError("hand_origin_max_fk_distance must be positive")
        self.contact = np.zeros(2, dtype=np.float64)
        self.last_surface_distance = np.full(2, np.inf, dtype=np.float64)
        self.last_fk_surface_distance = np.full(2, np.inf, dtype=np.float64)
        self.last_hand_origin_surface_distance = np.full(2, np.inf, dtype=np.float64)
        self.last_hand_origin_fk_distance = np.full(2, np.inf, dtype=np.float64)
        self.last_geometry_source = np.asarray(
            ["fk_capsule", "fk_capsule"], dtype=object
        )

    @staticmethod
    def _distance_pair(value: Any, name: str) -> np.ndarray:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if values.shape == (1,):
            values = np.repeat(values, 2)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be one finite value or a left/right pair")
        return values

    def set_hand_origin_thresholds(
        self, enter_distance: np.ndarray, exit_distance: np.ndarray
    ) -> None:
        enter = self._distance_pair(enter_distance, "hand_origin_enter_distance")
        exit_ = self._distance_pair(exit_distance, "hand_origin_exit_distance")
        if np.any(enter < 0.0) or np.any(enter >= exit_):
            raise ValueError(
                "hand-origin contact requires 0 <= enter_distance < exit_distance"
            )
        self.hand_origin_enter_distance = enter
        self.hand_origin_exit_distance = exit_

    def reset(self) -> None:
        self.contact.fill(0.0)
        self.last_surface_distance.fill(np.inf)
        self.last_fk_surface_distance.fill(np.inf)
        self.last_hand_origin_surface_distance.fill(np.inf)
        self.last_hand_origin_fk_distance.fill(np.inf)
        self.last_geometry_source[:] = "fk_capsule"

    def update(self, frame: CarryMocapFrame, hand_segments_world: np.ndarray) -> np.ndarray:
        segments = np.asarray(hand_segments_world, dtype=np.float64).reshape(2, 2, 3)
        object_quat = frame.object_quaternion_wxyz.reshape(1, 4)
        geom_quat = quat_mul(object_quat, self.geom_quaternion.reshape(1, 4))[0]
        geom_center = frame.object_position + quat_rotate_numpy(
            object_quat, self.geom_position.reshape(1, 3)
        )[0]
        geom_quats = np.repeat(geom_quat.reshape(1, 4), 4, axis=0)
        local_segments = quat_rotate_inverse_numpy(
            geom_quats, (segments - geom_center.reshape(1, 1, 3)).reshape(4, 3)
        ).reshape(2, 2, 3)
        fk_distance = np.asarray(
            [
                max(_segment_aabb_distance(local_segments[index], self.half_size) - self.capsule_radius, 0.0)
                for index in range(2)
            ],
            dtype=np.float64,
        )
        self.last_fk_surface_distance = fk_distance
        origin_positions = frame.hand_rigid_origin_positions_world
        origin_present = frame.hand_rigid_origin_present
        marker_available = np.zeros(2, dtype=bool)
        origin_distance = np.full(2, np.inf, dtype=np.float64)
        if (
            self.use_hand_rigid_origins
            and origin_positions is not None
            and origin_present is not None
        ):
            positions = np.asarray(origin_positions, dtype=np.float64).reshape(2, 3)
            present = np.asarray(origin_present, dtype=np.float64).reshape(2) > 0.5
            marker_available = present & np.all(np.isfinite(positions), axis=1)
            segment_start = segments[:, 0]
            segment_direction = segments[:, 1] - segment_start
            segment_length_sq = np.sum(segment_direction * segment_direction, axis=1)
            projection = np.divide(
                np.sum((positions - segment_start) * segment_direction, axis=1),
                segment_length_sq,
                out=np.zeros(2, dtype=np.float64),
                where=segment_length_sq > 1.0e-12,
            )
            closest = segment_start + np.clip(projection, 0.0, 1.0)[:, None] * segment_direction
            origin_fk_distance = np.linalg.norm(positions - closest, axis=1)
            self.last_hand_origin_fk_distance = origin_fk_distance
            marker_available &= origin_fk_distance <= self.hand_origin_max_fk_distance
            if bool(np.any(marker_available)):
                marker_local = quat_rotate_inverse_numpy(
                    np.repeat(geom_quat.reshape(1, 4), 2, axis=0),
                    positions - geom_center.reshape(1, 3),
                )
                outside = np.maximum(np.abs(marker_local) - self.half_size, 0.0)
                origin_distance = np.linalg.norm(outside, axis=1)
        self.last_hand_origin_surface_distance = origin_distance
        distance = np.where(marker_available, origin_distance, fk_distance)
        enter_distance = np.where(
            marker_available, self.hand_origin_enter_distance, self.enter_distance
        )
        exit_distance = np.where(
            marker_available, self.hand_origin_exit_distance, self.exit_distance
        )
        self.last_geometry_source = np.where(
            marker_available, "rigid_origin", "fk_capsule"
        )
        self.last_surface_distance = distance
        for index in range(2):
            if self.contact[index] > 0.5:
                self.contact[index] = float(distance[index] <= exit_distance[index])
            else:
                self.contact[index] = float(distance[index] < enter_distance[index])
        self.contact *= float(frame.object_present)
        return self.contact.copy()


def _rotation_tan_norm_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    rotation = matrix_from_quat(quaternion_wxyz.reshape(1, 4))[0]
    return np.concatenate((rotation[:, 0], rotation[:, 2]))


class CarryActorObservation:
    """Build the exact 98D state, 29D last action, and 372D history schema."""

    def __init__(self, default_joint_position: np.ndarray, hand_fk: G1HandForwardKinematics) -> None:
        self.default = np.asarray(default_joint_position, dtype=np.float64).reshape(29)
        self.hand_fk = hand_fk
        self.history = {
            "actions": np.zeros((4, 29), dtype=np.float32),
            "base_ang_vel": np.zeros((4, 3), dtype=np.float32),
            "dof_pos": np.zeros((4, 29), dtype=np.float32),
            "dof_vel": np.zeros((4, 29), dtype=np.float32),
            "projected_gravity": np.zeros((4, 3), dtype=np.float32),
        }

    def reset(self) -> None:
        for value in self.history.values():
            value.fill(0.0)

    def build(
        self,
        frame: CarryMocapFrame,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        imu_quaternion_wxyz: np.ndarray,
        imu_angular_velocity_body: np.ndarray,
        last_action: np.ndarray,
        hand_points_world: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        q = _vector(joint_position, 29, "joint_position")
        dq = _vector(joint_velocity, 29, "joint_velocity")
        imu_quat = _unit_quaternion_wxyz(imu_quaternion_wxyz, "imu_quaternion_wxyz")
        imu_omega = _vector(imu_angular_velocity_body, 3, "imu_angular_velocity_body")
        action = _vector(last_action, 29, "last_action")
        root_quat = frame.robot_root_quaternion_wxyz.reshape(1, 4)

        projected_gravity = quat_rotate_inverse_numpy(
            imu_quat.reshape(1, 4), np.asarray([[0.0, 0.0, -1.0]])
        )[0]
        base_ang_vel = 0.25 * imu_omega
        object_pos_local = quat_rotate_inverse_numpy(
            root_quat, (frame.object_position - frame.robot_root_position).reshape(1, 3)
        )[0]
        object_quat_local = quat_mul(
            quat_conjugate(root_quat), frame.object_quaternion_wxyz.reshape(1, 4)
        )[0]
        object_lin_vel_local = quat_rotate_inverse_numpy(
            root_quat, frame.object_linear_velocity_world.reshape(1, 3)
        )[0]
        object_ang_vel_local = quat_rotate_inverse_numpy(
            root_quat, frame.object_angular_velocity_world.reshape(1, 3)
        )[0]
        target_delta_local = quat_rotate_inverse_numpy(
            root_quat, (frame.target_position_world - frame.object_position).reshape(1, 3)
        )[0]
        hand_points = (
            self.hand_fk.hand_points(frame.robot_root_position, frame.robot_root_quaternion_wxyz, q)
            if hand_points_world is None
            else np.asarray(hand_points_world, dtype=np.float64).reshape(2, 3)
        )
        root_quats = np.repeat(root_quat, 2, axis=0)
        hand_pos_local = quat_rotate_inverse_numpy(
            root_quats, hand_points - frame.robot_root_position.reshape(1, 3)
        ).reshape(-1)
        hand_minus_object = quat_rotate_inverse_numpy(
            root_quats, hand_points - frame.object_position.reshape(1, 3)
        ).reshape(-1)
        present = float(frame.object_present)
        object_context = np.concatenate(
            (
                object_pos_local,
                _rotation_tan_norm_wxyz(object_quat_local),
                object_lin_vel_local,
                object_ang_vel_local,
                target_delta_local,
                frame.hand_contact,
                hand_pos_local,
                hand_minus_object,
                [present, float(frame.task_id)],
            )
        )
        object_context[:-2] *= present
        state = np.concatenate((q - self.default, dq, projected_gravity, base_ang_vel, object_context))
        raw = {
            "actions": action,
            "base_ang_vel": base_ang_vel,
            "dof_pos": q - self.default,
            "dof_vel": dq,
            "projected_gravity": projected_gravity,
        }
        history = np.concatenate([self.history[key].reshape(-1) for key in sorted(self.history)])
        for key, value in raw.items():
            self.history[key][1:] = self.history[key][:-1]
            self.history[key][0] = value
        result = {
            "state": state.astype(np.float32),
            "last_action": action.astype(np.float32),
            "history_actor": history.astype(np.float32),
        }
        if result["state"].shape != (STATE_DIM,) or history.shape != (HISTORY_DIM,):
            raise RuntimeError(f"Carry observation schema mismatch: state={state.shape}, history={history.shape}")
        return result


def project_z(value: np.ndarray) -> np.ndarray:
    z = np.asarray(value, dtype=np.float32).reshape(-1)
    if z.shape != (LATENT_DIM,) or not np.all(np.isfinite(z)):
        raise ValueError(f"latent must be finite {LATENT_DIM}D, got {z.shape}")
    norm = float(np.linalg.norm(z))
    if norm < 1.0e-8:
        raise ValueError("cannot project zero latent")
    return z * (math.sqrt(LATENT_DIM) / norm)


def _smoothstep01(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def load_phase_latents(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()

    def scalar(name: str) -> np.ndarray:
        value = joblib.load(root / f"z_{name}.pkl")
        if isinstance(value, Mapping):
            raise TypeError(f"z_{name}.pkl must contain one latent")
        return _vector(value, LATENT_DIM, f"z_{name}").astype(np.float32)

    def bank(name: str) -> dict[int, np.ndarray]:
        value = joblib.load(root / f"z_{name}.pkl")
        if not isinstance(value, Mapping):
            raise TypeError(f"z_{name}.pkl must contain a latent bank")
        return {direction: _vector(value[direction], LATENT_DIM, f"z_{name}[{direction}]").astype(np.float32)
                for direction in (-1, 1)}

    result = {name: scalar(name) for name in ("approach", "lift", "transport", "transport_slow", "align", "lower", "place")}
    result["turn"] = bank("turn")
    result["transport_turn"] = bank("transport_turn")
    return result


def load_hybrid_direct_place_latents(path: str | Path) -> dict[str, Any]:
    """Load the exact latent surface used by the isolated 80.1M hybrid rollout."""

    root = Path(path).expanduser().resolve()
    result = load_phase_latents(root)
    for name in (
        "hold",
        "direct_place",
        "direct_place_slow",
        "direct_place_lower",
        "direct_place_release",
        "direct_place_recover",
    ):
        value = joblib.load(root / f"z_{name}.pkl")
        if isinstance(value, Mapping):
            raise TypeError(f"z_{name}.pkl must contain one latent")
        result[name] = _vector(value, LATENT_DIM, f"z_{name}").astype(np.float32)
    result["direct_place_stand"] = result["hold"]
    return result


def load_fixed_goal_latent(path: str | Path) -> np.ndarray:
    """Load one checkpoint-specific goal latent and project it onto the FB sphere."""

    resolved = Path(path).expanduser().resolve()
    value = joblib.load(resolved)
    if isinstance(value, Mapping):
        raise TypeError(f"fixed goal latent must contain one vector, got a mapping from {resolved}")
    return project_z(_vector(value, LATENT_DIM, "fixed_goal_latent")).astype(np.float32)


def load_tracking_latents(path: str | Path) -> np.ndarray:
    """Load and validate one checkpoint-specific ``[T, 256]`` tracking sequence."""

    resolved = Path(path).expanduser().resolve()
    value = np.asarray(joblib.load(resolved), dtype=np.float32)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != LATENT_DIM:
        raise ValueError(
            f"tracking latent must have shape [T, {LATENT_DIM}], got {value.shape} from {resolved}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"tracking latent contains non-finite values: {resolved}")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError(f"tracking latent contains a zero row: {resolved}")
    return (value * (math.sqrt(LATENT_DIM) / norms)).astype(np.float32)


def load_get_up_latent(path: str | Path) -> np.ndarray:
    """Load a fixed reward-inference latent or a legacy tracking sequence."""

    resolved = Path(path).expanduser().resolve()
    value = np.asarray(joblib.load(resolved), dtype=np.float32)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != LATENT_DIM:
        raise ValueError(f"get-up latent must have shape [{LATENT_DIM}] or [T, {LATENT_DIM}], got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"get-up latent contains non-finite values: {resolved}")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError(f"get-up latent contains a zero row: {resolved}")
    return (value * (math.sqrt(LATENT_DIM) / norms)).astype(np.float32)


class TrackingReferenceAnchor:
    """Map one reference motion into the current world using root XY/yaw alignment."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.reference_root_position = _vector(
            config["root_position"], 3, "tracking_reference.root_position"
        )
        self.reference_root_quaternion = _unit_quaternion_wxyz(
            config["root_quaternion_wxyz"], "tracking_reference.root_quaternion_wxyz"
        )
        self.reference_object_position = _vector(
            config["object_position"], 3, "tracking_reference.object_position"
        )
        self.reference_object_quaternion = _unit_quaternion_wxyz(
            config["object_quaternion_wxyz"], "tracking_reference.object_quaternion_wxyz"
        )
        self.reference_target_position = _vector(
            config["target_position"], 3, "tracking_reference.target_position"
        )
        self.max_object_position_error = float(config.get("max_object_position_error", 0.12))
        orientation_limit = config.get("max_object_orientation_error_deg", 30.0)
        self.max_object_orientation_error_deg = (
            None if orientation_limit is None else float(orientation_limit)
        )
        if not np.isfinite(self.max_object_position_error) or self.max_object_position_error <= 0.0:
            raise ValueError("tracking_reference.max_object_position_error must be positive")
        if self.max_object_orientation_error_deg is not None and not (
            0.0 < self.max_object_orientation_error_deg <= 180.0
        ):
            raise ValueError(
                "tracking_reference.max_object_orientation_error_deg must be in (0, 180]"
            )
        self.alignment_quaternion: np.ndarray | None = None
        self.alignment_translation: np.ndarray | None = None
        self.expected_object_position: np.ndarray | None = None
        self.expected_object_quaternion: np.ndarray | None = None
        self.target_position_world: np.ndarray | None = None

    def reset(self) -> None:
        self.alignment_quaternion = None
        self.alignment_translation = None
        self.expected_object_position = None
        self.expected_object_quaternion = None
        self.target_position_world = None

    def align(self, frame: CarryMocapFrame) -> CarryMocapFrame:
        if self.alignment_quaternion is None:
            actual_yaw = yaw_quat(frame.robot_root_quaternion_wxyz.reshape(1, 4))[0]
            reference_yaw = yaw_quat(self.reference_root_quaternion.reshape(1, 4))[0]
            alignment = quat_mul(
                actual_yaw.reshape(1, 4), quat_conjugate(reference_yaw.reshape(1, 4))
            )[0]
            rotated_reference_root = quat_rotate_numpy(
                alignment.reshape(1, 4), self.reference_root_position.reshape(1, 3)
            )[0]
            translation = frame.robot_root_position - rotated_reference_root
            # Training and LuMo share the same Z-up floor frame. Align only root
            # XY and yaw so a different standing pelvis height cannot move the
            # box/goal vertically away from the ground.
            translation[2] = 0.0
            expected_object_position = translation + quat_rotate_numpy(
                alignment.reshape(1, 4), self.reference_object_position.reshape(1, 3)
            )[0]
            expected_object_quaternion = quat_mul(
                alignment.reshape(1, 4), self.reference_object_quaternion.reshape(1, 4)
            )[0]
            target_position = translation + quat_rotate_numpy(
                alignment.reshape(1, 4), self.reference_target_position.reshape(1, 3)
            )[0]

            position_error = float(np.linalg.norm(frame.object_position - expected_object_position))
            orientation_dot = float(
                np.clip(abs(np.dot(frame.object_quaternion_wxyz, expected_object_quaternion)), 0.0, 1.0)
            )
            orientation_error = math.degrees(2.0 * math.acos(orientation_dot))
            if position_error > self.max_object_position_error:
                raise CarryDataUnavailable(
                    "tracking start box position mismatch: "
                    f"error={position_error:.3f}m > {self.max_object_position_error:.3f}m; "
                    f"expected={expected_object_position.tolist()} actual={frame.object_position.tolist()}"
                )
            if (
                self.max_object_orientation_error_deg is not None
                and orientation_error > self.max_object_orientation_error_deg
            ):
                raise CarryDataUnavailable(
                    "tracking start box orientation mismatch: "
                    f"error={orientation_error:.1f}deg > {self.max_object_orientation_error_deg:.1f}deg"
                )
            self.alignment_quaternion = alignment
            self.alignment_translation = translation
            self.expected_object_position = expected_object_position
            self.expected_object_quaternion = expected_object_quaternion
            self.target_position_world = target_position
            logger.info(
                "Carry tracking anchor ready: box_error={:.3f}m orientation_error={:.1f}deg "
                "target={}".format(position_error, orientation_error, target_position.tolist())
            )
        assert self.target_position_world is not None
        return replace(frame, target_position_world=self.target_position_world.copy())


def _bearing_deg(rotation: np.ndarray, direction_xy: np.ndarray) -> float:
    local = rotation.T @ np.asarray((direction_xy[0], direction_xy[1], 0.0), dtype=np.float64)
    return math.degrees(math.atan2(float(local[1]), float(local[0])))


def _transport_turn_weight(
    *,
    goal_distance: float,
    bearing_deg: float,
    turn_boost_distance: float,
    place_distance: float,
    heading_blend_full_deg: float,
    far_max_turn_blend: float,
    near_max_turn_blend: float,
) -> float:
    if turn_boost_distance <= place_distance:
        raise ValueError("transport turn boost distance must exceed place distance")
    near = float(
        np.clip(
            (turn_boost_distance - goal_distance)
            / (turn_boost_distance - place_distance),
            0.0,
            1.0,
        )
    )
    near = near * near * (3.0 - 2.0 * near)
    turn_cap = (1.0 - near) * far_max_turn_blend + near * near_max_turn_blend
    heading_weight = float(np.clip(abs(bearing_deg) / heading_blend_full_deg, 0.0, 1.0))
    return heading_weight * turn_cap


class InitialCarryScene:
    """Validate initial geometry and freeze the selected target at enable time."""

    def __init__(
        self,
        config: Mapping[str, Any],
        box_geom_quaternion_wxyz: Any,
        box_geom_position: Any = (0.0, 0.0, 0.0),
    ) -> None:
        self.expected_box_distance = float(config["expected_box_distance"])
        self.box_distance_tolerance = float(config.get("box_distance_tolerance", 0.10))
        self.max_box_bearing_deg = float(config.get("max_box_bearing_deg", 8.0))
        self.expected_box_center_height = float(config["expected_box_center_height"])
        self.box_center_height_tolerance = float(
            config.get("box_center_height_tolerance", 0.03)
        )
        self.max_box_yaw_error_deg = float(config.get("max_box_yaw_error_deg", 10.0))
        self.target_mode = str(config.get("target_mode", "relative_forward")).lower()
        if self.target_mode == "relative_forward":
            self.target_forward_distance = float(config["target_forward_distance"])
        elif self.target_mode == "configured_world":
            self.target_forward_distance = None
        else:
            raise ValueError(
                "initial_scene.target_mode must be 'relative_forward' or 'configured_world'"
            )
        self.box_geom_quaternion = _unit_quaternion_wxyz(
            box_geom_quaternion_wxyz, "initial_scene.box_geom_quaternion_wxyz"
        )
        self.box_geom_position = _vector(
            box_geom_position, 3, "initial_scene.box_geom_position"
        )
        self.box_pose_semantics = str(
            config.get("box_pose_semantics", "training_reference")
        ).lower()
        if self.box_pose_semantics not in {"training_reference", "physical_geom"}:
            raise ValueError(
                "initial_scene.box_pose_semantics must be 'training_reference' "
                "or 'physical_geom'"
            )
        values = [
            self.expected_box_distance,
            self.box_distance_tolerance,
            self.max_box_bearing_deg,
            self.expected_box_center_height,
            self.box_center_height_tolerance,
            self.max_box_yaw_error_deg,
        ]
        if self.target_forward_distance is not None:
            values.append(self.target_forward_distance)
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("initial_scene distances, tolerances, heights, and angles must be positive")
        self.target_position_world: np.ndarray | None = None

    def reset(self) -> None:
        self.target_position_world = None

    def apply(self, frame: CarryMocapFrame) -> CarryMocapFrame:
        if self.target_position_world is None:
            root_rotation = matrix_from_quat(
                frame.robot_root_quaternion_wxyz.reshape(1, 4)
            )[0]
            physical_box_position = frame.object_position + quat_rotate_numpy(
                frame.object_quaternion_wxyz.reshape(1, 4),
                self.box_geom_position.reshape(1, 3),
            )[0]
            checked_box_position = (
                physical_box_position
                if self.box_pose_semantics == "physical_geom"
                else frame.object_position
            )
            box_delta = checked_box_position - frame.robot_root_position
            box_distance = float(np.linalg.norm(box_delta[:2]))
            box_bearing = abs(_bearing_deg(root_rotation, box_delta[:2]))
            box_height_error = abs(
                float(checked_box_position[2]) - self.expected_box_center_height
            )
            physical_box_quaternion = quat_mul(
                frame.object_quaternion_wxyz.reshape(1, 4),
                self.box_geom_quaternion.reshape(1, 4),
            )[0]
            physical_box_rotation = matrix_from_quat(
                physical_box_quaternion.reshape(1, 4)
            )[0]
            root_forward = root_rotation[:2, 0]
            root_forward /= max(float(np.linalg.norm(root_forward)), 1.0e-8)
            box_forward = physical_box_rotation[:2, 0]
            box_forward /= max(float(np.linalg.norm(box_forward)), 1.0e-8)
            yaw_dot = float(np.clip(np.dot(root_forward, box_forward), -1.0, 1.0))
            box_yaw_error = math.degrees(math.acos(yaw_dot))

            failures = []
            if abs(box_distance - self.expected_box_distance) > self.box_distance_tolerance:
                failures.append(
                    f"distance={box_distance:.3f}m expected={self.expected_box_distance:.3f}"
                    f"+/-{self.box_distance_tolerance:.3f}m"
                )
            if box_bearing > self.max_box_bearing_deg:
                failures.append(
                    f"bearing={box_bearing:.1f}deg > {self.max_box_bearing_deg:.1f}deg"
                )
            if box_height_error > self.box_center_height_tolerance:
                failures.append(
                    f"height={checked_box_position[2]:.3f}m expected="
                    f"{self.expected_box_center_height:.3f}+/-{self.box_center_height_tolerance:.3f}m"
                )
            if box_yaw_error > self.max_box_yaw_error_deg:
                failures.append(
                    f"box_yaw_error={box_yaw_error:.1f}deg > {self.max_box_yaw_error_deg:.1f}deg"
                )
            if failures:
                raise CarryDataUnavailable("initial Carry scene rejected: " + "; ".join(failures))

            if self.target_mode == "configured_world":
                target = frame.target_position_world.copy()
            else:
                target = frame.object_position.copy()
                target[:2] += root_forward * float(self.target_forward_distance)
            self.target_position_world = target
            logger.info(
                "Initial Carry scene accepted: box_distance={:.3f}m bearing={:.1f}deg "
                "box_yaw_error={:.1f}deg target={}".format(
                    box_distance,
                    box_bearing,
                    box_yaw_error,
                    target.tolist(),
                )
            )
        return replace(frame, target_position_world=self.target_position_world.copy())


class CarryPhaseController:
    """Deployment equivalent of the interactive MuJoCo phased controller."""

    def __init__(self, config: Mapping[str, Any], latents: dict[str, Any], dt: float) -> None:
        defaults = {
            "approach_switch_distance": 0.75, "heading_blend_full_deg": 60.0, "max_turn_blend": 0.90,
            "approach_turn_enter_deg": 12.0, "approach_turn_exit_deg": 5.0,
            "approach_turn_blend_seconds": 0.20,
            "approach_reacquire_min_seconds": 0.30,
            "approach_reacquire_max_seconds": 2.0,
            "approach_reacquire_max_hand_gap": 0.25,
            "approach_reacquire_max_far_hand_gap": 0.30,
            "approach_reacquire_min_root_height": 0.70,
            "approach_reacquire_min_upright": 0.90,
            "approach_reacquire_max_boxward_speed": 0.60,
            "approach_reacquire_ready_steps": 5,
            "initial_approach_from_live_pose": False,
            "approach_pickup_max_bearing_deg": 8.0,
            "transport_heading_blend_full_deg": 45.0, "transport_max_turn_blend": 0.45,
            "transport_slowdown_distance": 0.35, "place_switch_distance": 0.2,
            "place_align_distance": 0.35, "place_drift_distance": 0.35,
            "place_enter_steps": 5, "place_drift_steps": 10,
            "hold_switch_lift": 0.5, "hold_switch_upright": 0.85, "hold_release_lift": 0.25,
            "phase_hold_steps": 5, "ground_center_height": 0.175, "lift_height": 0.35,
            "lift_contact_timeout_seconds": 0.80,
            "place_settle_seconds": 0.2, "place_lower_seconds": 1.0, "place_release_seconds": 2.0,
            "place_retry_delay_seconds": 1.0, "place_max_attempt_seconds": 10.0,
            "place_release_height": 0.08, "place_success_distance": 0.2,
            "place_success_height": 0.05, "place_success_speed": 0.15,
            "place_success_upright": 0.85, "place_success_steps": 25,
            "place_abort_distance": 0.8, "align_goal_lookahead": 0.45,
            "fall_recovery_enabled": False, "get_up_distance": 1.5,
            "fall_upright_threshold": 0.20, "fall_root_height": 0.30,
            "fall_tilted_root_height": 0.50,
            "fall_detect_steps": 10, "get_up_upright_threshold": 0.85,
            "get_up_root_height": 0.65, "get_up_stable_steps": 25,
            "direct_place_terminal_blend_start_distance": 0.60,
            "direct_place_terminal_blend_end_distance": 0.30,
            "direct_place_profile_lower_full_distance": 0.25,
            "direct_place_profile_commit_seconds": 10.0,
            "direct_place_profile_commit_distance": 0.35,
            "direct_place_profile_commit_speed": 0.25,
            "direct_place_profile_commit_reset_distance": 0.60,
            "direct_place_release_lift_start": 0.183,
            "direct_place_release_lift_end": 0.02,
            "direct_place_release_latch_lift": 0.06,
            "direct_place_release_latch_distance": 0.38,
            "direct_place_release_latch_max_vertical_speed": 0.20,
            "direct_place_release_latch_max_speed": float("inf"),
            "direct_place_release_latch_min_upright": 0.90,
            "direct_place_release_commit_seconds": 0.0,
            "direct_place_contact_guard_steps": 2,
            "direct_place_contact_guard_lift": 0.10,
            "direct_place_success_distance": 0.30,
        }
        defaults.update(dict(config))
        self.cfg = SimpleNamespace(**defaults)
        self.latents = latents
        if self.cfg.fall_recovery_enabled and "get_up" not in self.latents:
            raise ValueError("fall_recovery_enabled requires a get_up latent sequence")
        self.dt = float(dt)
        self.settle_steps = max(0, round(self.cfg.place_settle_seconds / dt))
        self.lower_steps = max(1, round(self.cfg.place_lower_seconds / dt))
        self.release_steps = max(1, round(self.cfg.place_release_seconds / dt))
        self.retry_steps = max(1, round(self.cfg.place_retry_delay_seconds / dt))
        self.max_attempt_steps = max(1, round(self.cfg.place_max_attempt_seconds / dt))
        self.reset()

    def reset(self) -> None:
        self.phase = "approach"
        self.phase_steps = self.ready_steps = self.lost_steps = 0
        self.placed_steps = self.failure_steps = self.completion_lost_steps = 0
        self.fall_detect_steps = self.get_up_stable_steps = 0
        self.fall_event_count = 0
        self.release_started_step: int | None = None

    def _change_phase(self, phase: str) -> None:
        if phase == self.phase:
            return
        self.phase = phase
        self.phase_steps = 0
        self.ready_steps = 0
        self.lost_steps = 0
        self.release_started_step = None

    @staticmethod
    def _smooth(step: int, start: int, duration: int) -> float:
        x = float(np.clip((step - start) / max(duration, 1), 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def select(self, observation: dict[str, np.ndarray], metrics: Mapping[str, Any], frame: CarryMocapFrame) -> tuple[np.ndarray, np.ndarray, str]:
        rotation = matrix_from_quat(frame.robot_root_quaternion_wxyz.reshape(1, 4))[0]
        if self.phase == "get_up":
            sequence = self.latents["get_up"]
            sequence_step = min(self.phase_steps, sequence.shape[0] - 1)
            latent = sequence[sequence_step]
            actor = np.concatenate((observation["state"].copy(), observation["last_action"], observation["history_actor"]))
            actor[STATE_DIM - OBJECT_CONTEXT_DIM:STATE_DIM] = 0.0
            mode = (
                "get_up_reward"
                if sequence.shape[0] == 1
                else f"get_up[{sequence_step}/{sequence.shape[0] - 1}]"
            )
            return actor, latent, mode
        if self.phase == "approach":
            bearing = _bearing_deg(rotation, frame.object_position[:2] - frame.robot_root_position[:2])
            weight = min(abs(bearing) / self.cfg.heading_blend_full_deg, 1.0) * self.cfg.max_turn_blend
            direction = 1 if bearing >= 0.0 else -1
            latent = project_z((1.0 - weight) * self.latents["approach"] + weight * self.latents["turn"][direction])
            actor = np.concatenate((observation["state"].copy(), observation["last_action"], observation["history_actor"]))
            actor[STATE_DIM - OBJECT_CONTEXT_DIM:STATE_DIM] = 0.0
            return actor, latent, "approach"
        if self.phase == "lift":
            latent, mode = self.latents["lift"], "lift"
        elif self.phase == "transport":
            bearing = _bearing_deg(rotation, frame.target_position_world[:2] - frame.object_position[:2])
            turn = min(abs(bearing) / self.cfg.transport_heading_blend_full_deg, 1.0) * self.cfg.transport_max_turn_blend
            direction = 1 if bearing >= 0.0 else -1
            distance = float(metrics["goal_distance"])
            slow = float(np.clip((self.cfg.transport_slowdown_distance - distance) /
                                 max(self.cfg.transport_slowdown_distance - self.cfg.place_switch_distance, 1e-6), 0.0, 1.0))
            forward = project_z((1.0 - slow) * self.latents["transport"] + slow * self.latents["transport_slow"])
            latent = project_z((1.0 - turn) * forward + turn * self.latents["transport_turn"][direction])
            mode = "transport"
        elif self.phase == "place":
            if self.release_started_step is None:
                blend = self._smooth(self.phase_steps, self.settle_steps, self.lower_steps)
                latent = project_z((1.0 - blend) * self.latents["align"] + blend * self.latents["lower"])
                mode = "place_lower"
            else:
                blend = self._smooth(self.phase_steps - self.release_started_step, 0, self.release_steps)
                latent = project_z((1.0 - blend) * self.latents["lower"] + blend * self.latents["place"])
                mode = "place_release"
        else:
            latent, mode = self.latents["place"], "complete"
        actor = np.concatenate((observation["state"].copy(), observation["last_action"], observation["history_actor"]))
        if self.phase == "place" and self.cfg.align_goal_lookahead > 0.0:
            start = STATE_DIM - OBJECT_CONTEXT_DIM + 15
            delta = actor[start:start + 3]
            norm = float(np.linalg.norm(delta[:2]))
            if norm > 1.0e-6:
                delta[:2] += delta[:2] / norm * self.cfg.align_goal_lookahead
        return actor, np.asarray(latent, dtype=np.float32), mode

    def metrics(
        self,
        frame: CarryMocapFrame,
        hand_box_gaps: np.ndarray | None = None,
    ) -> dict[str, Any]:
        root_rot = matrix_from_quat(frame.robot_root_quaternion_wxyz.reshape(1, 4))[0]
        geom_quat = _unit_quaternion_wxyz(
            getattr(self.cfg, "box_geom_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
            "box_geom_quaternion_wxyz",
        )
        object_geom_quat = quat_mul(
            frame.object_quaternion_wxyz.reshape(1, 4), geom_quat.reshape(1, 4)
        )[0]
        object_rot = matrix_from_quat(object_geom_quat.reshape(1, 4))[0]
        robot_box_xy = frame.object_position[:2] - frame.robot_root_position[:2]
        robot_box_norm = max(float(np.linalg.norm(robot_box_xy)), 1.0e-8)
        root_linear_velocity = (
            np.zeros(3, dtype=np.float64)
            if frame.robot_root_linear_velocity_world is None
            else _vector(
                frame.robot_root_linear_velocity_world,
                3,
                "robot_root_linear_velocity_world",
            )
        )
        gaps = (
            np.full(2, np.inf, dtype=np.float64)
            if hand_box_gaps is None
            else _vector(hand_box_gaps, 2, "hand_box_gaps")
        )
        return {
            "horizontal_distance": float(np.linalg.norm(robot_box_xy)),
            "goal_distance": float(np.linalg.norm(frame.target_position_world[:2] - frame.object_position[:2])),
            "object_height": float(frame.object_position[2]),
            "object_speed": float(np.linalg.norm(frame.object_linear_velocity_world)),
            "object_vertical_speed": float(frame.object_linear_velocity_world[2]),
            "lift": float(np.clip((frame.object_position[2] - self.cfg.ground_center_height) / self.cfg.lift_height, 0.0, 1.0)),
            "root_height": float(frame.robot_root_position[2]),
            "pelvis_up_z": float(root_rot[2, 2]),
            "robot_upright": float(np.clip(root_rot[2, 2], 0.0, 1.0)),
            "box_upright": float(np.clip(object_rot[2, 2], 0.0, 1.0)),
            "bilateral_contact": bool(np.all(frame.hand_contact > 0.5)),
            "any_contact": bool(np.any(frame.hand_contact > 0.5)),
            "left_hand_box_gap": float(gaps[0]),
            "right_hand_box_gap": float(gaps[1]),
            "approach_boxward_speed": float(
                np.dot(root_linear_velocity[:2], robot_box_xy / robot_box_norm)
            ),
            "approach_bearing_deg": float(_bearing_deg(root_rot, robot_box_xy)),
        }

    def update(self, m: Mapping[str, Any]) -> None:
        self.phase_steps += 1
        c = self.cfg
        if c.fall_recovery_enabled:
            fallen = m["root_height"] < c.fall_root_height or (
                m["pelvis_up_z"] < c.fall_upright_threshold
                and m["root_height"] < c.fall_tilted_root_height
            )
            self.fall_detect_steps = self.fall_detect_steps + 1 if fallen else 0
            if self.phase == "get_up":
                if m["horizontal_distance"] <= c.get_up_distance:
                    self._change_phase("lift")
                    self.get_up_stable_steps = 0
                    return
                recovered = (
                    m["pelvis_up_z"] >= c.get_up_upright_threshold
                    and m["root_height"] >= c.get_up_root_height
                )
                self.get_up_stable_steps = self.get_up_stable_steps + 1 if recovered else 0
                if self.get_up_stable_steps >= c.get_up_stable_steps:
                    self._change_phase("approach")
                    self.fall_detect_steps = 0
                    self.get_up_stable_steps = 0
                return
            if self.fall_detect_steps >= c.fall_detect_steps:
                recovery_phase = "get_up" if m["horizontal_distance"] > c.get_up_distance else "lift"
                if recovery_phase != self.phase:
                    self.fall_event_count += 1
                    self._change_phase(recovery_phase)
                    self.get_up_stable_steps = 0
                return
            self.get_up_stable_steps = 0
        if self.phase == "approach":
            if m["horizontal_distance"] <= c.approach_switch_distance:
                self.phase, self.phase_steps = "lift", 0
            return
        if self.phase == "lift":
            ready = m["bilateral_contact"] and m["lift"] >= c.hold_switch_lift and m["robot_upright"] >= c.hold_switch_upright
            self.ready_steps = self.ready_steps + 1 if ready else 0
            if self.ready_steps >= c.phase_hold_steps:
                self.phase, self.phase_steps, self.lost_steps = "transport", 0, 0
            return
        if self.phase == "transport":
            lost = (not m["bilateral_contact"]) or m["lift"] < c.hold_release_lift
            self.lost_steps = self.lost_steps + 1 if lost else 0
            if self.lost_steps >= c.phase_hold_steps:
                self.phase = "lift" if m["horizontal_distance"] <= c.approach_switch_distance else "approach"
                self.phase_steps = self.ready_steps = 0
            elif m["goal_distance"] <= c.place_align_distance:
                self.phase, self.phase_steps = "place", 0
                self.release_started_step = None
            return
        ground_error = abs(m["object_height"] - c.ground_center_height)
        if self.phase == "place":
            if m["bilateral_contact"] and m["goal_distance"] > c.place_drift_distance:
                self.phase, self.phase_steps, self.release_started_step = "transport", 0, None
                return
            if self.release_started_step is None and m["bilateral_contact"] and ground_error <= c.place_release_height:
                self.release_started_step = self.phase_steps
            placed = (m["goal_distance"] <= c.place_success_distance and ground_error <= c.place_success_height
                      and m["object_speed"] <= c.place_success_speed and m["box_upright"] >= c.place_success_upright
                      and not m["bilateral_contact"])
            self.placed_steps = self.placed_steps + 1 if placed else 0
            if self.placed_steps >= c.place_success_steps:
                self.phase, self.phase_steps = "complete", 0
                return
            released = not m["bilateral_contact"]
            failed = (
                (released and ground_error > c.place_release_height)
                or (
                    released
                    and ground_error <= c.place_success_height
                    and m["object_speed"] <= c.place_success_speed
                    and m["goal_distance"] > c.place_success_distance
                )
                or self.phase_steps >= self.max_attempt_steps
                or m["goal_distance"] > c.place_abort_distance
            )
            self.failure_steps = self.failure_steps + 1 if failed else 0
            if self.failure_steps >= self.retry_steps:
                self.phase = "lift" if m["horizontal_distance"] <= c.approach_switch_distance else "approach"
                self.phase_steps = self.ready_steps = self.lost_steps = self.placed_steps = self.failure_steps = 0
                self.release_started_step = None
            return
        stable = (m["goal_distance"] <= c.place_success_distance and ground_error <= c.place_success_height
                  and m["object_speed"] <= c.place_success_speed and m["box_upright"] >= c.place_success_upright)
        self.completion_lost_steps = 0 if stable else self.completion_lost_steps + 1
        if self.completion_lost_steps >= self.retry_steps:
            self.phase = "lift" if m["horizontal_distance"] <= c.approach_switch_distance else "approach"
            self.phase_steps = self.ready_steps = self.lost_steps = self.completion_lost_steps = 0


class H20Reward736PhaseController(CarryPhaseController):
    """H20 pair-80M controller matching the calibrated MuJoCo reward736 flow."""

    def reset(self) -> None:
        super().reset()
        self.place_enter_steps = 0
        self.place_drift_steps = 0
        self.approach_turn_active = False
        self.approach_turn_direction = 1
        self.approach_turn_weight = 0.0

    def _change_phase(self, phase: str) -> None:
        previous = self.phase
        super()._change_phase(phase)
        if phase != previous:
            self.place_enter_steps = 0
            self.place_drift_steps = 0
            self.approach_turn_active = False
            self.approach_turn_weight = 0.0

    def select(
        self,
        observation: dict[str, np.ndarray],
        metrics: Mapping[str, Any],
        frame: CarryMocapFrame,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        rotation = matrix_from_quat(frame.robot_root_quaternion_wxyz.reshape(1, 4))[0]
        if self.phase == "approach":
            bearing = _bearing_deg(
                rotation, frame.object_position[:2] - frame.robot_root_position[:2]
            )
            direction = 1 if bearing >= 0.0 else -1
            turn_enter_deg = float(getattr(self.cfg, "approach_turn_enter_deg", 12.0))
            turn_exit_deg = float(getattr(self.cfg, "approach_turn_exit_deg", 5.0))
            if self.approach_turn_active:
                if abs(bearing) <= turn_exit_deg or direction != self.approach_turn_direction:
                    self.approach_turn_active = False
            elif self.approach_turn_weight <= 1.0e-6 and abs(bearing) >= turn_enter_deg:
                self.approach_turn_active = True
                self.approach_turn_direction = direction

            target_weight = self.cfg.max_turn_blend if self.approach_turn_active else 0.0
            blend_seconds = float(getattr(self.cfg, "approach_turn_blend_seconds", 0.20))
            if blend_seconds > 0.0:
                max_delta = self.cfg.max_turn_blend * self.dt / blend_seconds
                self.approach_turn_weight += float(
                    np.clip(
                        target_weight - self.approach_turn_weight,
                        -max_delta,
                        max_delta,
                    )
                )
            else:
                self.approach_turn_weight = target_weight
            turn_weight = self.approach_turn_weight
            direction = self.approach_turn_direction
            latent = project_z(
                (1.0 - turn_weight) * self.latents["approach"]
                + turn_weight * self.latents["turn"][direction]
            )
            actor = np.concatenate(
                (
                    observation["state"].copy(),
                    observation["last_action"],
                    observation["history_actor"],
                )
            )
            actor[STATE_DIM - OBJECT_CONTEXT_DIM : STATE_DIM] = 0.0
            mode = "approach" if turn_weight < 1.0e-3 else (
                "approach_left" if direction > 0 else "approach_right"
            )
            return actor, latent, mode

        if self.phase == "transport":
            bearing = _bearing_deg(
                rotation, frame.target_position_world[:2] - frame.object_position[:2]
            )
            turn_weight = _transport_turn_weight(
                goal_distance=float(metrics["goal_distance"]),
                bearing_deg=bearing,
                turn_boost_distance=float(
                    getattr(self.cfg, "transport_turn_boost_distance", 1.2)
                ),
                place_distance=float(self.cfg.place_align_distance),
                heading_blend_full_deg=float(self.cfg.transport_heading_blend_full_deg),
                far_max_turn_blend=float(self.cfg.transport_max_turn_blend),
                near_max_turn_blend=float(
                    getattr(self.cfg, "transport_near_max_turn_blend", 0.8)
                ),
            )
            direction = 1 if bearing >= 0.0 else -1
            latent = project_z(
                (1.0 - turn_weight) * self.latents["transport"]
                + turn_weight * self.latents["transport_turn"][direction]
            )
            actor = np.concatenate(
                (
                    observation["state"].copy(),
                    observation["last_action"],
                    observation["history_actor"],
                )
            )
            mode = "transport" if turn_weight < 1.0e-3 else (
                "transport_left" if direction > 0 else "transport_right"
            )
            return actor, latent, mode

        return super().select(observation, metrics, frame)

    def update(self, m: Mapping[str, Any]) -> None:
        self.phase_steps += 1
        c = self.cfg
        if c.fall_recovery_enabled:
            fallen = m["root_height"] < c.fall_root_height or (
                m["pelvis_up_z"] < c.fall_upright_threshold
                and m["root_height"] < c.fall_tilted_root_height
            )
            self.fall_detect_steps = self.fall_detect_steps + 1 if fallen else 0
            if self.phase == "get_up":
                if m["horizontal_distance"] <= c.get_up_distance:
                    self._change_phase("lift")
                    self.get_up_stable_steps = 0
                    return
                recovered = (
                    m["pelvis_up_z"] >= c.get_up_upright_threshold
                    and m["root_height"] >= c.get_up_root_height
                )
                self.get_up_stable_steps = self.get_up_stable_steps + 1 if recovered else 0
                if self.get_up_stable_steps >= c.get_up_stable_steps:
                    self._change_phase("approach")
                    self.fall_detect_steps = 0
                    self.get_up_stable_steps = 0
                return
            if self.fall_detect_steps >= c.fall_detect_steps:
                recovery_phase = (
                    "get_up" if m["horizontal_distance"] > c.get_up_distance else "lift"
                )
                if recovery_phase != self.phase:
                    self.fall_event_count += 1
                    self._change_phase(recovery_phase)
                    self.get_up_stable_steps = 0
                return
            self.get_up_stable_steps = 0

        if self.phase == "approach":
            if m["horizontal_distance"] <= c.approach_switch_distance:
                self._change_phase("lift")
            return
        if self.phase == "lift":
            ready = (
                m["bilateral_contact"]
                and m["lift"] >= c.hold_switch_lift
                and m["robot_upright"] >= c.hold_switch_upright
            )
            self.ready_steps = self.ready_steps + 1 if ready else 0
            if self.ready_steps >= getattr(c, "lift_ready_steps", c.phase_hold_steps):
                self._change_phase("transport")
            return
        if self.phase == "transport":
            lost = (not m["bilateral_contact"]) or m["lift"] < c.hold_release_lift
            self.lost_steps = self.lost_steps + 1 if lost else 0
            inside_place = (
                not lost and m["goal_distance"] <= c.place_align_distance
            )
            self.place_enter_steps = self.place_enter_steps + 1 if inside_place else 0
            if self.lost_steps >= c.phase_hold_steps:
                next_phase = (
                    "lift"
                    if m["horizontal_distance"] <= c.approach_switch_distance
                    else "approach"
                )
                self._change_phase(next_phase)
            elif self.place_enter_steps >= int(getattr(c, "place_enter_steps", 5)):
                self._change_phase("place")
                self.placed_steps = 0
                self.failure_steps = 0
            return

        ground_error = abs(m["object_height"] - c.ground_center_height)
        if self.phase == "place":
            drifted_while_holding = (
                self.release_started_step is None
                and m["bilateral_contact"]
                and m["goal_distance"] > c.place_drift_distance
            )
            self.place_drift_steps = self.place_drift_steps + 1 if drifted_while_holding else 0
            if self.place_drift_steps >= int(getattr(c, "place_drift_steps", 10)):
                self._change_phase("transport")
                return
            if (
                self.release_started_step is None
                and m["bilateral_contact"]
                and ground_error <= c.place_release_height
            ):
                self.release_started_step = self.phase_steps
            placed = (
                m["goal_distance"] <= c.place_success_distance
                and ground_error <= c.place_success_height
                and m["object_speed"] <= c.place_success_speed
                and m["box_upright"] >= c.place_success_upright
                and not m["bilateral_contact"]
            )
            self.placed_steps = self.placed_steps + 1 if placed else 0
            if self.placed_steps >= c.place_success_steps:
                self._change_phase("complete")
                self.completion_lost_steps = 0
                return
            released = not m["bilateral_contact"]
            failed = (
                (released and ground_error > c.place_release_height)
                or (
                    released
                    and ground_error <= c.place_success_height
                    and m["object_speed"] <= c.place_success_speed
                    and m["goal_distance"] > c.place_success_distance
                )
                or self.phase_steps >= self.max_attempt_steps
                or m["goal_distance"] > c.place_abort_distance
            )
            self.failure_steps = self.failure_steps + 1 if failed else 0
            if self.failure_steps >= self.retry_steps:
                next_phase = (
                    "lift"
                    if m["horizontal_distance"] <= c.approach_switch_distance
                    else "approach"
                )
                self._change_phase(next_phase)
                self.recovery_count = getattr(self, "recovery_count", 0) + 1
                self.placed_steps = 0
                self.failure_steps = 0
            return

        stable = (
            m["goal_distance"] <= c.place_success_distance
            and ground_error <= c.place_success_height
            and m["object_speed"] <= c.place_success_speed
            and m["box_upright"] >= c.place_success_upright
        )
        self.completion_lost_steps = 0 if stable else self.completion_lost_steps + 1
        if self.completion_lost_steps >= self.retry_steps:
            next_phase = (
                "lift"
                if m["horizontal_distance"] <= c.approach_switch_distance
                else "approach"
            )
            self._change_phase(next_phase)
            self.completion_lost_steps = 0


class BestCarry140935168StrongController(H20Reward736PhaseController):
    """Exact 140.935M strong phased flow with distance-independent get-up.

    The regular lift/transport/place transitions reuse the debounced MuJoCo
    equivalent in :class:`H20Reward736PhaseController`.  Approach keeps the
    continuous forward/turn blend used by the frozen 140.935M controller, and
    every confirmed fall is locked in ``get_up`` until the pelvis pose is
    stably recovered.  Only then does the 0.75 m box-distance threshold choose
    lift versus approach.
    """

    def select(
        self,
        observation: dict[str, np.ndarray],
        metrics: Mapping[str, Any],
        frame: CarryMocapFrame,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if self.phase != "approach":
            return super().select(observation, metrics, frame)

        rotation = matrix_from_quat(
            frame.robot_root_quaternion_wxyz.reshape(1, 4)
        )[0]
        bearing = _bearing_deg(
            rotation,
            frame.object_position[:2] - frame.robot_root_position[:2],
        )
        turn_weight = min(
            abs(bearing) / self.cfg.heading_blend_full_deg, 1.0
        ) * self.cfg.max_turn_blend
        direction = 1 if bearing >= 0.0 else -1
        latent = project_z(
            (1.0 - turn_weight) * self.latents["approach"]
            + turn_weight * self.latents["turn"][direction]
        )
        actor = np.concatenate(
            (
                observation["state"].copy(),
                observation["last_action"],
                observation["history_actor"],
            )
        )
        actor[STATE_DIM - OBJECT_CONTEXT_DIM : STATE_DIM] = 0.0
        mode = "forward" if turn_weight < 1.0e-3 else (
            "forward_left" if direction > 0 else "forward_right"
        )
        return actor, latent, mode

    def update(self, m: Mapping[str, Any]) -> None:
        c = self.cfg
        if not c.fall_recovery_enabled:
            super().update(m)
            return

        fallen = m["root_height"] < c.fall_root_height or (
            m["pelvis_up_z"] < c.fall_upright_threshold
            and m["root_height"] < c.fall_tilted_root_height
        )
        self.fall_detect_steps = self.fall_detect_steps + 1 if fallen else 0

        if self.phase == "get_up":
            self.phase_steps += 1
            recovered = (
                m["pelvis_up_z"] >= c.get_up_upright_threshold
                and m["root_height"] >= c.get_up_root_height
            )
            self.get_up_stable_steps = (
                self.get_up_stable_steps + 1 if recovered else 0
            )
            if self.get_up_stable_steps >= c.get_up_stable_steps:
                next_phase = (
                    "lift"
                    if m["horizontal_distance"] <= c.approach_switch_distance
                    else "approach"
                )
                self._change_phase(next_phase)
                self.fall_detect_steps = 0
                self.get_up_stable_steps = 0
            return

        if self.fall_detect_steps >= c.fall_detect_steps:
            self.phase_steps += 1
            self.fall_event_count += 1
            self._change_phase("get_up")
            self.get_up_stable_steps = 0
            return

        self.get_up_stable_steps = 0
        # Run the shared regular state machine without its legacy
        # distance-based fall routing.  Restore the immutable profile setting
        # immediately after this one update.
        c.fall_recovery_enabled = False
        try:
            super().update(m)
        finally:
            c.fall_recovery_enabled = True


class NewIntactHybridDirectPlaceController(H20Reward736PhaseController):
    """Exact no-handoff direct-place profile used by the cited 80.1M rollout."""

    def __init__(
        self, config: Mapping[str, Any], latents: dict[str, Any], dt: float
    ) -> None:
        super().__init__(config, latents, dt)
        self.approach_reacquire_min_steps = max(
            1,
            round(float(getattr(self.cfg, "approach_reacquire_min_seconds", 0.30)) / dt),
        )
        self.approach_reacquire_max_steps = max(
            self.approach_reacquire_min_steps,
            round(float(getattr(self.cfg, "approach_reacquire_max_seconds", 2.0)) / dt),
        )
        lift_contact_timeout_seconds = float(
            getattr(self.cfg, "lift_contact_timeout_seconds", 0.80)
        )
        if not math.isfinite(lift_contact_timeout_seconds) or lift_contact_timeout_seconds <= 0.0:
            raise ValueError("lift_contact_timeout_seconds must be finite and positive")
        self.lift_contact_timeout_steps = max(
            1, round(lift_contact_timeout_seconds / dt)
        )
        if not 0.0 <= self.cfg.approach_turn_exit_deg < self.cfg.approach_turn_enter_deg:
            raise ValueError("approach turn thresholds must satisfy 0 <= exit < enter")
        if not (
            0.0
            <= self.cfg.approach_reacquire_min_seconds
            <= self.cfg.approach_reacquire_max_seconds
        ):
            raise ValueError("approach reacquire seconds must satisfy 0 <= min <= max")
        if not (
            0.0 < self.cfg.approach_reacquire_max_hand_gap
            <= self.cfg.approach_reacquire_max_far_hand_gap
        ):
            raise ValueError("approach reacquire hand gaps must satisfy 0 < near <= far")
        if not (
            self.cfg.direct_place_terminal_blend_start_distance
            > self.cfg.direct_place_terminal_blend_end_distance
            > self.cfg.direct_place_profile_lower_full_distance
            > 0.0
        ):
            raise ValueError("direct-place distances must satisfy start > end > lower > 0")
        if not (
            0.0
            <= self.cfg.direct_place_release_lift_end
            < self.cfg.direct_place_release_lift_start
        ):
            raise ValueError("direct-place release lifts must satisfy 0 <= end < start")

    def reset(self) -> None:
        super().reset()
        live_initial_pose = bool(
            getattr(self.cfg, "initial_approach_from_live_pose", False)
        )
        # A live initial pose first uses H20 to face the measured box. Once
        # aligned, the first pickup keeps the validated MuJoCo distance-only
        # approach->lift transition. Hand-gap gating remains exclusive to a
        # dropped-box/get-up reacquisition started by _start_approach_reacquire.
        self.approach_reorient = live_initial_pose
        self.approach_reacquire_active = False
        self.approach_reacquire_steps = 0
        self.approach_reacquire_ready_steps = 0
        self.direct_place_lower_commitment = 0.0
        self.direct_place_release_commitment = 0.0
        self.direct_place_release_latched = False
        self.direct_place_seen_bilateral = False
        self.direct_place_contact_loss_steps = 0
        self.direct_place_initial_goal_distance: float | None = None

    def _reset_direct_place_state(self) -> None:
        self.direct_place_lower_commitment = 0.0
        self.direct_place_release_commitment = 0.0
        self.direct_place_release_latched = False
        self.direct_place_seen_bilateral = False
        self.direct_place_contact_loss_steps = 0
        self.direct_place_initial_goal_distance = None

    def _change_phase(self, phase: str) -> None:
        previous = self.phase
        super()._change_phase(phase)
        if phase != previous:
            self.approach_reorient = False
        if phase != "approach":
            self.approach_reacquire_active = False
            self.approach_reacquire_steps = 0
            self.approach_reacquire_ready_steps = 0
        if phase != previous and phase == "place_direct":
            self._reset_direct_place_state()

    def _start_approach_reacquire(self) -> None:
        self._change_phase("approach")
        self.approach_reacquire_active = True
        self.approach_reacquire_steps = 0
        self.approach_reacquire_ready_steps = 0

    def select(
        self,
        observation: dict[str, np.ndarray],
        metrics: Mapping[str, Any],
        frame: CarryMocapFrame,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if self.phase == "approach" and self.approach_reorient:
            rotation = matrix_from_quat(
                frame.robot_root_quaternion_wxyz.reshape(1, 4)
            )[0]
            bearing = _bearing_deg(
                rotation,
                frame.object_position[:2] - frame.robot_root_position[:2],
            )
            if abs(bearing) <= float(self.cfg.approach_turn_exit_deg):
                self.approach_reorient = False
            else:
                direction = 1 if bearing >= 0.0 else -1
                actor = np.concatenate(
                    (
                        observation["state"].copy(),
                        observation["last_action"],
                        observation["history_actor"],
                    )
                )
                actor[STATE_DIM - OBJECT_CONTEXT_DIM : STATE_DIM] = 0.0
                return (
                    actor,
                    self.latents["turn"][direction],
                    "approach_reorient_left"
                    if direction > 0
                    else "approach_reorient_right",
                )
        if self.phase not in {"place_direct", "complete"}:
            return super().select(observation, metrics, frame)

        actor = np.concatenate(
            (
                observation["state"].copy(),
                observation["last_action"],
                observation["history_actor"],
            )
        )
        if self.phase == "complete":
            blend = self._smooth(self.phase_steps, 0, self.release_steps)
            latent = project_z(
                (1.0 - blend) * self.latents["direct_place_release"]
                + blend * self.latents["direct_place_stand"]
            )
            return actor, latent, f"place_direct_complete_stand_{blend:.2f}"

        c = self.cfg
        distance = float(metrics["goal_distance"])
        if self.direct_place_initial_goal_distance is None:
            self.direct_place_initial_goal_distance = distance

        blend_start = float(c.direct_place_terminal_blend_start_distance)
        blend_end = float(c.direct_place_terminal_blend_end_distance)
        lower_full = float(c.direct_place_profile_lower_full_distance)
        slow_progress = _smoothstep01(
            (blend_start - distance) / (blend_start - blend_end)
        )
        lower_progress = _smoothstep01(
            (blend_end - distance) / (blend_end - lower_full)
        )

        commit_seconds = float(c.direct_place_profile_commit_seconds)
        can_commit = (
            distance <= float(c.direct_place_profile_commit_distance)
            and float(metrics["object_speed"])
            <= float(c.direct_place_profile_commit_speed)
        )
        if commit_seconds > 0.0:
            delta = self.dt / commit_seconds
            if can_commit:
                self.direct_place_lower_commitment = min(
                    1.0, self.direct_place_lower_commitment + delta
                )
            elif distance > float(c.direct_place_profile_commit_reset_distance):
                self.direct_place_lower_commitment = max(
                    0.0, self.direct_place_lower_commitment - delta
                )
            lower_progress = max(lower_progress, self.direct_place_lower_commitment)
            slow_progress = max(slow_progress, self.direct_place_lower_commitment)

        lift_start = float(c.direct_place_release_lift_start)
        lift_end = float(c.direct_place_release_lift_end)
        release_progress = _smoothstep01(
            (lift_start - float(metrics["lift"])) / (lift_start - lift_end)
        )
        release_ready = (
            distance <= float(c.direct_place_release_latch_distance)
            and float(metrics["object_speed"])
            <= float(c.direct_place_release_latch_max_speed)
            and float(metrics["box_upright"])
            >= float(c.direct_place_release_latch_min_upright)
        )
        can_latch_release = (
            release_ready
            and float(c.direct_place_release_latch_lift) > 0.0
            and float(metrics["lift"]) <= float(c.direct_place_release_latch_lift)
            and abs(float(metrics["object_vertical_speed"]))
            <= float(c.direct_place_release_latch_max_vertical_speed)
        )
        release_commit_seconds = float(c.direct_place_release_commit_seconds)
        if can_latch_release and release_commit_seconds > 0.0:
            self.direct_place_release_commitment = min(
                1.0,
                self.direct_place_release_commitment
                + self.dt / release_commit_seconds,
            )
        elif can_latch_release:
            self.direct_place_release_commitment = 1.0
        elif release_commit_seconds > 0.0:
            self.direct_place_release_commitment = max(
                0.0,
                self.direct_place_release_commitment
                - self.dt / release_commit_seconds,
            )
        release_progress = max(
            release_progress, self.direct_place_release_commitment
        )
        if self.direct_place_release_commitment >= 1.0:
            self.direct_place_release_latched = True
        if self.direct_place_release_latched:
            release_progress = 1.0

        lower_or_release = project_z(
            (1.0 - release_progress) * self.latents["direct_place_lower"]
            + release_progress * self.latents["direct_place_release"]
        )
        near_latent = project_z(
            (1.0 - lower_progress) * self.latents["direct_place_slow"]
            + lower_progress * lower_or_release
        )
        latent = project_z(
            (1.0 - slow_progress) * self.latents["direct_place"]
            + slow_progress * near_latent
        )

        if bool(metrics["bilateral_contact"]):
            self.direct_place_seen_bilateral = True
            self.direct_place_contact_loss_steps = 0
        elif (
            self.direct_place_seen_bilateral
            and float(metrics["lift"]) > float(c.direct_place_contact_guard_lift)
        ):
            self.direct_place_contact_loss_steps += 1
        else:
            self.direct_place_contact_loss_steps = 0
        contact_recovery = (
            self.direct_place_contact_loss_steps
            >= int(c.direct_place_contact_guard_steps)
        )
        if self.direct_place_release_latched:
            latent = self.latents["direct_place_release"]
        elif contact_recovery:
            latent = self.latents["direct_place_recover"]

        mode = (
            "place_direct_profile_"
            f"s{slow_progress:.2f}_l{lower_progress:.2f}_r{release_progress:.2f}_"
            f"c{self.direct_place_lower_commitment:.2f}_g0.00_"
            f"rec{int(contact_recovery)}_rl{int(self.direct_place_release_latched)}"
        )
        return actor, np.asarray(latent, dtype=np.float32), mode

    def update(self, m: Mapping[str, Any]) -> None:
        self.phase_steps += 1
        c = self.cfg
        fallen = m["root_height"] < c.fall_root_height or (
            m["pelvis_up_z"] < c.fall_upright_threshold
            and m["root_height"] < c.fall_tilted_root_height
        )
        self.fall_detect_steps = self.fall_detect_steps + 1 if fallen else 0

        if self.phase == "get_up":
            recovered = (
                m["pelvis_up_z"] >= c.get_up_upright_threshold
                and m["root_height"] >= c.get_up_root_height
            )
            self.get_up_stable_steps = self.get_up_stable_steps + 1 if recovered else 0
            if self.get_up_stable_steps >= c.get_up_stable_steps:
                self._start_approach_reacquire()
                self.fall_detect_steps = 0
                self.get_up_stable_steps = 0
            return

        if c.fall_recovery_enabled and self.fall_detect_steps >= c.fall_detect_steps:
            self.fall_event_count += 1
            self._change_phase("get_up")
            self._reset_direct_place_state()
            self.placed_steps = 0
            self.failure_steps = 0
            self.completion_lost_steps = 0
            self.get_up_stable_steps = 0
            return

        self.get_up_stable_steps = 0
        if self.phase == "approach":
            distance = float(m["horizontal_distance"])
            if self.approach_reacquire_active:
                if distance <= c.approach_switch_distance:
                    self.approach_reacquire_steps += 1
                else:
                    self.approach_reacquire_steps = 0
                    self.approach_reacquire_ready_steps = 0

                near_hand_gap = min(
                    float(m["left_hand_box_gap"]),
                    float(m["right_hand_box_gap"]),
                )
                far_hand_gap = max(
                    float(m["left_hand_box_gap"]),
                    float(m["right_hand_box_gap"]),
                )
                pregrasp_ready = (
                    self.approach_reacquire_steps
                    >= self.approach_reacquire_min_steps
                    and near_hand_gap <= c.approach_reacquire_max_hand_gap
                    and far_hand_gap <= c.approach_reacquire_max_far_hand_gap
                    and float(m["root_height"])
                    >= c.approach_reacquire_min_root_height
                    and float(m["pelvis_up_z"])
                    >= c.approach_reacquire_min_upright
                    and abs(float(m["approach_boxward_speed"]))
                    <= c.approach_reacquire_max_boxward_speed
                    and abs(float(m.get("approach_bearing_deg", 0.0)))
                    <= c.approach_pickup_max_bearing_deg
                    and not self.approach_reorient
                )
                self.approach_reacquire_ready_steps = (
                    self.approach_reacquire_ready_steps + 1
                    if pregrasp_ready
                    else 0
                )
                if (
                    self.approach_reacquire_ready_steps
                    >= c.approach_reacquire_ready_steps
                ):
                    self._change_phase("lift")
                elif (
                    self.approach_reacquire_steps
                    >= self.approach_reacquire_max_steps
                    and not pregrasp_ready
                ):
                    self.approach_reorient = True
                    self.approach_reacquire_steps = 0
                    self.approach_reacquire_ready_steps = 0
                return

            if (
                not self.approach_reorient
                and distance <= c.approach_switch_distance
            ):
                self._change_phase("lift")
            return

        if self.phase == "lift":
            ready = (
                bool(m["bilateral_contact"])
                and float(m["lift"]) >= c.hold_switch_lift
                and float(m["robot_upright"]) >= c.hold_switch_upright
            )
            self.ready_steps = self.ready_steps + 1 if ready else 0
            if self.ready_steps >= c.phase_hold_steps:
                self._change_phase("transport")
                return
            lost_contact = not bool(m["bilateral_contact"])
            self.lost_steps = self.lost_steps + 1 if lost_contact else 0
            if self.lost_steps >= self.lift_contact_timeout_steps:
                unstable = (
                    float(m["root_height"]) < c.get_up_root_height
                    or float(m["pelvis_up_z"]) < c.get_up_upright_threshold
                )
                if unstable:
                    self._change_phase("get_up")
                    self.get_up_stable_steps = 0
                else:
                    self._start_approach_reacquire()
            return

        if self.phase == "transport":
            lost = (
                not bool(m["bilateral_contact"])
                or float(m["lift"]) < c.hold_release_lift
            )
            self.lost_steps = self.lost_steps + 1 if lost else 0
            inside_place = (
                not lost and float(m["goal_distance"]) <= c.place_align_distance
            )
            self.place_enter_steps = self.place_enter_steps + 1 if inside_place else 0
            if self.lost_steps >= c.phase_hold_steps:
                self._change_phase(
                    "lift"
                    if float(m["horizontal_distance"])
                    <= c.approach_switch_distance
                    else "approach"
                )
            elif self.place_enter_steps >= c.place_enter_steps:
                self._change_phase("place_direct")
                self.placed_steps = 0
                self.failure_steps = 0
            return

        ground_error = abs(float(m["object_height"]) - c.ground_center_height)
        placed = (
            float(m["goal_distance"]) <= c.direct_place_success_distance
            and ground_error <= c.place_success_height
            and float(m["object_speed"]) <= c.place_success_speed
            and float(m["box_upright"]) >= c.place_success_upright
            and not bool(m["any_contact"])
        )
        self.placed_steps = self.placed_steps + 1 if placed else 0
        if self.phase == "complete":
            self.completion_lost_steps = (
                0 if placed else self.completion_lost_steps + 1
            )
            if self.completion_lost_steps >= self.retry_steps:
                still_carrying = (
                    bool(m["bilateral_contact"])
                    and float(m["lift"]) >= c.hold_release_lift
                    and float(m["robot_upright"]) >= c.hold_switch_upright
                )
                if still_carrying:
                    self._change_phase("transport")
                else:
                    self._start_approach_reacquire()
                self._reset_direct_place_state()
                self.placed_steps = 0
                self.completion_lost_steps = 0
                self.recovery_count = getattr(self, "recovery_count", 0) + 1
            return
        if self.placed_steps >= c.place_success_steps:
            self._change_phase("complete")
            self.completion_lost_steps = 0
            return

        settled_release = (
            not bool(m["any_contact"])
            and ground_error <= c.place_success_height
            and float(m["object_speed"]) <= c.place_success_speed
        )
        placement_failed = (
            float(m["goal_distance"]) > c.direct_place_success_distance
            or float(m["box_upright"]) < c.place_success_upright
        )
        self.failure_steps = (
            self.failure_steps + 1
            if settled_release and placement_failed
            else 0
        )
        if self.failure_steps >= self.retry_steps:
            self._start_approach_reacquire()
            self._reset_direct_place_state()
            self.placed_steps = 0
            self.failure_steps = 0
            self.recovery_count = getattr(self, "recovery_count", 0) + 1


class CarryRuntime:
    def __init__(self, exp_config: Mapping[str, Any], policy_config: Mapping[str, Any], joint_names: list[str], default: np.ndarray, dt: float) -> None:
        # Pelvis-height calibration depends on the support posture, not on how
        # closely unloaded arm joints settle at their nominal zero angles.
        # Keep the full-body default-pose check in the policy start safety gate,
        # but use only both legs and the waist while accepting pelvis samples.
        self.support_pose_joint_indices = np.asarray(
            [
                index
                for index, name in enumerate(joint_names)
                if name.startswith("waist_")
                or any(token in name for token in ("_hip_", "_knee_", "_ankle_"))
            ],
            dtype=np.int64,
        )
        if self.support_pose_joint_indices.size == 0:
            raise ValueError(
                "no leg/waist joints were found for pelvis deployment calibration"
            )
        self.mocap_source = str(exp_config.get("mocap_source", "json_zmq")).lower()
        self.actor_inertial_source = str(
            exp_config.get("actor_inertial_source", "onboard_imu")
        ).lower()
        if self.actor_inertial_source not in {"onboard_imu", "mocap"}:
            raise ValueError(
                "actor_inertial_source must be 'onboard_imu' or 'mocap'"
            )
        timeout_ms = float(exp_config.get("mocap_timeout_ms", 100.0))
        if self.mocap_source == "json_zmq":
            self.subscriber = CarryMocapSubscriber(exp_config["mocap_zmq_addr"], timeout_ms)
        elif self.mocap_source == "lumo":
            from rl_policy.lumo_mocap import LuMoCarrySource

            self.subscriber = LuMoCarrySource(exp_config.get("lumo", {}), timeout_ms)
        elif self.mocap_source == "cmavatar":
            from rl_policy.cmavatar_mocap import CMAvatarCarrySource

            self.subscriber = CMAvatarCarrySource(
                exp_config.get("cmavatar", {}), timeout_ms
            )
        else:
            raise ValueError(f"unsupported mocap_source={self.mocap_source!r}")
        offsets = exp_config.get("hand_contact_offsets", [[0.11, -0.01, 0.0], [0.11, 0.01, 0.0]])
        capsule_segments = exp_config.get(
            "hand_capsule_segments",
            [[[0.07, 0.0, 0.0], [0.15, -0.02, 0.0]], [[0.07, 0.0, 0.0], [0.15, 0.02, 0.0]]],
        )
        sphere_log_actor = bool(exp_config.get("sphere_log_actor", False))
        privileged_body_names = exp_config.get("privileged_body_names")
        if sphere_log_actor and privileged_body_names is None:
            raise ValueError("sphere_log_actor requires privileged_body_names")
        self.sphere_log_actor = sphere_log_actor
        self.observer = CarryActorObservation(
            default,
            G1HandForwardKinematics(
                exp_config["kinematics_xml"],
                joint_names,
                offsets,
                capsule_segments,
                privileged_body_names=privileged_body_names,
                privileged_torso_body_name=str(
                    exp_config.get("privileged_torso_body_name", "torso_link")
                ),
            ),
        )
        contact_source = str(exp_config.get("contact_source", "packet")).lower()
        if contact_source == "packet":
            self.contact_estimator = None
        elif contact_source == "box_hysteresis":
            self.contact_estimator = HandBoxContactEstimator(exp_config.get("contact_estimator", {}))
        else:
            raise ValueError(f"unsupported contact_source={contact_source!r}")
        self.controller_profile = str(
            exp_config.get("phase_controller_profile", "legacy_deploy")
        ).lower()
        if self.controller_profile == "newintact80m_hybrid_direct_place":
            phase_latents = load_hybrid_direct_place_latents(exp_config["latent_dir"])
            approach_latents = load_phase_latents(exp_config["approach_latent_dir"])
            phase_latents["approach"] = approach_latents["approach"]
            phase_latents["turn"] = approach_latents["turn"]
        else:
            phase_latents = load_phase_latents(exp_config["latent_dir"])
        get_up_path = exp_config.get("get_up_latent")
        if get_up_path is not None:
            phase_latents["get_up"] = load_get_up_latent(get_up_path)
        controller_types = {
            "legacy_deploy": CarryPhaseController,
            "h20_reward736": H20Reward736PhaseController,
            "bestcarry140935168_strong": BestCarry140935168StrongController,
            "intact_logmap150544384_strong": BestCarry140935168StrongController,
            "newintact80m_hybrid_direct_place": NewIntactHybridDirectPlaceController,
        }
        try:
            controller_type = controller_types[self.controller_profile]
        except KeyError as exc:
            raise ValueError(
                "phase_controller_profile must be 'legacy_deploy', 'h20_reward736', "
                "'bestcarry140935168_strong', 'intact_logmap150544384_strong', or "
                "'newintact80m_hybrid_direct_place'"
            ) from exc
        phase_config = exp_config.get("phase", {})
        self.controller = controller_type(phase_config, phase_latents, dt)
        self.initial_scene = (
            InitialCarryScene(
                exp_config["initial_scene"],
                phase_config.get(
                    "box_geom_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]
                ),
                phase_config.get("box_geom_position", [0.0, 0.0, 0.0]),
            )
            if exp_config.get("initial_scene") is not None
            else None
        )
        fixed_goal_path = exp_config.get("fixed_goal_latent")
        self.fixed_goal_latent = (
            load_fixed_goal_latent(fixed_goal_path) if fixed_goal_path is not None else None
        )
        tracking_path = exp_config.get("tracking_latent")
        self.tracking_latents = (
            load_tracking_latents(tracking_path) if tracking_path is not None else None
        )
        if self.fixed_goal_latent is not None and self.tracking_latents is not None:
            raise ValueError("fixed_goal_latent and tracking_latent are mutually exclusive")
        self.tracking_loop = bool(exp_config.get("tracking_loop", False))
        self.tracking_reference = (
            TrackingReferenceAnchor(exp_config["tracking_reference"])
            if self.tracking_latents is not None
            else None
        )
        self.runtime_mode = (
            "tracking"
            if self.tracking_latents is not None
            else "fixed_goal"
            if self.fixed_goal_latent is not None
            else "phased"
        )
        if self.initial_scene is not None and self.runtime_mode != "phased":
            raise ValueError("initial_scene is supported only by phased Carry")
        self.fixed_goal_target_mode = str(
            exp_config.get("fixed_goal_target_mode", "configured")
        ).lower()
        if self.fixed_goal_target_mode not in {"configured", "follow_object"}:
            raise ValueError(
                "fixed_goal_target_mode must be 'configured' or 'follow_object'"
            )
        if self.fixed_goal_latent is None and self.fixed_goal_target_mode != "configured":
            raise ValueError("fixed_goal_target_mode requires fixed_goal_latent")
        self.expected_input_dim = int(exp_config.get("expected_input_dim", ONNX_INPUT_DIM))
        required_input_dim = (
            SPHERE_LOG_ONNX_INPUT_DIM if self.sphere_log_actor else ONNX_INPUT_DIM
        )
        if self.expected_input_dim != required_input_dim:
            raise ValueError(
                f"expected_input_dim must be {required_input_dim} when "
                f"sphere_log_actor={self.sphere_log_actor}, got {self.expected_input_dim}"
            )
        approach_model_path = exp_config.get("approach_model_path")
        self.approach_model_path = (
            None
            if approach_model_path is None
            else str(Path(str(approach_model_path)).expanduser().resolve())
        )
        if (
            self.controller_profile == "newintact80m_hybrid_direct_place"
            and self.approach_model_path is None
        ):
            raise ValueError(
                "newintact80m_hybrid_direct_place requires approach_model_path"
            )
        self.active_model_role = "approach" if self.approach_model_path else "main"
        self.pending_metrics: dict[str, Any] | None = None
        self.pending_inference_trace: dict[str, np.ndarray | None] | None = None
        self.last_mode = "starting"
        self.step = 0
        self.tracking_index = 0
        self.tracking_completion_logged = False
        calibration = exp_config.get("deployment_calibration", {})
        if not isinstance(calibration, Mapping):
            raise ValueError("deployment_calibration must be a mapping")
        self.deployment_calibration_enabled = bool(calibration.get("enabled", False))
        self.deployment_contact_calibration_enabled = bool(
            calibration.get("contact_calibration_enabled", True)
        )
        self.calibration_stage = (
            "pelvis" if self.deployment_calibration_enabled else "disabled"
        )
        # Pelvis and box-height calibration use explicit, one-shot capture
        # windows. Entering the default pose with A must never start or replace
        # calibration data; B starts a 50 Hz window which auto-finalizes after
        # the configured number of valid frames.
        self.calibration_capture_active = False
        self.calibration_contact_label: int | None = None
        self.calibration_cfg = SimpleNamespace(
            expected_standing_pelvis_height=float(
                calibration.get("expected_standing_pelvis_height", 0.80)
            ),
            min_pelvis_samples=int(calibration.get("min_pelvis_samples", 50)),
            min_box_height_samples=int(
                calibration.get("min_box_height_samples", 50)
            ),
            min_contact_samples_per_label=int(
                calibration.get("min_contact_samples_per_label", 25)
            ),
            max_samples_per_bucket=int(
                calibration.get("max_samples_per_bucket", 5000)
            ),
            max_default_joint_error=float(
                calibration.get("max_default_joint_error", 0.15)
            ),
            min_pelvis_upright=float(
                calibration.get("min_pelvis_upright", 0.90)
            ),
            max_pelvis_height_std=float(
                calibration.get("max_pelvis_height_std", 0.015)
            ),
            max_pelvis_height_correction=float(
                calibration.get("max_pelvis_height_correction", 0.25)
            ),
            min_box_upright=float(calibration.get("min_box_upright", 0.90)),
            max_box_speed=float(calibration.get("max_box_speed", 0.10)),
            max_box_height_std=float(
                calibration.get("max_box_height_std", 0.010)
            ),
            max_box_height_correction=float(
                calibration.get("max_box_height_correction", 0.15)
            ),
            contact_quantile=float(calibration.get("contact_quantile", 0.90)),
            no_contact_quantile=float(
                calibration.get("no_contact_quantile", 0.10)
            ),
            contact_min_separation=float(
                calibration.get("contact_min_separation", 0.010)
            ),
            contact_exit_margin=float(
                calibration.get("contact_exit_margin", 0.015)
            ),
            contact_max_enter_distance=float(
                calibration.get("contact_max_enter_distance", 0.15)
            ),
        )
        positive_integer_fields = (
            "min_pelvis_samples",
            "min_box_height_samples",
            "min_contact_samples_per_label",
            "max_samples_per_bucket",
        )
        if any(getattr(self.calibration_cfg, name) <= 0 for name in positive_integer_fields):
            raise ValueError("deployment calibration sample counts must be positive")
        if not 0.5 <= self.calibration_cfg.min_pelvis_upright <= 1.0:
            raise ValueError("deployment_calibration.min_pelvis_upright must be in [0.5, 1]")
        if not 0.5 <= self.calibration_cfg.min_box_upright <= 1.0:
            raise ValueError("deployment_calibration.min_box_upright must be in [0.5, 1]")
        if not 0.0 < self.calibration_cfg.contact_quantile < 1.0:
            raise ValueError("deployment_calibration.contact_quantile must be in (0, 1)")
        if not 0.0 < self.calibration_cfg.no_contact_quantile < 1.0:
            raise ValueError("deployment_calibration.no_contact_quantile must be in (0, 1)")
        self.calibration_samples: dict[str, Any] = {
            "pelvis_height": [],
            "pelvis_upright": [],
            "box_height": [],
            "contact": [[[], []], [[], []]],
        }
        self.calibration_results: dict[str, Any] = {}
        saved_result_path = calibration.get("saved_result_path")
        self.calibration_result_path = (
            None
            if saved_result_path is None
            else Path(str(saved_result_path)).expanduser().resolve(strict=False)
        )
        self.saved_calibration_source_path = self._locate_saved_calibration()
        self.saved_calibration_detected = self.saved_calibration_source_path is not None
        self.saved_calibration_last_error: str | None = None
        self._calibration_base_root_thresholds = {
            name: float(getattr(self.controller.cfg, name))
            for name in (
                "fall_root_height",
                "fall_tilted_root_height",
                "get_up_root_height",
            )
        }
        logger.info(
            "Carry runtime configured: mode={} controller_profile={}".format(
                self.runtime_mode, self.controller_profile
            )
        )

    @property
    def deployment_calibration_complete(self) -> bool:
        return self.calibration_stage == "complete"

    def _append_calibration_sample(self, bucket: list[float], value: float) -> None:
        bucket.append(float(value))
        overflow = len(bucket) - self.calibration_cfg.max_samples_per_bucket
        if overflow > 0:
            del bucket[:overflow]

    def set_calibration_contact_label(self, contact: bool) -> tuple[bool, str]:
        if not self.deployment_calibration_enabled:
            return False, "deployment calibration is disabled"
        if not self.deployment_contact_calibration_enabled:
            return False, "contact calibration is disabled; FK capsule thresholds are fixed"
        if self.calibration_stage != "contact":
            return False, f"contact label is unavailable during {self.calibration_stage} calibration"
        self.calibration_contact_label = int(bool(contact))
        label = "CONTACT" if contact else "NO_CONTACT"
        return True, f"queued one {label} annotation for the next control frame"

    def clear_calibration_contact_label(self) -> None:
        self.calibration_contact_label = None

    @property
    def saved_deployment_calibration_available(self) -> bool:
        return self.saved_calibration_detected

    def _locate_saved_calibration(self) -> Path | None:
        canonical = self.calibration_result_path
        if canonical is None:
            return None
        if canonical.is_file():
            return canonical
        # Backward-compatible discovery for a calibration completed before the
        # fixed latest_calibration.json copy was introduced.
        try:
            candidates = [
                path
                for path in canonical.parent.glob("*/calibration.json")
                if path.is_file()
            ]
            return max(candidates, key=lambda item: item.stat().st_mtime_ns, default=None)
        except OSError:
            return None

    def activate_saved_deployment_calibration(self) -> tuple[bool, str]:
        """Validate and apply a previously completed on-site calibration."""
        if self.calibration_result_path is None:
            return False, "deployment_calibration.saved_result_path is not configured"
        path = self._locate_saved_calibration()
        self.saved_calibration_source_path = path
        if path is None:
            self.saved_calibration_detected = False
            return False, (
                "saved calibration does not exist: "
                f"{self.calibration_result_path}"
            )
        self.saved_calibration_detected = True
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("root must be a JSON object")
            if int(payload.get("schema_version", -1)) != 1:
                raise ValueError("unsupported schema_version")
            if not bool(payload.get("complete", False)):
                raise ValueError("saved calibration is not marked complete")
            saved_profile = payload.get("controller_profile")
            if saved_profile is not None and str(saved_profile) != self.controller_profile:
                raise ValueError(
                    "controller profile mismatch: "
                    f"{saved_profile!r} != {self.controller_profile!r}"
                )
            results = payload.get("results")
            if not isinstance(results, Mapping):
                raise ValueError("results must be a JSON object")

            pelvis = results.get("pelvis")
            box_height = results.get("box_height")
            contact = results.get("contact")
            if not isinstance(pelvis, Mapping) or not isinstance(box_height, Mapping):
                raise ValueError("pelvis and box_height results are required")
            if self.deployment_contact_calibration_enabled and not isinstance(contact, Mapping):
                raise ValueError("contact result is required")

            correction = float(pelvis["height_correction_m"])
            standing_height = float(pelvis["standing_height_m"])
            if not np.isfinite(correction) or not np.isfinite(standing_height):
                raise ValueError("pelvis values must be finite")
            if abs(correction) > self.calibration_cfg.max_pelvis_height_correction:
                raise ValueError("pelvis correction exceeds the current safety limit")
            if not np.isclose(
                standing_height,
                self.calibration_cfg.expected_standing_pelvis_height + correction,
                atol=0.005,
            ):
                raise ValueError("pelvis standing height and correction are inconsistent")

            ground_center_height = float(box_height["ground_center_height_m"])
            current_ground_height = float(self.controller.cfg.ground_center_height)
            if not np.isfinite(ground_center_height):
                raise ValueError("box ground-center height must be finite")
            if (
                abs(ground_center_height - current_ground_height)
                > self.calibration_cfg.max_box_height_correction
            ):
                raise ValueError("box-height correction exceeds the current safety limit")

            if self.contact_estimator is None:
                raise ValueError("contact estimator is unavailable")
            if not self.deployment_contact_calibration_enabled:
                if self.contact_estimator.use_hand_rigid_origins:
                    raise ValueError(
                        "FK-only calibration cannot use hand rigid origins"
                    )
            else:
                enter_values = np.zeros(2, dtype=np.float64)
                exit_values = np.zeros(2, dtype=np.float64)
                for hand_index, side in enumerate(("left", "right")):
                    hand = contact.get(side)
                    if not isinstance(hand, Mapping):
                        raise ValueError(f"missing {side} contact result")
                    enter = float(hand["enter_distance_m"])
                    exit_ = float(hand["exit_distance_m"])
                    if not np.isfinite(enter) or not np.isfinite(exit_):
                        raise ValueError(f"{side} contact thresholds must be finite")
                    if not 0.0 <= enter < exit_:
                        raise ValueError(f"{side} contact hysteresis is invalid")
                    if enter > self.calibration_cfg.contact_max_enter_distance:
                        raise ValueError(f"{side} enter threshold exceeds the safety limit")
                    if exit_ > enter + self.calibration_cfg.contact_exit_margin + 1.0e-6:
                        raise ValueError(f"{side} exit margin exceeds the safety limit")
                    enter_values[hand_index] = enter
                    exit_values[hand_index] = exit_
            for name, base in self._calibration_base_root_thresholds.items():
                setattr(self.controller.cfg, name, base + correction)
            self.controller.cfg.ground_center_height = ground_center_height
            if hasattr(self.subscriber, "target_position_world"):
                self.subscriber.target_position_world[2] = ground_center_height
            if self.deployment_contact_calibration_enabled:
                self.contact_estimator.set_hand_origin_thresholds(
                    enter_values, exit_values
                )
            self.calibration_results = dict(results)
            self.calibration_results["reuse"] = {
                "source_path": str(path),
                "loaded_at_ns": time.time_ns(),
            }
            self.calibration_contact_label = None
            self.calibration_capture_active = False
            self.calibration_stage = "complete"
            self.saved_calibration_last_error = None
            return True, f"loaded and applied saved calibration: {path}"
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            self.saved_calibration_last_error = str(exc)
            return False, f"saved calibration rejected: {exc}"

    def observe_deployment_calibration(
        self,
        frame: CarryMocapFrame,
        metrics: Mapping[str, Any],
        joint_position: np.ndarray,
        default_joint_position: np.ndarray,
        *,
        default_pose_active: bool,
    ) -> None:
        if not self.deployment_calibration_enabled or self.deployment_calibration_complete:
            return
        cfg = self.calibration_cfg
        if self.calibration_stage == "pelvis":
            if not self.calibration_capture_active:
                return
            joint_error = self.support_pose_joint_error(
                joint_position, default_joint_position
            )
            if (
                default_pose_active
                and joint_error <= cfg.max_default_joint_error
                and float(metrics["pelvis_up_z"]) >= cfg.min_pelvis_upright
            ):
                self._append_calibration_sample(
                    self.calibration_samples["pelvis_height"], metrics["root_height"]
                )
                self._append_calibration_sample(
                    self.calibration_samples["pelvis_upright"], metrics["pelvis_up_z"]
                )
            return
        if self.calibration_stage == "box_height":
            if not self.calibration_capture_active:
                return
            if (
                default_pose_active
                and float(metrics["box_upright"]) >= cfg.min_box_upright
                and float(metrics["object_speed"]) <= cfg.max_box_speed
            ):
                self._append_calibration_sample(
                    self.calibration_samples["box_height"], frame.object_position[2]
                )
            return
        if self.calibration_stage != "contact" or default_pose_active:
            return
        if self.calibration_contact_label is None or self.contact_estimator is None:
            return
        label = self.calibration_contact_label
        # X/Y are one-shot annotations. Consume the event on this frame even if
        # one or both hand rigid bodies are unavailable, so a stale label can
        # never be applied to a later physical hand/box configuration.
        self.calibration_contact_label = None
        for hand_index in range(2):
            if self.contact_estimator.last_geometry_source[hand_index] != "rigid_origin":
                continue
            distance = float(
                self.contact_estimator.last_hand_origin_surface_distance[hand_index]
            )
            if np.isfinite(distance):
                self._append_calibration_sample(
                    self.calibration_samples["contact"][hand_index][label], distance
                )

    def support_pose_joint_error(
        self,
        joint_position: np.ndarray,
        default_joint_position: np.ndarray,
    ) -> float:
        """Return the max default-pose error over both legs and the waist."""
        current = np.asarray(joint_position, dtype=np.float64).reshape(-1)
        default = np.asarray(default_joint_position, dtype=np.float64).reshape(-1)
        if current.shape != default.shape:
            raise ValueError(
                "joint and default positions must have the same shape for pelvis calibration"
            )
        indices = np.asarray(
            self.support_pose_joint_indices, dtype=np.int64
        ).reshape(-1)
        if indices.size == 0 or int(np.max(indices)) >= current.size:
            raise ValueError("pelvis calibration joint indices are invalid")
        return float(np.max(np.abs(current[indices] - default[indices])))

    def start_deployment_calibration_capture(
        self, *, default_pose_active: bool
    ) -> tuple[bool, str]:
        """Start a fresh fixed-length pelvis or box-height capture window."""
        if not self.deployment_calibration_enabled:
            return False, "deployment calibration is disabled"
        if self.calibration_stage not in {"pelvis", "box_height"}:
            return False, f"stage {self.calibration_stage} does not use continuous capture"
        if not default_pose_active:
            return False, "press A and settle in the default pose before starting capture"
        if self.calibration_capture_active:
            return False, f"{self.calibration_stage} capture is already running"

        if self.calibration_stage == "pelvis":
            self.calibration_samples["pelvis_height"].clear()
            self.calibration_samples["pelvis_upright"].clear()
            required = self.calibration_cfg.min_pelvis_samples
        else:
            self.calibration_samples["box_height"].clear()
            required = self.calibration_cfg.min_box_height_samples
        self.calibration_capture_active = True
        return True, (
            f"{self.calibration_stage} 50 Hz capture started; "
            f"auto-finalizing after {required} valid frames"
        )

    def maybe_finalize_deployment_calibration_capture(
        self, *, default_pose_active: bool
    ) -> tuple[bool, str] | None:
        """Finalize an active capture as soon as its valid-frame quota is met."""
        if not self.calibration_capture_active:
            return None
        if self.calibration_stage == "pelvis":
            count = len(self.calibration_samples["pelvis_height"])
            required = self.calibration_cfg.min_pelvis_samples
        elif self.calibration_stage == "box_height":
            count = len(self.calibration_samples["box_height"])
            required = self.calibration_cfg.min_box_height_samples
        else:
            self.calibration_capture_active = False
            return False, f"capture cancelled in unexpected stage {self.calibration_stage}"
        if count < required:
            return None

        completed_stage = self.calibration_stage
        self.calibration_capture_active = False
        accepted, message = self.advance_deployment_calibration(
            default_pose_active=default_pose_active
        )
        prefix = "completed" if accepted else "rejected"
        return accepted, f"{completed_stage} capture {prefix}: {message}"

    def advance_deployment_calibration(
        self, *, default_pose_active: bool
    ) -> tuple[bool, str]:
        if not self.deployment_calibration_enabled:
            return False, "deployment calibration is disabled"
        cfg = self.calibration_cfg
        if self.calibration_stage == "pelvis":
            if not default_pose_active:
                return False, "press A to enter and hold the default pose before finishing pelvis calibration"
            values = np.asarray(self.calibration_samples["pelvis_height"], dtype=np.float64)
            if values.size < cfg.min_pelvis_samples:
                return False, f"pelvis samples {values.size}/{cfg.min_pelvis_samples}; keep standing still"
            median = float(np.median(values))
            std = float(np.std(values))
            correction = median - cfg.expected_standing_pelvis_height
            if std > cfg.max_pelvis_height_std:
                self.calibration_samples["pelvis_height"].clear()
                self.calibration_samples["pelvis_upright"].clear()
                return False, f"pelvis height std {std:.4f}m is unstable; samples cleared"
            if abs(correction) > cfg.max_pelvis_height_correction:
                return False, f"pelvis height correction {correction:+.3f}m exceeds safety limit"
            updated = {}
            for name, base in self._calibration_base_root_thresholds.items():
                value = base + correction
                setattr(self.controller.cfg, name, value)
                updated[name] = value
            self.calibration_results["pelvis"] = {
                "samples": int(values.size),
                "standing_height_m": median,
                "height_std_m": std,
                "expected_standing_height_m": cfg.expected_standing_pelvis_height,
                "height_correction_m": correction,
                "updated_root_thresholds_m": updated,
            }
            self.calibration_capture_active = False
            self.calibration_stage = "box_height"
            return True, f"pelvis calibrated at {median:.3f}m; place the upright box on the ground"

        if self.calibration_stage == "box_height":
            if not default_pose_active:
                return False, "default pose must remain active during box-height calibration"
            values = np.asarray(self.calibration_samples["box_height"], dtype=np.float64)
            if values.size < cfg.min_box_height_samples:
                return False, f"box-height samples {values.size}/{cfg.min_box_height_samples}; keep box still"
            median = float(np.median(values))
            std = float(np.std(values))
            previous = float(self.controller.cfg.ground_center_height)
            if std > cfg.max_box_height_std:
                self.calibration_samples["box_height"].clear()
                return False, f"box height std {std:.4f}m is unstable; samples cleared"
            if abs(median - previous) > cfg.max_box_height_correction:
                return False, f"box height correction {median - previous:+.3f}m exceeds safety limit"
            self.controller.cfg.ground_center_height = median
            if hasattr(self.subscriber, "target_position_world"):
                self.subscriber.target_position_world[2] = median
            self.calibration_results["box_height"] = {
                "samples": int(values.size),
                "ground_center_height_m": median,
                "height_std_m": std,
                "previous_ground_center_height_m": previous,
            }
            self.calibration_capture_active = False
            if not self.deployment_contact_calibration_enabled:
                if self.contact_estimator is None:
                    return False, "FK contact estimator is unavailable"
                if self.contact_estimator.use_hand_rigid_origins:
                    return False, "FK-only profile unexpectedly enables hand rigid origins"
                self.calibration_results["contact"] = {
                    "mode": "fk_capsule_fixed",
                    "calibrated": False,
                    "left": {
                        "enter_distance_m": self.contact_estimator.enter_distance,
                        "exit_distance_m": self.contact_estimator.exit_distance,
                    },
                    "right": {
                        "enter_distance_m": self.contact_estimator.enter_distance,
                        "exit_distance_m": self.contact_estimator.exit_distance,
                    },
                }
                self.calibration_contact_label = None
                self.calibration_stage = "complete"
                return True, (
                    "box height calibrated; FK capsule contact uses fixed "
                    f"{self.contact_estimator.enter_distance:.3f}/"
                    f"{self.contact_estimator.exit_distance:.3f}m enter/exit thresholds; "
                    "calibration complete"
                )
            self.calibration_stage = "contact"
            self.calibration_contact_label = None
            return True, "box height calibrated; suspend robot, press A to release default pose, then label X/Y"

        if self.calibration_stage == "contact":
            if default_pose_active:
                return False, "press A to release the default pose before finishing contact calibration"
            if self.contact_estimator is None or not self.contact_estimator.use_hand_rigid_origins:
                return False, "hand rigid-origin contact estimator is not enabled"
            enter_values = np.zeros(2, dtype=np.float64)
            exit_values = np.zeros(2, dtype=np.float64)
            diagnostics = {}
            for hand_index, side in enumerate(("left", "right")):
                no_contact = np.asarray(
                    self.calibration_samples["contact"][hand_index][0], dtype=np.float64
                )
                contact = np.asarray(
                    self.calibration_samples["contact"][hand_index][1], dtype=np.float64
                )
                if (
                    no_contact.size < cfg.min_contact_samples_per_label
                    or contact.size < cfg.min_contact_samples_per_label
                ):
                    return False, (
                        f"{side} contact samples no/contact={no_contact.size}/{contact.size}; "
                        f"need at least {cfg.min_contact_samples_per_label} each"
                    )
                contact_edge = float(np.quantile(contact, cfg.contact_quantile))
                no_contact_edge = float(
                    np.quantile(no_contact, cfg.no_contact_quantile)
                )
                separation = no_contact_edge - contact_edge
                if separation < cfg.contact_min_separation:
                    return False, (
                        f"{side} labels are not separable: contact_q={contact_edge:.3f}m "
                        f"no_contact_q={no_contact_edge:.3f}m"
                    )
                enter = 0.5 * (contact_edge + no_contact_edge)
                if enter > cfg.contact_max_enter_distance:
                    return False, f"{side} learned enter distance {enter:.3f}m exceeds safety limit"
                exit_ = min(
                    no_contact_edge - 1.0e-4,
                    enter + cfg.contact_exit_margin,
                )
                if exit_ <= enter:
                    return False, f"{side} learned hysteresis interval is invalid"
                enter_values[hand_index] = enter
                exit_values[hand_index] = exit_
                diagnostics[side] = {
                    "no_contact_samples": int(no_contact.size),
                    "contact_samples": int(contact.size),
                    "contact_quantile_m": contact_edge,
                    "no_contact_quantile_m": no_contact_edge,
                    "enter_distance_m": enter,
                    "exit_distance_m": exit_,
                }
            self.contact_estimator.set_hand_origin_thresholds(
                enter_values, exit_values
            )
            self.calibration_results["contact"] = diagnostics
            self.calibration_contact_label = None
            self.calibration_capture_active = False
            self.calibration_stage = "complete"
            return True, (
                "contact calibrated: enter="
                f"{enter_values.tolist()}m exit={exit_values.tolist()}m; press A for default pose"
            )

        return False, "deployment calibration is already complete"

    def deployment_calibration_snapshot(self) -> dict[str, Any]:
        contact_counts = {
            side: {
                "no_contact": len(self.calibration_samples["contact"][index][0]),
                "contact": len(self.calibration_samples["contact"][index][1]),
            }
            for index, side in enumerate(("left", "right"))
        }
        return {
            "enabled": self.deployment_calibration_enabled,
            "contact_calibration_enabled": self.deployment_contact_calibration_enabled,
            "stage": self.calibration_stage,
            "complete": self.deployment_calibration_complete,
            "capture_active": self.calibration_capture_active,
            "contact_label": self.calibration_contact_label,
            "sample_counts": {
                "pelvis": len(self.calibration_samples["pelvis_height"]),
                "box_height": len(self.calibration_samples["box_height"]),
                "contact": contact_counts,
            },
            "results": self.calibration_results,
            "saved_result_path": (
                None
                if self.calibration_result_path is None
                else str(self.calibration_result_path)
            ),
            "saved_result_available": self.saved_deployment_calibration_available,
            "saved_result_source_path": (
                None
                if self.saved_calibration_source_path is None
                else str(self.saved_calibration_source_path)
            ),
            "saved_result_last_error": self.saved_calibration_last_error,
        }

    def reset(self) -> None:
        self.observer.reset()
        if self.contact_estimator is not None:
            self.contact_estimator.reset()
        self.controller.reset()
        if self.tracking_reference is not None:
            self.tracking_reference.reset()
        if self.initial_scene is not None:
            self.initial_scene.reset()
        self.pending_metrics = None
        self.pending_inference_trace = None
        self.step = 0
        self.tracking_index = 0
        self.tracking_completion_logged = False
        self.active_model_role = "approach" if self.approach_model_path else "main"

    def actor_inertial_measurement(
        self, frame: CarryMocapFrame, state_processor: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the quaternion and body-frame angular velocity seen by the actor."""

        if self.actor_inertial_source == "mocap":
            quaternion = _unit_quaternion_wxyz(
                frame.robot_root_quaternion_wxyz,
                "mocap_robot_root_quaternion_wxyz",
            )
            if frame.robot_root_angular_velocity_world is None:
                raise CarryDataUnavailable(
                    "actor_inertial_source=mocap requires fresh robot-root angular velocity"
                )
            angular_velocity_body = quat_rotate_inverse_numpy(
                quaternion.reshape(1, 4),
                _vector(
                    frame.robot_root_angular_velocity_world,
                    3,
                    "mocap_robot_root_angular_velocity_world",
                ).reshape(1, 3),
            )[0]
            return quaternion, angular_velocity_body

        return (
            _unit_quaternion_wxyz(
                state_processor.root_quat_b,
                "onboard_imu_quaternion_wxyz",
            ),
            _vector(
                state_processor.root_ang_vel_b,
                3,
                "onboard_imu_angular_velocity_body",
            ),
        )

    def build_input(self, state_processor: Any, last_action: np.ndarray) -> tuple[np.ndarray, CarryMocapFrame]:
        source_frame = self.subscriber.poll()
        frame = (
            source_frame
            if isinstance(source_frame, CarryMocapFrame)
            else CarryMocapFrame.from_mapping(source_frame)
        )
        if self.tracking_reference is not None:
            frame = self.tracking_reference.align(frame)
        if self.fixed_goal_latent is not None and self.fixed_goal_target_mode == "follow_object":
            # The symmetric static goal was encoded with object_target_pos equal
            # to object_pos. Preserve that zero target-delta semantic online.
            frame = replace(frame, target_position_world=frame.object_position.copy())
        if self.initial_scene is not None:
            frame = self.initial_scene.apply(frame)
        actor_root_quaternion, actor_root_angular_velocity = (
            self.actor_inertial_measurement(frame, state_processor)
        )
        hand_points, hand_segments = self.observer.hand_fk.hand_geometry(
            frame.robot_root_position,
            frame.robot_root_quaternion_wxyz,
            _vector(state_processor.joint_pos, 29, "joint_position"),
            root_linear_velocity_world=frame.robot_root_linear_velocity_world,
            root_angular_velocity_body=actor_root_angular_velocity,
            joint_velocity=_vector(state_processor.joint_vel, 29, "joint_velocity"),
        )
        if self.contact_estimator is not None:
            frame = replace(
                frame,
                hand_contact=self.contact_estimator.update(frame, hand_segments),
            )
        observation = self.observer.build(
            frame, state_processor.joint_pos, state_processor.joint_vel,
            actor_root_quaternion, actor_root_angular_velocity, last_action,
            hand_points_world=hand_points,
        )
        hand_box_gaps = (
            None
            if self.contact_estimator is None
            else self.contact_estimator.last_surface_distance
        )
        metrics = self.controller.metrics(frame, hand_box_gaps)
        if self.contact_estimator is not None:
            metrics.update(
                {
                    "left_contact_geometry_source": str(
                        self.contact_estimator.last_geometry_source[0]
                    ),
                    "right_contact_geometry_source": str(
                        self.contact_estimator.last_geometry_source[1]
                    ),
                    "left_fk_hand_box_gap": float(
                        self.contact_estimator.last_fk_surface_distance[0]
                    ),
                    "right_fk_hand_box_gap": float(
                        self.contact_estimator.last_fk_surface_distance[1]
                    ),
                    "left_hand_origin_box_gap": float(
                        self.contact_estimator.last_hand_origin_surface_distance[0]
                    ),
                    "right_hand_origin_box_gap": float(
                        self.contact_estimator.last_hand_origin_surface_distance[1]
                    ),
                    "left_hand_origin_fk_distance": float(
                        self.contact_estimator.last_hand_origin_fk_distance[0]
                    ),
                    "right_hand_origin_fk_distance": float(
                        self.contact_estimator.last_hand_origin_fk_distance[1]
                    ),
                }
            )
        if self.tracking_latents is not None:
            actor = np.concatenate(
                (
                    observation["state"].copy(),
                    observation["last_action"],
                    observation["history_actor"],
                )
            )
            latent = self.tracking_latents[self.tracking_index]
            mode = f"tracking[{self.tracking_index}/{self.tracking_latents.shape[0] - 1}]"
        elif self.fixed_goal_latent is None:
            actor, latent, mode = self.controller.select(observation, metrics, frame)
            self.active_model_role = (
                "approach"
                if self.approach_model_path is not None
                and self.controller.phase == "approach"
                else "main"
            )
        else:
            actor = np.concatenate(
                (
                    observation["state"].copy(),
                    observation["last_action"],
                    observation["history_actor"],
                )
            )
            latent = self.fixed_goal_latent
            mode = "fixed_goal"
        if self.sphere_log_actor:
            # Match the frozen simulator: the privileged object context is taken
            # before the phase controller masks/modifies actor state.
            privileged_state = self.observer.hand_fk.privileged_state(
                observation["state"][-OBJECT_CONTEXT_DIM:]
            )
            inputs = np.concatenate((actor, privileged_state, latent))
        else:
            inputs = np.concatenate((actor, latent))
        inputs = inputs.reshape(1, -1).astype(np.float32)
        if inputs.shape != (1, self.expected_input_dim) or not np.all(np.isfinite(inputs)):
            raise CarryDataUnavailable(f"invalid Carry ONNX input shape/value: {inputs.shape}")
        self.pending_inference_trace = build_carry_inference_trace(
            actor,
            privileged_state if self.sphere_log_actor else None,
            latent,
            inputs,
        )
        self.pending_metrics = metrics
        self.last_mode = mode
        return inputs, frame

    def advance(self) -> None:
        if self.pending_metrics is not None:
            previous_phase = self.controller.phase
            if self.tracking_latents is not None:
                last_index = self.tracking_latents.shape[0] - 1
                if self.tracking_loop:
                    self.tracking_index = (self.tracking_index + 1) % self.tracking_latents.shape[0]
                elif self.tracking_index < last_index:
                    self.tracking_index += 1
                elif not self.tracking_completion_logged:
                    logger.info(
                        "Carry tracking sequence complete at step={}; holding final latent".format(
                            self.step + 1
                        )
                    )
                    self.tracking_completion_logged = True
            elif self.fixed_goal_latent is None:
                self.controller.update(self.pending_metrics)
            self.step += 1
            if self.tracking_latents is not None:
                phase_changed = False
                phase_name = "tracking"
            elif self.fixed_goal_latent is not None:
                phase_changed = False
                phase_name = "goal"
            else:
                phase_changed = self.controller.phase != previous_phase
                phase_name = self.controller.phase
            if phase_changed or self.step % 50 == 0:
                metrics = self.pending_metrics
                logger.info(
                    "Carry phase={} mode={} robot_box={:.3f}m box_goal={:.3f}m "
                    "lift={:.2f} pelvis_z={:.3f}m pelvis_up={:.3f} contact={}".format(
                        phase_name,
                        self.last_mode,
                        metrics["horizontal_distance"],
                        metrics["goal_distance"],
                        metrics["lift"],
                        metrics["root_height"],
                        metrics["pelvis_up_z"],
                        metrics["bilateral_contact"],
                    )
                )
                if self.contact_estimator is not None:
                    logger.info(
                        "Carry hand-box surface distance left={:.3f}m right={:.3f}m "
                        "source={}/{}".format(
                            *self.contact_estimator.last_surface_distance,
                            *self.contact_estimator.last_geometry_source,
                        )
                    )
            self.pending_metrics = None
