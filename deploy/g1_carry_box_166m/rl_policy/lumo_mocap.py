"""Direct LuMo/FZMotion rigid-body input for the Carry deployment runtime."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import time
from types import ModuleType
from typing import Any, Mapping

import numpy as np

from utils.math import quat_mul, quat_rotate_numpy


class LuMoDataUnavailable(RuntimeError):
    """Raised when LuMo cannot provide two safe, current rigid-body poses."""


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values, got {result.shape}")
    return result


def _quaternion(value: Any, name: str) -> np.ndarray:
    result = _vector(value, 4, name)
    norm = float(np.linalg.norm(result))
    if norm < 1.0e-6:
        raise ValueError(f"{name} has zero norm")
    return result / norm


def _rotate(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return quat_rotate_numpy(quaternion_wxyz.reshape(1, 4), vector.reshape(1, 3))[0]


def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _quaternion(
        quat_mul(left.reshape(1, 4), right.reshape(1, 4))[0],
        "composed quaternion",
    )


@dataclass(frozen=True)
class RigidBodySelector:
    name: str | None = None
    rigid_id: int | None = None

    @classmethod
    def from_value(cls, value: Any, label: str) -> "RigidBodySelector":
        if isinstance(value, str) and value:
            return cls(name=value)
        if isinstance(value, (int, np.integer)):
            return cls(rigid_id=int(value))
        if isinstance(value, Mapping):
            name = value.get("name")
            rigid_id = value.get("id")
            if name is not None and (not isinstance(name, str) or not name):
                raise ValueError(f"{label}.name must be a non-empty string")
            if rigid_id is not None:
                rigid_id = int(rigid_id)
            if name is None and rigid_id is None:
                raise ValueError(f"{label} requires name or id")
            return cls(name=name, rigid_id=rigid_id)
        raise ValueError(f"{label} must be a rigid-body name, ID, or mapping")

    def matches(self, rigid: Any) -> bool:
        return (
            (self.name is None or str(rigid.Name) == self.name)
            and (self.rigid_id is None or int(rigid.Id) == self.rigid_id)
        )

    def describe(self) -> str:
        fields = []
        if self.name is not None:
            fields.append(f"name={self.name!r}")
        if self.rigid_id is not None:
            fields.append(f"id={self.rigid_id}")
        return ", ".join(fields)


@dataclass(frozen=True)
class FrameExtrinsic:
    """Pose of the desired training frame expressed in the tracked rigid frame."""

    position: np.ndarray
    quaternion_wxyz: np.ndarray

    @classmethod
    def from_mapping(cls, value: Any, label: str) -> "FrameExtrinsic":
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be a mapping")
        return cls(
            position=_vector(value.get("position", [0.0, 0.0, 0.0]), 3, f"{label}.position"),
            quaternion_wxyz=_quaternion(
                value.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
                f"{label}.quaternion_wxyz",
            ),
        )


@dataclass(frozen=True)
class RigidBodySample:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    quality_grade: int


def load_lumo_proto_module(path: str | Path) -> ModuleType:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LuMo protobuf module not found: {path}")
    spec = importlib.util.spec_from_file_location("ufo_lumo_frame_pb2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load LuMo protobuf module from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(
            f"Failed to import LuMo protobuf definitions from {path}. "
            "Install the SDK-compatible protobuf runtime from requirements/runtime.txt."
        ) from exc
    if not hasattr(module, "Frame"):
        raise ImportError(f"LuMo protobuf module {path} does not define Frame")
    return module


class LuMoCarrySource:
    """Consume conflated LuMo protobuf frames and emit the Carry JSON schema."""

    def __init__(
        self,
        config: Mapping[str, Any],
        timeout_ms: float,
        *,
        connect: bool = True,
    ) -> None:
        self.config = dict(config)
        calibration = self.config.get("calibration", {})
        if not isinstance(calibration, Mapping):
            raise ValueError("lumo.calibration must be a mapping")
        if bool(self.config.get("require_calibrated", True)) and not bool(calibration.get("confirmed", False)):
            raise ValueError(
                "Mocap marker-to-training-frame calibration is not confirmed. "
                "Measure both extrinsics, set calibration.confirmed=true, then rerun preflight."
            )

        self.root_selector = RigidBodySelector.from_value(
            self.config.get("root_rigid_body"), "lumo.root_rigid_body"
        )
        self.object_selector = RigidBodySelector.from_value(
            self.config.get("object_rigid_body"), "lumo.object_rigid_body"
        )
        self.hand_origin_selectors = (
            (
                None
                if self.config.get("left_hand_rigid_body") is None
                else RigidBodySelector.from_value(
                    self.config.get("left_hand_rigid_body"),
                    "lumo.left_hand_rigid_body",
                )
            ),
            (
                None
                if self.config.get("right_hand_rigid_body") is None
                else RigidBodySelector.from_value(
                    self.config.get("right_hand_rigid_body"),
                    "lumo.right_hand_rigid_body",
                )
            ),
        )
        self.root_extrinsic = FrameExtrinsic.from_mapping(
            calibration.get("root_rigid_to_training_root"),
            "lumo.calibration.root_rigid_to_training_root",
        )
        self.object_extrinsic = FrameExtrinsic.from_mapping(
            calibration.get("object_rigid_to_training_object"),
            "lumo.calibration.object_rigid_to_training_object",
        )
        world = calibration.get(
            "world_to_ufo_world",
            calibration.get("lumo_world_to_ufo_world", {}),
        )
        if not isinstance(world, Mapping):
            raise ValueError("lumo.calibration.lumo_world_to_ufo_world must be a mapping")
        self.world_translation = _vector(
            world.get("translation", [0.0, 0.0, 0.0]), 3,
            "lumo.calibration.lumo_world_to_ufo_world.translation",
        )
        self.world_quaternion = _quaternion(
            world.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
            "lumo.calibration.lumo_world_to_ufo_world.quaternion_wxyz",
        )
        self.position_scale = float(self.config.get("position_scale", 0.001))
        self.velocity_scale = float(self.config.get("velocity_scale", self.position_scale))
        self.angular_velocity_scale = float(self.config.get("angular_velocity_scale", 1.0))
        if not all(
            np.isfinite(value) and value > 0.0
            for value in (self.position_scale, self.velocity_scale, self.angular_velocity_scale)
        ):
            raise ValueError("LuMo position/velocity scales must be positive finite values")
        self.angular_velocity_frame = str(self.config.get("angular_velocity_frame", "world")).lower()
        if self.angular_velocity_frame not in {"world", "rigid"}:
            raise ValueError("lumo.angular_velocity_frame must be 'world' or 'rigid'")
        self.enforce_quality_grade = bool(self.config.get("enforce_quality_grade", True))
        self.max_quality_grade = int(self.config.get("max_quality_grade", 2))
        if self.max_quality_grade not in (1, 2, 3):
            raise ValueError("lumo.max_quality_grade must be 1, 2, or 3")
        self.velocity_alpha = float(self.config.get("velocity_filter_alpha", 0.35))
        if not 0.0 < self.velocity_alpha <= 1.0:
            raise ValueError("lumo.velocity_filter_alpha must be in (0, 1]")
        self.object_linear_velocity_source = str(
            self.config.get("object_linear_velocity_source", "sdk")
        ).lower()
        self.root_linear_velocity_source = str(
            self.config.get("root_linear_velocity_source", "sdk")
        ).lower()
        if self.object_linear_velocity_source not in {"sdk", "position_delta"}:
            raise ValueError(
                "lumo.object_linear_velocity_source must be 'sdk' or 'position_delta'"
            )
        if self.root_linear_velocity_source not in {"sdk", "position_delta"}:
            raise ValueError(
                "lumo.root_linear_velocity_source must be 'sdk' or 'position_delta'"
            )
        self.position_delta_min_dt_s = float(
            self.config.get("position_delta_min_dt_seconds", 0.001)
        )
        self.position_delta_max_dt_s = float(
            self.config.get("position_delta_max_dt_seconds", 0.25)
        )
        self.position_delta_max_speed_mps = float(
            self.config.get("position_delta_max_speed_mps", 5.0)
        )
        if not (
            np.isfinite(self.position_delta_min_dt_s)
            and np.isfinite(self.position_delta_max_dt_s)
            and 0.0 < self.position_delta_min_dt_s < self.position_delta_max_dt_s
        ):
            raise ValueError(
                "lumo position-delta time limits must be finite and satisfy "
                "0 < min_dt < max_dt"
            )
        if not (
            np.isfinite(self.position_delta_max_speed_mps)
            and self.position_delta_max_speed_mps > 0.0
        ):
            raise ValueError("lumo.position_delta_max_speed_mps must be positive and finite")
        target_position = _vector(
            self.config.get("target_position_world"), 3, "lumo.target_position_world"
        )
        self.target_pose_semantics = str(
            self.config.get("target_pose_semantics", "training_reference")
        ).lower()
        if self.target_pose_semantics in {"physical_geom", "physical_ground"}:
            target_quaternion = _quaternion(
                self.config.get("target_quaternion_world", [1.0, 0.0, 0.0, 0.0]),
                "lumo.target_quaternion_world",
            )
            if self.target_pose_semantics == "physical_ground":
                center_height = float(self.config.get("target_box_center_height", 0.0))
                if not np.isfinite(center_height) or center_height <= 0.0:
                    raise ValueError(
                        "lumo.target_box_center_height must be positive for physical_ground"
                    )
                target_position = target_position.copy()
                target_position[2] += center_height
            # Apply the same physical-geom -> training-reference transform to
            # the goal that is applied to each measured physical box pose.
            self.target_position_world = target_position + _rotate(
                target_quaternion, self.object_extrinsic.position
            )
        elif self.target_pose_semantics == "training_reference":
            self.target_position_world = target_position
        else:
            raise ValueError(
                "lumo.target_pose_semantics must be 'physical_geom', "
                "'physical_ground', or 'training_reference'"
            )
        self.timeout_s = float(timeout_ms) / 1000.0
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("mocap timeout must be positive")

        self.latest_payload: dict[str, Any] | None = None
        # This timestamp is updated only by a fully valid frame. Rejected
        # quality/untracked/corrupt frames must not extend the hold-last window.
        self.last_receive_monotonic: float | None = None
        self.last_frame_id: int | None = None
        self.last_rejected_reason: str | None = None
        self.filtered_root_linear_velocity: np.ndarray | None = None
        self.filtered_root_angular_velocity: np.ndarray | None = None
        self.filtered_linear_velocity: np.ndarray | None = None
        self.filtered_angular_velocity: np.ndarray | None = None
        self.last_root_position_world: np.ndarray | None = None
        self.last_root_position_monotonic: float | None = None
        self.last_object_position_world: np.ndarray | None = None
        self.last_object_position_monotonic: float | None = None
        self.proto: ModuleType | None = None
        self.socket: Any | None = None
        if connect:
            import zmq

            proto_path = self.config.get(
                "protobuf_module",
                "动捕SDK/LuMoSDK_ubuntu_V200R001C00B002/Sample_Python/LusterFrameStruct_pb2.py",
            )
            self.proto = load_lumo_proto_module(proto_path)
            context = zmq.Context.instance()
            self.socket = context.socket(zmq.SUB)
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.socket.setsockopt(zmq.CONFLATE, 1)
            self.socket.setsockopt(zmq.RCVTIMEO, 0)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(str(self.config.get("address", "tcp://10.43.16.129:6868")))

    @staticmethod
    def available_rigid_bodies(frame: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": int(rigid.Id),
                "name": str(rigid.Name),
                "tracked": bool(rigid.IsTrack),
                "quality_grade": int(rigid.QualityGrade),
                "position": [float(rigid.X), float(rigid.Y), float(rigid.Z)],
            }
            for rigid in frame.rigidBodys
        ]

    @staticmethod
    def _select_rigid(frame: Any, selector: RigidBodySelector) -> Any:
        matches = [rigid for rigid in frame.rigidBodys if selector.matches(rigid)]
        if len(matches) != 1:
            available = [(int(item.Id), str(item.Name)) for item in frame.rigidBodys]
            raise LuMoDataUnavailable(
                f"expected one LuMo rigid body ({selector.describe()}), found {len(matches)}; "
                f"available={available}"
            )
        rigid = matches[0]
        if not bool(rigid.IsTrack):
            raise LuMoDataUnavailable(f"LuMo rigid body {selector.describe()} is not tracked")
        return rigid

    def _sample(self, rigid: Any, extrinsic: FrameExtrinsic, label: str) -> RigidBodySample:
        quality = int(rigid.QualityGrade)
        if self.enforce_quality_grade and (quality < 1 or quality > self.max_quality_grade):
            raise LuMoDataUnavailable(
                f"LuMo {label} rigid quality grade {quality} exceeds allowed {self.max_quality_grade}"
            )
        marker_position_lumo = np.asarray([rigid.X, rigid.Y, rigid.Z], dtype=np.float64) * self.position_scale
        marker_quaternion_lumo = _quaternion(
            [rigid.qw, rigid.qx, rigid.qy, rigid.qz], f"LuMo {label} quaternion"
        )
        marker_position_world = self.world_translation + _rotate(self.world_quaternion, marker_position_lumo)
        marker_quaternion_world = _multiply(self.world_quaternion, marker_quaternion_lumo)
        offset_world = _rotate(marker_quaternion_world, extrinsic.position)
        body_position_world = marker_position_world + offset_world
        body_quaternion_world = _multiply(marker_quaternion_world, extrinsic.quaternion_wxyz)

        linear_lumo = np.asarray(
            [rigid.speeds.XfSpeed, rigid.speeds.YfSpeed, rigid.speeds.ZfSpeed], dtype=np.float64
        ) * self.velocity_scale
        angular_raw = np.asarray(
            [rigid.palstance.fXPalstance, rigid.palstance.fYPalstance, rigid.palstance.fZPalstance],
            dtype=np.float64,
        ) * self.angular_velocity_scale
        linear_world = _rotate(self.world_quaternion, linear_lumo)
        if self.angular_velocity_frame == "rigid":
            angular_world = _rotate(marker_quaternion_world, angular_raw)
        else:
            angular_world = _rotate(self.world_quaternion, angular_raw)
        # The requested frame origin can be offset from the tracked rigid origin.
        linear_world = linear_world + np.cross(angular_world, offset_world)
        if not np.all(np.isfinite(linear_world)) or not np.all(np.isfinite(angular_world)):
            raise LuMoDataUnavailable(f"LuMo {label} velocity contains non-finite values")
        return RigidBodySample(
            position=body_position_world,
            quaternion_wxyz=body_quaternion_world,
            linear_velocity=linear_world,
            angular_velocity=angular_world,
            quality_grade=quality,
        )

    def _optional_rigid_origin(
        self,
        frame: Any,
        selector: RigidBodySelector | None,
        label: str,
    ) -> dict[str, Any]:
        """Return one optional rigid origin without invalidating the LuMo frame.

        Hand rigid bodies are an enhancement to the encoder/FK contact proxy.
        Unlike the robot root and box, an absent, untracked, or low-quality hand
        must fall back locally instead of stopping the policy.
        """

        if selector is None:
            return {"present": False, "reason": "not configured"}
        matches = [rigid for rigid in frame.rigidBodys if selector.matches(rigid)]
        if len(matches) != 1:
            return {
                "present": False,
                "reason": f"expected one {selector.describe()}, found {len(matches)}",
            }
        rigid = matches[0]
        quality = int(rigid.QualityGrade)
        if not bool(rigid.IsTrack):
            return {
                "present": False,
                "id": int(rigid.Id),
                "name": str(rigid.Name),
                "quality_grade": quality,
                "reason": "not tracked",
            }
        if self.enforce_quality_grade and (quality < 1 or quality > self.max_quality_grade):
            return {
                "present": False,
                "id": int(rigid.Id),
                "name": str(rigid.Name),
                "quality_grade": quality,
                "reason": f"quality grade exceeds allowed {self.max_quality_grade}",
            }
        position_lumo = np.asarray(
            [rigid.X, rigid.Y, rigid.Z], dtype=np.float64
        ) * self.position_scale
        if not np.all(np.isfinite(position_lumo)):
            return {
                "present": False,
                "id": int(rigid.Id),
                "name": str(rigid.Name),
                "quality_grade": quality,
                "reason": "origin contains non-finite values",
            }
        position_world = self.world_translation + _rotate(
            self.world_quaternion, position_lumo
        )
        return {
            "present": True,
            "id": int(rigid.Id),
            "name": str(rigid.Name),
            "quality_grade": quality,
            # Intentionally use only the rigid origin. Its quaternion is neither
            # parsed nor consumed by the hand contact estimator.
            "position": position_world.tolist(),
            "reason": None,
        }

    def _object_linear_velocity(
        self,
        obj: RigidBodySample,
        receipt_monotonic: float,
    ) -> tuple[np.ndarray, np.ndarray | None, bool]:
        """Return the configured velocity measurement and position-delta diagnostics."""
        source = getattr(self, "object_linear_velocity_source", "sdk")
        if source == "sdk":
            return obj.linear_velocity.copy(), None, False

        previous_position = self.last_object_position_world
        previous_time = self.last_object_position_monotonic
        self.last_object_position_world = obj.position.copy()
        self.last_object_position_monotonic = receipt_monotonic

        reset_filter = previous_position is None or previous_time is None
        position_delta_velocity = np.zeros(3, dtype=np.float64)
        if not reset_filter:
            dt = receipt_monotonic - previous_time
            min_dt = self.position_delta_min_dt_s
            max_dt = self.position_delta_max_dt_s
            if np.isfinite(dt) and min_dt <= dt <= max_dt:
                position_delta_velocity = (obj.position - previous_position) / dt
                speed = float(np.linalg.norm(position_delta_velocity))
                if speed > self.position_delta_max_speed_mps:
                    position_delta_velocity *= self.position_delta_max_speed_mps / speed
            else:
                # A first/reordered/long-gap frame starts a fresh derivative window.
                # Resetting the filter prevents a stale velocity from leaking forward.
                reset_filter = True
        return position_delta_velocity, position_delta_velocity.copy(), reset_filter

    def _root_linear_velocity(
        self,
        root: RigidBodySample,
        receipt_monotonic: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return root velocity, using published-frame position deltas when configured."""
        source = getattr(self, "root_linear_velocity_source", "sdk")
        if source == "sdk":
            return root.linear_velocity.copy(), None

        previous_position = self.last_root_position_world
        previous_time = self.last_root_position_monotonic
        self.last_root_position_world = root.position.copy()
        self.last_root_position_monotonic = receipt_monotonic

        reset_filter = previous_position is None or previous_time is None
        position_delta_velocity = np.zeros(3, dtype=np.float64)
        if not reset_filter:
            dt = receipt_monotonic - previous_time
            if (
                np.isfinite(dt)
                and self.position_delta_min_dt_s <= dt <= self.position_delta_max_dt_s
            ):
                position_delta_velocity = (root.position - previous_position) / dt
                speed = float(np.linalg.norm(position_delta_velocity))
                if speed > self.position_delta_max_speed_mps:
                    position_delta_velocity *= self.position_delta_max_speed_mps / speed
            else:
                reset_filter = True

        if self.filtered_root_linear_velocity is None or reset_filter:
            self.filtered_root_linear_velocity = position_delta_velocity.copy()
        else:
            self.filtered_root_linear_velocity += self.velocity_alpha * (
                position_delta_velocity - self.filtered_root_linear_velocity
            )
        return self.filtered_root_linear_velocity.copy(), position_delta_velocity

    def frame_to_payload(
        self,
        frame: Any,
        *,
        receipt_monotonic: float | None = None,
    ) -> dict[str, Any]:
        root_rigid = self._select_rigid(frame, self.root_selector)
        object_rigid = self._select_rigid(frame, self.object_selector)
        if int(root_rigid.Id) == int(object_rigid.Id):
            raise LuMoDataUnavailable(
                "robot root and object selectors resolved to the same LuMo rigid body "
                f"(id={int(root_rigid.Id)}, name={str(root_rigid.Name)!r})"
            )
        root = self._sample(root_rigid, self.root_extrinsic, "robot root")
        obj = self._sample(object_rigid, self.object_extrinsic, "object")
        hand_origins = {
            side: self._optional_rigid_origin(frame, selector, f"{side} hand")
            for side, selector in zip(
                ("left", "right"),
                getattr(self, "hand_origin_selectors", (None, None)),
            )
        }
        if receipt_monotonic is None:
            receipt_monotonic = time.monotonic()
        if not np.isfinite(receipt_monotonic):
            raise LuMoDataUnavailable("LuMo local receipt timestamp is non-finite")
        root_linear_velocity, root_position_delta_velocity = (
            self._root_linear_velocity(root, receipt_monotonic)
        )
        if self.filtered_root_angular_velocity is None:
            self.filtered_root_angular_velocity = root.angular_velocity.copy()
        else:
            self.filtered_root_angular_velocity += self.velocity_alpha * (
                root.angular_velocity - self.filtered_root_angular_velocity
            )
        linear_velocity, position_delta_velocity, reset_linear_filter = (
            self._object_linear_velocity(obj, receipt_monotonic)
        )
        alpha = self.velocity_alpha
        if self.filtered_linear_velocity is None or reset_linear_filter:
            self.filtered_linear_velocity = linear_velocity.copy()
        else:
            self.filtered_linear_velocity += alpha * (
                linear_velocity - self.filtered_linear_velocity
            )
        if self.filtered_angular_velocity is None:
            self.filtered_angular_velocity = obj.angular_velocity.copy()
        else:
            self.filtered_angular_velocity += alpha * (obj.angular_velocity - self.filtered_angular_velocity)
        return {
            "sequence": int(frame.FrameId),
            # LuMo's TimeStamp unit is not specified by the bundled API. The Carry
            # watchdog uses local receipt time, so publish an unambiguous local ns value.
            "timestamp_ns": time.time_ns(),
            "robot_root": {
                "position": root.position.tolist(),
                "quaternion_wxyz": root.quaternion_wxyz.tolist(),
                "linear_velocity_world": root_linear_velocity.tolist(),
                "angular_velocity_world": self.filtered_root_angular_velocity.tolist(),
            },
            "object": {
                "position": obj.position.tolist(),
                "quaternion_wxyz": obj.quaternion_wxyz.tolist(),
                "linear_velocity_world": self.filtered_linear_velocity.tolist(),
                "angular_velocity_world": self.filtered_angular_velocity.tolist(),
            },
            "hand_rigid_origins": hand_origins,
            # Replaced by the FK/box estimator in CarryRuntime for lumo input.
            "contacts": {"left": False, "right": False},
            "target_position_world": self.target_position_world.tolist(),
            "object_present": 1,
            "task_id": 1,
            "lumo": {
                "source_timestamp": int(frame.TimeStamp),
                "root_quality_grade": root.quality_grade,
                "object_quality_grade": obj.quality_grade,
                "root_linear_velocity_source": getattr(
                    self, "root_linear_velocity_source", "sdk"
                ),
                "root_linear_velocity_sdk_world": root.linear_velocity.tolist(),
                "root_linear_velocity_position_delta_world": (
                    root_position_delta_velocity.tolist()
                    if root_position_delta_velocity is not None
                    else None
                ),
                "object_linear_velocity_source": getattr(
                    self, "object_linear_velocity_source", "sdk"
                ),
                "object_linear_velocity_sdk_world": obj.linear_velocity.tolist(),
                "object_linear_velocity_position_delta_world": (
                    position_delta_velocity.tolist()
                    if position_delta_velocity is not None
                    else None
                ),
            },
        }

    def poll(self) -> dict[str, Any]:
        import zmq

        if self.socket is None or self.proto is None:
            raise RuntimeError("LuMoCarrySource was created with connect=False")

        # A policy cycle can find several LuMo packets queued at once. Only the
        # newest packet represents the observation we will actually publish.
        # Differentiating every queued packet against host recv() timestamps
        # makes their dt artificially sub-millisecond and used to reset the
        # position-delta velocity to zero intermittently.
        latest_message: bytes | None = None
        while True:
            try:
                latest_message = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        if latest_message is not None:
            receipt_monotonic = time.monotonic()
            frame = self.proto.Frame()
            try:
                frame.ParseFromString(latest_message)
                frame_id = int(frame.FrameId)
                if self.last_frame_id is not None and frame_id <= self.last_frame_id:
                    raise LuMoDataUnavailable(
                        f"non-increasing LuMo frame ID {frame_id} <= {self.last_frame_id}"
                    )
                payload = self.frame_to_payload(
                    frame, receipt_monotonic=receipt_monotonic
                )
            except LuMoDataUnavailable as exc:
                self.last_rejected_reason = str(exc)
            except Exception as exc:
                self.last_rejected_reason = f"invalid LuMo frame: {exc}"
            else:
                self.latest_payload = payload
                self.last_frame_id = frame_id
                self.last_receive_monotonic = receipt_monotonic
                self.last_rejected_reason = None

        if self.latest_payload is None or self.last_receive_monotonic is None:
            detail = (
                f"; latest rejected: {self.last_rejected_reason}"
                if self.last_rejected_reason
                else ""
            )
            raise LuMoDataUnavailable(f"no valid LuMo frame received{detail}")
        age = time.monotonic() - self.last_receive_monotonic
        if age > self.timeout_s:
            detail = (
                f"; latest rejected: {self.last_rejected_reason}"
                if self.last_rejected_reason
                else ""
            )
            raise LuMoDataUnavailable(
                f"LuMo valid frame is stale ({age * 1000.0:.0f} ms{detail})"
            )
        return self.latest_payload
