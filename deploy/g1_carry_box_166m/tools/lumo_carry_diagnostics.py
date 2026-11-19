"""Read-only LuMo rigid-body diagnostics before enabling the G1 Carry policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml
import zmq

from rl_policy.carry_runtime import CarryMocapFrame, InitialCarryScene
from rl_policy.lumo_mocap import LuMoCarrySource, load_lumo_proto_module
from utils.math import quat_mul, quat_rotate_numpy


DEFAULT_PROTO = Path(
    "动捕SDK/LuMoSDK_ubuntu_V200R001C00B002/Sample_Python/LusterFrameStruct_pb2.py"
)


def rigid_record(rigid, position_scale: float) -> dict:
    return {
        "id": int(rigid.Id),
        "name": str(rigid.Name),
        "tracked": bool(rigid.IsTrack),
        "quality_grade": int(rigid.QualityGrade),
        "position_raw": [float(rigid.X), float(rigid.Y), float(rigid.Z)],
        "position_m": [
            float(rigid.X) * position_scale,
            float(rigid.Y) * position_scale,
            float(rigid.Z) * position_scale,
        ],
        "quaternion_wxyz": [float(rigid.qw), float(rigid.qx), float(rigid.qy), float(rigid.qz)],
        "linear_velocity_raw": [
            float(rigid.speeds.XfSpeed),
            float(rigid.speeds.YfSpeed),
            float(rigid.speeds.ZfSpeed),
        ],
        "angular_velocity_raw": [
            float(rigid.palstance.fXPalstance),
            float(rigid.palstance.fYPalstance),
            float(rigid.palstance.fZPalstance),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://10.43.16.129:6868")
    parser.add_argument("--protobuf-module", type=Path, default=DEFAULT_PROTO)
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--root-name", default=None)
    parser.add_argument("--object-name", default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional Carry YAML; also print poses after its calibration transforms.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.position_scale <= 0.0 or args.seconds <= 0.0:
        parser.error("--position-scale and --seconds must be positive")

    proto = load_lumo_proto_module(args.protobuf_module)
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.address)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    started = time.monotonic()
    received = 0
    first_frame_id = None
    last_frame_id = None
    latest = None
    while time.monotonic() - started < args.seconds:
        if not dict(poller.poll(timeout=500)).get(socket, 0) & zmq.POLLIN:
            continue
        frame = proto.Frame()
        frame.ParseFromString(socket.recv())
        received += 1
        first_frame_id = int(frame.FrameId) if first_frame_id is None else first_frame_id
        last_frame_id = int(frame.FrameId)
        latest = frame

    elapsed = time.monotonic() - started
    if latest is None:
        raise RuntimeError(f"No LuMo frames received from {args.address} in {elapsed:.1f}s")
    records = [rigid_record(rigid, args.position_scale) for rigid in latest.rigidBodys]
    selected = {}
    for label, name in (("root", args.root_name), ("object", args.object_name)):
        if name is None:
            continue
        matches = [record for record in records if record["name"] == name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {label} rigid body named {name!r}, found {len(matches)}")
        selected[label] = matches[0]

    report = {
        "address": args.address,
        "elapsed_s": elapsed,
        "received_frames": received,
        "receive_rate_hz": received / elapsed,
        "first_frame_id": first_frame_id,
        "last_frame_id": last_frame_id,
        "source_timestamp": int(latest.TimeStamp),
        "position_scale": args.position_scale,
        "rigid_bodies": records,
        "selected": selected,
    }
    if args.config is not None:
        config = yaml.safe_load(args.config.read_text())
        if not isinstance(config, dict) or not isinstance(config.get("lumo"), dict):
            raise RuntimeError(f"{args.config} does not contain a lumo mapping")
        converter = LuMoCarrySource(
            config["lumo"],
            float(config.get("mocap_timeout_ms", 100.0)),
            connect=False,
        )
        transformed = converter.frame_to_payload(latest)
        report["transformed"] = transformed
        phase = config.get("phase", {})
        contact = config.get("contact_estimator", {})
        geom_position = np.asarray(
            phase.get(
                "box_geom_position", contact.get("geom_position", [0.0, 0.0, 0.0])
            ),
            dtype=np.float64,
        )
        geom_quaternion = np.asarray(
            phase.get(
                "box_geom_quaternion_wxyz",
                contact.get("geom_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
            ),
            dtype=np.float64,
        )
        reference_position = np.asarray(transformed["object"]["position"])
        reference_quaternion = np.asarray(
            transformed["object"]["quaternion_wxyz"]
        )
        physical_position = reference_position + quat_rotate_numpy(
            reference_quaternion.reshape(1, 4), geom_position.reshape(1, 3)
        )[0]
        physical_quaternion = quat_mul(
            reference_quaternion.reshape(1, 4),
            geom_quaternion.reshape(1, 4),
        )[0]
        physical_quaternion /= np.linalg.norm(physical_quaternion)
        report["physical_box"] = {
            "position": physical_position.tolist(),
            "quaternion_wxyz": physical_quaternion.tolist(),
        }
        if config.get("initial_scene") is not None:
            scene = InitialCarryScene(
                config["initial_scene"], geom_quaternion, geom_position
            )
            scene.apply(CarryMocapFrame.from_mapping(transformed))
            report["initial_scene_valid"] = True
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print(
        f"LuMo {args.address}: received={received} rate={received / elapsed:.1f}Hz "
        f"frames={first_frame_id}..{last_frame_id}"
    )
    print("ID   tracked quality name                         position_m                 quaternion_wxyz")
    for record in records:
        position = np.asarray(record["position_m"])
        quaternion = np.asarray(record["quaternion_wxyz"])
        print(
            f"{record['id']:<4d} {str(record['tracked']):<7s} {record['quality_grade']:<7d} "
            f"{record['name'][:28]:<28s} "
            f"[{position[0]:+.3f},{position[1]:+.3f},{position[2]:+.3f}] "
            f"[{quaternion[0]:+.3f},{quaternion[1]:+.3f},{quaternion[2]:+.3f},{quaternion[3]:+.3f}]"
        )
    if selected:
        print("Selected rigid bodies are present. This does not validate marker extrinsics or world axes.")
    if "transformed" in report:
        transformed = report["transformed"]
        root_position = np.asarray(transformed["robot_root"]["position"])
        object_position = np.asarray(transformed["object"]["position"])
        print(f"UFO robot_root position_m={root_position.tolist()}")
        print(f"UFO object training-reference position_m={object_position.tolist()}")
        if "physical_box" in report:
            physical = report["physical_box"]
            physical_position = np.asarray(physical["position"])
            print(f"UFO object physical position_m={physical['position']}")
            print(f"UFO object physical quaternion_wxyz={physical['quaternion_wxyz']}")
            print(
                "UFO robot-physical-box distance_m="
                f"{np.linalg.norm(physical_position[:2] - root_position[:2]):.4f}"
            )
        else:
            print(f"UFO robot-object distance_m={np.linalg.norm(object_position - root_position):.4f}")
        print(f"UFO target training-reference position_m={transformed['target_position_world']}")
        for side, hand in transformed.get("hand_rigid_origins", {}).items():
            if hand.get("present", False):
                print(
                    f"UFO {side}_hand rigid-origin position_m={hand['position']} "
                    f"quality={hand.get('quality_grade')}"
                )
            else:
                print(
                    f"UFO {side}_hand rigid-origin unavailable; FK fallback "
                    f"reason={hand.get('reason')}"
                )
        if report.get("initial_scene_valid"):
            print("PASS initial physical box distance/bearing/height/yaw alignment")
    print("Before motor enable: physically move a rigid body exactly 1 m and verify position_m changes by 1 m.")


if __name__ == "__main__":
    main()
