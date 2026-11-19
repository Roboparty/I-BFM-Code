#!/usr/bin/env python3
"""Offline workstation/SDK checks that never initialize DDS or publish commands."""

from __future__ import annotations

import argparse
import ipaddress
import os
import platform
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail(f"{path} must contain a YAML mapping")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path, default=ROOT / "config/robot/g1_carry_real.yaml")
    parser.add_argument("--skip-interface", action="store_true")
    args = parser.parse_args()

    print("NO-ACTUATION: no DDS initialization, subscriber, publisher, mode switch, or LowCmd")
    if sys.version_info[:2] != (3, 10):
        fail(f"expected Python 3.10, got {platform.python_version()}")
    if platform.machine() != "x86_64":
        fail(f"workstation backend expects x86_64, got {platform.machine()}")

    config = load_yaml(args.robot_config)
    if config.get("ROBOT_TYPE") != "g1_real":
        fail("ROBOT_TYPE must be g1_real")
    if str(config.get("CONTROL_BACKEND", "")).lower() != "unitree_sdk2py":
        fail("CONTROL_BACKEND must be unitree_sdk2py for workstation deployment")
    for key in ("ORIN_HOST", "MOTOR_CONTROLLER_HOST"):
        ipaddress.ip_address(str(config.get(key, "")))
    if config["ORIN_HOST"] == config["MOTOR_CONTROLLER_HOST"]:
        fail("ORIN_HOST and MOTOR_CONTROLLER_HOST must describe different endpoints")
    if bool(config.get("RELEASE_MOTION_SERVICE", False)) and not bool(
        config.get("ENABLE_LOW_LEVEL_COMMANDS", False)
    ):
        fail("RELEASE_MOTION_SERVICE cannot be enabled while LowCmd is disabled")

    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    command = unitree_hg_msg_dds__LowCmd_()
    if len(command.motor_cmd) != 35:
        fail(f"official G1 LowCmd has {len(command.motor_cmd)} motor slots, expected 35")
    command.crc = CRC().Crc(command)
    print(
        f"PASS Python={platform.python_version()} arch={platform.machine()} "
        f"UnitreeHG slots={len(command.motor_cmd)} CRC={int(command.crc)}"
    )

    if not args.skip_interface:
        from scripts.onboard.interface_config import Reporter, validate_interface

        reporter = Reporter()
        if not validate_interface(str(config.get("INTERFACE", "")), reporter):
            fail("configured G1 DDS interface is not ready")
    print(
        f"PASS backend={config['CONTROL_BACKEND']} interface={config['INTERFACE']} "
        f"orin={config['ORIN_HOST']} motor={config['MOTOR_CONTROLLER_HOST']}"
    )
    print(
        "INFO safety gates: config_lowcmd={} env_lowcmd={} release_motion_service={}".format(
            bool(config.get("ENABLE_LOW_LEVEL_COMMANDS", False)),
            os.environ.get("UFO_ALLOW_LOW_LEVEL_COMMANDS") == "1",
            bool(config.get("RELEASE_MOTION_SERVICE", False)),
        )
    )


if __name__ == "__main__":
    main()
