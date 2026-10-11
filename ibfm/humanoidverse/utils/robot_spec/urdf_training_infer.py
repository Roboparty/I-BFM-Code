"""Lightweight URDF parsing helpers for robot config draft generation.

MuJoCo XML remains the source of truth for training layout. URDF data parsed
here is used only as optional auxiliary metadata for hardware limits, dynamics,
and semantic hints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET


_SEMANTIC_TOKENS = (
    "left",
    "right",
    "hip",
    "knee",
    "ankle",
    "shoulder",
    "elbow",
    "wrist",
    "foot",
    "toe",
    "hand",
    "torso",
    "pelvis",
    "chest",
    "trunk",
)


@dataclass(frozen=True)
class UrdfJointInfo:
    name: str
    type: str
    parent: str | None = None
    child: str | None = None
    axis: list[float] | None = None
    limit_lower: float | None = None
    limit_upper: float | None = None
    limit_effort: float | None = None
    limit_velocity: float | None = None
    dynamics_damping: float | None = None
    dynamics_friction: float | None = None
    mimic_joint: str | None = None
    mimic_multiplier: float | None = None
    mimic_offset: float | None = None
    semantic_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UrdfLinkInfo:
    name: str
    visual_meshes: list[str] = field(default_factory=list)
    collision_meshes: list[str] = field(default_factory=list)
    semantic_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UrdfTransmissionInfo:
    name: str
    joint_name: str | None = None
    actuator_name: str | None = None


@dataclass(frozen=True)
class UrdfRobotInfo:
    source_path: str
    joints: dict[str, UrdfJointInfo]
    links: dict[str, UrdfLinkInfo]
    transmissions: list[UrdfTransmissionInfo]
    warnings: list[str] = field(default_factory=list)


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _tag_name(child.tag) == name]


def _first_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if _tag_name(child.tag) == name:
            return child
    return None


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _axis(value: str | None) -> list[float] | None:
    if not value:
        return None
    parts = value.split()
    if len(parts) != 3:
        return None
    try:
        return [float(item) for item in parts]
    except ValueError:
        return None


def _semantic_hints(*names: str | None) -> list[str]:
    text = " ".join(name or "" for name in names).lower()
    hints = [token for token in _SEMANTIC_TOKENS if token in text]
    if text.startswith("l_") or "_l_" in text:
        hints.append("left")
    if text.startswith("r_") or "_r_" in text:
        hints.append("right")
    return sorted(set(hints))


def _mesh_paths(element: ET.Element, section_name: str) -> list[str]:
    paths: list[str] = []
    for section in _children(element, section_name):
        geometry = _first_child(section, "geometry")
        if geometry is None:
            continue
        mesh = _first_child(geometry, "mesh")
        if mesh is None:
            continue
        filename = mesh.attrib.get("filename")
        if filename:
            paths.append(str(filename))
    return paths


def _parse_joint(element: ET.Element) -> UrdfJointInfo | None:
    name = element.attrib.get("name")
    if not name:
        return None
    parent = _first_child(element, "parent")
    child = _first_child(element, "child")
    axis = _first_child(element, "axis")
    limit = _first_child(element, "limit")
    dynamics = _first_child(element, "dynamics")
    mimic = _first_child(element, "mimic")
    parent_link = parent.attrib.get("link") if parent is not None else None
    child_link = child.attrib.get("link") if child is not None else None
    return UrdfJointInfo(
        name=str(name),
        type=str(element.attrib.get("type", "")),
        parent=parent_link,
        child=child_link,
        axis=_axis(axis.attrib.get("xyz") if axis is not None else None),
        limit_lower=_to_float(limit.attrib.get("lower") if limit is not None else None),
        limit_upper=_to_float(limit.attrib.get("upper") if limit is not None else None),
        limit_effort=_to_float(limit.attrib.get("effort") if limit is not None else None),
        limit_velocity=_to_float(limit.attrib.get("velocity") if limit is not None else None),
        dynamics_damping=_to_float(dynamics.attrib.get("damping") if dynamics is not None else None),
        dynamics_friction=_to_float(dynamics.attrib.get("friction") if dynamics is not None else None),
        mimic_joint=mimic.attrib.get("joint") if mimic is not None else None,
        mimic_multiplier=_to_float(mimic.attrib.get("multiplier") if mimic is not None else None),
        mimic_offset=_to_float(mimic.attrib.get("offset") if mimic is not None else None),
        semantic_hints=_semantic_hints(name, parent_link, child_link),
    )


def _parse_link(element: ET.Element) -> UrdfLinkInfo | None:
    name = element.attrib.get("name")
    if not name:
        return None
    return UrdfLinkInfo(
        name=str(name),
        visual_meshes=_mesh_paths(element, "visual"),
        collision_meshes=_mesh_paths(element, "collision"),
        semantic_hints=_semantic_hints(name),
    )


def _parse_transmission(element: ET.Element) -> UrdfTransmissionInfo:
    joint = _first_child(element, "joint")
    actuator = _first_child(element, "actuator")
    return UrdfTransmissionInfo(
        name=str(element.attrib.get("name", "")),
        joint_name=joint.attrib.get("name") if joint is not None else None,
        actuator_name=actuator.attrib.get("name") if actuator is not None else None,
    )


def parse_urdf_robot(path: str | Path) -> UrdfRobotInfo:
    urdf_path = Path(path).expanduser()
    root = ET.parse(urdf_path).getroot()
    warnings: list[str] = []
    if _tag_name(root.tag) != "robot":
        warnings.append(f"URDF root tag is {_tag_name(root.tag)!r}, expected 'robot'.")
    joints: dict[str, UrdfJointInfo] = {}
    links: dict[str, UrdfLinkInfo] = {}
    transmissions: list[UrdfTransmissionInfo] = []
    for child in list(root):
        tag = _tag_name(child.tag)
        if tag == "joint":
            joint = _parse_joint(child)
            if joint is not None:
                if joint.name in joints:
                    warnings.append(f"Duplicate URDF joint {joint.name!r}; keeping the last definition.")
                joints[joint.name] = joint
        elif tag == "link":
            link = _parse_link(child)
            if link is not None:
                if link.name in links:
                    warnings.append(f"Duplicate URDF link {link.name!r}; keeping the last definition.")
                links[link.name] = link
        elif tag == "transmission":
            transmissions.append(_parse_transmission(child))
    return UrdfRobotInfo(
        source_path=str(urdf_path),
        joints=joints,
        links=links,
        transmissions=transmissions,
        warnings=warnings,
    )
