from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from humanoidverse.utils.robot_spec import load_robot_training_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_robot_xml(root: Path) -> Path:
    xml_path = root / "tiny_urdf_assisted.xml"
    xml_path.write_text(
        """
<mujoco model="tiny_urdf_assisted">
  <compiler autolimits="true" angle="radian"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="left_hip_link" pos="0 0.1 -0.1">
        <joint name="left_hip_joint" type="hinge" axis="0 1 0" range="-0.7 0.7" armature="0.03" damping="0.2" frictionloss="0.01"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
        <body name="left_knee_link" pos="0 0 -0.2">
          <joint name="left_knee_joint" type="hinge" axis="0 1 0" range="-1.5 0.1"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
          <body name="left_ankle_roll_link" pos="0 0 -0.2">
            <joint name="left_ankle_joint" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.03" mass="0.1"/>
          </body>
        </body>
      </body>
      <body name="right_hip_link" pos="0 -0.1 -0.1">
        <joint name="right_hip_joint" type="hinge" axis="0 1 0" range="-0.8 0.8"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
        <body name="right_knee_link" pos="0 0 -0.2">
          <joint name="right_knee_joint" type="hinge" axis="0 1 0" range="-1.6 0.2"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
          <body name="right_ankle_roll_link" pos="0 0 -0.2">
            <joint name="right_ankle_joint" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.03" mass="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key name="stand" qpos="0 0 0.8 1 0 0 0 0.1 -0.2 0.05 0.11 -0.21 0.06"/>
  </keyframe>
  <actuator>
    <motor name="right_hip_motor" joint="right_hip_joint" forcerange="-111 111"/>
    <motor name="left_hip_motor" joint="left_hip_joint" forcerange="-101 101"/>
    <motor name="right_knee_motor" joint="right_knee_joint"/>
    <motor name="left_knee_motor" joint="left_knee_joint"/>
    <motor name="right_ankle_motor" joint="right_ankle_joint"/>
    <motor name="left_ankle_motor" joint="left_ankle_joint"/>
  </actuator>
</mujoco>
""".strip()
    )
    return xml_path


def _write_robot_urdf(root: Path) -> Path:
    urdf_path = root / "tiny_urdf_assisted.urdf"
    urdf_path.write_text(
        """
<robot name="tiny_urdf_assisted">
  <link name="pelvis"/>
  <link name="left_hip_link"/>
  <link name="left_knee_link"/>
  <link name="left_ankle_roll_link"/>
  <link name="right_hip_link"/>
  <link name="right_knee_link"/>
  <link name="right_ankle_roll_link"/>
  <joint name="left_hip_joint" type="revolute">
    <parent link="pelvis"/>
    <child link="left_hip_link"/>
    <limit lower="-1.2" upper="1.3" effort="210" velocity="12"/>
    <dynamics damping="0.7" friction="0.17"/>
  </joint>
  <joint name="left_knee_joint" type="revolute">
    <parent link="left_hip_link"/>
    <child link="left_knee_link"/>
    <limit lower="-1.5" upper="0.1" effort="220" velocity="13"/>
    <dynamics damping="0.8" friction="0.18"/>
  </joint>
  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_knee_link"/>
    <child link="left_ankle_roll_link"/>
    <limit lower="-0.4" upper="0.5" effort="230" velocity="14"/>
    <dynamics damping="0.9" friction="0.19"/>
  </joint>
  <joint name="right_hip_joint" type="revolute">
    <parent link="pelvis"/>
    <child link="right_hip_link"/>
    <limit lower="-1.4" upper="1.5" effort="240" velocity="15"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>
  <joint name="rknee_urdf" type="revolute">
    <parent link="right_hip_link"/>
    <child link="right_knee_link"/>
    <limit lower="-1.6" upper="0.2" effort="250" velocity="16"/>
    <dynamics damping="1.1" friction="0.21"/>
  </joint>
  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_knee_link"/>
    <child link="right_ankle_roll_link"/>
    <limit lower="-0.6" upper="0.7" effort="260" velocity="17"/>
    <dynamics damping="1.2" friction="0.22"/>
  </joint>
</robot>
""".strip()
    )
    return urdf_path


def _run_robot_inspect(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", "-m", "humanoidverse.tools.robot_inspect", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class UrdfAssistedRobotInspectTest(unittest.TestCase):
    def test_urdf_enriches_training_draft_without_reordering_xml_layout(self) -> None:
        hydra_dir = REPO_ROOT / "humanoidverse" / "config" / "robot" / "__tmp_urdf_assisted_test__"
        if hydra_dir.exists():
            shutil.rmtree(hydra_dir)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                xml_path = _write_robot_xml(root)
                urdf_path = _write_robot_urdf(root)
                map_path = root / "joint_map.json"
                map_path.write_text(json.dumps({"right_knee_joint": "rknee_urdf"}))
                robot_yaml = root / "tiny_urdf_assisted.yaml"
                hydra_out = hydra_dir / "tiny_urdf_assisted_auto.yaml"

                result = _run_robot_inspect(
                    [
                        "--xml",
                        str(xml_path),
                        "--urdf",
                        str(urdf_path),
                        "--urdf-joint-name-map",
                        str(map_path),
                        "--name",
                        "tiny_urdf_assisted",
                        "--out",
                        str(robot_yaml),
                        "--hydra-out",
                        str(hydra_out),
                        "--force",
                    ]
                )

                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                robot_config = OmegaConf.to_container(OmegaConf.load(robot_yaml), resolve=True)
                hydra_config = OmegaConf.to_container(OmegaConf.load(hydra_out), resolve=True)

                control_joints = robot_config["control_joints"]["names"]
                self.assertEqual(
                    control_joints,
                    [
                        "right_hip_joint",
                        "left_hip_joint",
                        "right_knee_joint",
                        "left_knee_joint",
                        "right_ankle_joint",
                        "left_ankle_joint",
                    ],
                )

                metadata = robot_config["metadata"]
                self.assertEqual(metadata["merge_policy"]["source_of_truth"], "mujoco_xml")
                self.assertIn("rknee_urdf", metadata["matched_urdf_joints"])
                self.assertIn("left", metadata["semantic_hints"]["joints"]["left_hip_joint"])
                self.assertIn("ankle", metadata["semantic_hints"]["links"]["left_ankle_roll_link"])
                self.assertEqual(metadata["merge_policy"]["prefer_urdf_limits"], False)
                self.assertTrue(any("keeping XML limits" in item for item in metadata["warnings"]))

                training = robot_config["training"]
                self.assertEqual(training["control"]["effort_limit"], [240.0, 210.0, 250.0, 220.0, 260.0, 230.0])
                self.assertEqual(training["control"]["velocity_limit"], [15.0, 12.0, 16.0, 13.0, 17.0, 14.0])
                self.assertEqual(training["actuator"]["joints"]["right_knee_joint"]["friction"], 0.21)
                self.assertEqual(training["actuator"]["joints"]["right_knee_joint"]["damping"], 1.1)

                robot = hydra_config["robot"]
                self.assertEqual(robot["dof_names"], control_joints)
                self.assertEqual(robot["dof_pos_lower_limit_list"][1], -0.7)
                self.assertEqual(robot["dof_pos_upper_limit_list"][1], 0.7)
                self.assertEqual(robot["dof_pos_lower_limit_list"][5], -0.4)
                self.assertEqual(robot["dof_pos_upper_limit_list"][5], 0.5)
                self.assertEqual(robot["foot_name"], "ankle_roll_link")
                self.assertEqual(robot["knee_name"], "knee_link")
                self.assertFalse(robot["motion"]["motion_sym_loss"])

                symmetry = robot["metadata"]["symmetric_dofs_idx_draft"]
                self.assertIn(["left_hip_joint", "right_hip_joint"], symmetry["pairs"])
                self.assertIn([1, 0], symmetry["index_pairs"])

                spec = load_robot_training_spec(robot_yaml)
                self.assertEqual(spec.robot.control_joint_names, control_joints)
        finally:
            if hydra_dir.exists():
                shutil.rmtree(hydra_dir)

    def test_xml_only_output_does_not_include_urdf_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_robot_xml(root)
            robot_yaml = root / "tiny_xml_only.yaml"

            result = _run_robot_inspect(
                [
                    "--xml",
                    str(xml_path),
                    "--name",
                    "tiny_xml_only",
                    "--out",
                    str(robot_yaml),
                    "--with-training",
                    "--hydra-robot",
                    "tiny/tiny_xml_only_auto",
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            robot_config = OmegaConf.to_container(OmegaConf.load(robot_yaml), resolve=True)
            self.assertNotIn("urdf_source", robot_config["metadata"])
            self.assertNotIn("merge_policy", robot_config["metadata"])
            self.assertTrue(
                any("symmetric_dofs_idx was not generated" in item for item in robot_config["metadata"]["warnings"])
            )
            self.assertEqual(robot_config["training"]["control"]["velocity_limit"], [20.0] * 6)


if __name__ == "__main__":
    unittest.main()
