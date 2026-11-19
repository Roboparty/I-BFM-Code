"""Check the RP1 release recipe without starting distributed training."""
import json
from pathlib import Path
import re
import shlex
import subprocess
import unittest
import xml.etree.ElementTree as ET

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


class RP1ReleaseInputsTest(unittest.TestCase):
    def test_launcher_selects_rp1_configs_and_preserves_replay(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/train_rp1.sh"),
            "rp1_recipe_test", "--dry-run"], capture_output=True, text=True, check=True)
        command = shlex.split(result.stdout)
        expected = {"--robot-config": "configs/robots/rp1_28dof_official_pd.yaml",
            "--carry-config": "configs/tasks/carry_box_rp1_full_dr.yaml",
            "--data-manifest": "configs/data/rp1_lafan_carry_kick_30_40_30.yaml",
            "--num-envs": "512", "--buffer-size": "2560000",
            "--sphere-log-mode": "actor"}
        for key, value in expected.items():
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertNotIn("--disable-checkpoint-buffer", command)
        self.assertIn("--checkpoint-buffer-final-only", command)
        self.assertNotIn("--init-model-checkpoint", command)
        self.assertFalse((ROOT / "runs/rp1_recipe_test").exists())

    def test_launcher_rejects_path_run_name(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/train_rp1.sh"),
            "../bad", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_pd_groups_cover_all_28_actuated_joints(self):
        config = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/robots/rp1_28dof_official_pd.yaml"))
        xml = ET.parse(ROOT / config["xml_path"]).getroot()
        joints = [motor.attrib["joint"] for motor in xml.find("actuator")]
        self.assertEqual(len(joints), 28)
        control = config["training"]["control"]
        for joint in joints:
            group = next(k for k in control["stiffness"] if k in joint)
            if group in ("waist_roll", "waist_yaw", "hip_pitch", "hip_roll", "knee"):
                expected = 120., 7.5
            elif group in ("hip_yaw", "shoulder_pitch"):
                expected = 35., 2.5
            elif group in ("ankle_pitch", "ankle_roll"):
                expected = 25., 1.5
            else:
                expected = 12.5, .75
            self.assertEqual((control["stiffness"][group], control["damping"][group]), expected)
        compiler = xml.find("compiler")
        mesh_base = (ROOT / config["xml_path"]).parent / compiler.attrib["meshdir"]
        for mesh in xml.findall("asset/mesh"):
            self.assertTrue((mesh_base / mesh.attrib["file"]).is_file())
        task = OmegaConf.load(ROOT / "configs/tasks/carry_box_rp1_full_dr.yaml")
        expression = task.domain_rand_overrides.friction_geom_names
        foot_geoms = [g for g in xml.findall(".//geom") if re.fullmatch(expression, g.attrib.get("name", ""))]
        self.assertEqual(len(foot_geoms), 14)

    def test_manifest_and_checksum_lock_describe_same_six_inputs(self):
        manifest = OmegaConf.load(ROOT / "configs/data/rp1_lafan_carry_kick_30_40_30.yaml")
        lock = json.loads((ROOT / "configs/data/rp1_assets.json").read_text())
        self.assertEqual([s.weight for s in manifest.datasets], [.3, .4, .3])
        self.assertEqual(manifest.robot_config, "configs/robots/rp1_28dof_official_pd.yaml")
        inputs = set()
        for source in manifest.datasets:
            for key in ("train_path", "inference_path"):
                path = (ROOT / "configs/data" / source[key]).resolve()
                self.assertEqual(path.parent, ROOT / lock["cache_directory"])
                inputs.add(path.name)
        self.assertEqual(inputs, set(lock["sha256"]))


if __name__ == "__main__":
    unittest.main()
