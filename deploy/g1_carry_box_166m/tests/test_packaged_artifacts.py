import hashlib
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np
import yaml

from rl_policy.carry_solve_close_166559744_runtime import validate_task_config


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model/g1_carry_intact_logmap166559744_solve_close_gpu"


class PackagedArtifactsTest(unittest.TestCase):
    def test_manifest_hashes_match_packaged_artifacts(self):
        manifest = json.loads((MODEL / "manifest.json").read_text())
        self.assertEqual(manifest["checkpoint_step"], 166559744)
        for relative, expected in manifest["sha256"].items():
            with self.subTest(path=relative):
                digest = hashlib.sha256()
                with (ROOT / relative).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                self.assertEqual(digest.hexdigest(), expected)

    def test_export_matches_selected_checkpoint_and_input_schema(self):
        metadata = json.loads(
            (MODEL / "exported/FBcprAuxModel_spherelog_actor.meta.json").read_text()
        )
        self.assertEqual(metadata["checkpoint_step"], 166559744)
        self.assertEqual(metadata["sphere_log_mode"], "actor")
        self.assertEqual(metadata["input_dim"], 1252)
        self.assertEqual(metadata["output_dim"], 29)
        self.assertEqual(
            metadata["source_model_sha256"],
            "25b0d98c83f059289e97a3634cd04dd211dafb674c341e783fd86cb1ffd41cd9",
        )

    def test_shared_meshes_load_and_fk_joint_order_matches_policy(self):
        model = mujoco.MjModel.from_xml_path(
            str(ROOT / "data/robots/g1_carry_training/g1_29dof.xml")
        )
        config = yaml.safe_load(
            (ROOT / "config/policy/g1_carry_intact_logmap166559744_solve_close_gpu.yaml").read_text()
        )
        joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
            if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.assertEqual(joint_names, config["isaac_joint_names"])
        self.assertEqual(len(joint_names), 29)
        self.assertTrue(np.all(np.isfinite(model.body_pos)))

    def test_selected_latents_and_solve_close_configuration_validate(self):
        validate_task_config(
            ROOT / "config/exp/carry/g1_carry_intact_logmap166559744_solve_close_gpu_lumo.yaml"
        )


if __name__ == "__main__":
    unittest.main()
