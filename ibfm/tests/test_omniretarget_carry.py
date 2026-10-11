from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoidverse.tools.omniretarget_hand_contact import contact_hysteresis
from humanoidverse.utils.motion_data.omniretarget_carry import (
    load_omniretarget_carry_npz,
    omniretarget_qpos_components,
    omniretarget_qpos_to_mujoco,
)
from humanoidverse.utils.robot_spec import load_robot_spec


class OmniRetargetCarryTest(unittest.TestCase):
    @staticmethod
    def _qpos(frames: int = 3) -> np.ndarray:
        qpos = np.zeros((frames, 43), dtype=np.float64)
        qpos[:, 0] = 1.0
        qpos[:, 6] = 0.8
        qpos[:, 36] = 1.0
        qpos[:, 40:43] = np.array([0.5, 0.0, 0.2])
        return qpos

    def test_quat_first_qpos_is_decoded_and_reordered(self) -> None:
        qpos = self._qpos()
        components = omniretarget_qpos_components(qpos)
        mujoco_qpos = omniretarget_qpos_to_mujoco(qpos)

        np.testing.assert_allclose(components["root_pos"][0], [0.0, 0.0, 0.8])
        np.testing.assert_allclose(components["root_quat"][0], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(components["object_pos"][0], [0.5, 0.0, 0.2])
        np.testing.assert_allclose(mujoco_qpos[0, :7], [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(mujoco_qpos[0, -7:], [0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])

    def test_two_four_centimeter_hysteresis_is_independent_per_hand(self) -> None:
        distance = np.array(
            [
                [0.03, 0.01],
                [0.01, 0.03],
                [0.03, 0.05],
                [0.05, 0.01],
            ]
        )
        contact = contact_hysteresis(distance, enter=0.02, exit_=0.04)

        np.testing.assert_array_equal(
            contact,
            np.array([[False, True], [True, True], [True, False], [False, True]]),
        )

    def test_labeled_npz_import_preserves_hand_contacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npz"
            np.savez_compressed(
                path,
                qpos=self._qpos(),
                fps=np.asarray(30),
                hand_box_contact=np.array([[0, 0], [1, 0], [1, 1]], dtype=np.uint8),
                hand_box_surface_distance=np.array([[0.1, 0.1], [0.01, 0.05], [0.0, 0.0]], dtype=np.float32),
            )
            robot_spec = load_robot_spec(Path("configs/robots/g1_29dof.yaml"))
            data = load_omniretarget_carry_npz(
                path,
                source_name="test",
                robot_spec=robot_spec,
            )

        motion = data["sample"]
        self.assertEqual(motion["pose_aa"].shape, (3, 30, 3))
        np.testing.assert_array_equal(motion["hand_box_contact"], [[0, 0], [1, 0], [1, 1]])
        np.testing.assert_allclose(motion["object_pos"], [[0.5, 0.0, 0.2]] * 3)


if __name__ == "__main__":
    unittest.main()
