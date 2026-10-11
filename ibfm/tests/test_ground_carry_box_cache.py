from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from humanoidverse.tools.ground_carry_box_cache import box_bottom_height, ground_carry_motion


class GroundCarryBoxCacheTest(unittest.TestCase):
    def test_grounding_removes_penetration_and_updates_dependent_fields(self) -> None:
        half_size = np.asarray([0.1675266, 0.1688563, 0.1800429])
        geom_pos = np.asarray([0.00102584, -0.00160677, 0.00637628])
        geom_quat_wxyz = np.asarray([0.07016202, 0.97153219, 0.22227911, 0.04236142])
        geom_quat_xyzw = geom_quat_wxyz[[1, 2, 3, 0]]
        marker_quat = Rotation.from_quat(geom_quat_xyzw).inv().as_quat().astype(np.float32)
        object_pos = np.asarray([[0.0, 0.0, 0.18], [0.0, 0.0, 0.19]], dtype=np.float32)
        motion = {
            "root_trans_offset": np.zeros((2, 3), dtype=np.float32),
            "pose_aa": np.zeros((2, 30, 3), dtype=np.float32),
            "fps": 30,
            "object_pos": object_pos,
            "object_quat": np.repeat(marker_quat[None], 2, axis=0),
            "object_lin_vel": np.zeros((2, 3), dtype=np.float32),
            "object_ang_vel": np.zeros((2, 3), dtype=np.float32),
            "object_target_pos": object_pos[-1].copy(),
            "object_target_quat": marker_quat.copy(),
            "hand_box_contact": np.zeros((2, 2), dtype=np.float32),
            "hand_box_surface_distance": np.ones((2, 2), dtype=np.float32),
        }

        grounded, stats = ground_carry_motion(
            motion,
            half_size=half_size,
            geom_pos=geom_pos,
            geom_quat_wxyz=geom_quat_wxyz,
            carry_evaluation_eligible=False,
        )
        bottom = box_bottom_height(
            grounded["object_pos"],
            grounded["object_quat"],
            half_size=half_size,
            geom_pos=geom_pos,
            geom_quat_wxyz=geom_quat_wxyz,
        )

        self.assertGreaterEqual(float(bottom.min()), -1.0e-6)
        self.assertGreater(stats["max_correction_m"], 0.0)
        self.assertFalse(grounded["carry_evaluation_eligible"])
        self.assertNotIn("hand_box_surface_distance", grounded)
        self.assertAlmostEqual(float(grounded["object_target_pos"][2]), float(grounded["object_pos"][-1, 2]))
        self.assertTrue(np.all(np.isfinite(grounded["object_lin_vel"])))


if __name__ == "__main__":
    unittest.main()
