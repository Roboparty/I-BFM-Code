from __future__ import annotations

import unittest

import numpy as np

from humanoidverse.tools.lafan_approach_box import ApproachBuildConfig, build_lafan_approach_dataset


def _motion(*, towards_box: bool = True, motion_name: str = "walk2_subject1") -> dict[str, object]:
    num_frames = 300
    end_x = 2.0 if towards_box else -2.0
    root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    root_pos[:, 0] = np.linspace(0.0, end_x, num_frames)
    root_pos[:, 2] = 0.8
    root_quat = np.zeros((num_frames, 4), dtype=np.float32)
    root_quat[:, 3] = 1.0
    return {
        "root_trans_offset": root_pos,
        "pose_aa": np.zeros((num_frames, 30, 3), dtype=np.float32),
        "root_rot": root_quat,
        "fps": 30,
        "motion_name": motion_name,
    }


def _box_config() -> dict[str, object]:
    return {
        "half_size": [0.1675266, 0.1688563, 0.1800429],
        "geom_pos": [0.00102584, -0.00160677, 0.00637628],
        "geom_quat_wxyz": [0.07016202, 0.97153219, 0.22227911, 0.04236142],
        "ground_center_height": 0.185,
    }


class LafanApproachBoxTest(unittest.TestCase):
    def test_builds_static_object_track_with_carry_semantics(self) -> None:
        result, summary = build_lafan_approach_dataset(
            {"walk2_subject1_clip0": _motion()},
            _box_config(),
            ApproachBuildConfig(),
        )

        self.assertEqual(len(result), 1)
        record = next(iter(result.values()))
        expected_position = np.repeat(record["object_pos"][0:1], len(record["object_pos"]), axis=0)
        np.testing.assert_allclose(record["object_pos"], expected_position)
        np.testing.assert_allclose(record["object_lin_vel"], 0.0)
        np.testing.assert_allclose(record["object_ang_vel"], 0.0)
        np.testing.assert_allclose(record["hand_box_contact"], 0.0)
        self.assertFalse(record["carry_evaluation_eligible"])
        self.assertGreaterEqual(summary["initial_distance_range"][0], 2.0)
        self.assertLessEqual(summary["final_distance_range"][1], 1.5)

    def test_rejects_motion_that_moves_away_from_final_forward_box(self) -> None:
        with self.assertRaisesRegex(ValueError, "produced no motions"):
            build_lafan_approach_dataset(
                {"walk2_subject1_clip0": _motion(towards_box=False)},
                _box_config(),
                ApproachBuildConfig(),
            )

    def test_rejects_fall_named_clip_without_low_recovery_pose(self) -> None:
        with self.assertRaisesRegex(ValueError, "produced no motions"):
            build_lafan_approach_dataset(
                {"fallAndGetUp1_subject1_clip0": _motion(motion_name="fallAndGetUp1_subject1")},
                _box_config(),
                ApproachBuildConfig(),
            )


if __name__ == "__main__":
    unittest.main()
