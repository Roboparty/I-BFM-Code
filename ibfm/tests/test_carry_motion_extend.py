import unittest

import numpy as np

from humanoidverse.tools.carry_motion_extend import extend_carry_motion, pause_carry_motion


class CarryMotionExtendTests(unittest.TestCase):
    def test_repeats_walk_and_translates_suffix_and_target(self):
        frames = 8
        root = np.zeros((frames, 3), dtype=np.float32)
        root[:, 0] = np.arange(frames)
        obj = root.copy()
        obj[:, 1] = -0.5
        motion = {
            "root_trans_offset": root,
            "pose_aa": np.zeros((frames, 2, 3), dtype=np.float32),
            "dof_pos": np.zeros((frames, 1), dtype=np.float32),
            "root_quat": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)).astype(np.float32),
            "object_pos": obj,
            "object_quat": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)).astype(np.float32),
            "object_lin_vel": np.zeros((frames, 3), dtype=np.float32),
            "object_ang_vel": np.zeros((frames, 3), dtype=np.float32),
            "object_target_pos": obj[-1].copy(),
            "fps": 10.0,
        }

        result = extend_carry_motion(motion, walk_start=2, walk_end=4, repeats=3, transition_frames=1)

        self.assertEqual(result["root_trans_offset"].shape[0], 16)
        np.testing.assert_allclose(result["root_trans_offset"][-1], [11.0, 0.0, 0.0])
        np.testing.assert_allclose(result["object_target_pos"], [7.0, -0.5, 0.0])
        for field in ("pose_aa", "dof_pos", "root_quat", "object_pos", "object_quat"):
            self.assertEqual(len(result[field]), 16)

    def test_can_translate_target_for_a_spatially_consistent_dataset(self):
        frames = 6
        root = np.zeros((frames, 3), dtype=np.float32)
        root[:, 0] = np.arange(frames)
        motion = {
            "root_trans_offset": root,
            "pose_aa": np.zeros((frames, 2, 3), dtype=np.float32),
            "dof_pos": np.zeros((frames, 1), dtype=np.float32),
            "root_quat": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)).astype(np.float32),
            "object_pos": root.copy(),
            "object_quat": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)).astype(np.float32),
            "object_lin_vel": np.zeros((frames, 3), dtype=np.float32),
            "object_ang_vel": np.zeros((frames, 3), dtype=np.float32),
            "object_target_pos": root[-1].copy(),
            "fps": 10.0,
        }
        result = extend_carry_motion(
            motion, walk_start=1, walk_end=3, repeats=2, transition_frames=0, preserve_target=False
        )
        np.testing.assert_allclose(result["object_target_pos"], [7.0, 0.0, 0.0])

    def test_rejects_invalid_range(self):
        motion = {"root_trans_offset": np.zeros((2, 3), dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, "Invalid walk range"):
            extend_carry_motion(motion, walk_start=1, walk_end=2, repeats=2, transition_frames=1)

    def test_pauses_reference_and_zeros_object_velocity(self):
        frames = 6
        root = np.zeros((frames, 3), dtype=np.float32)
        root[:, 0] = np.arange(frames)
        velocity = np.ones((frames, 3), dtype=np.float32)
        motion = {
            "root_trans_offset": root,
            "pose_aa": np.zeros((frames, 2, 3), dtype=np.float32),
            "dof_pos": np.zeros((frames, 1), dtype=np.float32),
            "root_quat": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)).astype(np.float32),
            "object_pos": root.copy(),
            "object_quat": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)).astype(np.float32),
            "object_lin_vel": velocity.copy(),
            "object_ang_vel": velocity.copy(),
            "object_target_pos": root[-1].copy(),
            "object_target_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "fps": 10.0,
        }

        result = pause_carry_motion(motion, pause_frame=2, pause_frames=3)

        self.assertEqual(len(result["root_trans_offset"]), 9)
        np.testing.assert_allclose(result["root_trans_offset"][2:6], np.tile(root[2], (4, 1)))
        np.testing.assert_allclose(result["root_trans_offset"][6:], root[3:])
        np.testing.assert_allclose(result["object_lin_vel"][3:6], 0.0)
        np.testing.assert_allclose(result["object_ang_vel"][3:6], 0.0)
        np.testing.assert_allclose(result["object_target_pos"], motion["object_target_pos"])


if __name__ == "__main__":
    unittest.main()
