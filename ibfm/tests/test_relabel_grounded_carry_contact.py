from __future__ import annotations

import unittest

import numpy as np

from humanoidverse.tools.relabel_grounded_carry_contact import contact_hysteresis


class RelabelGroundedCarryContactTest(unittest.TestCase):
    def test_contact_hysteresis_uses_enter_and_exit_thresholds(self) -> None:
        distance = np.asarray(
            [
                [0.07, 0.03],
                [0.04, 0.05],
                [0.05, 0.07],
                [0.061, 0.04],
            ],
            dtype=np.float32,
        )

        contact = contact_hysteresis(distance, enter=0.045, exit_=0.06)

        np.testing.assert_array_equal(
            contact,
            [
                [False, True],
                [True, True],
                [True, False],
                [False, True],
            ],
        )


if __name__ == "__main__":
    unittest.main()
