from __future__ import annotations

import unittest

import torch

from humanoidverse.carry_stand_latent_sweep import _stand_source_mask


class CarryStandLatentSweepTest(unittest.TestCase):
    def test_carry_precontact_source_rejects_contact_and_near_box_rows(self) -> None:
        state = torch.zeros(4, 98)
        state[:, -2] = 1.0
        state[:, 64] = torch.tensor([1.5, 1.5, 0.5, 1.5])
        state[1, 82] = 1.0
        aux = {
            "carry_lift": torch.zeros(4),
            "carry_robot_upright": torch.ones(4),
            "carry_upright": torch.ones(4),
            "carry_drop": torch.zeros(4),
        }
        aux["carry_lift"][3] = 0.5

        mask = _stand_source_mask(
            state,
            aux,
            num_dof=29,
            source="carry-precontact",
            distance_min=1.0,
            distance_max=2.5,
        )

        self.assertEqual(mask.tolist(), [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
