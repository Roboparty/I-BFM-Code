#!/usr/bin/env python3
"""Require fresh bilateral rear-hand contact after each push attempt."""

import importlib.util
from pathlib import Path


bundle = Path(__file__).resolve().parent
source = bundle / "push_box_sim2sim.py"
spec = importlib.util.spec_from_file_location("push166_success_source", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen controller: {source}")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)

controller = frozen.module.DirectPushController
original_update = controller.update


def update_with_reacquire(self, metrics, *, goal_height):
    original_update(self, metrics, goal_height=goal_height)
    if self.phase in (
        "push_direct_release",
        "push_direct_backoff",
        "push_direct_reposition",
    ):
        self._v2_drive_seen = False
        self.contact_drive_latched = False
        self.valid_rear_contact_steps = 0
        self.invalid_rear_contact_steps = 0
        self._v2_drive_blend_steps = 0


controller.update = update_with_reacquire


if __name__ == "__main__":
    frozen.module.run(frozen.module.parse_args())
