"""Velocity and position metrics required by the Carry Box pivot controller."""

from __future__ import annotations

from typing import Any

import numpy as np


def _install_metric_probe(module: Any) -> None:
    original = module._controller_metrics

    def augmented(model, data, **kwargs):
        result = original(model, data, **kwargs)
        base_body_id = int(kwargs["base_body_id"])
        box_body_id = int(kwargs["box_body_id"])
        box_qpos_address = int(kwargs["box_qpos_address"])
        base_lin, base_ang = module._world_body_velocity(model, data, base_body_id)
        box_lin, box_ang = module._world_body_velocity(model, data, box_body_id)
        result.update(
            {
                "probe_base_yaw_rate": float(base_ang[2]),
                "probe_box_yaw_rate": float(box_ang[2]),
                "probe_base_planar_speed": float(np.linalg.norm(base_lin[:2])),
                "probe_box_planar_speed": float(np.linalg.norm(box_lin[:2])),
                "probe_base_x": float(data.xpos[base_body_id, 0]),
                "probe_base_y": float(data.xpos[base_body_id, 1]),
                "probe_box_x": float(data.qpos[box_qpos_address]),
                "probe_box_y": float(data.qpos[box_qpos_address + 1]),
            }
        )
        return result

    module._controller_metrics = augmented
