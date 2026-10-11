#!/usr/bin/env python3
"""Kick Box controller with reposition navigation.

Reposition uses the frozen ``better_fix_best`` controller flow (backoff,
collision-aware bypass, staging, and independent align pivot).  Reposition
always exits directly to the Kick drive; acquire is never re-entered.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import joblib
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "controller_snapshot/carry_phased_sim2sim.py"
SPEC = importlib.util.spec_from_file_location("push166_kick_box_source", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load frozen controller: {SOURCE}")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

START_WITH_REPOSITION = os.environ.get("UFO_KICK_START_WITH_REPOSITION", "0") == "1"
HOLD_DRIVE_AFTER_CONTACT = os.environ.get("UFO_PUSH_V2_HOLD_DRIVE", "1") == "1"
REACQUIRE_BLEND_MAX = float(os.environ.get("UFO_PUSH_V2_REACQUIRE_BLEND_MAX", "0.25"))
STOP_SPEED = float(os.environ.get("UFO_PUSH_V2_STOP_SPEED", "0.06"))
STALL_PULSE_BLEND_MAX = float(os.environ.get("UFO_PUSH_V2_STALL_PULSE_BLEND_MAX", "0.40"))
STALL_PULSE_DELAY_SECONDS = float(os.environ.get("UFO_PUSH_V2_STALL_PULSE_DELAY_SECONDS", "0.50"))
STALL_PULSE_RAMP_SECONDS = float(os.environ.get("UFO_PUSH_V2_STALL_PULSE_RAMP_SECONDS", "0.50"))
NOHAND_REKICK_GUARD = os.environ.get("UFO_KICK_NOHAND_REKICK_GUARD", "0") == "1"
NOHAND_GUARD_ROOT_HEIGHT = float(os.environ.get("UFO_KICK_NOHAND_GUARD_ROOT_HEIGHT_M", "0.76"))
NOHAND_GUARD_BOX_DISTANCE = float(os.environ.get("UFO_KICK_NOHAND_GUARD_BOX_DISTANCE_M", "0.85"))
NOHAND_GUARD_HAND_GAP = float(os.environ.get("UFO_KICK_NOHAND_GUARD_HAND_GAP_M", "0.15"))
NOHAND_GUARD_REQUIRE_FOOT = os.environ.get("UFO_KICK_NOHAND_GUARD_REQUIRE_FOOT", "0") == "1"
REKICK_LATENT_PATH = os.environ.get("UFO_KICK_REKICK_LATENT_PATH", "")
REKICK_LATENT_WEIGHT = float(os.environ.get("UFO_KICK_REKICK_LATENT_WEIGHT", "0.5"))
if not 0.0 <= REKICK_LATENT_WEIGHT <= 1.0:
    raise ValueError("UFO_KICK_REKICK_LATENT_WEIGHT must be in [0, 1]")
_rekick_latent = None
if REKICK_LATENT_PATH:
    _rekick_latent = np.asarray(joblib.load(REKICK_LATENT_PATH), dtype=np.float32).reshape(1, -1)
    if _rekick_latent.shape != (1, 256):
        raise ValueError(f"expected 1x256 rekick latent, got {_rekick_latent.shape}")
NOHAND_GUARD_STEPS = int(os.environ.get("UFO_KICK_NOHAND_GUARD_STEPS", "3"))
if NOHAND_GUARD_STEPS <= 0:
    raise ValueError("UFO_KICK_NOHAND_GUARD_STEPS must be positive")


_original_reset = module.DirectPushController.reset
_original_latent = module.DirectPushController.latent
_original_update = module.DirectPushController.update


def _v2_reset(self: object) -> None:
    _original_reset(self)
    # A locked drive flag is intentional for Kick: once navigation reaches the
    # staging/alignment gate, the next phase must be drive rather than acquire.
    self._v2_drive_seen = True
    self._v2_stop_steps = 0
    self._v2_stall_steps = 0
    self._v2_stall_blend = 0.0
    self._v2_foot_seen = False
    self._v2_guard_steps = 0
    self._v2_guard_count = 0
    self._v2_rekick_latent_weight = 0.0
    if START_WITH_REPOSITION:
        self.phase = "push_direct_reposition"
        self.pre_fall_phase = "push_direct_reposition"
        self.phase_steps = 0
        self.ready_steps = 0
        self.reposition_side = 0.0
        self.reposition_turn_direction = 0
        self.reposition_centering_steps = 0
        self.reposition_alignment_latched = False


def _v2_latent(self: object, *args: object, **kwargs: object):
    observation, latent, mode = _original_latent(self, *args, **kwargs)
    self._v2_rekick_latent_weight = 0.0
    if (
        HOLD_DRIVE_AFTER_CONTACT
        and self.phase == "push_direct"
        and self._v2_drive_seen
    ):
        policy = args[0]
        metrics = args[2]
        z = kwargs["z"]
        # Keep the successful drive as the dominant behavior.  As either hand
        # starts to separate, add only a bounded amount of the acquire latent;
        # the old 100% acquire switch is what produced the 0.37 <-> 0.64 m
        # repeated squat/stand motion in continuous-v1.
        if bool(metrics["bilateral_contact"]):
            blend = 0.0
        elif bool(metrics["any_contact"]):
            blend = 0.5 * REACQUIRE_BLEND_MAX
        else:
            blend = REACQUIRE_BLEND_MAX
        blend = max(blend, self._v2_stall_blend)
        latent = policy.project_z(
            (1.0 - blend) * z["push_direct"]
            + blend * z["push_contact_acquire"]
        )
        mode = f"push_direct_drive_contacthold_{blend:.2f}"
    if _rekick_latent is not None and self.phase == "push_direct" and self._v2_guard_count > 0:
        kick = torch.as_tensor(_rekick_latent, device=latent.device, dtype=latent.dtype)
        latent = args[0].project_z((1.0 - REKICK_LATENT_WEIGHT) * latent
                                   + REKICK_LATENT_WEIGHT * kick)
        self._v2_rekick_latent_weight = REKICK_LATENT_WEIGHT
    return observation, latent, mode


def _v2_update(
    self: object,
    metrics: dict[str, float | bool],
    *,
    goal_height: float,
) -> None:
    _original_update(self, metrics, goal_height=goal_height)
    if NOHAND_REKICK_GUARD and self.phase == "push_direct":
        self._v2_foot_seen |= float(metrics.get("foot_box_contact_count", 0.0)) > 0.0
        risk = (
            (self._v2_foot_seen or not NOHAND_GUARD_REQUIRE_FOOT)
            and float(metrics["root_height"]) < NOHAND_GUARD_ROOT_HEIGHT
            and float(metrics["horizontal_distance"]) < NOHAND_GUARD_BOX_DISTANCE
            and float(metrics.get("min_hand_gap_m", float("inf"))) < NOHAND_GUARD_HAND_GAP
            and float(metrics["goal_distance"]) > self.args.direct_push_stop_distance
        )
        self._v2_guard_steps = self._v2_guard_steps + 1 if risk else 0
        if self._v2_guard_steps >= NOHAND_GUARD_STEPS:
            self._change_phase("push_direct_backoff")
            self._v2_guard_count += 1
            self._v2_foot_seen = False
            self._v2_guard_steps = 0
            self.reposition_side = 0.0
            self.reposition_turn_direction = 0
            self.reposition_centering_steps = 0
            self.reposition_alignment_latched = False
            return
    elif self.phase != "push_direct":
        self._v2_foot_seen = False
        self._v2_guard_steps = 0
    if self.contact_drive_latched:
        self._v2_drive_seen = True
    if self.phase == "push_direct":
        slow_and_not_arrived = (
            self._v2_drive_seen
            and float(metrics["goal_distance"]) > self.args.direct_push_stop_distance
            and float(metrics["transport_goalward_speed"])
            < self.args.direct_push_stall_speed
        )
        self._v2_stall_steps = self._v2_stall_steps + 1 if slow_and_not_arrived else 0
        pulse_delay_steps = max(1, round(STALL_PULSE_DELAY_SECONDS / self.dt))
        target_pulse = (
            STALL_PULSE_BLEND_MAX
            if self._v2_stall_steps >= pulse_delay_steps
            else 0.0
        )
        pulse_step = STALL_PULSE_BLEND_MAX * min(
            self.dt / max(STALL_PULSE_RAMP_SECONDS, self.dt), 1.0
        )
        if self._v2_stall_blend < target_pulse:
            self._v2_stall_blend = min(target_pulse, self._v2_stall_blend + pulse_step)
        else:
            self._v2_stall_blend = max(target_pulse, self._v2_stall_blend - pulse_step)
        safely_inside_goal = (
            float(metrics["goal_distance"]) <= self.args.direct_push_stop_distance
            and float(metrics["lift"]) <= self.args.push_max_lift
            and float(metrics["box_upright"]) >= self.args.push_upright_min
            and float(metrics["object_speed"]) <= STOP_SPEED
        )
        self._v2_stop_steps = self._v2_stop_steps + 1 if safely_inside_goal else 0
        if self._v2_stop_steps >= int(self.args.direct_push_stop_enter_steps):
            # Requiring bilateral contact here is backwards: once the box has
            # arrived and stopped, releasing it is a valid and safer terminal
            # condition than crouching to touch it again.
            self._change_phase("push_direct_stop")
            self._v2_stop_steps = 0
    else:
        self._v2_stop_steps = 0
        self._v2_stall_steps = 0
        self._v2_stall_blend = 0.0


module.DirectPushController.reset = _v2_reset
module.DirectPushController.latent = _v2_latent
module.DirectPushController.update = _v2_update


if __name__ == "__main__":
    module.run(module.parse_args())
