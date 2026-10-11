#!/usr/bin/env python3
"""Frozen push-box behavior flow driven by the 166.560M actor and latents.

The local controller snapshot is imported read-only.  This intentionally
preserves the continuous-v2 behavior observed on 2026-09-10, including the
cross-attempt drive hold that can produce lower-body box contact after a box
is externally moved away from the first completed goal.

The inherited overlay behavior is:

* stage behind the box before the first acquire, while the robot is still far
  enough away, instead of waiting for a one-handed acquire to time out;
* after bilateral rear contact has selected the drive latent once, keep using
  that same drive latent through short contact dropouts.  Contact hysteresis is
  still updated for telemetry and the existing stall recovery remains active.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import joblib
import numpy as np
import torch

from humanoidverse.envs.carry_observations import (
    CARRY_HAND_CONTACT_DIM,
    CARRY_OBJECT_OBSERVATION_DIM,
    OBJECT_CONTEXT_DIM,
)


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "controller_snapshot/carry_phased_sim2sim.py"
SPEC = importlib.util.spec_from_file_location("push166_success_controller", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load frozen controller: {SOURCE}")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

START_WITH_REPOSITION = os.environ.get("UFO_PUSH_V2_START_WITH_REPOSITION", "1") == "1"
HOLD_DRIVE_AFTER_CONTACT = os.environ.get("UFO_PUSH_V2_HOLD_DRIVE", "1") == "1"
REACQUIRE_BLEND_MAX = float(os.environ.get("UFO_PUSH_V2_REACQUIRE_BLEND_MAX", "0.25"))
STOP_SPEED = float(os.environ.get("UFO_PUSH_V2_STOP_SPEED", "0.06"))
STALL_PULSE_BLEND_MAX = float(os.environ.get("UFO_PUSH_V2_STALL_PULSE_BLEND_MAX", "0.40"))
STALL_PULSE_DELAY_SECONDS = float(os.environ.get("UFO_PUSH_V2_STALL_PULSE_DELAY_SECONDS", "0.50"))
STALL_PULSE_RAMP_SECONDS = float(os.environ.get("UFO_PUSH_V2_STALL_PULSE_RAMP_SECONDS", "0.50"))
ACQUIRE_DRIVE_BLEND_SECONDS = float(
    os.environ.get("UFO_PUSH_V2_ACQUIRE_DRIVE_BLEND_SECONDS", "3.0")
)
HANDPUSH_HOLD_LATENT_DIR = os.environ.get(
    "UFO_PUSH_V2_HANDPUSH_HOLD_LATENT_DIR", ""
)
HANDPUSH_HOLD_BLEND_BASE = float(
    os.environ.get("UFO_PUSH_V2_HANDPUSH_HOLD_BLEND_BASE", "0.0")
)
HANDPUSH_HOLD_BLEND_MAX = float(
    os.environ.get("UFO_PUSH_V2_HANDPUSH_HOLD_BLEND_MAX", "0.30")
)
HANDPUSH_HOLD_HEIGHT_ENTER = float(
    os.environ.get("UFO_PUSH_V2_HANDPUSH_HOLD_HEIGHT_ENTER", "0.65")
)
HANDPUSH_HOLD_HEIGHT_EXIT = float(
    os.environ.get("UFO_PUSH_V2_HANDPUSH_HOLD_HEIGHT_EXIT", "0.58")
)
HANDPUSH_HOLD_RISE_SECONDS = float(
    os.environ.get("UFO_PUSH_V2_HANDPUSH_HOLD_RISE_SECONDS", "0.50")
)
HANDPUSH_HOLD_FALL_SECONDS = float(
    os.environ.get("UFO_PUSH_V2_HANDPUSH_HOLD_FALL_SECONDS", "1.00")
)
HANDPUSH_HOLD_ON_CONTACT_LOSS = (
    os.environ.get("UFO_PUSH_V2_HANDPUSH_HOLD_ON_CONTACT_LOSS", "0") == "1"
)
FORCE_DRIVE_CONTACT_OBS = (
    os.environ.get("UFO_PUSH_V2_FORCE_DRIVE_CONTACT_OBS", "1") == "1"
)
if not 0.0 <= HANDPUSH_HOLD_BLEND_BASE <= HANDPUSH_HOLD_BLEND_MAX <= 1.0:
    raise ValueError(
        "hand-push hold weights must satisfy 0 <= base <= max <= 1; "
        f"got base={HANDPUSH_HOLD_BLEND_BASE}, max={HANDPUSH_HOLD_BLEND_MAX}"
    )

_HANDPUSH_HOLD_LATENT = None
if HANDPUSH_HOLD_LATENT_DIR:
    hold_path = Path(HANDPUSH_HOLD_LATENT_DIR) / "z_push_direct.pkl"
    _HANDPUSH_HOLD_LATENT = np.asarray(joblib.load(hold_path), dtype=np.float32).reshape(
        1, -1
    )
    if _HANDPUSH_HOLD_LATENT.shape != (1, 256):
        raise ValueError(
            f"expected hand-push hold latent [1, 256], got "
            f"{_HANDPUSH_HOLD_LATENT.shape}: {hold_path}"
        )

_original_reset = module.DirectPushController.reset
_original_latent = module.DirectPushController.latent
_original_update = module.DirectPushController.update


def _v2_reset(self: object) -> None:
    _original_reset(self)
    self._v2_drive_seen = False
    self._v2_stop_steps = 0
    self._v2_stall_steps = 0
    self._v2_stall_blend = 0.0
    self._v2_drive_blend_steps = 0
    self._v2_stall_recovery_deferred = True
    self._v2_handpush_hold_blend = 0.0
    self._v2_handpush_hold_latched = False
    if START_WITH_REPOSITION:
        # At reset the configured robot-box distance is 1 m.  Perform the
        # behind-box alignment here, before either knees or torso can touch the
        # box.  This replaces the bad acquire -> release -> reposition cycle.
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
        transition_steps = max(
            1, round(max(ACQUIRE_DRIVE_BLEND_SECONDS, 0.0) / self.dt)
        )
        transition_progress = (
            1.0
            if ACQUIRE_DRIVE_BLEND_SECONDS <= 0.0
            else min(float(self._v2_drive_blend_steps) / transition_steps, 1.0)
        )
        if transition_progress < 1.0:
            latent = policy.project_z(
                (1.0 - transition_progress) * z["push_contact_acquire"]
                + transition_progress * latent
            )
            mode += f"_acquire_to_drive_{transition_progress:.2f}"
        if _HANDPUSH_HOLD_LATENT is not None:
            bilateral = bool(metrics["bilateral_contact"])
            root_height = float(metrics["root_height"])
            if (
                (HANDPUSH_HOLD_ON_CONTACT_LOSS and not bilateral)
                or root_height >= HANDPUSH_HOLD_HEIGHT_ENTER
            ):
                self._v2_handpush_hold_latched = True
            elif root_height <= HANDPUSH_HOLD_HEIGHT_EXIT:
                self._v2_handpush_hold_latched = False
            target_hold = (
                HANDPUSH_HOLD_BLEND_MAX
                if self._v2_handpush_hold_latched
                else HANDPUSH_HOLD_BLEND_BASE
            )
            duration = (
                HANDPUSH_HOLD_RISE_SECONDS
                if target_hold > self._v2_handpush_hold_blend
                else HANDPUSH_HOLD_FALL_SECONDS
            )
            if self._v2_handpush_hold_blend < HANDPUSH_HOLD_BLEND_BASE:
                blend_span = HANDPUSH_HOLD_BLEND_BASE
            else:
                blend_span = HANDPUSH_HOLD_BLEND_MAX - HANDPUSH_HOLD_BLEND_BASE
            hold_step = blend_span * min(self.dt / max(duration, self.dt), 1.0)
            if self._v2_handpush_hold_blend < target_hold:
                self._v2_handpush_hold_blend = min(
                    target_hold, self._v2_handpush_hold_blend + hold_step
                )
            else:
                self._v2_handpush_hold_blend = max(
                    target_hold, self._v2_handpush_hold_blend - hold_step
                )
            if self._v2_handpush_hold_blend > 0.0:
                hold_latent = torch.as_tensor(
                    _HANDPUSH_HOLD_LATENT,
                    dtype=latent.dtype,
                    device=latent.device,
                )
                latent = policy.project_z(
                    (1.0 - self._v2_handpush_hold_blend) * latent
                    + self._v2_handpush_hold_blend * hold_latent
                )
                mode += f"_handpush_hold_{self._v2_handpush_hold_blend:.2f}"
        if "direct_turn" in z:
            bearing = module._bearing_deg(
                kwargs["base_rotation"], kwargs["goal_direction"]
            )
            turn_weight = (
                min(
                    abs(bearing) / self.args.direct_push_heading_blend_full_deg,
                    1.0,
                )
                * self.args.direct_push_max_turn_blend
            )
            self.last_transport_turn_weight = turn_weight
            if turn_weight >= 1e-3:
                direction = 1 if bearing >= 0.0 else -1
                latent = policy.project_z(
                    (1.0 - turn_weight) * latent
                    + turn_weight * z["direct_turn"][direction]
                )
                mode += "_left" if direction > 0 else "_right"
        else:
            self.last_transport_turn_weight = 0.0
        if FORCE_DRIVE_CONTACT_OBS:
            actor_observation = dict(observation)
            state = observation["state"].clone()
            object_context_start = int(state.shape[-1]) - OBJECT_CONTEXT_DIM
            contact_start = object_context_start + CARRY_OBJECT_OBSERVATION_DIM
            contact_stop = contact_start + CARRY_HAND_CONTACT_DIM
            if object_context_start < 0 or contact_stop > int(state.shape[-1]):
                raise RuntimeError(
                    "cannot locate hand-contact channels in actor state: "
                    f"state_dim={int(state.shape[-1])}, "
                    f"object_context_dim={OBJECT_CONTEXT_DIM}"
                )
            state[..., contact_start:contact_stop] = 1.0
            actor_observation["state"] = state
            observation = actor_observation
            mode += "_contactobs11"
    return observation, latent, mode


def _v2_update(
    self: object,
    metrics: dict[str, float | bool],
    *,
    goal_height: float,
) -> None:
    transition_steps = max(
        1, round(max(ACQUIRE_DRIVE_BLEND_SECONDS, 0.0) / self.dt)
    )
    # Acquiring the box is intentionally stationary. Do not classify that
    # period, or the one-way acquire-to-drive transition, as a failed push.
    self._v2_stall_recovery_deferred = (
        self.phase == "push_direct"
        and (
            not self._v2_drive_seen
            or self._v2_drive_blend_steps < transition_steps
        )
    )
    _original_update(self, metrics, goal_height=goal_height)
    if self.contact_drive_latched:
        self._v2_drive_seen = True
    if self.phase == "push_direct":
        if self._v2_drive_seen:
            self._v2_drive_blend_steps = min(
                self._v2_drive_blend_steps + 1, transition_steps
            )
        slow_and_not_arrived = (
            self._v2_drive_seen
            and not self._v2_stall_recovery_deferred
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
        self._v2_drive_blend_steps = 0
        self._v2_handpush_hold_blend = 0.0
        self._v2_handpush_hold_latched = False


module.DirectPushController.reset = _v2_reset
module.DirectPushController.latent = _v2_latent
module.DirectPushController.update = _v2_update


if __name__ == "__main__":
    module.run(module.parse_args())
