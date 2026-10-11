#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle="${here}"
runner="${RUNNER:-${here}/run_record_cpu_compatible.sh}"

# The evaluated 0.6 s / 0.8 s pulse.  This launches the bundled runner
# with the isolated controller and writes output only in this folder.
exec env \
  CONTROLLER_SCRIPT="${CONTROLLER_SCRIPT:-${here}/kick_box_sim2sim.py}" \
  DRIVE_DIR="${bundle}/latents/candidates/kick25_min75" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-${here}/interactive_rollouts}" \
  START_WITH_REPOSITION="${START_WITH_REPOSITION:-1}" \
  VIEWER="${VIEWER:-1}" \
  MUJOCO_GL="${MUJOCO_GL:-glfw}" \
  MAX_STEPS="${MAX_STEPS:-0}" \
  UFO_PUSH_CUDA_VISIBLE_DEVICES="${UFO_PUSH_CUDA_VISIBLE_DEVICES-0}" \
  UFO_PUSH_DEVICE="${UFO_PUSH_DEVICE:-cuda:0}" \
  UFO_KICK_ARM_STAND_WEIGHT=0 \
  UFO_KICK_ARM_STAND_NEAR_WEIGHT=0 \
  UFO_KICK_WAIST_STAND_WEIGHT=0 \
  UFO_KICK_STOP_STAND_WEIGHT=0 \
  UFO_KICK_RELEASE_ARM_STAND_WEIGHT=0 \
  UFO_KICK_NOHAND_REKICK_GUARD=0 \
  UFO_KICK_REKICK_LATENT_PATH= \
  UFO_KICK_HEIGHT_GUARD_ARM_WEIGHT=1.00 \
  UFO_KICK_HEIGHT_GUARD_WAIST_WEIGHT=0.65 \
  UFO_KICK_HEIGHT_GUARD_LEG_WEIGHT=0.35 \
  UFO_KICK_HEIGHT_GUARD_ENTER_M=0.75 \
  UFO_KICK_HEIGHT_GUARD_EXIT_M=0.80 \
  UFO_KICK_HEIGHT_GUARD_DWELL_STEPS=1 \
  UFO_KICK_HEIGHT_GUARD_RAMP_S=0.12 \
  UFO_KICK_HEIGHT_GUARD_MAX_ACTIVE_S=0.6 \
  UFO_KICK_HEIGHT_GUARD_COOLDOWN_S=0.8 \
  "${runner}" "$@"
