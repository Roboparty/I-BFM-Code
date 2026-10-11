#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the bundled controller and latents with CPU policy inference.
exec env \
  UFO_PUSH_DEVICE=cpu \
  UFO_PUSH_CUDA_VISIBLE_DEVICES= \
  RUNNER="${here}/run_record_cpu_compatible.sh" \
  SAVE_MP4="${SAVE_MP4:-0}" \
  VIEWER="${VIEWER:-1}" \
  MUJOCO_GL="${MUJOCO_GL:-glfw}" \
  MAX_STEPS="${MAX_STEPS:-0}" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-${here}/local_cpu_rollouts}" \
  "${here}/run_interactive_height_guard_e.sh" "$@"
