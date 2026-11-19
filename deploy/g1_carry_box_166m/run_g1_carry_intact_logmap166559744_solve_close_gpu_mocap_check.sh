#!/usr/bin/env bash
set -euo pipefail

UFO_ROOT="${UFO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TASK_CONFIG="${TASK_CONFIG:-${UFO_ROOT}/config/exp/carry/g1_carry_intact_logmap166559744_solve_close_gpu_lumo.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${UFO_ROOT}"
export PYTHONPATH="${UFO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" tools/lumo_carry_diagnostics.py \
  --config "${TASK_CONFIG}" \
  --root-name G1_WAIST \
  --object-name CARRY_BOX \
  --seconds "${MOCAP_CHECK_SECONDS:-10}"
