#!/usr/bin/env bash
set -euo pipefail

UFO_ROOT="${UFO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROBOT_CONFIG="${ROBOT_CONFIG:-${UFO_ROOT}/config/robot/g1_carry_real.yaml}"
TASK_CONFIG="${TASK_CONFIG:-${UFO_ROOT}/config/exp/carry/g1_carry_intact_logmap166559744_solve_close_gpu_lumo.yaml}"
MANIFEST="${MANIFEST:-${UFO_ROOT}/model/g1_carry_intact_logmap166559744_solve_close_gpu/manifest.json}"
G1_INTERFACE="${G1_INTERFACE:-enp131s0}"
G1_ORIN_HOST="${G1_ORIN_HOST:-192.168.123.164}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${UFO_GPU_ID:-0}"

cd "${UFO_ROOT}"
export PYTHONPATH="${UFO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
OVERLAY="$(mktemp /tmp/ufo-g1-intact166-solve-close-preflight.XXXXXX.yaml)"
trap 'rm -f "${OVERLAY}"' EXIT
"${PYTHON_BIN}" scripts/onboard/interface_config.py \
  --robot-config "${ROBOT_CONFIG}" \
  --interface "${G1_INTERFACE}" \
  --orin-host "${G1_ORIN_HOST}" \
  --output "${OVERLAY}"
"${PYTHON_BIN}" tools/preflight_g1_workstation.py --robot-config "${OVERLAY}"
"${PYTHON_BIN}" tools/preflight_g1_carry.py \
  --manifest "${MANIFEST}" \
  --task-config "${TASK_CONFIG}" \
  --require-runtime-mode phased \
  --require-controller-profile intact_logmap150544384_strong \
  --require-deployment-calibration \
  --require-onnx-provider CUDAExecutionProvider
"${PYTHON_BIN}" -m rl_policy.carry_solve_close_166559744_runtime \
  --task-config "${TASK_CONFIG}"

echo "PASS 166.560M solve-close GPU preflight; no DDS or LowCmd was initialized."
