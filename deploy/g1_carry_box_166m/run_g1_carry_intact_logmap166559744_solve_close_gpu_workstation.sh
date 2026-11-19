#!/usr/bin/env bash
set -euo pipefail

if [[ "${UFO_REAL_ROBOT_OK:-}" != "1" ]]; then
  echo "Refusing real G1 launch without UFO_REAL_ROBOT_OK=1." >&2
  exit 2
fi
if [[ "${UFO_ALLOW_LOW_LEVEL_COMMANDS:-}" != "1" ]]; then
  echo "Refusing LowCmd publisher without UFO_ALLOW_LOW_LEVEL_COMMANDS=1." >&2
  exit 2
fi
if [[ "${UFO_INTACT_LOGMAP166559744_SOLVE_CLOSE_GPU_EXPERIMENTAL_OK:-}" != "1" ]]; then
  echo "Refusing solve-close launch without explicit experimental acknowledgement." >&2
  exit 2
fi

UFO_ROOT="${UFO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROBOT_CONFIG="${ROBOT_CONFIG:-${UFO_ROOT}/config/robot/g1_carry_real.yaml}"
POLICY_CONFIG="${POLICY_CONFIG:-${UFO_ROOT}/config/policy/g1_carry_intact_logmap166559744_solve_close_gpu.yaml}"
TASK_CONFIG="${TASK_CONFIG:-${UFO_ROOT}/config/exp/carry/g1_carry_intact_logmap166559744_solve_close_gpu_lumo.yaml}"
MODEL_PATH="${MODEL_PATH:-${UFO_ROOT}/model/g1_carry_intact_logmap166559744_solve_close_gpu/exported/FBcprAuxModel_spherelog_actor.onnx}"
MANIFEST="${MANIFEST:-${UFO_ROOT}/model/g1_carry_intact_logmap166559744_solve_close_gpu/manifest.json}"
G1_INTERFACE="${G1_INTERFACE:-enp131s0}"
G1_ORIN_HOST="${G1_ORIN_HOST:-192.168.123.164}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CARRY_LOG_DIR="${UFO_CARRY_LOG_DIR:-${UFO_ROOT}/logs/g1_carry_intact_logmap166559744_solve_close_gpu}"
export CUDA_VISIBLE_DEVICES="${UFO_GPU_ID:-0}"
export UFO_INTACT_LOGMAP150544384_STRONG_GPU_EXPERIMENTAL_OK=1

cd "${UFO_ROOT}"
export PYTHONPATH="${UFO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
OVERLAY="$(mktemp /tmp/ufo-g1-intact166-solve-close.XXXXXX.yaml)"
trap 'rm -f "${OVERLAY}"' EXIT
interface_flags=(--enable-low-level-commands)
if [[ "${UFO_RELEASE_MOTION_SERVICE:-0}" == "1" ]]; then
  interface_flags+=(--release-motion-service)
fi
"${PYTHON_BIN}" scripts/onboard/interface_config.py \
  --robot-config "${ROBOT_CONFIG}" \
  --interface "${G1_INTERFACE}" \
  --orin-host "${G1_ORIN_HOST}" \
  --output "${OVERLAY}" \
  "${interface_flags[@]}"
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
exec "${PYTHON_BIN}" -m rl_policy.ufo_policy_solve_close_166559744 \
  --robot_config "${OVERLAY}" \
  --policy_config "${POLICY_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --task "${TASK_CONFIG}" \
  --record-dir "${CARRY_LOG_DIR}"
