#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${ROOT}/checkpoints/ibfm-166m/model/model.safetensors"
PART0="${MODEL}.part-000"
PART1="${MODEL}.part-001"
EXPECTED_MODEL="25b0d98c83f059289e97a3634cd04dd211dafb674c341e783fd86cb1ffd41cd9"
EXPECTED_PART0="567aba5f5a26377ed7bd16932346fe313ec524b7680a71dd1dc34a297761da6a"
EXPECTED_PART1="eb31133c466e352d1d57cec908376b957772dea92ef42efc12cb4667c6138c3c"

verify_file() {
  local expected="$1"
  local path="$2"
  printf '%s  %s\n' "${expected}" "${path}" | sha256sum -c -
}

if [[ -f "${MODEL}" ]]; then
  verify_file "${EXPECTED_MODEL}" "${MODEL}"
  echo "Checkpoint is already restored."
  exit 0
fi

for part in "${PART0}" "${PART1}"; do
  if [[ ! -f "${part}" ]]; then
    echo "ERROR: missing checkpoint part: ${part}" >&2
    echo "Run 'git lfs pull' and try again." >&2
    exit 1
  fi
done

verify_file "${EXPECTED_PART0}" "${PART0}"
verify_file "${EXPECTED_PART1}" "${PART1}"

tmp_model="${MODEL}.tmp.$$"
cleanup() {
  rm -f "${tmp_model}"
}
trap cleanup EXIT

cat "${PART0}" "${PART1}" > "${tmp_model}"
verify_file "${EXPECTED_MODEL}" "${tmp_model}"
mv "${tmp_model}" "${MODEL}"
trap - EXIT

echo "Restored ${MODEL}"

