#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${root}/checkpoints/ibfm-166m/model/model.safetensors"
carry="${root}/ibfm/cache/motion_data/g1_carry_box_omniretarget_no_foot/g1_carry_box_omniretarget_no_foot_full_grounded_mujoco_hysteresis_4p5_6cm_ufo.pkl"
model_sha=25b0d98c83f059289e97a3634cd04dd211dafb674c341e783fd86cb1ffd41cd9
carry_sha=6b0b76e3cb722d6e3f620ae073e41e3e83bd55457d3c4924d0913cd32c2c5a4e

install_checked() {
  local source="$1" target="$2" expected="$3"
  [[ -f "$source" ]] || { echo "Missing asset: $source" >&2; exit 2; }
  printf '%s  %s\n' "$expected" "$source" | sha256sum -c --quiet -
  mkdir -p "$(dirname "$target")"
  if [[ -e "$target" || -L "$target" ]]; then
    printf '%s  %s\n' "$expected" "$target" | sha256sum -c --quiet -
    echo "Already installed: $target"
    return
  fi
  ln -s "$(realpath "$source")" "$target"
  echo "Linked: $target"
}

if [[ -n "${MODEL_FILE:-}" ]]; then
  install_checked "$MODEL_FILE" "$model" "$model_sha"
fi
if [[ -n "${CARRY_CACHE_FILE:-}" ]]; then
  install_checked "$CARRY_CACHE_FILE" "$carry" "$carry_sha"
fi
if [[ -z "${MODEL_FILE:-}" && -z "${CARRY_CACHE_FILE:-}" ]]; then
  echo 'Set MODEL_FILE and/or CARRY_CACHE_FILE to install a verified local asset.' >&2
  exit 2
fi

