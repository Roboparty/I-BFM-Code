#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${1:-ibfm_$(date +%Y%m%d_%H%M%S)}"
run_dir="$root/runs/$run_id"
if [[ "$run_id" == */* ]]; then
  echo "Run name must not contain a slash." >&2
  exit 2
fi
if [[ -e "$run_dir" && "${RESUME:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing run: $run_dir" >&2
  exit 3
fi
if [[ "${RESUME:-0}" == "1" && ! -f "$run_dir/checkpoint/train_status.json" ]]; then
  echo "No rolling checkpoint to resume in $run_dir" >&2
  exit 3
fi
if [[ ! -x "$root/.venv/bin/python" ]]; then
  echo "Run uv sync from ibfm/ before training." >&2
  exit 2
fi
mkdir -p "$run_dir"
cd "$root"
export PYTHONPATH="$root"
export UFO_CACHE_DIR="$root/cache"
export BFMZERO_MJLAB_CACHE_DIR="$UFO_CACHE_DIR"
export MPLCONFIGDIR="$UFO_CACHE_DIR/matplotlib"
export UV_OFFLINE=1 UV_NO_SYNC=1 WANDB_MODE=offline NCCL_NVLS_ENABLE=0

"$root/.venv/bin/python" "$root/scripts/retain_policies.py" --run-dir "$run_dir" \
  > "$run_dir/policy_retention.log" 2>&1 &
watcher_pid=$!
trap 'kill "$watcher_pid" 2>/dev/null || true; wait "$watcher_pid" 2>/dev/null || true' EXIT

if "$root/.venv/bin/python" -m humanoidverse.train \
  --agent fb \
  --task carry_box \
  --carry-config configs/tasks/carry_box_mass0p7_range0p5_1p0_pd_delay_rfi_footmu0p4_1p25.yaml \
  --data-manifest configs/data/g1_lafan_carry_kick_30_40_30.yaml \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps "${NUM_ENV_STEPS:-250000000}" \
  --checkpoint-every-steps 3200000 \
  --disable-checkpoint-buffer \
  --checkpoint-buffer-final-only \
  --update-z-every-step 100 \
  --expert-z-mode mean \
  --expert-z-horizon 8 \
  --intact-goal-horizon 8 \
  --intact-native-goal-coef 0.4285714286 \
  --intact-native-local-coef 0.5714285714 \
  --intact-native-residual-coef 0.005 \
  --sphere-log-mode actor \
  --buffer-size 5120000 \
  --num-agent-updates 16 \
  --seed 4728 \
  --use-wandb \
  --wandb-run-name "$run_id" \
  --work-dir "$run_dir" \
  2>&1 | tee "$run_dir/launcher.log"; then
  train_status=0
else
  train_status=$?
fi

kill "$watcher_pid" 2>/dev/null || true
wait "$watcher_pid" 2>/dev/null || true
"$root/.venv/bin/python" "$root/scripts/retain_policies.py" --run-dir "$run_dir" --once \
  >> "$run_dir/policy_retention.log" 2>&1
exit "$train_status"
