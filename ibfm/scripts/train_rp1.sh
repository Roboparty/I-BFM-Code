#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${1:-rp1_official_pd_$(date +%Y%m%d_%H%M%S)}"
if [[ "$run_id" == "--dry-run" ]]; then
  run_id=rp1_official_pd_preview
else
  if [[ $# -gt 0 ]]; then shift; fi
fi
dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=1; shift; fi
if [[ $# -gt 0 || ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo 'Usage: bash scripts/train_rp1.sh [RUN_NAME] [--dry-run]' >&2
  exit 2
fi
run_dir="$root/runs/$run_id"
cd "$root"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export UFO_CACHE_DIR="$root/cache" BFMZERO_MJLAB_CACHE_DIR="$root/cache"
export MPLCONFIGDIR="$root/cache/matplotlib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=1 WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
command=("$root/.venv/bin/python" -m humanoidverse.train
  --agent fb --task carry_box
  --robot-config configs/robots/rp1_28dof_official_pd.yaml
  --carry-config configs/tasks/carry_box_rp1_full_dr.yaml
  --data-manifest configs/data/rp1_lafan_carry_kick_30_40_30.yaml
  --gpu-ids all --num-envs "${NUM_ENVS:-512}"
  --num-env-steps "${NUM_ENV_STEPS:-250000000}"
  --checkpoint-every-steps 3200000 --checkpoint-buffer-final-only
  --update-z-every-step 100 --expert-z-mode mean --expert-z-horizon 8
  --intact-goal-horizon 8 --intact-native-goal-coef 0.4285714286
  --intact-native-local-coef 0.5714285714 --intact-native-residual-coef 0.005
  --intact-grad-diagnostic-samples 0 --sphere-log-mode actor
  --buffer-size "${BUFFER_SIZE:-2560000}" --buffer-device "${BUFFER_DEVICE:-auto}"
  --num-agent-updates "${NUM_AGENT_UPDATES:-8}" --seed 4728
  --work-dir "$run_dir")
if [[ "$dry_run" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
if [[ ! -x "$root/.venv/bin/python" ]]; then
  echo 'Run uv sync in ibfm/ first.' >&2
  exit 2
fi
if [[ -e "$run_dir" && "${RESUME:-0}" != 1 ]]; then
  echo "Refusing to overwrite existing run: $run_dir" >&2
  exit 3
fi
if [[ "${RESUME:-0}" == 1 && ! -f "$run_dir/checkpoint/train_status.json" ]]; then
  echo "No rolling checkpoint to resume: $run_dir" >&2
  exit 3
fi
report="$("$root/.venv/bin/python" scripts/check_rp1_training.py)"
printf '%s\n' "$report"
expected_rows="$("$root/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["expected_eval_rows"])' <<< "$report")"
mkdir -p "$run_dir" "$MPLCONFIGDIR"
printf '%s\n' "$report" > "$run_dir/rp1_inputs.json"
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
"$root/.venv/bin/python" scripts/retain_policies.py --run-dir "$run_dir" \
  --expected-eval-rows "$expected_rows" > "$run_dir/policy_retention.log" 2>&1 &
watcher_pid=$!
trap 'kill "$watcher_pid" 2>/dev/null || true; wait "$watcher_pid" 2>/dev/null || true' EXIT
if "${command[@]}" 2>&1 | tee -a "$run_dir/launcher.log"; then
  train_status=0
else
  train_status=$?
fi
kill "$watcher_pid" 2>/dev/null || true
wait "$watcher_pid" 2>/dev/null || true
"$root/.venv/bin/python" scripts/retain_policies.py --run-dir "$run_dir" \
  --expected-eval-rows "$expected_rows" --once >> "$run_dir/policy_retention.log" 2>&1
exit "$train_status"
