#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
scene_dir="${script_dir}/random_height_scene"
record_tag="${RECORD_TAG:-$(date +%Y%m%d_%H%M%S)}"
output_dir="${script_dir}/interactive/random_height_${record_tag}"
mkdir -p "${output_dir}"
viewer_args=(
  --viewer --max-steps 0 --save-mp4
  --render-size "${RENDER_SIZE:-720}" --video-fps "${VIDEO_FPS:-30}"
  --video-label "166.560M z-aware | R = score and randomize box height"
  --output-path "${output_dir}/session.mp4"
)
if [[ "${HEADLESS:-0}" == "1" ]]; then
  viewer_args=(--no-viewer --max-steps "${MAX_STEPS:-100}")
fi

cd "${repo_root}"
TARGET_HEIGHT="${TARGET_HEIGHT:-0.65}" \
ROBOT_BOX_DISTANCE_MIN="${ROBOT_BOX_DISTANCE_MIN:-1.5}" \
ROBOT_BOX_DISTANCE_MAX="${ROBOT_BOX_DISTANCE_MAX:-2.0}" \
SCENE_SEED="${SCENE_SEED:-166559744}" \
PYTHONPATH="${repo_root}:${PYTHONPATH:-}" \
  .venv/bin/python "${script_dir}/tools/prepare_random_height_scene.py"

initial_distance="$(.venv/bin/python -c 'import json, sys; print(json.load(open(sys.argv[1]))["initial_robot_box_distance"])' "${scene_dir}/scene.json")"

CUDA_VISIBLE_DEVICES="${GPU_ID:-}" \
DISPLAY="${DISPLAY:-:1}" \
MUJOCO_GL="${MUJOCO_GL_BACKEND:-glfw}" \
MPLCONFIGDIR=/tmp/ufo-mpl-solve-close166-height-reset \
XDG_CACHE_HOME=/tmp/ufo-cache-solve-close166-height-reset \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="${script_dir}/../common/code_overlay:${repo_root}:${PYTHONPATH:-}" \
UFO_Z3D_RANDOM_HEIGHT_RESET=1 \
UFO_Z3D_RESET_HEIGHT_MIN="${HEIGHT_MIN:-0.15}" \
UFO_Z3D_RESET_HEIGHT_MAX="${HEIGHT_MAX:-0.80}" \
UFO_Z3D_RESET_SEED="${HEIGHT_SEED:-166559744}" \
UFO_Z3D_RESET_SUCCESS_RADIUS=0.20 \
UFO_Z3D_RESET_SUMMARY_PATH="${output_dir}/height_trials.json" \
UFO_Z3D_INITIAL_BOX_Z=0.65 \
UFO_Z3D_TARGET_BOX_Z="${TARGET_HEIGHT:-0.65}" \
UFO_Z3D_TARGET_X=2.50 UFO_Z3D_TARGET_Y=0.0 \
UFO_Z3D_XY_ALIGN_DISTANCE_OVERRIDE="${XY_ALIGN_DISTANCE:-0.20}" \
UFO_Z3D_XY_ALIGN_DISTANCE=0.20 \
UFO_Z3D_XY_ALIGN_EXIT_DISTANCE_OVERRIDE="${XY_ALIGN_EXIT_DISTANCE:-0.22}" \
UFO_Z3D_XY_ALIGN_EXIT_DISTANCE=0.22 \
UFO_Z3D_PLACE_ENTRY_DISTANCE=0.22 \
UFO_Z3D_PLACE_ENTRY_SPEED_OVERRIDE="${PLACE_ENTRY_SPEED:-0.50}" \
UFO_Z3D_PLACE_ENTRY_SPEED=0.50 \
UFO_Z3D_TERMINAL_RADIUS=0.195 \
UFO_Z3D_TERMINAL_EXIT_RADIUS=0.20 \
UFO_Z3D_TERMINAL_SPEED=0.15 \
UFO_Z3D_TERMINAL_STEPS=10 \
UFO_Z3D_TERMINAL_EXIT_STEPS=1 \
UFO_Z3D_MAX_LIFT_SWITCH_HEIGHT=0.78 \
UFO_Z3D_LIFT_HYSTERESIS_BAND=0.50 \
UFO_Z3D_TRANSPORT_FIXED_BAND=original \
UFO_PIVOT_ARRIVAL_SPEED=0.15 UFO_PIVOT_RESPONSE_SECONDS=0.50 \
UFO_PIVOT_MAX_SECONDS=5.0 UFO_PIVOT_COOLDOWN_SECONDS=0.50 \
UFO_PIVOT_STALL_WINDOW_SECONDS=0.50 UFO_PIVOT_MIN_RECENT_YAW_RAD=0.03 \
UFO_PIVOT_MIN_BEARING_IMPROVEMENT_DEG=3.0 \
UFO_PIVOT_MAX_HEIGHT_DRIFT_M=0.08 UFO_PIVOT_MIN_BOX_UPRIGHT=0.85 \
.venv/bin/python "${script_dir}/z_aware_height_banded_sim2sim.py" \
  --model-folder "${script_dir}/checkpoint_ref" \
  --latent-dir "${script_dir}/latents" \
  --get-up-latent "${script_dir}/latents/z_get_up_reward.pkl" \
  --motion-cache "${scene_dir}/standing_front_box.pkl" \
  --motion-id 0 --initial-frame 0 --initial-distance "${initial_distance}" \
  --initial-box-center-height 0.65 \
  --goal-offset-x 1.5 --goal-offset-y 0.0 \
  --robot-config "${scene_dir}/robot_config.yaml" \
  --box-config "${scene_dir}/box_config.yaml" \
  --box-size 0.30 0.30 0.30 --box-mass 0.70 \
  --box-condim 3 --hand-friction 1.0 0.005 0.0001 \
  --contact-source physical --device cpu \
  --fall-detect-steps 10 --get-up-root-height 0.65 \
  --get-up-upright-threshold 0.85 --get-up-stable-steps 25 \
  --approach-switch-distance 0.75 --phase-hold-steps 10 \
  --transport-heading-blend-full-deg 45 --transport-max-turn-blend 0.45 \
  --transport-near-max-turn-blend 0.80 --transport-turn-boost-distance 1.20 \
  --place-switch-distance 0.195 --place-align-distance 0.22 --place-enter-steps 8 \
  --place-drift-distance 0.30 --place-drift-steps 10 --place-abort-distance 0.60 \
  --place-success-distance 0.195 --place-success-height 0.05 \
  --place-release-height 0.06 --place-lower-seconds 2.0 \
  --place-max-attempt-seconds 30.0 \
  --show-coordinate-frames \
  --coordinate-frame-size "${COORDINATE_FRAME_SIZE:-0.65}" \
  --coordinate-frame-width "${COORDINATE_FRAME_WIDTH:-0.012}" \
  --camera-distance "${CAMERA_DISTANCE:-7.5}" \
  "${viewer_args[@]}" \
  --trace-csv "${output_dir}/session.csv" \
  --summary-path "${output_dir}/controller_summary.json"

if [[ "${HEADLESS:-0}" != "1" ]]; then
  echo "Video:        ${output_dir}/session.mp4"
fi
echo "Trace:        ${output_dir}/session.csv"
echo "Controller:   ${output_dir}/controller_summary.json"
echo "Height tests: ${output_dir}/height_trials.json"
