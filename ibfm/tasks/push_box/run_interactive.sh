#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
carry_bundle="${repo_root}/tasks/common"
runtime_overlay="${script_dir}/runtime_dependencies/code_overlay"
drive_dir="${DRIVE_DIR:-${script_dir}/latents/selected/drive}"
acquire_dir="${ACQUIRE_DIR:-${script_dir}/latents/acquire_openloop_grid_v1/crouch_sync_0p55}"
tag="${1:-$(date +%Y%m%d_%H%M%S)}"
output_root="${OUTPUT_ROOT:-${script_dir}/rollouts}"
output_dir="${output_root}/${tag}"
mkdir -p "${output_dir}"

test -f "${drive_dir}/z_push_direct.pkl"
test -f "${acquire_dir}/z_push_direct.pkl"

viewer="${VIEWER:-1}"
viewer_args=(--no-viewer)
default_gl=egl
default_max_steps=3000
if [[ "$viewer" == "1" ]]; then
  viewer_args=(--viewer)
  default_gl=glfw
  default_max_steps=0
fi
video_args=()
if [[ "${SAVE_MP4:-1}" == "1" ]]; then
  video_args=(
    --save-mp4
    --render-size "${RENDER_SIZE:-720}"
    --video-fps "${VIDEO_FPS:-30}"
    --video-label "${VIDEO_LABEL-166.560M push box | better_fix_best}"
    --output-path "${output_dir}/push_box.mp4"
  )
fi
reposition_marker_args=()
if [[ "${SHOW_REPOSITION_MARKERS:-1}" == "1" ]]; then
  reposition_marker_args=(--show-reposition-markers)
fi

cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${UFO_PUSH_CUDA_VISIBLE_DEVICES:-}" \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ufo-mpl-push166-success \
MESA_SHADER_CACHE_DIR=/tmp/ufo-mesa-push166-success \
MUJOCO_GL="${MUJOCO_GL:-$default_gl}" \
PYTHONPATH="${runtime_overlay}:${carry_bundle}/code_overlay:${repo_root}:${PYTHONPATH:-}" \
UFO_PUSH_V2_START_WITH_REPOSITION="${START_WITH_REPOSITION:-1}" \
UFO_PUSH_V2_HOLD_DRIVE="${HOLD_DRIVE:-1}" \
UFO_PUSH_V2_REACQUIRE_BLEND_MAX="${REACQUIRE_BLEND_MAX:-0.0}" \
UFO_PUSH_V2_STOP_SPEED="${STOP_SPEED:-0.06}" \
UFO_PUSH_V2_STALL_PULSE_BLEND_MAX="${STALL_PULSE_BLEND_MAX:-0.0}" \
UFO_PUSH_V2_STALL_PULSE_DELAY_SECONDS="${STALL_PULSE_DELAY_SECONDS:-0.50}" \
UFO_PUSH_V2_STALL_PULSE_RAMP_SECONDS="${STALL_PULSE_RAMP_SECONDS:-0.50}" \
UFO_PUSH_V2_ACQUIRE_DRIVE_BLEND_SECONDS="${ACQUIRE_DRIVE_BLEND_SECONDS:-0.40}" \
UFO_PUSH_TIMED_ACQUIRE_SECONDS="${TIMED_ACQUIRE_SECONDS:-0.65}" \
UFO_PUSH_TIMED_ACQUIRE_CROUCH_SECONDS="${TIMED_ACQUIRE_CROUCH_SECONDS:-0.65}" \
UFO_PUSH_TIMED_ACQUIRE_SETTLE_SECONDS="${TIMED_ACQUIRE_SETTLE_SECONDS:-0.25}" \
UFO_PUSH_TIMED_ACQUIRE_SETTLE_LATENT_DIR="${TIMED_ACQUIRE_SETTLE_DIR:-${script_dir}/latents/selected/acquire_hand_layout_v2}" \
UFO_PUSH_V2_HANDPUSH_HOLD_LATENT_DIR="${HANDPUSH_HOLD_DIR:-${script_dir}/latents/selected/drive_handpush_hold_v1}" \
UFO_PUSH_V2_HANDPUSH_HOLD_BLEND_BASE="${HANDPUSH_HOLD_BLEND_BASE:-0.0}" \
UFO_PUSH_V2_HANDPUSH_HOLD_BLEND_MAX="${HANDPUSH_HOLD_BLEND_MAX:-0.30}" \
UFO_PUSH_V2_HANDPUSH_HOLD_HEIGHT_ENTER="${HANDPUSH_HOLD_HEIGHT_ENTER:-0.65}" \
UFO_PUSH_V2_HANDPUSH_HOLD_HEIGHT_EXIT="${HANDPUSH_HOLD_HEIGHT_EXIT:-0.58}" \
UFO_PUSH_V2_HANDPUSH_HOLD_RISE_SECONDS="${HANDPUSH_HOLD_RISE_SECONDS:-0.50}" \
UFO_PUSH_V2_HANDPUSH_HOLD_FALL_SECONDS="${HANDPUSH_HOLD_FALL_SECONDS:-0.50}" \
UFO_PUSH_V2_HANDPUSH_HOLD_ON_CONTACT_LOSS="${HANDPUSH_HOLD_ON_CONTACT_LOSS:-0}" \
UFO_PUSH_V2_FORCE_DRIVE_CONTACT_OBS="${FORCE_DRIVE_CONTACT_OBS:-1}" \
.venv/bin/python "${CONTROLLER_SCRIPT:-${script_dir}/two_stage_timed_acquire_sim2sim.py}" \
  --controller-mode push-direct \
  --model-folder "${carry_bundle}/checkpoint_ref" \
  --latent-dir "${drive_dir}" \
  --direct-push-contact-acquire-latent-dir "${acquire_dir}" \
  --direct-push-contact-drive-enter-steps "${CONTACT_ENTER_STEPS:-5}" \
  --direct-push-contact-drive-exit-steps "${CONTACT_EXIT_STEPS:-30}" \
  --direct-push-turn-latent-dir "${script_dir}/latents/turn_stop" \
  --direct-push-align-pivot-latent-dir "${ALIGN_PIVOT_DIR:-${script_dir}/latents/selected/align_pivot_no_contact_v1}" \
  --direct-push-stop-latent-dir "${script_dir}/latents/turn_stop" \
  --get-up-latent "${script_dir}/latents/get_up/z_get_up_reward.pkl" \
  --get-up-stand-latent "${script_dir}/latents/get_up/z_get_up_stand.pkl" \
  --get-up-stand-seconds 0.60 \
  --direct-push-heading-blend-full-deg 45 \
  --direct-push-max-turn-blend "${DRIVE_TURN_BLEND_MAX:-0.05}" \
  --direct-push-speed-target 0.0 \
  --direct-push-speed-deadband 0.05 \
  --direct-push-speed-filter-seconds 0.15 \
  --direct-push-speed-min-drive 0.25 \
  --direct-push-stall-speed 0.10 \
  --direct-push-stall-recovery-seconds "${STALL_SECONDS:-5}" \
  --direct-push-stop-distance "${STOP_DISTANCE:-0.40}" \
  --direct-push-stop-enter-steps 5 \
  --direct-push-stop-exit-distance "${STOP_EXIT_DISTANCE:-0.55}" \
  --direct-push-stop-exit-steps 10 \
  --push-stop-enter-distance "${SUCCESS_DISTANCE:-0.40}" \
  --push-stop-exit-distance "${STOP_EXIT_DISTANCE:-0.55}" \
  --push-success-distance "${SUCCESS_DISTANCE:-0.40}" \
  --goal-marker-radius "${GOAL_MARKER_RADIUS:-0.40}" \
  --camera-distance "${CAMERA_DISTANCE:-5.0}" \
  --direct-push-require-rear-hand-contact \
  --direct-push-hand-rear-min 0.04 \
  --direct-push-hand-rear-max 0.38 \
  --direct-push-hand-lateral-abs-max "${HAND_LATERAL_MAX:-0.13}" \
  --direct-push-hand-height-abs-max "${HAND_HEIGHT_MAX:-0.30}" \
  --direct-push-reposition-enter-steps 5 \
  --direct-push-release-seconds 0.60 \
  --direct-push-release-speed 0.08 \
  --direct-push-release-upright-min 0.90 \
  --direct-push-backoff-distance "${BACKOFF_DISTANCE:-1.0}" \
  --direct-push-backoff-max-seconds "${BACKOFF_MAX_SECONDS:-1.5}" \
  --direct-push-backoff-stable-steps "${BACKOFF_STABLE_STEPS:-1}" \
  --direct-push-backoff-blend-seconds "${BACKOFF_BLEND_SECONDS:-0.4}" \
  --direct-push-backoff-latent-dir "${BACKOFF_LATENT_DIR:-${script_dir}/latents/selected/backoff_replay_v1}" \
  --direct-push-backoff-latent-mode "${BACKOFF_LATENT_MODE:-replay}" \
  --direct-push-reposition-blend-seconds 0.80 \
  --direct-push-reposition-max-turn-blend 0.75 \
  --direct-push-reposition-turn-cap-min-recovery 2 \
  --direct-push-reposition-turn-hysteresis-deg 150 \
  --direct-push-reposition-side-clearance 0.75 \
  --direct-push-reposition-behind-offset 0.60 \
  --direct-push-reposition-min-path-clearance "${BYPASS_MIN_PATH_CLEARANCE:-0.50}" \
  --direct-push-reposition-replan-box-displacement "${BYPASS_REPLAN_BOX_DISPLACEMENT:-0.10}" \
  --direct-push-reposition-side-switch-clearance-gain "${BYPASS_SIDE_SWITCH_CLEARANCE_GAIN:-0.10}" \
  --push-standoff-distance "${STAGING_DISTANCE:-0.80}" \
  --push-staging-tolerance "${STAGING_TOLERANCE:-0.20}" \
  --push-lateral-max "${STAGING_LATERAL_MAX:-0.30}" \
  --direct-push-reposition-heading-tolerance-deg "${ALIGN_HEADING_ENTER_DEG:-20}" \
  --direct-push-align-heading-exit-tolerance-deg "${ALIGN_HEADING_EXIT_DEG:-25}" \
  --direct-push-align-ready-distance "${ALIGN_READY_DISTANCE:-0.35}" \
  --direct-push-reposition-align-exit-distance "${ALIGN_EXIT_DISTANCE:-0.35}" \
  --direct-push-reposition-align-exit-lateral "${ALIGN_EXIT_LATERAL:-0.40}" \
  --direct-push-reposition-align-exit-behind-margin 0.10 \
  --direct-push-overshoot-margin 0.10 \
  --direct-push-overshoot-steps 10 \
  --motion-cache cache/motion_data/g1_carry_box_omniretarget_no_foot/g1_carry_box_omniretarget_no_foot_full_grounded_mujoco_hysteresis_4p5_6cm_ufo.pkl \
  --motion-id 67 \
  --initial-frame 0 \
  --initial-distance "${INITIAL_DISTANCE:-1.0}" \
  --initial-box-center-height 0.186028293 \
  --goal-offset-x "${GOAL_OFFSET_X:-2.0}" \
  --goal-offset-y "${GOAL_OFFSET_Y:-0.0}" \
  --relocate-box-offset "${RELOCATE_BOX_X:-0}" "${RELOCATE_BOX_Y:-0}" \
  --relocate-after-complete-seconds "${RELOCATE_AFTER_SECONDS:-2}" \
  --contact-source physical \
  --device "${UFO_PUSH_DEVICE:-cpu}" \
  --mask-approach-object-context \
  "${reposition_marker_args[@]}" \
  "${viewer_args[@]}" \
  --max-steps "${MAX_STEPS:-$default_max_steps}" \
  "${video_args[@]}" \
  --trace-csv "${output_dir}/rollout.csv" \
  --summary-path "${output_dir}/summary.json"

if [[ "${SAVE_MP4:-1}" == "1" ]]; then
  echo "Video:   ${output_dir}/push_box.mp4"
fi
echo "Trace:   ${output_dir}/rollout.csv"
echo "Summary: ${output_dir}/summary.json"
