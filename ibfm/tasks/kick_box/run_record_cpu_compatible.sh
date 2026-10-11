#!/usr/bin/env bash
set -euo pipefail

variant_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle="${variant_dir}"
repo_root="$(cd "${bundle}/../.." && pwd)"
carry_bundle="${repo_root}/tasks/common"
drive_dir="${DRIVE_DIR:-${bundle}/latents/candidates/kick25_min75}"
tag="${1:-$(date +%Y%m%d_%H%M%S)}"
output_dir="${OUTPUT_ROOT:-${variant_dir}/rollouts}/${tag}"
mkdir -p "${output_dir}"
video_filename="${VIDEO_FILENAME:-kick_box.mp4}"
viewer_args=(--no-viewer)
if [[ "${VIEWER:-0}" == "1" ]]; then
  viewer_args=(--viewer)
fi
record_args=(--save-mp4)
if [[ "${SAVE_MP4:-1}" == "0" ]]; then
  record_args=(--no-save-mp4)
fi
reposition_marker_args=()
if [[ "${SHOW_REPOSITION_MARKERS:-1}" == "1" ]]; then
  reposition_marker_args=(--show-reposition-markers)
fi

cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${UFO_PUSH_CUDA_VISIBLE_DEVICES:-}" \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ufo-mpl-push166-kick-sucess \
MESA_SHADER_CACHE_DIR=/tmp/ufo-mesa-push166-kick-sucess \
MUJOCO_GL="${MUJOCO_GL:-egl}" \
PYTHONPATH="${bundle}/runtime_dependencies/code_overlay:${carry_bundle}/code_overlay:${repo_root}:${PYTHONPATH:-}" \
UFO_KICK_START_WITH_REPOSITION="${START_WITH_REPOSITION:-0}" \
UFO_PUSH_V2_HOLD_DRIVE=1 \
UFO_PUSH_V2_REACQUIRE_BLEND_MAX=0.0 \
UFO_PUSH_V2_STALL_PULSE_BLEND_MAX=0.0 \
.venv/bin/python "${CONTROLLER_SCRIPT:-${bundle}/kick_box_sim2sim.py}" \
  --controller-mode push-direct \
  --model-folder "${repo_root}/tasks/common/checkpoint_ref" \
  --latent-dir "${drive_dir}" \
  --direct-push-contact-acquire-latent-dir "${bundle}/latents/selected/acquire" \
  --direct-push-contact-drive-enter-steps 5 \
  --direct-push-contact-drive-exit-steps 30 \
  --direct-push-turn-latent-dir "${bundle}/latents/turn_stop" \
  --direct-push-align-pivot-latent-dir "${ALIGN_PIVOT_DIR:-${bundle}/latents/selected/align_pivot_no_contact_v1}" \
  --direct-push-stop-latent-dir "${bundle}/latents/turn_stop" \
  --get-up-latent "${bundle}/latents/get_up/z_get_up_reward.pkl" \
  --get-up-stand-latent "${bundle}/latents/get_up/z_get_up_stand.pkl" \
  --get-up-stand-seconds 0.60 \
  --direct-push-heading-blend-full-deg 45 \
  --direct-push-max-turn-blend 0.0 \
  --direct-push-speed-target 0.0 \
  --direct-push-speed-deadband 0.05 \
  --direct-push-speed-filter-seconds 0.15 \
  --direct-push-speed-min-drive 0.25 \
  --direct-push-stall-speed 0.03 \
  --direct-push-stall-recovery-seconds "${STALL_SECONDS:-10}" \
  --direct-push-stop-distance 0.20 \
  --direct-push-stop-enter-steps 5 \
  --direct-push-stop-exit-distance 0.40 \
  --direct-push-stop-exit-steps 10 \
  --direct-push-require-rear-hand-contact \
  --direct-push-hand-rear-min 0.04 \
  --direct-push-hand-rear-max 0.38 \
  --direct-push-hand-lateral-abs-max 0.25 \
  --direct-push-hand-height-abs-max 0.20 \
  --direct-push-reposition-enter-steps 5 \
  --direct-push-release-seconds 0.60 \
  --direct-push-release-speed 0.08 \
  --direct-push-release-upright-min 0.90 \
  --direct-push-backoff-distance "${BACKOFF_DISTANCE:-1.0}" \
  --direct-push-backoff-max-seconds "${BACKOFF_MAX_SECONDS:-1.5}" \
  --direct-push-backoff-stable-steps "${BACKOFF_STABLE_STEPS:-1}" \
  --direct-push-backoff-blend-seconds "${BACKOFF_BLEND_SECONDS:-0.4}" \
  --direct-push-backoff-latent-dir "${BACKOFF_LATENT_DIR:-${bundle}/latents/selected/backoff_replay_v1}" \
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
  --initial-distance 1.0 \
  --initial-box-center-height 0.186028293 \
  --goal-offset-x 2.0 \
  --goal-offset-y 0.0 \
  --relocate-box-offset "${RELOCATE_BOX_X:-0}" "${RELOCATE_BOX_Y:-0}" \
  --relocate-after-complete-seconds "${RELOCATE_AFTER_SECONDS:-2}" \
  --contact-source physical \
  --device "${UFO_PUSH_DEVICE:-cpu}" \
  --mask-approach-object-context \
  --goal-marker-radius 0.20 \
  --camera-distance "${CAMERA_DISTANCE:-5.0}" \
  --camera-azimuth "${CAMERA_AZIMUTH:-135}" \
  "${reposition_marker_args[@]}" \
  "${viewer_args[@]}" \
  --max-steps "${MAX_STEPS:-3000}" \
  "${record_args[@]}" \
  --render-size "${RENDER_SIZE:-720}" \
  --video-fps "${VIDEO_FPS:-30}" \
  --video-label "${VIDEO_LABEL-Kick Box simulation}" \
  --output-path "${output_dir}/${video_filename}" \
  --trace-csv "${output_dir}/rollout.csv" \
  --summary-path "${output_dir}/summary.json"

if [[ "${SAVE_MP4:-1}" != "0" ]]; then
  echo "Video:   ${output_dir}/${video_filename}"
fi
echo "Trace:   ${output_dir}/rollout.csv"
echo "Summary: ${output_dir}/summary.json"
