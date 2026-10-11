"""Tracking inference and video export for UFO policies.

Policy rollout is rendered from the training environment, while the reference
motion is rendered from the configured robot MJCF with pure MuJoCo qpos playback.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import mediapy as media
import mujoco
import mujoco.viewer
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.export.backward_encoder import (
    UnsupportedBackwardEncoderExport,
    export_backward_encoder_from_model,
)
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    add_bool_arg,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    policy_qpos_from_env,
    render_policy_frame,
)
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, get_backward_observation
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from humanoidverse.utils.robot_spec import assert_robot_configs_compatible, load_robot_training_spec, resolve_robot_config_path
from humanoidverse.utils.torch_utils import quat_from_angle_axis, quat_mul


DEFAULT_ROBOT_CONFIG = "configs/robots/g1_29dof.yaml"


def _sync_passive_viewer(
    wrapped_env: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer: Any,
    *,
    dof_qpos_order_indices: np.ndarray,
    num_dof: int,
) -> None:
    """Copy the single MJLab world into an independent native MuJoCo viewer."""
    qpos = policy_qpos_from_env(wrapped_env, expected_qpos_size=model.nq)
    qpos[7 : 7 + num_dof] = qpos[7 : 7 + num_dof][dof_qpos_order_indices]
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    viewer.cam.lookat[:] = [float(qpos[0]), float(qpos[1]), max(float(qpos[2]), 0.75)]
    viewer.sync()


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    y_idx = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
    return frame[y_idx[:, None], x_idx[None, :]]


def _control_to_qpos_order_indices(robot_training: Any) -> tuple[np.ndarray, list[str]]:
    control_joint_names = list(robot_training.robot.control_joint_names)
    qpos_joint_names = sorted(control_joint_names, key=lambda joint: robot_training.robot.joint_qpos_addr[joint])
    qpos_addrs = [int(robot_training.robot.joint_qpos_addr[joint]) for joint in qpos_joint_names]
    if len(set(qpos_addrs)) != len(qpos_addrs):
        raise ValueError(f"Duplicate MuJoCo qpos addresses for control joints: {list(zip(qpos_joint_names, qpos_addrs))}")
    index_by_control_joint = {joint: idx for idx, joint in enumerate(control_joint_names)}
    return np.asarray([index_by_control_joint[joint] for joint in qpos_joint_names], dtype=np.int64), qpos_joint_names


def _expert_qpos_from_obs(
    obs_dict: dict[str, torch.Tensor],
    *,
    num_dof: int,
    dof_qpos_order_indices: np.ndarray,
) -> np.ndarray:
    root_pos = obs_dict["ref_body_pos"][:, 0].detach().cpu().numpy()
    root_quat_wxyz = np.roll(obs_dict["ref_body_rots"][:, 0].detach().cpu().numpy(), 1, axis=-1)
    # MotionLib stores dof_pos in policy/control-joint order. MuJoCo qpos playback
    # expects hinge joints sorted by qpos address, which can differ for robots such
    # as X2 where actuator order places head joints before arm joints.
    dof_pos = obs_dict["dof_pos"].detach().cpu().numpy()[:, dof_qpos_order_indices]
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=-1)
    expected = 7 + int(num_dof)
    if "object_pos" in obs_dict:
        object_pos = obs_dict["object_pos"].detach().cpu().numpy()
        object_quat_wxyz = np.roll(obs_dict["object_quat"].detach().cpu().numpy(), 1, axis=-1)
        qpos = np.concatenate([qpos, object_pos, object_quat_wxyz], axis=-1)
        expected += 7
    if qpos.shape[-1] != expected:
        raise ValueError(f"Expected expert qpos shape (*, {expected}), got {qpos.shape}")
    return qpos


def _target_states_from_obs(obs_dict: dict[str, torch.Tensor], device: str, *, num_dof: int) -> dict[str, torch.Tensor]:
    root_state_xyzw = torch.cat(
        [
            obs_dict["ref_body_pos"][0, 0],
            obs_dict["ref_body_rots"][0, 0],
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ],
        dim=-1,
    ).to(device=device, dtype=torch.float32)
    dof_state = torch.zeros((int(num_dof), 2), device=device, dtype=torch.float32)
    dof_state[:, 0] = obs_dict["dof_pos"][0].to(device=device, dtype=torch.float32)
    dof_state[:, 1] = obs_dict["ref_dof_vel"][0].to(device=device, dtype=torch.float32)
    target_states = {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}
    carry_fields = {
        "object_pos",
        "object_quat",
        "object_lin_vel",
        "object_ang_vel",
        "object_target_pos",
        "object_target_quat",
        "object_present",
    }
    present_carry_fields = carry_fields.intersection(obs_dict)
    if present_carry_fields:
        missing_carry_fields = carry_fields - present_carry_fields
        if missing_carry_fields:
            raise ValueError(f"Incomplete carry state in tracking observation; missing {sorted(missing_carry_fields)}")

        def initial(value: torch.Tensor) -> torch.Tensor:
            return value if value.ndim == 1 else value[0]

        object_state = torch.cat(
            [
                initial(obs_dict["object_pos"]),
                initial(obs_dict["object_quat"]),
                initial(obs_dict["object_lin_vel"]),
                initial(obs_dict["object_ang_vel"]),
            ],
            dim=-1,
        ).to(device=device, dtype=torch.float32)
        target_states.update(
            {
                "object_states": object_state.unsqueeze(0),
                "target_object_pos": initial(obs_dict["object_target_pos"])
                .to(device=device, dtype=torch.float32)
                .unsqueeze(0),
                "target_object_quat": initial(obs_dict["object_target_quat"])
                .to(device=device, dtype=torch.float32)
                .unsqueeze(0),
                "object_present": initial(obs_dict["object_present"])
                .to(device=device, dtype=torch.float32)
                .reshape(1, 1),
            }
        )
    return target_states


def _apply_lie_down_initialization(
    target_states: dict[str, torch.Tensor],
    *,
    side: str,
    root_height: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return target states with the robot root initialized on one side."""

    if side not in {"left", "right"}:
        raise ValueError(f"lie-down side must be 'left' or 'right', got {side!r}")
    result = {key: value.clone() if torch.is_tensor(value) else value for key, value in target_states.items()}
    root_states = result["root_states"]
    if root_states.ndim != 2 or root_states.shape[-1] < 7:
        raise ValueError(f"Expected root_states shape (N, >=7), got {tuple(root_states.shape)}")

    root_states[:, 2] = float(root_height)
    angle = torch.pi / 2 if side == "left" else -torch.pi / 2
    rotation = quat_from_angle_axis(
        torch.tensor(angle, device=root_states.device, dtype=root_states.dtype),
        torch.tensor([1.0, 0.0, 0.0], device=root_states.device, dtype=root_states.dtype),
        w_last=True,
    )
    root_states[:, 3:7] = quat_mul(
        rotation.expand_as(root_states[:, 3:7]),
        root_states[:, 3:7],
        w_last=True,
    )
    return result


def _downward_force_active(step: int, *, dt: float, start_s: float, duration_s: float) -> bool:
    """Return whether a deterministic external-force pulse is active."""

    if dt <= 0.0:
        raise ValueError(f"environment dt must be positive, got {dt}")
    if start_s < 0.0 or duration_s < 0.0:
        raise ValueError(f"force timing must be non-negative, got start_s={start_s}, duration_s={duration_s}")
    time_s = float(step) * float(dt)
    return duration_s > 0.0 and start_s <= time_s < start_s + duration_s


def _write_downward_root_force(env: Any, force_n: float) -> None:
    """Apply a world-frame downward force to the robot root body."""

    force = torch.zeros((env.num_envs, 1, 3), device=env.device, dtype=torch.float32)
    force[:, :, 2] = -float(force_n)
    torque = torch.zeros_like(force)
    env.robot.write_external_wrench_to_sim(force, torque, body_ids=[0])


@torch.no_grad()
def _tracking_z(
    model: torch.nn.Module,
    obs: Any,
    *,
    seq_length: int | None = None,
    expert_z_mode: str | None = None,
    expert_z_horizon: int | None = None,
) -> torch.Tensor:
    train_cfg = getattr(model.cfg, "train", None)
    if expert_z_mode is None:
        expert_z_mode = getattr(train_cfg, "expert_z_mode", "mean")
    if expert_z_mode == "direct":
        horizon = int(
            expert_z_horizon
            if expert_z_horizon is not None
            else getattr(train_cfg, "expert_z_horizon", model.cfg.seq_length)
        )
        if horizon <= 0:
            raise ValueError(f"tracking expert_z_horizon must be positive, got {horizon}")
        z = model.backward_map(obs)
        # Training samples goal_observation at offset horizon - 1 from the
        # next-observation sequence.  Preserve the rollout length by holding
        # the final available goal latent near the end of a finite motion.
        offset = horizon - 1
        if offset > 0 and z.shape[0] > 0:
            if offset < z.shape[0]:
                z = torch.cat((z[offset:], z[-1:].expand(offset, -1)), dim=0)
            else:
                z = z[-1:].expand(z.shape[0], -1).clone()
        return model.project_z(z)
    if expert_z_mode != "mean":
        raise ValueError(f"Unsupported tracking expert_z_mode: {expert_z_mode!r}")
    if seq_length is None:
        seq_length = int(model.cfg.seq_length)
    if seq_length <= 0:
        raise ValueError(f"tracking seq_length must be positive, got {seq_length}")
    z = model.backward_map(obs)
    for step in range(z.shape[0]):
        end_idx = min(step + seq_length, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    return model.project_z(z)


def _export_policy_model(model: torch.nn.Module, output_dir: Path, robot_training: Any) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.__class__.__name__
    output_name = f"{model_name}.onnx"
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    export_metadata = export_meta_policy_as_onnx(
        model,
        output_dir,
        output_name,
        z_dim=model.cfg.archi.z_dim,
    )
    if int(export_metadata["output_action_dim"]) != num_dof:
        raise ValueError(
            "Policy action dim does not match robot control joint count: "
            f"output_action_dim={export_metadata['output_action_dim']}, num_dof={num_dof}"
        )
    export_metadata.update(
        {
            "robot_name": robot_training.robot.name,
            "robot_config_path": str(Path(robot_training.config_path).expanduser().resolve()),
            "xml_path": str(Path(robot_training.robot.xml_path).expanduser().resolve()),
            "num_dof": num_dof,
            "control_joint_names": control_joint_names,
        }
    )
    metadata_path = output_dir / f"{model_name}.meta.json"
    metadata_path.write_text(json.dumps(export_metadata, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Exported model to {output_dir / output_name}")
    print(f"[INFO] Wrote policy ONNX metadata to {metadata_path}")
    return export_metadata


def _export_tracking_onnx(model: torch.nn.Module, output_dir: Path, robot_training: Any) -> None:
    _export_policy_model(model, output_dir, robot_training)
    try:
        export_backward_encoder_from_model(model, output_dir / "backward_encoder.onnx")
    except UnsupportedBackwardEncoderExport as exc:
        print(f"[INFO] Skip backward encoder ONNX export: {exc}")


def _resolve_tracking_robot_config(
    cli_robot_config: str | Path | None,
    manifest_robot_config: str | Path | None,
) -> Path:
    if cli_robot_config is not None and manifest_robot_config is not None:
        return assert_robot_configs_compatible(cli_robot_config, manifest_robot_config)
    if cli_robot_config is not None:
        return resolve_robot_config_path(cli_robot_config)
    if manifest_robot_config is not None:
        return resolve_robot_config_path(manifest_robot_config)
    return resolve_robot_config_path(DEFAULT_ROBOT_CONFIG)


def run_tracking_inference(
    *,
    model_folder: Path,
    output_dir: Path | None,
    data_path: Path | None,
    robot_config: Path | None,
    headless: bool,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    motion_list: list[int],
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    fps: int,
    max_steps: int | None,
    log_every_steps: int,
    max_episode_length_s: float,
    tracking_seq_length: int | None,
    lie_down_init: bool,
    lie_down_side: str,
    downward_force_n: float,
    force_start_s: float,
    force_duration_s: float,
    export_onnx: bool,
) -> None:
    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    robot_config = _resolve_tracking_robot_config(robot_config, None)
    robot_training = load_robot_training_spec(robot_config)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    if not robot_xml.exists():
        raise FileNotFoundError(f"Missing robot XML: {robot_xml}")
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    dof_qpos_order_indices, qpos_joint_names = _control_to_qpos_order_indices(robot_training)

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()

    # The serialized model config only describes the network.  Tracking-z
    # sampling semantics live in the full training config stored beside the
    # snapshot, so read them explicitly for INTACT/direct checkpoints.
    full_config_path = model_folder / "config.json"
    full_config = json.loads(full_config_path.read_text()) if full_config_path.exists() else {}
    saved_train_cfg = full_config.get("agent", {}).get("train", {})
    expert_z_mode = str(saved_train_cfg.get("expert_z_mode", "mean"))
    expert_z_horizon = int(saved_train_cfg.get("expert_z_horizon", model.cfg.seq_length))

    if export_onnx:
        _export_tracking_onnx(model, model_folder / "exported", robot_training)

    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=device,
        headless=headless,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=max_episode_length_s,
    )
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env

    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else model_folder / "tracking_inference"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] UFO tracking inference model_folder={model_folder}")
    print(f"[INFO] Rollout XML={env_cfg.mjcf_path}")
    print(f"[INFO] Motion data={env_cfg.lafan_tail_path}")
    print(f"[INFO] Expert renderer XML={robot_xml}")
    if qpos_joint_names != control_joint_names:
        print(f"[INFO] Expert qpos joint order differs from control order: {qpos_joint_names}")
    effective_tracking_seq_length = (
        int(model.cfg.seq_length) if tracking_seq_length is None else int(tracking_seq_length)
    )
    print(f"[INFO] device={device} disable_dr={disable_dr} disable_obs_noise={disable_obs_noise} save_mp4={save_mp4}")
    if expert_z_mode == "direct":
        print(f"[INFO] tracking_latent_mode=direct expert_z_horizon={expert_z_horizon}")
    else:
        print(f"[INFO] tracking_latent_mode=mean tracking_seq_length={effective_tracking_seq_length}")
    print(f"[INFO] lie_down_init={lie_down_init} lie_down_side={lie_down_side}")
    print(
        f"[INFO] downward_force_n={downward_force_n} force_start_s={force_start_s} "
        f"force_duration_s={force_duration_s}"
    )
    if downward_force_n < 0.0:
        raise ValueError(f"downward_force_n must be non-negative, got {downward_force_n}")
    _downward_force_active(0, dt=env.dt, start_s=force_start_s, duration_s=force_duration_s)

    env._motion_lib.load_all_motions()
    env.is_evaluating = True
    box_half_size = (
        tuple(float(value) for value in env.carry_box_cfg.get("half_size", [0.2356, 0.2294, 0.1850]))
        if env.carry_box_enabled
        else None
    )
    box_geom_pos = tuple(float(value) for value in env.carry_box_cfg.get("geom_pos", [0.0, 0.0, 0.0]))
    box_geom_quat_wxyz = tuple(
        float(value) for value in env.carry_box_cfg.get("geom_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
    )
    expert_renderer = (
        MujocoQposRenderer(
            robot_xml,
            render_size=render_size,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            expected_qpos_size=7 + num_dof + (7 if box_half_size is not None else 0),
            box_half_size=box_half_size,
            box_geom_pos=box_geom_pos,
            box_geom_quat_wxyz=box_geom_quat_wxyz,
        )
        if save_mp4
        else None
    )
    passive_model = env.mjlab_env.sim.mj_model
    passive_data = mujoco.MjData(passive_model) if not headless else None
    passive_viewer = mujoco.viewer.launch_passive(passive_model, passive_data) if not headless else None
    if passive_viewer is not None:
        passive_viewer.cam.distance = float(camera_distance)
        passive_viewer.cam.azimuth = float(camera_azimuth)
        passive_viewer.cam.elevation = float(camera_elevation)
    try:
        for motion_id in motion_list:
            backward_obs, obs_dict = get_backward_observation(env, motion_id, use_root_height_obs=use_root_height_obs)
            z = _tracking_z(
                model,
                tree_map(lambda x: x[1:].to(device) if hasattr(x, "to") else x, backward_obs),
                seq_length=tracking_seq_length,
                expert_z_mode=expert_z_mode,
                expert_z_horizon=expert_z_horizon,
            )
            output_suffix = f"_lie_{lie_down_side}" if lie_down_init else ""
            if downward_force_n > 0.0 and force_duration_s > 0.0:
                output_suffix += "_force_down"
            z_path = output_dir / f"zs_{motion_id}{output_suffix}.pkl"
            joblib.dump(z.detach().cpu().numpy(), z_path)
            print(f"[INFO] Saved z embedding: {z_path}")

            target_states = _target_states_from_obs(obs_dict, device=device, num_dof=num_dof)
            if lie_down_init:
                target_states = _apply_lie_down_initialization(target_states, side=lie_down_side)
            observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)
            if passive_viewer is not None:
                _sync_passive_viewer(
                    wrapped_env,
                    passive_model,
                    passive_data,
                    passive_viewer,
                    dof_qpos_order_indices=dof_qpos_order_indices,
                    num_dof=num_dof,
                )
            episode_len = int(z.shape[0])
            if max_steps is not None:
                episode_len = min(episode_len, int(max_steps))
            expert_qpos = _expert_qpos_from_obs(
                obs_dict,
                num_dof=num_dof,
                dof_qpos_order_indices=dof_qpos_order_indices,
            )
            frames: list[np.ndarray] = []
            use_env_render = True

            print(f"[INFO] Running policy rollout for motion_id={motion_id}, steps={episode_len}", flush=True)
            for step in range(episode_len):
                if passive_viewer is not None and not passive_viewer.is_running():
                    print("[INFO] Viewer closed; stopping tracking rollout.", flush=True)
                    return
                step_start = time.perf_counter()
                force_active = downward_force_n > 0.0 and _downward_force_active(
                    step,
                    dt=env.dt,
                    start_s=force_start_s,
                    duration_s=force_duration_s,
                )
                _write_downward_root_force(env, downward_force_n if force_active else 0.0)
                if force_active and not _downward_force_active(
                    step - 1,
                    dt=env.dt,
                    start_s=force_start_s,
                    duration_s=force_duration_s,
                ):
                    print(
                        f"[INFO] Applying downward root force: {downward_force_n:.1f} N "
                        f"at step={step} for {force_duration_s:.3f}s",
                        flush=True,
                    )
                action = model.act(observation, z[step].unsqueeze(0), mean=True)
                observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)

                if passive_viewer is not None:
                    _sync_passive_viewer(
                        wrapped_env,
                        passive_model,
                        passive_data,
                        passive_viewer,
                        dof_qpos_order_indices=dof_qpos_order_indices,
                        num_dof=num_dof,
                    )
                    remaining = float(env.dt) - (time.perf_counter() - step_start)
                    if remaining > 0.0:
                        time.sleep(remaining)

                if save_mp4:
                    policy_frame, use_env_render = render_policy_frame(
                        wrapped_env,
                        expert_renderer,
                        use_env_render=use_env_render,
                    )
                    expert_frame = expert_renderer.render_qpos(expert_qpos[min(step + 1, len(expert_qpos) - 1)])
                    expert_frame = _resize_nearest(expert_frame, policy_frame.shape[0], policy_frame.shape[1])
                    frames.append(np.concatenate([expert_frame, policy_frame], axis=1))

                if step == 0 or (step + 1) == episode_len or (log_every_steps > 0 and (step + 1) % log_every_steps == 0):
                    print(f"[INFO] motion_id={motion_id} rollout/render progress {step + 1}/{episode_len}", flush=True)

                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    print(f"[INFO] Episode ended at step={step}; stopping rollout for motion_id={motion_id}")
                    break

            if save_mp4:
                video_path = output_dir / f"tracking_{motion_id}{output_suffix}.mp4"
                if not frames:
                    raise RuntimeError(f"No frames were rendered for motion_id={motion_id}")
                media.write_video(str(video_path), frames, fps=fps)
                print(f"[INFO] Saved side-by-side video: {video_path}")
            _write_downward_root_force(env, 0.0)
        if passive_viewer is not None and passive_viewer.is_running():
            print("[INFO] Tracking finished; close the viewer window to exit.", flush=True)
            while passive_viewer.is_running():
                passive_viewer.sync()
                time.sleep(0.05)
    finally:
        if passive_viewer is not None:
            passive_viewer.close()
        if expert_renderer is not None:
            expert_renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFO tracking inference with MuJoCo expert rendering.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for latent files and videos; defaults to MODEL_FOLDER/tracking_inference.",
    )
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=None, help="Robot YAML for rollout and expert rendering.")
    parser.add_argument("--data-manifest", type=Path, default=None, help="Motion data manifest. Use with --dataset.")
    parser.add_argument("--dataset", default=None, help="Dataset name inside --data-manifest for tracking inference.")
    parser.add_argument("--rebuild-motion-cache", action="store_true", help="Rebuild manifest-generated motion pkl cache.")
    add_bool_arg(parser, "--headless", True, "Run MuJoCo in headless mode.")
    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save side-by-side expert/policy MP4.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--motion-list", type=int, nargs="+", default=[20])
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rollout/video frames for quick previews.")
    parser.add_argument("--log-every-steps", type=int, default=100, help="Print rollout/render progress every N steps; 0 disables periodic logs.")
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    parser.add_argument(
        "--tracking-seq-length",
        type=int,
        default=None,
        help="Future reference frames averaged into each tracking latent; defaults to the checkpoint model seq_length.",
    )
    add_bool_arg(
        parser,
        "--lie-down-init",
        False,
        "Initialize the policy robot on its side while keeping the selected reference trajectory.",
    )
    parser.add_argument(
        "--lie-down-side",
        choices=("left", "right"),
        default="left",
        help="Side used by --lie-down-init.",
    )
    parser.add_argument(
        "--downward-force-n",
        type=float,
        default=0.0,
        help="World-frame downward force applied to the robot root; zero disables the pulse.",
    )
    parser.add_argument("--force-start-s", type=float, default=2.4, help="Simulation time at which the force starts.")
    parser.add_argument("--force-duration-s", type=float, default=0.25, help="Duration of the force pulse.")
    add_bool_arg(parser, "--export-onnx", True, "Export ONNX next to the checkpoint before inference.")
    args = parser.parse_args()
    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        if args.dataset is None:
            parser.error("--dataset is required when --data-manifest is provided")
        manifest_robot_config = prepare_manifest_robot_config_path(args.data_manifest)
        args.data_path = Path(
            prepare_manifest_dataset_path(
                args.data_manifest,
                args.dataset,
                split="inference",
                rebuild_cache=bool(args.rebuild_motion_cache),
            )
        )
    args.robot_config = _resolve_tracking_robot_config(args.robot_config, manifest_robot_config)
    return args


def main() -> None:
    args = parse_args()
    run_tracking_inference(
        model_folder=args.model_folder,
        output_dir=args.output_dir,
        data_path=args.data_path,
        robot_config=args.robot_config,
        headless=args.headless,
        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        motion_list=args.motion_list,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        fps=args.fps,
        max_steps=args.max_steps,
        log_every_steps=args.log_every_steps,
        max_episode_length_s=args.max_episode_length_s,
        tracking_seq_length=args.tracking_seq_length,
        lie_down_init=args.lie_down_init,
        lie_down_side=args.lie_down_side,
        downward_force_n=args.downward_force_n,
        force_start_s=args.force_start_s,
        force_duration_s=args.force_duration_s,
        export_onnx=args.export_onnx,
    )


if __name__ == "__main__":
    main()
