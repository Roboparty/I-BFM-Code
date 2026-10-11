"""Interactive MuJoCo test for the body-only LAFAN T-pose latent.

This deliberately uses the 98D object-aware actor schema with all 34 object/task
channels masked by ``object_present=0``.  It is separate from Carry goal
sim2sim, whose normal mode assumes a live box state.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import torch

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.carry_goal_sim2sim import _patch_mode15_joint_physics
from humanoidverse.envs.carry_observations import (
    build_object_context,
    compute_carry_object_observation,
    compute_hand_contact_points,
    compute_hand_spatial_observation,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import quat_rotate_inverse


CONTROL_DT = 0.02
SIM_DT = 0.005
HISTORY_LENGTH = 4
OBJECT_CONTEXT_DIM = 34


def _np(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _joint_addresses(model: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([model.joint(name).id for name in names], dtype=np.int32)
    return model.jnt_qposadr[ids].astype(np.int64), model.jnt_dofadr[ids].astype(np.int64)


def _actuator_indices(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    result: list[int] = []
    for name in names:
        jid = model.joint(name).id
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == jid)
        if matches.size != 1:
            raise ValueError(f"Expected one actuator for {name}, found {matches.tolist()}")
        result.append(int(matches[0]))
    return np.asarray(result, dtype=np.int64)


def _patch_model(robot_xml: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.option.timestep = SIM_DT
    spec.worldbody.add_geom(
        name="tpose_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, 0.0),
        size=(20.0, 20.0, 0.05),
        rgba=(0.42, 0.44, 0.47, 1.0),
        friction=(0.9, 0.01, 0.001),
    )
    spec.worldbody.add_light(name="tpose_key", pos=(1.0, -2.0, 4.0), dir=(-0.2, 0.4, -1.0))
    model = spec.compile()
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 720)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 720)
    return model


def _load_latent(path: Path, device: torch.device) -> torch.Tensor:
    value = joblib.load(path)
    if isinstance(value, dict) and set(value) == {"z"}:
        value = value["z"]
    latent = np.asarray(value, dtype=np.float32).reshape(-1)
    if latent.shape != (256,) or not np.all(np.isfinite(latent)):
        raise ValueError(f"Expected one finite 256D latent, got {latent.shape}")
    latent = latent * (16.0 / max(float(np.linalg.norm(latent)), 1e-8))
    return torch.as_tensor(latent, device=device).reshape(1, -1)


def _load_latent_sequence(path: Path, device: torch.device) -> torch.Tensor:
    """Load a tracking-inference ``zs_<motion>.pkl`` sequence."""

    value = joblib.load(path)
    if isinstance(value, dict) and set(value) == {"z"}:
        value = value["z"]
    latent = np.asarray(value, dtype=np.float32)
    if latent.ndim == 1:
        latent = latent.reshape(1, -1)
    if latent.ndim != 2 or latent.shape[1] != 256 or latent.shape[0] == 0:
        raise ValueError(f"Expected a finite [T, 256] latent sequence, got {latent.shape}")
    if not np.all(np.isfinite(latent)):
        raise ValueError(f"Latent sequence contains non-finite values: {path}")
    norms = np.linalg.norm(latent, axis=1, keepdims=True)
    latent = latent * (16.0 / np.maximum(norms, 1.0e-8))
    return torch.as_tensor(latent, device=device)


def _load_reference(path: Path, motion_id: int, frame: int) -> tuple[dict[str, Any], int]:
    motions = joblib.load(path)
    keys = list(motions)
    if not 0 <= motion_id < len(keys):
        raise IndexError(f"motion_id={motion_id} outside [0, {len(keys)})")
    motion = motions[keys[motion_id]]
    dof = _np(motion["dof"])[frame]
    root = _np(motion["root_trans_offset"])[frame].copy()
    quat = _np(motion["root_rot"])[frame].copy()
    if quat.shape != (4,):
        raise ValueError(f"Expected root_rot shape [4], got {quat.shape}")
    return {"dof": dof, "root": root, "quat_xyzw": quat, "key": keys[motion_id]}, frame


def _build_observation(
    *,
    policy: Any,
    device: torch.device,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    qpos_ids: np.ndarray,
    qvel_ids: np.ndarray,
    defaults: np.ndarray,
    action: torch.Tensor,
    history: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    dof_pos = torch.as_tensor(data.qpos[qpos_ids], device=device, dtype=torch.float32).reshape(1, -1)
    dof_vel = torch.as_tensor(data.qvel[qvel_ids], device=device, dtype=torch.float32).reshape(1, -1)
    base_pos = torch.as_tensor(data.qpos[:3], device=device, dtype=torch.float32).reshape(1, 3)
    # MuJoCo stores free-joint and body quaternions as wxyz, while the
    # project's torch quaternion utilities use xyzw (w_last=True).
    base_quat = torch.as_tensor(
        np.roll(data.qpos[3:7], -1), device=device, dtype=torch.float32
    ).reshape(1, 4)
    base_ang_world = torch.as_tensor(data.qvel[3:6], device=device, dtype=torch.float32).reshape(1, 3)
    gravity = quat_rotate_inverse(base_quat, torch.tensor([[0.0, 0.0, -1.0]], device=device), w_last=True)
    base_ang = 0.25 * quat_rotate_inverse(base_quat, base_ang_world, w_last=True)

    # These inputs are intentionally dummy: object_present=0 masks all 34
    # object/contact/spatial/task channels before they reach the actor.
    object_pos = torch.zeros((1, 3), device=device)
    object_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    object_obs = compute_carry_object_observation(
        base_pos, base_quat, object_pos, object_quat,
        torch.zeros((1, 3), device=device), torch.zeros((1, 3), device=device), object_pos,
    )
    body_ids = [model.body(name).id for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    hand_pos = torch.as_tensor(np.asarray([data.xpos[i] for i in body_ids]), device=device, dtype=torch.float32).reshape(1, 2, 3)
    hand_quat = torch.as_tensor(
        np.roll(np.asarray([data.xquat[i] for i in body_ids]), -1, axis=-1),
        device=device,
        dtype=torch.float32,
    ).reshape(1, 2, 4)
    hand_spatial = compute_hand_spatial_observation(
        base_pos, base_quat,
        compute_hand_contact_points(hand_pos, hand_quat, torch.tensor([[0.11, -0.01, 0.0], [0.11, 0.01, 0.0]], device=device)),
        object_pos,
    )
    context = build_object_context(
        object_obs,
        torch.zeros((1, 1), device=device),
        torch.zeros((1, 2), device=device),
        hand_spatial,
        task_id=torch.zeros((1, 1), device=device),
        include_hand_spatial=True,
        observe_target_delta=True,
        observe_object_present=True,
        observe_task_id=True,
    )
    if context.shape[-1] != OBJECT_CONTEXT_DIM or torch.count_nonzero(context).item() != 0:
        raise RuntimeError(f"body-only context is not masked: shape={context.shape}, nonzero={torch.count_nonzero(context)}")
    state = torch.cat((dof_pos - torch.as_tensor(defaults, device=device).reshape(1, -1), dof_vel, gravity, base_ang, context), dim=-1)
    raw = {"actions": action, "base_ang_vel": base_ang, "dof_pos": state[:, :29], "dof_vel": dof_vel, "projected_gravity": gravity}
    history_actor = torch.cat([history[key].reshape(1, -1) for key in sorted(history)], dim=-1)
    for key, value in raw.items():
        history[key][:, 1:] = history[key][:, :-1].clone()
        history[key][:, 0] = value
    obs = {"state": state, "last_action": action, "history_actor": history_actor}
    if tuple(state.shape) != (1, int(policy.obs_space["state"].shape[0])):
        raise RuntimeError(f"unexpected state shape {state.shape}")
    return obs, raw


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    robot = load_robot_training_spec(args.robot_config)
    joints = list(robot.robot.control_joint_names)
    defaults = np.asarray([robot.default_joint_angles[name] for name in joints], dtype=np.float64)
    model = _patch_model(Path(robot.robot.xml_path))
    _patch_mode15_joint_physics(model, joints)
    data = mujoco.MjData(model)
    qpos_ids, qvel_ids = _joint_addresses(model, joints)
    act_ids = _actuator_indices(model, joints)
    ref, _ = _load_reference(args.motion_cache, args.motion_id, args.frame)
    data.qpos[:3] = ref["root"]
    data.qpos[3:7] = np.roll(ref["quat_xyzw"], 1)
    data.qpos[qpos_ids] = ref["dof"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    device = torch.device(args.device)
    policy = load_model_from_checkpoint_dir(args.model_folder / "checkpoint", device=device.type, strict=False)
    # The checkpoint is float32 and CPU BatchNorm rejects mixed input/running-stat
    # dtypes.  Keep this offline viewer deterministic across CPU and CUDA.
    policy.to(device).float().eval()
    z = _load_latent(args.z_latent, device)
    tracking_z = (
        _load_latent_sequence(args.tracking_latent, device)
        if args.tracking_latent is not None
        else None
    )
    if int(policy.obs_space["state"].shape[0]) != 98:
        raise ValueError("This viewer expects the 98D hand-spatial Carry actor")
    history = {
        "actions": torch.zeros(1, HISTORY_LENGTH, 29, device=device),
        "base_ang_vel": torch.zeros(1, HISTORY_LENGTH, 3, device=device),
        "dof_pos": torch.zeros(1, HISTORY_LENGTH, 29, device=device),
        "dof_vel": torch.zeros(1, HISTORY_LENGTH, 29, device=device),
        "projected_gravity": torch.zeros(1, HISTORY_LENGTH, 3, device=device),
    }
    action = torch.zeros((1, 29), device=device)
    interaction = {"started": bool(args.start_immediately)}
    interaction_lock = threading.Lock()

    def key_callback(keycode: int) -> None:
        key = "SPACE" if keycode == 32 else (chr(keycode).upper() if 0 <= keycode < 256 else "")
        with interaction_lock:
            if key == args.start_key.upper():
                interaction["started"] = True

    kp = np.asarray([max(float(robot.stiffness.get("hip_pitch", 99.09843)), 1.0)] * 29)
    kd = np.asarray([1.0] * 29)
    # Match the training robot's per-group gains and action-to-target mapping.
    def match(values: dict[str, float], name: str, fallback: float) -> float:
        hit = [(k, float(v)) for k, v in values.items() if k in name]
        return max(hit, key=lambda item: len(item[0]))[1] if hit else fallback
    kp = np.asarray([match(robot.stiffness, n, 40.0) for n in joints])
    kd = np.asarray([match(robot.damping, n, 2.0) for n in joints])
    effort = np.asarray(robot.effort_limits, dtype=np.float64)
    scales = np.asarray([robot.action_scale_overrides.get(n, robot.action_scale) for n in joints], dtype=np.float64)
    target_scale = scales * effort / kp
    viewer = (
        mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        if args.viewer
        else None
    )
    try:
        steps = max(1, round(args.episode_seconds / CONTROL_DT))
        tracking_step = 0
        printed_start = bool(args.start_immediately)
        if tracking_z is not None and not printed_start:
            print(
                f"[interactive] paused at motion {args.motion_id} frame {args.frame}; "
                f"press {args.start_key} to start motion tracking",
                flush=True,
            )
        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                break
            with interaction_lock:
                started = bool(interaction["started"])
            if started and not printed_start:
                print(
                    f"[interactive] tracking started; sequence length={tracking_z.shape[0] if tracking_z is not None else 0}",
                    flush=True,
                )
                printed_start = True
            obs, _ = _build_observation(
                policy=policy, device=device, model=model, data=data, joint_names=joints,
                qpos_ids=qpos_ids, qvel_ids=qvel_ids, defaults=defaults, action=action, history=history,
            )
            obs = {key: value.to(device=device, dtype=torch.float32) for key, value in obs.items()}
            active_z = z
            if started and tracking_z is not None:
                if args.loop_tracking:
                    tracking_index = tracking_step % tracking_z.shape[0]
                else:
                    tracking_index = min(tracking_step, tracking_z.shape[0] - 1)
                active_z = tracking_z[tracking_index].reshape(1, -1)
                tracking_step += 1
            active_z = active_z.to(device=device, dtype=torch.float32)
            raw_action = policy.act(obs, active_z, mean=True)[0].cpu().numpy()
            current_action = np.clip(
                raw_action * float(robot.normalize_action_to),
                -float(robot.action_clip_value),
                float(robot.action_clip_value),
            )
            action = torch.as_tensor(current_action, device=device, dtype=torch.float32).reshape(1, -1)
            target = defaults + current_action * target_scale
            target = np.clip(target, np.asarray([robot.robot.joint_ranges[n][0] for n in joints]), np.asarray([robot.robot.joint_ranges[n][1] for n in joints]))
            for _ in range(round(CONTROL_DT / SIM_DT)):
                data.ctrl[act_ids] = np.clip(kp * (target - data.qpos[qpos_ids]) - kd * data.qvel[qvel_ids], -effort, effort)
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.cam.lookat[:] = data.xpos[model.body(robot.robot.base_body).id]
                viewer.cam.lookat[2] = 0.75
                viewer.sync()
                time.sleep(max(0.0, CONTROL_DT))
    finally:
        if viewer is not None:
            viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--z-latent", type=Path, required=True)
    parser.add_argument(
        "--tracking-latent",
        type=Path,
        default=None,
        help="Optional [T,256] tracking latent sequence activated by --start-key.",
    )
    parser.add_argument("--motion-cache", type=Path, required=True)
    parser.add_argument(
        "--motion-id",
        type=int,
        default=0,
        help="Reference motion used for the paused initial pose (default: motion 0).",
    )
    parser.add_argument(
        "--frame",
        "--initial-frame",
        dest="frame",
        type=int,
        default=0,
        help="Reference frame used for the paused initial pose (default: frame 0).",
    )
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start-key", default="SPACE", help="Key that starts tracking (default: SPACE).")
    parser.add_argument(
        "--start-immediately",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start tracking immediately instead of waiting for --start-key.",
    )
    parser.add_argument(
        "--loop-tracking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Loop the tracking latent sequence after its final frame.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
