"""Expert trajectory loading helpers shared by MJLab training."""

import numpy as np
import torch

from humanoidverse.envs.carry_observations import (
    build_object_context,
    compute_carry_object_observation,
    compute_hand_contact_points,
    compute_hand_spatial_observation,
    empty_object_context,
    infer_carry_phase_contact,
    select_reference_hand_box_contact,
)
from humanoidverse.envs.env_utils.history_handler import HistoryHandler as HVHistoryHandler
from humanoidverse.envs.motion_observations import compute_humanoid_observations_max
from humanoidverse.utils.reference_observations import reference_base_ang_vel
from humanoidverse.utils.torch_utils import quat_rotate_inverse

from ..buffers.trajectory import TrajectoryDictBuffer


def apply_source_mixed_expert_priorities(expert_buffer: TrajectoryDictBuffer, motion_lib) -> None:
    """Match expert sampling mass to the manifest source weights."""

    source_priorities = motion_lib._sampling_prob.to(device=expert_buffer.priorities.device)
    if source_priorities.shape != expert_buffer.priorities.shape:
        raise ValueError(
            "Expert motion priorities must match the loaded trajectories: "
            f"priorities={tuple(source_priorities.shape)} trajectories={tuple(expert_buffer.priorities.shape)}"
        )
    expert_buffer.priorities.copy_(source_priorities / source_priorities.sum())


def load_expert_trajectories_from_motion_lib(env, agent_cfg, device="cpu", add_history_noaction: bool = False):
    """Load expert trajectories directly from an environment motion library."""
    env._motion_lib.load_motions_for_training()
    episodes = []
    file_names = []
    history_handler = HVHistoryHandler(1, env.config.obs.obs_auxiliary, env.config.obs.obs_dims, device)
    history_config = env.config.obs.obs_auxiliary["history_actor"]
    for i in range(env._motion_lib._num_unique_motions):
        motion_times = torch.arange(int(np.ceil((env._motion_lib._motion_lengths[i] / env.dt).cpu()))).to(env.device) * env.dt
        motion_id = torch.tensor([i]).to(env.device).repeat(motion_times.shape[0])
        motion_res = env._motion_lib.get_motion_state(motion_id, motion_times)
        file_names.append(env._motion_lib._motion_data_keys[i])

        ref_body_pos = motion_res["rg_pos_t"]
        ref_body_rots = motion_res["rg_rot_t"]
        ref_body_vels = motion_res["body_vel_t"]
        ref_body_angular_vels = motion_res["body_ang_vel_t"]

        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
        )
        max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

        base_quat = ref_body_rots[:, 0]
        ref_dof_pos = motion_res["dof_pos"] - env.default_dof_pos[0]
        ref_dof_vel = motion_res["dof_vel"]
        ref_ang_vel = reference_base_ang_vel(env, base_quat, ref_body_angular_vels[:, 0])
        projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True)
        bogus_actions = ref_dof_pos * 0

        state = torch.cat(
            [
                ref_dof_pos,
                ref_dof_vel,
                projected_gravity,
                ref_ang_vel,
            ],
            dim=-1,
        )
        if env.carry_box_enabled:
            object_obs = compute_carry_object_observation(
                motion_res["root_pos"],
                motion_res["root_rot"],
                motion_res["object_pos"],
                motion_res["object_quat"],
                motion_res["object_lin_vel"],
                motion_res["object_ang_vel"],
                motion_res["object_target_pos"],
            )
            phase_contact = infer_carry_phase_contact(
                motion_res["object_pos"],
                motion_res["object_quat"],
                motion_res["object_present"],
                env.box_half_size,
                env.box_geom_pos,
                env.box_geom_quat_xyzw,
                ground_surface_height=float(env.carry_box_cfg.get("ground_surface_height", 0.0)),
                lift_clearance=float(env.carry_box_cfg.get("phase_lift_clearance", 0.005)),
            )
            hand_box_contact = select_reference_hand_box_contact(
                phase_contact,
                motion_res.get("hand_box_contact"),
                motion_res.get("hand_box_contact_available"),
            )
            hand_contact_points = compute_hand_contact_points(
                ref_body_pos[:, env.hand_indices],
                ref_body_rots[:, env.hand_indices],
                env.hand_contact_offsets,
            )
            hand_spatial_observation = compute_hand_spatial_observation(
                motion_res["root_pos"],
                motion_res["root_rot"],
                hand_contact_points,
                motion_res["object_pos"],
            )
            object_context = build_object_context(
                object_obs,
                motion_res["object_present"],
                hand_box_contact,
                hand_spatial_observation,
                include_hand_spatial=bool(env.carry_box_cfg.get("observe_hand_spatial", False)),
                observe_target_delta=bool(env.carry_box_cfg.get("observe_target_delta", True)),
                observe_object_present=bool(env.carry_box_cfg.get("observe_object_present", True)),
                observe_task_id=bool(env.carry_box_cfg.get("observe_task_id", True)),
            )
        else:
            object_context = empty_object_context(state, include_hand_spatial=False)
        state = torch.cat([state, object_context], dim=-1)
        max_local_self_obs = torch.cat([max_local_self_obs, object_context], dim=-1)

        data = {
            "base_ang_vel": ref_ang_vel,
            "projected_gravity": projected_gravity,
            "dof_pos": ref_dof_pos,
            "dof_vel": ref_dof_vel,
        }

        if add_history_noaction:
            history_handler.reset([0])
            history_actor = []
            for ii in range(state.shape[0]):
                history_tensors = []
                for key in sorted(history_config.keys()):
                    if key not in ["action", "actions"]:
                        history_length = history_config[key]
                        history_tensor = history_handler.query(key)[:, :history_length]
                        history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
                        history_tensors.append(history_tensor)
                history_tensors = torch.cat(history_tensors, dim=1)
                history_actor.append(history_tensors)

                for key in history_handler.history.keys():
                    if key not in ["action", "actions"]:
                        history_handler.add(key, data[key][ii][None, ...])
            history_actor = torch.stack(history_actor, dim=0).squeeze(1)

        curr_motion_len = state.shape[0]
        truncated = torch.zeros(curr_motion_len, dtype=bool).to(env.device)
        truncated[-1] = True

        assert state.shape[0] == curr_motion_len, f"{env._motion_lib._motion_data_keys[i]}: {state.shape[0]} vs {curr_motion_len}"
        assert max_local_self_obs.shape[0] == curr_motion_len, (
            f"{env._motion_lib._motion_data_keys[i]}: {max_local_self_obs.shape[0]} vs {curr_motion_len}"
        )
        assert bogus_actions.shape[0] == curr_motion_len, (
            f"{env._motion_lib._motion_data_keys[i]}: {bogus_actions.shape[0]} vs {curr_motion_len}"
        )
        assert truncated.shape[0] == curr_motion_len, f"{env._motion_lib._motion_data_keys[i]}: {truncated.shape[0]} vs {curr_motion_len}"
        if add_history_noaction:
            assert history_actor.shape[0] == curr_motion_len, (
                f"{env._motion_lib._motion_data_keys[i]}: {history_actor.shape[0]} vs {curr_motion_len}"
            )

        ep = {
            "observation": {
                "state": state,
                "last_action": bogus_actions,
                "privileged_state": max_local_self_obs,
            },
            "terminated": torch.zeros(curr_motion_len, dtype=bool).to(env.device),
            "truncated": truncated,
            "motion_id": torch.ones(curr_motion_len, dtype=torch.long) * i,
        }
        if add_history_noaction:
            ep["observation"]["history_noaction"] = history_actor
        episodes.append(ep)

    expert_buffer = TrajectoryDictBuffer(
        episodes=episodes,
        seq_length=agent_cfg.model.seq_length,
        device=device,
    )
    apply_source_mixed_expert_priorities(expert_buffer, env._motion_lib)

    assert expert_buffer.storage["observation"]["state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["observation"]["last_action"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["observation"]["privileged_state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["terminated"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["motion_id"].shape[0] == expert_buffer.storage["truncated"].shape[0]

    expert_buffer.file_names = file_names
    return expert_buffer
