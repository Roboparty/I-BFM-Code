
import collections
import copy
import dataclasses
import functools
import numbers
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, Mapping

import numpy as np
import ot
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humanoidverse.envs.carry_observations import (
    build_object_context,
    compute_carry_object_observation,
    compute_hand_contact_points,
    compute_hand_spatial_observation,
    empty_object_context,
    infer_carry_phase_contact,
    select_reference_hand_box_contact,
)
from humanoidverse.utils.reference_observations import reference_base_ang_vel

from ..buffers.trajectory import TrajectoryDictBufferMultiDim
from ..envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
from .base import BaseEvalConfig, extract_model


def get_next(field: str, data: Any):
    if "next" in data and field in data["next"]:
        return data["next"][field]
    elif f"next_{field}" in data:
        return data[f"next_{field}"]
    else:
        raise ValueError(f"No next of {field} found in data.")


@dataclasses.dataclass
class Episode:
    storage: Dict | None = None

    def initialise(self, observation: np.ndarray | Dict[str, np.ndarray], info: Dict) -> None:
        if self.storage is None:
            self.storage = defaultdict(list)
        if isinstance(observation, Mapping):
            self.storage["observation"] = {k: [copy.deepcopy(v)] for k, v in observation.items()}
        else:
            self.storage["observation"].append(copy.deepcopy(observation))
        self.storage["info"] = {
            k: [copy.deepcopy(v)] for k, v in info.items() if not k.startswith("_") and k not in ["final_observation", "final_info"]
        }

    def add(
        self,
        observation: np.ndarray | Dict[str, np.ndarray],
        reward: np.ndarray,
        action: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: Dict,
    ) -> None:
        if isinstance(observation, Mapping):
            for k, v in observation.items():
                self.storage["observation"][k].append(copy.deepcopy(v))
        else:
            self.storage["observation"].append(copy.deepcopy(observation))
        for k, v in info.items():
            if not k.startswith("_") and k not in ["final_observation", "final_info"]:
                if k in self.storage["info"]:
                    # this is a quick fix but not robust fix to the problem that certain environments may return keys that are different from the one in the first step
                    # We may want to find a different solution to deal with this problem
                    self.storage["info"][k].append(copy.deepcopy(v))
                else:
                    self.storage["info"][k] = [copy.deepcopy(v)]
        self.storage["reward"].append(copy.deepcopy(reward))
        self.storage["action"].append(copy.deepcopy(action))
        self.storage["terminated"].append(copy.deepcopy(terminated))
        self.storage["truncated"].append(copy.deepcopy(truncated))

    def get(self) -> Dict[str, np.ndarray]:
        output = {}
        for k, v in self.storage.items():
            if k in ["observation", "info"]:
                if isinstance(v, Mapping):
                    output[k] = {}
                    for k2, v2 in v.items():
                        output[k][k2] = np.array(v2)
                else:
                    output[k] = np.array(v)
            else:
                output[k] = np.array(v)
        return output

xpos_bodies = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    # "left_wrist_roll_link",
    # "left_wrist_pitch_link",
    # "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    # "right_wrist_roll_link",
    # "right_wrist_pitch_link",
    # "right_wrist_yaw_link",
]

def get_backward_observation(env, motion_id, include_last_action, velocity_multiplier: float = 1.0) -> torch.Tensor:
    import numpy as np

    from humanoidverse.envs.motion_observations import (
        compute_humanoid_observations_max,
        compute_humanoid_observations_max_with_contact,
    )
    from humanoidverse.utils.torch_utils import quat_rotate_inverse

    if not env.config.obs.use_obs_filter:
        raise ValueError("MJLab tracking evaluation requires obs.use_obs_filter=True")

    motion_times = torch.arange(int(np.ceil((env._motion_lib._motion_lengths[motion_id] / env.dt).cpu()))).to(env.device) * env.dt
    # motion_times = torch.arange(0, env._motion_lib._motion_num_frames.item() * frame_interval, frame_interval, device=env.device)

    # get blend motion state
    motion_state = env._motion_lib.get_motion_state(motion_id, motion_times)

    ref_body_pos = motion_state["rg_pos_t"]
    ref_body_rots = motion_state["rg_rot_t"]
    ref_body_vels = motion_state["body_vel_t"] * velocity_multiplier
    ref_body_angular_vels = motion_state["body_ang_vel_t"] * velocity_multiplier
    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
    ref_dof_vel = motion_state["dof_vel"] * velocity_multiplier

    # npz_path = "dataprocess/push_door-3steps-no_door_amass-robot_only-processed/motion.npz"
    # data_npz = np.load(npz_path)
    # data_dict = {key: data_npz[key] for key in data_npz.files}

    def process_body_data(data_dict, device="cuda"):
        def extract_and_concat_vec3(arr: torch.Tensor) -> torch.Tensor:
            arr = arr.float()
            # arr: (T, N, 3)
            return torch.cat(
                [
                    arr[:, 0:13],
                    torch.zeros_like(arr[:, 14:16]),  # 两个补 0
                    arr[:, 13:18],
                    arr[:, 21:25],
                    arr[:, 20:21],
                    arr[:, 27:28],
                    arr[:, 12:13],
                ],
                dim=1,
            )

        def extract_and_concat_quat_torch(arr: torch.Tensor) -> torch.Tensor:
            arr = arr.float()
            # arr: (T, N, 4), wxyz
            seg1 = arr[:, 0:13]
            seg2 = torch.tensor([1, 0, 0, 0], dtype=arr.dtype, device=arr.device).view(1, 1, 4).repeat(arr.shape[0], 2, 1)
            seg3 = arr[:, 13:18]
            seg4 = arr[:, 21:25]
            seg7 = arr[:, 12:13]
            seg5 = arr[:, 20:21]
            seg6 = arr[:, 27:28]
            quat_wxyz = torch.cat([seg1, seg2, seg3, seg4, seg5, seg6, seg7], dim=1)  # (T, 24, 4)
            quat_xyzw = torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], dim=-1)  # (T, 24, 4)
            return quat_xyzw

        # 转为 torch tensor 并送到 GPU
        body_pos = torch.from_numpy(data_dict["body_pos_w"]).to(device)
        body_quat = torch.from_numpy(data_dict["body_quat_w"]).to(device)
        body_lin_vel = torch.from_numpy(data_dict["body_lin_vel_w"]).to(device)
        body_ang_vel = torch.from_numpy(data_dict["body_ang_vel_w"]).to(device)
        joint_pos = torch.from_numpy(data_dict["joint_pos"]).to(device)[:, 1:]
        joint_vel = torch.from_numpy(data_dict["joint_vel"]).to(device)[:, 1:]

        # 处理
        ref_body_pos = extract_and_concat_vec3(body_pos)
        ref_body_vels = extract_and_concat_vec3(body_lin_vel)
        ref_body_angular_vels = extract_and_concat_vec3(body_ang_vel)
        ref_body_rots = extract_and_concat_quat_torch(body_quat)
        ref_dof_pos = joint_pos.float() - env.default_dof_pos[0]
        ref_dof_vel = joint_vel.float()

        return ref_body_pos, ref_body_rots, ref_body_vels, ref_body_angular_vels, ref_dof_pos, ref_dof_vel

    # import ipdb; ipdb.set_trace()
    # ref_body_pos, ref_body_rots, ref_body_vels, ref_body_angular_vels, ref_dof_pos, ref_dof_vel = process_body_data(data_dict)

    # construct observation
    if env.use_contact_in_obs_max:
        contact_binary = env.foot_contact_detect(ref_body_pos, ref_body_vels)
        obs_dict = compute_humanoid_observations_max_with_contact(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
            contact_binary=contact_binary,
        )
    else:
        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
        )
    max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

    if env.config.obs.use_obs_filter:
        base_quat = ref_body_rots[:, 0]  # root orientation
        # ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
        # ref_dof_vel = motion_state["dof_vel"]
        ref_ang_vel = reference_base_ang_vel(env, base_quat, ref_body_angular_vels[:, 0])
        projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True)
        bogus_actions = torch.zeros_like(ref_dof_pos)

        bogus_history_actor = torch.cat([bogus_actions, ref_ang_vel, ref_dof_pos, ref_dof_vel, projected_gravity], dim=-1).repeat(1, 4)
        # obs = torch.cat([bogus_actions, ref_ang_vel, ref_dof_pos, ref_dof_vel, bogus_history_actor, max_local_self_obs, projected_gravity], dim=-1)
        ref_dict = {
            "actions": bogus_actions,
            "ref_ang_vel": ref_ang_vel,
            "ref_dof_pos": ref_dof_pos,
            "dof_pos": motion_state["dof_pos"],
            "ref_dof_vel": ref_dof_vel,
            "dof_vel": motion_state["dof_vel"],
            "fake_history": bogus_history_actor,
            "max_local_self_obs": max_local_self_obs,
            "projected_gravity": projected_gravity,
            "ref_body_pos": ref_body_pos,
            "ref_body_rots": ref_body_rots,
            "ref_body_vels": ref_body_vels,
            "ref_body_angular_vels": ref_body_angular_vels,
            "root_pos": motion_state["root_pos"],
            "root_rot": motion_state["root_rot"],
            "root_vel": motion_state["root_vel"],
            "root_ang_vel": motion_state["root_ang_vel"],
        }
    else:
        # obs = max_local_self_obs
        ref_dict = {
            "max_local_self_obs": max_local_self_obs,
        }

    projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True)

    # TODO ensure this is correct
    g1env_state = torch.cat(
        [
            ref_dof_pos,
            ref_dof_vel,
            projected_gravity,
            ref_ang_vel,
        ],
        dim=-1,
    )
    g1env_last_action = torch.zeros_like(ref_dof_pos)

    g1env_privileged_obs = ref_dict["max_local_self_obs"]

    if env.carry_box_enabled:
        object_obs = compute_carry_object_observation(
            motion_state["root_pos"],
            motion_state["root_rot"],
            motion_state["object_pos"],
            motion_state["object_quat"],
            motion_state["object_lin_vel"],
            motion_state["object_ang_vel"],
            motion_state["object_target_pos"],
        )
        phase_contact = infer_carry_phase_contact(
            motion_state["object_pos"],
            motion_state["object_quat"],
            motion_state["object_present"],
            env.box_half_size,
            env.box_geom_pos,
            env.box_geom_quat_xyzw,
            ground_surface_height=float(env.carry_box_cfg.get("ground_surface_height", 0.0)),
            lift_clearance=float(env.carry_box_cfg.get("phase_lift_clearance", 0.005)),
        )
        hand_box_contact = select_reference_hand_box_contact(
            phase_contact,
            motion_state.get("hand_box_contact"),
            motion_state.get("hand_box_contact_available"),
        )
        hand_contact_points = compute_hand_contact_points(
            ref_body_pos[:, env.hand_indices],
            ref_body_rots[:, env.hand_indices],
            env.hand_contact_offsets,
        )
        hand_spatial_observation = compute_hand_spatial_observation(
            motion_state["root_pos"],
            motion_state["root_rot"],
            hand_contact_points,
            motion_state["object_pos"],
        )
        object_context = build_object_context(
            object_obs,
            motion_state["object_present"],
            hand_box_contact,
            hand_spatial_observation,
            include_hand_spatial=bool(env.carry_box_cfg.get("observe_hand_spatial", False)),
            observe_target_delta=bool(env.carry_box_cfg.get("observe_target_delta", True)),
            observe_object_present=bool(env.carry_box_cfg.get("observe_object_present", True)),
            observe_task_id=bool(env.carry_box_cfg.get("observe_task_id", True)),
        )
        ref_dict.update(
            {
                "object_pos": motion_state["object_pos"],
                "object_quat": motion_state["object_quat"],
                "object_lin_vel": motion_state["object_lin_vel"],
                "object_ang_vel": motion_state["object_ang_vel"],
                "object_target_pos": motion_state["object_target_pos"],
                "object_target_quat": motion_state["object_target_quat"],
                "object_present": motion_state["object_present"],
                "carry_evaluation_eligible": motion_state["carry_evaluation_eligible"],
            }
        )
    else:
        object_context = empty_object_context(g1env_state, include_hand_spatial=False)
    g1env_state = torch.cat([g1env_state, object_context], dim=-1)
    g1env_privileged_obs = torch.cat([g1env_privileged_obs, object_context], dim=-1)

    g1env_obs = {
        "state": g1env_state,
        "privileged_state": g1env_privileged_obs,
    }

    if include_last_action:
        g1env_obs["last_action"] = g1env_last_action

    return g1env_obs, ref_dict


class HumanoidVerseMjlabTrackingEvaluationConfig(BaseEvalConfig):
    name_in_logs: str = "humanoidverse_tracking_eval"
    env: HumanoidVerseMjlabConfig | None = None
    # Used if above env is provided
    num_envs: int = 1024

    n_episodes_per_motion: int = 1
    include_results_from_all_envs: bool = False  # If True, include results from all num_envs envs, not just first num_motions

    disable_tqdm: bool = True

    def build(self):
        return HumanoidVerseMjlabTrackingEvaluation(self)


class HumanoidVerseMjlabTrackingEvaluation:
    def __init__(self, config: HumanoidVerseMjlabTrackingEvaluationConfig):
        self.cfg = config

    def summarize_and_log(self, metrics: Dict[str, Any], *, timestep: int, logger) -> Dict[str, Any]:
        """Aggregate and optionally log an already-complete metric dictionary."""

        aggregate = collections.defaultdict(list)
        wandb_dict = {}
        for metr in metrics.values():
            for key, value in metr.items():
                if isinstance(value, numbers.Number) and np.isfinite(value):
                    aggregate[key].append(value)
        for key, values in aggregate.items():
            wandb_dict[key] = np.mean(values)
            wandb_dict[f"{key}#std"] = np.std(values)

        if logger is not None:
            for motion_name, metric in metrics.items():
                metric["motion_name"] = motion_name
                metric["timestep"] = timestep
                logger.log(metric)
        return wandb_dict

    def run(
        self,
        *,
        timestep,
        agent_or_model,
        logger,
        env: Any | None = None,
        motion_ids: list[int] | None = None,
        finalize: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        if env is None:
            if self.cfg.env is None:
                raise ValueError("Either env or cfg.env must be provided")
            env, _ = self.cfg.env.build(num_envs=self.cfg.num_envs)
        else:
            if self.cfg.env is not None:
                raise ValueError("Both env and cfg.env are provided, please provide only one (which one you want to evaluate with?)")
        # Do not call set_is_evaluating(): it reloads/reset motions and changes the
        # evaluation setup. The flag alone preserves terminal states for metrics.
        env._env.is_evaluating = True

        core_env = env._env
        motion_lib = core_env._motion_lib
        shared_training_env = self.cfg.env is None
        total_motions = int(motion_lib._num_unique_motions)
        global_motion_ids = list(range(total_motions)) if motion_ids is None else list(motion_ids)
        local_to_global_motion_id: dict[int, int] | None = None
        motions_preloaded = False

        if motion_ids is not None:
            if not global_motion_ids:
                raise ValueError("Distributed tracking-evaluation shard cannot be empty")
            expected = list(range(global_motion_ids[0], global_motion_ids[0] + len(global_motion_ids)))
            if global_motion_ids != expected:
                raise ValueError("Tracking-evaluation motion shards must be contiguous")
            if global_motion_ids[0] < 0 or global_motion_ids[-1] >= total_motions:
                raise ValueError(
                    f"Tracking-evaluation shard {global_motion_ids[0]}:{global_motion_ids[-1] + 1} "
                    f"is outside the {total_motions} available motions"
                )

            # Load only this rank's contiguous shard.  The motion library uses
            # local indices after loading, so retain an explicit local->global
            # map for CSV output and replay-buffer prioritization.
            motion_lib.all_motions_loaded = False
            motion_lib.load_motions(
                random_sample=False,
                num_motions_to_load=len(global_motion_ids),
                start_idx=global_motion_ids[0],
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            local_motion_ids = list(range(len(global_motion_ids)))
            local_to_global_motion_id = dict(enumerate(global_motion_ids))
            motions_preloaded = True
        else:
            local_motion_ids = global_motion_ids

        self.motion_ids = global_motion_ids
        n_envs = core_env.num_envs

        try:
            # "Parallelization" is handled on the distributed-rank level.
            metrics = {}
            for repetition_i in range(self.cfg.n_episodes_per_motion):
                run_metrics = {}
                # If we have less envs than motions, run iteratively over chunks.
                for motion_id_chunk_start in range(0, len(local_motion_ids), n_envs):
                    motion_id_chunk = local_motion_ids[motion_id_chunk_start : motion_id_chunk_start + n_envs]
                    motion_chunk_results = _async_tracking_worker(
                        (motion_id_chunk, 0, agent_or_model),
                        env=env,
                        disable_tqdm=self.cfg.disable_tqdm,
                        include_results_from_all_envs=self.cfg.include_results_from_all_envs,
                        motion_id_map=local_to_global_motion_id,
                        motions_preloaded=motions_preloaded,
                    )
                    for key, value in motion_chunk_results.items():
                        assert key not in run_metrics, "Tried to override existing metric"
                        run_metrics[key] = value

                if self.cfg.n_episodes_per_motion == 1:
                    metrics = run_metrics
                else:
                    for key, value in run_metrics.items():
                        metrics[f"{key}_repetition#{repetition_i}"] = value

            wandb_dict = self.summarize_and_log(metrics, timestep=timestep, logger=logger) if finalize else {}
            return metrics, wandb_dict
        finally:
            # A shared training environment must not retain the full evaluation
            # dataset (or a rank-local evaluation shard).  Restore exactly the
            # normal per-rank training working set before the next rollout.
            # set_is_training() calls resample_motion(), which performs the load
            # and reset; do not load here as well or every evaluation allocates
            # two independent 1024-motion batches back-to-back.
            if shared_training_env:
                motion_lib.all_motions_loaded = False
            core_env.set_is_training()
            if shared_training_env and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self) -> None:
        mp_manager = getattr(self, "mp_manager", None)
        if mp_manager is not None:
            mp_manager.shutdown()


def group_assign_motions_to_envs_with_map(motion_ids, num_envs, device=None):
    motion_ids = torch.tensor(motion_ids, device=device)
    num_motions = len(motion_ids)

    env_idxs = list(range(num_envs))
    # Shuffle the environment indices so repeated tracking evaluations do not
    # bind the same motion to the same per-env randomized state every time.
    random.shuffle(env_idxs)
    shuffled_env_idxs = torch.tensor(env_idxs, device=device, dtype=torch.long)

    assigned = motion_ids[shuffled_env_idxs % num_motions]

    motion_to_envs = {}
    for env_id, motion_id in enumerate(assigned.tolist()):
        motion_to_envs.setdefault(motion_id, []).append(env_id)

    return assigned, motion_to_envs


def _initial_carry_target(value: torch.Tensor, width: int, field: str) -> torch.Tensor:
    """Return a static carry target, accepting either [D] or time-varying [T, D] data."""
    target = value if value.ndim == 1 else value[0]
    if target.shape != (width,):
        raise ValueError(f"{field} must resolve to shape ({width},), got {tuple(target.shape)}")
    return target


def _add_env_origin(value: torch.Tensor, env_origin: torch.Tensor) -> torch.Tensor:
    """Convert a local pose/position to world coordinates without mutating the source."""
    if value.shape[-1] < 3 or env_origin.shape != (3,):
        raise ValueError(
            f"Expected value[..., >=3] and env_origin[3], got {tuple(value.shape)} and {tuple(env_origin.shape)}"
        )
    result = value.clone()
    result[..., :3] += env_origin.to(device=value.device, dtype=value.dtype)
    return result


def _async_tracking_worker(
    inputs,
    env: Any,
    disable_tqdm: bool = False,
    include_results_from_all_envs=False,
    motion_id_map: dict[int, int] | None = None,
    motions_preloaded: bool = False,
):
    motion_ids, pos, agent = inputs
    model = extract_model(agent)
    # import ipdb; ipdb.set_trace()
    # motion_ids = torch.tensor(motion_ids, device=env.device)
    core_env = env._env

    if not motions_preloaded and not core_env._motion_lib.all_motions_loaded:
        core_env._motion_lib.all_motions_loaded = True
        core_env._motion_lib.load_motions(random_sample=False, num_motions_to_load=core_env._motion_lib._num_unique_motions, start_idx=0)

    metrics = {}
    num_envs = core_env.num_envs

    # Assign motions to envs in grouped fashion
    assigned_motions, motion_to_envs = group_assign_motions_to_envs_with_map(motion_ids, num_envs)

    # A shared training environment retains motion IDs sampled from its prior
    # working set.  Distributed evaluation replaces that set with a rank-local
    # shard, so stale training IDs can be outside [0, shard_size) and trigger a
    # CUDA indexing failure while reset constructs its first observation.  A
    # target-state reset intentionally does not resample IDs; bind every env to
    # its locally assigned motion explicitly before that reset.
    assigned_motions = assigned_motions.to(device=core_env.device, dtype=torch.long)
    core_env.motion_ids.copy_(assigned_motions)
    core_env.motion_len.copy_(core_env._motion_lib.get_motion_length(assigned_motions))
    core_env.motion_start_times.zero_()
    core_env.motion_dt = core_env._motion_lib._motion_dt

    ctx_dict = {}
    tracking_targets = {}
    # target_xpos_dict = {}
    tracking_joint_pos = {}
    dof_states_list = [None] * num_envs
    root_states_list = [None] * num_envs
    object_states_list = [None] * num_envs if core_env.carry_box_enabled else None
    target_object_pos_list = [None] * num_envs if core_env.carry_box_enabled else None
    target_object_quat_list = [None] * num_envs if core_env.carry_box_enabled else None
    object_present_list = [None] * num_envs if core_env.carry_box_enabled else None
    carry_evaluation_eligible_list = [None] * num_envs if core_env.carry_box_enabled else None
    tracking_object_pos = {}

    # Precompute context and target states per motion
    for m_id in motion_ids:
        tracking_target, tracking_target_dict = get_backward_observation(env._env, m_id, include_last_action=env.include_last_action)
        # first z should try to reach the next state, ie 1:
        z = model.backward_map(tree_map(lambda x: x[1:], tracking_target)).clone()
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        ctx = model.project_z(z)
        ctx_dict[m_id] = ctx
        tracking_targets[m_id] = tree_map(lambda x: x.cpu(), tracking_target)
        tracking_joint_pos[m_id] = tracking_target_dict["dof_pos"].clone()
        if core_env.carry_box_enabled:
            tracking_object_pos[m_id] = tracking_target_dict["object_pos"].clone()
        # import ipdb; ipdb.set_trace()

        ref_body_rots = tracking_target_dict["ref_body_rots"][0, 0]

        ref_root_init_state = torch.cat(
            [
                tracking_target_dict["ref_body_pos"][0, 0],
                ref_body_rots,
                tracking_target_dict["ref_body_vels"][0, 0],
                tracking_target_dict["ref_body_angular_vels"][0, 0],
            ]
        )
        # import ipdb; ipdb.set_trace()
        for env_id in motion_to_envs[m_id]:
            env_origin = core_env.env_origins[env_id]
            dof_init_state = torch.zeros_like(core_env.simulator.dof_state.view(num_envs, -1, 2)[0])
            dof_init_state[..., 0] = tracking_target_dict["dof_pos"][0]
            dof_init_state[..., 1] = tracking_target_dict["ref_dof_vel"][0]
            dof_states_list[env_id] = dof_init_state
            root_states_list[env_id] = _add_env_origin(ref_root_init_state, env_origin)
            if core_env.carry_box_enabled:
                object_states_list[env_id] = _add_env_origin(
                    torch.cat(
                        [
                            tracking_target_dict["object_pos"][0],
                            tracking_target_dict["object_quat"][0],
                            tracking_target_dict["object_lin_vel"][0],
                            tracking_target_dict["object_ang_vel"][0],
                        ]
                    ),
                    env_origin,
                )
                target_object_pos_list[env_id] = _add_env_origin(
                    _initial_carry_target(tracking_target_dict["object_target_pos"], 3, "object_target_pos"),
                    env_origin,
                )
                target_object_quat_list[env_id] = _initial_carry_target(
                    tracking_target_dict["object_target_quat"], 4, "object_target_quat"
                )
                object_present_list[env_id] = tracking_target_dict["object_present"][0]
                carry_evaluation_eligible_list[env_id] = tracking_target_dict[
                    "carry_evaluation_eligible"
                ][0]
            # target_xpos_dict[env_id] = tracking_target_dict["ref_body_pos"][:, : len(xpos_bodies)]

    # this is for environments that are not initialized
    for i in range(num_envs):
        if dof_states_list[i] is None:
            dof_states_list[i] = torch.zeros_like(core_env.simulator.dof_state.view(num_envs, -1, 2)[0])
        if root_states_list[i] is None:
            root_states_list[i] = torch.zeros_like(root_states_list[0])
        if core_env.carry_box_enabled and object_states_list[i] is None:
            object_states_list[i] = object_states_list[0].clone()
            target_object_pos_list[i] = target_object_pos_list[0].clone()
            target_object_quat_list[i] = target_object_quat_list[0].clone()
            object_present_list[i] = object_present_list[0].clone()
            carry_evaluation_eligible_list[i] = carry_evaluation_eligible_list[0].clone()

    target_states = {"dof_states": torch.stack(dof_states_list), "root_states": torch.stack(root_states_list)}
    if core_env.carry_box_enabled:
        target_states.update(
            {
                "object_states": torch.stack(object_states_list),
                "target_object_pos": torch.stack(target_object_pos_list),
                "target_object_quat": torch.stack(target_object_quat_list),
                "object_present": torch.stack(object_present_list),
            }
        )

    env_ids = list(range(num_envs))
    env_ids = torch.tensor(env_ids, dtype=torch.long, device=core_env.device)

    with torch.no_grad():
        observation, info = env.reset(target_states=target_states, to_numpy=False)
        assert torch.allclose(env._env.simulator.dof_pos.clone()[env_ids], target_states["dof_states"][..., 0])

        if env.context_length:
            env._update_history(torch.arange(num_envs, device=core_env.device), observation, None)
            observation["history"] = {"observation": [], "action": []}

        # info = {}
        obs_cpu = tree_map(lambda x: x.cpu() if torch.is_tensor(x) else x, observation)
        info_cpu = tree_map(lambda x: x.cpu() if torch.is_tensor(x) else x, info)

        _episode = Episode()
        ooo = {k: v for k, v in obs_cpu.items() if k != "history"}
        _episode.initialise(ooo, info_cpu)

        xpos_log = [core_env.simulator._rigid_body_pos.reshape(num_envs, -1, 3)]
        joint_pos = [core_env.simulator.dof_state[..., 0]]
        joint_vel = [core_env.simulator.dof_state[..., 1]]
        object_pos_log = [core_env.object_pos.clone()] if core_env.carry_box_enabled else None
        object_vel_log = [core_env.object_lin_vel.clone()] if core_env.carry_box_enabled else None

        max_ctx_len = max([ctx.shape[0] for ctx in ctx_dict.values()])
        for step in tqdm(range(max_ctx_len), desc="Tracking Evaluation", disable=disable_tqdm):
            ctx_batch = []
            for env_id in range(num_envs):
                m_id = assigned_motions[env_id].item()
                ctx = ctx_dict[m_id]
                ctx_t = ctx[step % ctx.shape[0]]
                ctx_batch.append(ctx_t)
            ctx_batch = torch.stack(ctx_batch)
            action = agent.act(observation, ctx_batch, mean=True)
            observation, reward, terminated, truncated, info = env.step(action, to_numpy=False)
            joint_pos.append(core_env.simulator.dof_state[..., 0])
            joint_vel.append(core_env.simulator.dof_state[..., 1])
            xpos_log.append(core_env.simulator._rigid_body_pos.reshape(num_envs, -1, 3))
            if core_env.carry_box_enabled:
                object_pos_log.append(core_env.object_pos.clone())
                object_vel_log.append(core_env.object_lin_vel.clone())

            ooo = {k: v for k, v in observation.items() if k != "history"}
            _episode.add(
                tree_map(lambda x: x.cpu(), ooo),
                reward.cpu(),
                action.cpu(),
                terminated.cpu(),
                truncated.cpu(),
                tree_map(lambda x: x.cpu(), info),
            )

        episode_data = _episode.get()
        episode_data["xpos"] = torch.stack(xpos_log)[:-1]
        joint_pos = torch.stack(joint_pos)
        joint_vel = torch.stack(joint_vel)

        for m_id, envs_with_current_motion in motion_to_envs.items():
            for motion_repetition, env_id in enumerate(envs_with_current_motion):
                # A motion was executed on multiple environments
                # we average the results
                _joint_pos = joint_pos[: ctx_dict[m_id].shape[0] + 1, env_id]
                _target_joint_pos = tracking_joint_pos[m_id]
                local_metrics = _calc_metrics(
                    {
                        # "xpos": episode_data["xpos"][0 : ctx_dict[m_id].shape[0], env_id],
                        # "target_xpos": target_xpos_dict[env_id],
                        "tracking_target": tracking_targets[m_id],
                        "motion_id": motion_id_map[m_id] if motion_id_map is not None else m_id,
                        "motion_file": core_env._motion_lib.curr_motion_keys[m_id],
                        "observation": tree_map(lambda x: x[0 : ctx_dict[m_id].shape[0] + 1, env_id], episode_data["observation"]),
                        "joint_pos": _joint_pos,
                        "target_joint_pos": _target_joint_pos,
                        **(
                            {
                                "object_pos": torch.stack(object_pos_log)[: ctx_dict[m_id].shape[0] + 1, env_id],
                                "object_lin_vel": torch.stack(object_vel_log)[: ctx_dict[m_id].shape[0] + 1, env_id],
                                "target_object_pos_track": _add_env_origin(
                                    tracking_object_pos[m_id], core_env.env_origins[env_id]
                                ),
                                "target_object_pos": target_states["target_object_pos"][env_id],
                                "carry_cfg": core_env.carry_box_cfg,
                                "object_present": object_present_list[env_id],
                                "carry_evaluation_eligible": carry_evaluation_eligible_list[env_id],
                            }
                            if core_env.carry_box_enabled
                            else {}
                        ),
                    },
                )

                # Rename the metrics so that repetitions do not overlap
                assert len(local_metrics) == 1
                metric_key = list(local_metrics.keys())[0]

                if include_results_from_all_envs:
                    metrics[f"{metric_key}_repetition#{motion_repetition}"] = local_metrics[metric_key]
                else:
                    # Retain the original motion name without "repetition#"
                    # Note that this means that motions with the same name get overriden, so you only get one result
                    metrics[metric_key] = local_metrics[metric_key]
                    break

    return metrics


def distance_matrix(X: torch.Tensor, Y: torch.Tensor):
    X_norm = X.pow(2).sum(1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    return torch.sqrt(torch.clamp(val, min=0))


def emd_numpy(next_obs: torch.Tensor, tracking_target: torch.Tensor, prefix=""):
    # keep only pose part of the observations
    agent_obs = next_obs.to("cpu")
    tracked_obs = tracking_target.to("cpu")
    # compute optimal transport cost
    cost_matrix = distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    X_pot = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    Y_pot = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    return {f"{prefix}emd": transport_cost}


def distance_proximity(next_obs: torch.Tensor, tracking_target: torch.Tensor, bound: float = 2.0, margin: float = 2, prefix=""):
    stats = {}
    dist = torch.norm(next_obs - tracking_target, dim=-1)
    in_bounds_mask = dist <= bound
    out_bounds_mask = dist > bound + margin
    stats[f"{prefix}proximity"] = (in_bounds_mask + ((bound + margin - dist) / margin) * (~in_bounds_mask) * (~out_bounds_mask)).mean()
    stats[f"{prefix}distance"] = dist.mean()
    return stats


def _calc_metrics(ep):
    metr = {}
    observation_state = torch.as_tensor(ep["observation"]["state"], dtype=torch.float32)
    target_state = torch.as_tensor(ep["tracking_target"]["state"], dtype=torch.float32)
    joint_pos = torch.as_tensor(ep["joint_pos"], dtype=torch.float32)
    target_joint_pos = torch.as_tensor(ep["target_joint_pos"], dtype=torch.float32)
    num_dofs = target_joint_pos.shape[-1] if target_joint_pos.ndim > 0 else 0

    shape_errors = []
    if num_dofs <= 0:
        shape_errors.append("num_dofs must be greater than zero")
    if observation_state.ndim != 2:
        shape_errors.append("observation state must be two-dimensional [T, D]")
    elif observation_state.shape[-1] < num_dofs:
        shape_errors.append("observation state feature dimension is smaller than num_dofs")
    if target_state.ndim != 2:
        shape_errors.append("tracking target state must be two-dimensional [T, D]")
    elif target_state.shape[-1] < num_dofs:
        shape_errors.append("tracking target state feature dimension is smaller than num_dofs")
    if joint_pos.ndim != 2:
        shape_errors.append("joint_pos must be two-dimensional [T, D]")
    elif joint_pos.shape[-1] <= 0:
        shape_errors.append("joint_pos feature dimension must be greater than zero")
    if target_joint_pos.ndim != 2:
        shape_errors.append("target_joint_pos must be two-dimensional [T, D]")
    elif target_joint_pos.shape[-1] <= 0:
        shape_errors.append("target_joint_pos feature dimension must be greater than zero")
    if joint_pos.ndim == 2 and target_joint_pos.ndim == 2 and joint_pos.shape != target_joint_pos.shape:
        shape_errors.append("joint_pos and target_joint_pos shapes must match")
    if joint_pos.ndim == 2 and joint_pos.shape[0] < 3:
        shape_errors.append("joint_pos must contain at least 3 time steps")
    if target_joint_pos.ndim == 2 and target_joint_pos.shape[0] < 3:
        shape_errors.append("target_joint_pos must contain at least 3 time steps")
    if observation_state.ndim == 2 and target_state.ndim == 2 and observation_state.shape[0] != target_state.shape[0]:
        shape_errors.append("observation and tracking target time dimensions must match")
    if observation_state.ndim == 2 and joint_pos.ndim == 2 and observation_state.shape[0] != joint_pos.shape[0]:
        shape_errors.append("observation state and joint_pos time dimensions must match")
    if target_state.ndim == 2 and target_joint_pos.ndim == 2 and target_state.shape[0] != target_joint_pos.shape[0]:
        shape_errors.append("tracking target state and target_joint_pos time dimensions must match")
    if shape_errors:
        raise ValueError(
            "Invalid tracking metric shapes: "
            f"observation_state.shape={tuple(observation_state.shape)}, "
            f"tracking_target_state.shape={tuple(target_state.shape)}, "
            f"joint_pos.shape={tuple(joint_pos.shape)}, "
            f"target_joint_pos.shape={tuple(target_joint_pos.shape)}, "
            f"num_dofs={num_dofs}; "
            + "; ".join(shape_errors)
        )

    next_qpos = observation_state[..., :num_dofs]
    target_qpos = target_state[..., :num_dofs]
    dist_prox_res = distance_proximity(next_obs=next_qpos, tracking_target=target_qpos, prefix="obs_state_")
    metr.update(dist_prox_res)
    emd_res = emd_numpy(next_obs=next_qpos, tracking_target=target_qpos, prefix="obs_state_")
    metr.update(emd_res)
    # # add distance wrt xpos
    # xpos = torch.tensor(ep["xpos"], dtype=torch.float32).view(ep["xpos"].shape[0], -1)  # [keyframes, 72]
    # target_xpos = torch.tensor(ep["target_xpos"], dtype=torch.float32).view(ep["target_xpos"].shape[0], -1)
    # target_xpos = target_xpos[:tracking_length]

    # dist_prox_res = distance_proximity(next_obs=xpos, tracking_target=target_xpos)
    # for k, v in dist_prox_res.items():
    #     metr[f"{k}_xpos"] = v
    # # import ipdb; ipdb.set_trace()
    # metr["emd_xpos"] = emd_numpy(next_obs=xpos, tracking_target=target_xpos)["emd"]
    # jpos_pred = xpos.reshape(xpos.shape[0], -1, 3)  # length x 23 x 3
    # jpos_gt = target_xpos.reshape(target_xpos.shape[0], -1, 3)  # length x 23 x 3
    # metr["success_phc_linf_xpos"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1) <= 0.5).float()
    # metr["success_phc_mean_xpos"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1).mean(dim=-1) <= 0.5).float()

    # phc metrics
    phc_metrics = compute_joint_pos_metrics(joint_pos=joint_pos, target_joint_pos=target_joint_pos)
    metr.update(phc_metrics)
    object_present = bool(torch.as_tensor(ep.get("object_present", 1.0)).reshape(-1)[0] > 0.5)
    carry_evaluation_eligible = bool(
        torch.as_tensor(ep.get("carry_evaluation_eligible", object_present)).reshape(-1)[0] > 0.5
    )
    if "object_pos" in ep:
        metr["object_present"] = float(object_present)
        metr["carry_evaluation_eligible"] = float(carry_evaluation_eligible)
    if "object_pos" in ep and object_present and carry_evaluation_eligible:
        object_pos = torch.as_tensor(ep["object_pos"], dtype=torch.float32)
        object_lin_vel = torch.as_tensor(ep["object_lin_vel"], dtype=torch.float32)
        target_track = torch.as_tensor(ep["target_object_pos_track"], dtype=torch.float32)
        target_pos = torch.as_tensor(ep["target_object_pos"], dtype=torch.float32)
        carry_cfg = ep["carry_cfg"]
        object_error = torch.linalg.vector_norm(object_pos - target_track, dim=-1)
        final_distance = torch.linalg.vector_norm(object_pos[-1] - target_pos)
        final_speed = torch.linalg.vector_norm(object_lin_vel[-1])
        ground_height = float(carry_cfg.get("ground_center_height", 0.185))
        lift_height = torch.max(object_pos[:, 2] - ground_height)
        lifted = lift_height > float(carry_cfg.get("lifted_threshold", 0.18))
        placed = final_distance < float(carry_cfg.get("success_position_tolerance", 0.15))
        stable = final_speed < float(carry_cfg.get("success_speed_tolerance", 0.15))
        metr.update(
            {
                "object_track_error": object_error.mean(),
                "object_final_distance": final_distance,
                "object_final_speed": final_speed,
                "object_max_lift": lift_height,
                "carry_task_success": (lifted & placed & stable).float(),
            }
        )
    elif "object_pos" in ep:
        metr.update(
            {
                "object_track_error": float("nan"),
                "object_final_distance": float("nan"),
                "object_final_speed": float("nan"),
                "object_max_lift": float("nan"),
                "carry_task_success": float("nan"),
            }
        )
    for k, v in metr.items():
        if isinstance(v, torch.Tensor):
            metr[k] = v.tolist()
    metr["motion_id"] = ep["motion_id"]
    metr["motion_file"] = ep["motion_file"]
    return {ep["motion_file"]: metr}


def compute_joint_pos_metrics(joint_pos, target_joint_pos):
    stats = {}
    # Next observation should match the desired target (if possible in 1 step)
    stats["mpjpe_l"] = torch.norm(joint_pos - target_joint_pos, dim=-1).mean(-1) * 1000

    # We compute velocity as a finite difference along the time dimension.
    vel_gt = torch.diff(target_joint_pos, dim=-2)
    vel_pred = torch.diff(joint_pos, dim=-2)
    stats["vel_dist"] = torch.norm(vel_pred - vel_gt, dim=-1).mean(-1) * 1000

    # We compute acceleration as a second finite difference along time.
    accel_gt = torch.diff(target_joint_pos, n=2, dim=-2)
    accel_pred = torch.diff(joint_pos, n=2, dim=-2)
    stats["accel_dist"] = torch.norm(accel_pred - accel_gt, dim=-1).mean(-1) * 100

    stats.update(
        distance_proximity(next_obs=joint_pos, tracking_target=target_joint_pos, prefix="")
    )
    stats.update(
        emd_numpy(next_obs=joint_pos, tracking_target=target_joint_pos, prefix="")
    )
    return stats
