"""FB agent preset for UFO training."""

from __future__ import annotations

from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    RewardNormalizerConfig,
)
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig, TaskBatchNormNormalizerConfig

TRAIN_RUNTIME = {
    "log_every_updates": 384000,
    "update_agent_every": 1024,
    "num_seed_steps": 10240,
    "num_agent_updates": 16,
    "checkpoint_buffer": True,
    "use_trajectory_buffer": True,
    "buffer_size": 5120000,
    "eval_every_steps": 3200000,
}


def build_fb_agent(
    device: str = "cuda",
    compile: bool = True,
    update_z_every_step: int = 100,
    expert_z_mode: str = "mean",
    expert_z_horizon: int = 8,
    lr_scale: float = 1.0,
    clip_grad_norm: float = 0.0,
    cartwheel_aux_safe: bool = False,
    carry_box: bool = False,
    intact_native_goal_coef: float = 0.0,
    intact_native_local_coef: float = 0.0,
    intact_native_residual_coef: float = 0.0,
    intact_goal_horizon: int = 8,
    intact_grad_diagnostic_samples: int = 0,
    sphere_log_mode: str = "off",
    carry_aux_terms: bool = False,
    task_bn_normalizer: bool = False,
) -> FBcprAuxAgentConfig:
    if intact_native_goal_coef < 0 or intact_native_local_coef < 0:
        raise ValueError("INTACT native coefficients must be non-negative")
    if intact_native_residual_coef < 0:
        raise ValueError("intact_native_residual_coef must be non-negative")
    if intact_native_residual_coef > 0 and intact_native_goal_coef + intact_native_local_coef <= 0:
        raise ValueError("A positive Scheme 4 residual requires a positive local or goal coefficient")
    if intact_goal_horizon <= 0:
        raise ValueError("intact_goal_horizon must be positive")
    if intact_grad_diagnostic_samples < 0:
        raise ValueError("intact_grad_diagnostic_samples must be non-negative")
    if sphere_log_mode not in ("off", "actor", "all"):
        raise ValueError("sphere_log_mode must be 'off', 'actor', or 'all'")
    if cartwheel_aux_safe:
        aux_rewards = [
            "penalty_torques",
            "penalty_action_rate",
            "limits_dof_pos",
            "limits_torque",
        ]
        aux_rewards_scaling = {
            "penalty_action_rate": -0.03,
            "limits_dof_pos": -10.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        }
    else:
        aux_rewards = [
            "penalty_torques",
            "penalty_action_rate",
            "limits_dof_pos",
            "limits_torque",
            "penalty_undesired_contact",
            "penalty_feet_ori",
            "penalty_ankle_roll",
            "penalty_slippage",
        ]
        aux_rewards_scaling = {
            "penalty_action_rate": -0.1,
            "penalty_feet_ori": -0.4,
            "penalty_ankle_roll": -4.0,
            "limits_dof_pos": -10.0,
            "penalty_slippage": -4.0 if carry_box else -2.0,
            "penalty_undesired_contact": -1.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        }
    if carry_box and carry_aux_terms:
        # INTACT-BFM closed-loop carry object/contact + body/root terms. The env already
        # computes and stores every aux key; these entries are what make them carry
        # gradient into the objective via the safety-auxiliary-critic channel. Signs:
        # positive = reward (contact/lift/progress/place/success), negative = penalty.
        aux_rewards += [
            "carry_bilateral_contact",
            "carry_lift",
            "carry_progress",
            "carry_place",
            "carry_success",
            "carry_drop",
            "carry_body_mpjpe",
            "carry_root_pos_err",
            "carry_root_rot_err",
        ]
        aux_rewards_scaling.update(
            {
                "carry_bilateral_contact": 0.5,
                "carry_lift": 0.5,
                "carry_progress": 4.0,
                "carry_place": 1.0,
                "carry_success": 2.0,
                "carry_drop": -4.0,
                "carry_body_mpjpe": -2.0,
                "carry_root_pos_err": -2.0,
                "carry_root_rot_err": -0.02,
            }
        )
    return FBcprAuxAgentConfig(
        name="FBcprAuxAgent",
        model=FBcprAuxModelConfig(
            name="FBcprAuxModel",
            device=device,
            archi=FBcprAuxModelArchiConfig(
                name="FBcprAuxModelArchiConfig",
                z_dim=256,
                norm_z=True,
                f=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]
                    ),
                ),
                b=BackwardArchiConfig(
                    name="BackwardArchi",
                    hidden_dim=256,
                    hidden_layers=1,
                    norm=True,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                actor=ActorArchiConfig(
                    name="actor",
                    model="residual",
                    hidden_dim=2048,
                    hidden_layers=6,
                    embedding_layers=2,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "last_action", "history_actor"]),
                ),
                critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]
                    ),
                ),
                discriminator=DiscriminatorArchiConfig(
                    name="DiscriminatorArchi",
                    hidden_dim=1024,
                    hidden_layers=3,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                aux_critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]
                    ),
                ),
            ),
            obs_normalizer=_build_obs_normalizer(task_bn_normalizer),
            inference_batch_size=500000,
            seq_length=8,
            actor_std=0.05,
            amp=False,
            sphere_log_mode=sphere_log_mode,
            norm_aux_reward=RewardNormalizerConfig(name="RewardNormalizer", translate=False, scale=True),
        ),
        train=FBcprAuxAgentTrainConfig(
            name="FBcprAuxAgentTrainConfig",
            lr_f=0.0003 * lr_scale,
            lr_b=1e-05 * lr_scale,
            lr_actor=0.0003 * lr_scale,
            weight_decay=0.0,
            clip_grad_norm=clip_grad_norm,
            fb_target_tau=0.01,
            ortho_coef=100.0,
            train_goal_ratio=0.2,
            fb_pessimism_penalty=0.0,
            actor_pessimism_penalty=0.5,
            stddev_clip=0.3,
            q_loss_coef=0.0,
            batch_size=1024,
            discount=0.98,
            use_mix_rollout=True,
            update_z_every_step=int(update_z_every_step),
            expert_z_mode=expert_z_mode,
            expert_z_horizon=int(expert_z_horizon),
            z_buffer_size=8192,
            rollout_expert_trajectories=True,
            rollout_expert_trajectories_length=250,
            rollout_expert_trajectories_percentage=0.5,
            lr_discriminator=1e-05 * lr_scale,
            lr_critic=0.0003 * lr_scale,
            critic_target_tau=0.005,
            critic_pessimism_penalty=0.5,
            reg_coeff=0.05,
            scale_reg=True,
            expert_asm_ratio=0.6,
            relabel_ratio=0.8,
            grad_penalty_discriminator=10.0,
            weight_decay_discriminator=0.0,
            lr_aux_critic=0.0003 * lr_scale,
            reg_coeff_aux=0.02,
            aux_critic_pessimism_penalty=0.5,
            intact_native_goal_coef=float(intact_native_goal_coef),
            intact_native_local_coef=float(intact_native_local_coef),
            intact_native_residual_coef=float(intact_native_residual_coef),
            intact_goal_horizon=int(intact_goal_horizon),
            intact_grad_diagnostic_samples=int(intact_grad_diagnostic_samples),
        ),
        aux_rewards=aux_rewards,
        aux_rewards_scaling=aux_rewards_scaling,
        cudagraphs=False,
        compile=compile,
    )

def _build_obs_normalizer(task_bn_normalizer: bool) -> ObsNormalizerConfig:
    normalizers = {
        "state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
        "privileged_state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
        "last_action": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
        "history_actor": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
    }
    if task_bn_normalizer:
        for key in ("state", "privileged_state"):
            normalizers[key] = TaskBatchNormNormalizerConfig(
                name="TaskBatchNormNormalizerConfig", momentum=0.01, num_tasks=2, task_channel=-1, task_threshold=0.5
            )
    return ObsNormalizerConfig(
        name="ObsNormalizerConfig",
        normalizers=normalizers,
        allow_mismatching_keys=True,
    )
