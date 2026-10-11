# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from typing import Dict

import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ...distributed import average_gradients
from ..base import BaseConfig
from ..fb_cpr.agent import FBcprAgent, FBcprAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode
from .model import FBcprAuxModelConfig


def _combine_native_actor_losses(
    local_loss: torch.Tensor,
    goal_loss: torch.Tensor,
    local_coef: float,
    goal_coef: float,
) -> torch.Tensor:
    """Combine aligned objectives after normalizing local/goal relative weights."""
    weight_sum = local_coef + goal_coef
    if weight_sum <= 0:
        raise ValueError("At least one native actor coefficient must be positive")
    return (local_coef * local_loss + goal_coef * goal_loss) / weight_sum


def _combine_baseline_and_native_actor_losses(
    base_loss: torch.Tensor,
    native_loss: torch.Tensor,
    residual_coef: float,
) -> torch.Tensor:
    """Keep the baseline actor loss and add the aligned native loss as a residual."""
    return base_loss + residual_coef * native_loss


class FBcprAuxAgentTrainConfig(FBcprAgentTrainConfig):
    lr_aux_critic: float = 1e-4
    reg_coeff_aux: float = 1.0
    aux_critic_pessimism_penalty: float = 0.5


class FBcprAuxAgentConfig(BaseConfig):
    name: tp.Literal["FBcprAuxAgent"] = "FBcprAuxAgent"

    model: FBcprAuxModelConfig = FBcprAuxModelConfig()
    train: FBcprAuxAgentTrainConfig = FBcprAuxAgentTrainConfig()
    aux_rewards: list[str] = pydantic.Field(default_factory=list)
    aux_rewards_scaling: dict[str, float] = pydantic.Field(default_factory=dict)
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim: int) -> "FBcprAuxAgent":
        return self.object_class(
            obs_space=obs_space,
            action_dim=action_dim,
            cfg=self,
        )

    @property
    def object_class(self):
        return FBcprAuxAgent


class FBcprAuxAgent(FBcprAgent):
    config_class = FBcprAuxAgentConfig

    def setup_training(self) -> None:
        super().setup_training()

        # prepare parameter list
        self._aux_critic_map_paramlist = tuple(x for x in self._model._aux_critic.parameters())
        self._aux_target_critic_map_paramlist = tuple(x for x in self._model._target_aux_critic.parameters())

        self.aux_critic_optimizer = torch.optim.Adam(
            self._model._aux_critic.parameters(),
            lr=self.cfg.train.lr_aux_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers["aux_critic_optimizer"] = self.aux_critic_optimizer.state_dict()
        return optimizers

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_aux_critic = torch.compile(self.update_aux_critic, mode=mode)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_aux_critic = CudaGraphModule(self.update_aux_critic, warmup=5)

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = self._sample_expert_batch(replay_buffer["expert_slicer"])
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        intact_batch = None
        native_enabled = self.cfg.train.intact_native_residual_coef > 0 and (
            self.cfg.train.intact_native_goal_coef > 0
            or self.cfg.train.intact_native_local_coef > 0
        )
        if native_enabled or self.cfg.train.intact_grad_diagnostic_samples > 0:
            try:
                intact_batch = self._sample_intact_batch(replay_buffer["train"])
            except ValueError as exc:
                # Transient replay state (trajectories shorter than the sampling window,
                # e.g. right after seeding). Skip this update rather than crash the run.
                if "No trajectories with length" not in str(exc):
                    raise
                if not getattr(self, "_intact_skip_warned", False):
                    print("[INTACT] replay trajectories shorter than the sampling window; skipping intact losses this update", flush=True)
                    self._intact_skip_warned = True

        train_obs, train_action, train_next_obs = (
            tree_map(lambda x: x.to(self.device), train_batch["observation"]),
            train_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"]),
        )
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)
        expert_obs, expert_goal_obs = (
            tree_map(lambda x: x.to(self.device), expert_batch["observation"]),
            tree_map(lambda x: x.to(self.device), expert_batch["goal_observation"]),
        )

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs, train_next_obs = (
                self._model._obs_normalizer(train_obs),
                self._model._obs_normalizer(train_next_obs),
            )
            expert_obs, expert_goal_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_goal_obs),
            )
            if intact_batch is not None:
                future_keys = sorted(
                    (key for key in intact_batch if key.startswith("future_observation_")),
                    key=lambda key: int(key.rsplit("_", 1)[1]),
                )
                for intact_key in ("observation", *future_keys):
                    intact_batch[intact_key] = self._model._obs_normalizer(
                        tree_map(lambda x: x.to(self.device), intact_batch[intact_key])
                    )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_goal_obs)
        train_z = train_batch["z"].to(self.device)

        # train the discriminator
        grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=grad_penalty,
        )

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                goal=train_next_obs,
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        # compute scalar auxiliary reward as a weighted sum of the auxiliary rewards
        aux_reward = torch.zeros(
            (self.cfg.train.batch_size, 1),
            device=self.device,
            dtype=torch.float32,
        )
        for aux_reward_name in self.cfg.aux_rewards:
            # let's log even this information
            metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
            aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name].to(self.device)

        aux_reward = self._model._aux_reward_normalizer(aux_reward)

        metrics.update(
            self.update_aux_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                aux_reward=aux_reward,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        metrics.update(
            self.update_actor(
                obs=train_obs,
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
                intact_batch=intact_batch,
            )
        )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._critic_map_paramlist,
                self._target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )
            _soft_update_params(
                self._aux_critic_map_paramlist,
                self._aux_target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )

        return metrics

    def update_aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            # compute target critic
            with torch.no_grad():
                next_actor_prompt = self._model.actor_prompt(next_obs, z)
                next_value_prompt = self._model.value_prompt(next_obs, z)
                dist = self._model._actor(next_obs, next_actor_prompt, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_aux_critic(
                    next_obs, next_value_prompt, next_action
                )  # num_parallel x batch x 1
                # TODO AL: should we have aux_critic parameters here?
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.aux_critic_pessimism_penalty)
                target_Q = aux_reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            # compute critic loss
            current_value_prompt = self._model.value_prompt(obs, z)
            Qs = self._model._aux_critic(
                obs, current_value_prompt, action
            )  # num_parallel x batch x (1 or n_bins)
            aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        average_gradients(self._model._aux_critic.parameters())
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_auxQ": target_Q.mean().detach(),
                "auxQ1": Qs.mean().detach(),
                "mean_next_auxQ": Q_mean.mean().detach(),
                "unc_auxQ": Q_unc.mean().detach(),
                "aux_critic_loss": aux_critic_loss.mean().detach(),
                "mean_aux_reward": aux_reward.mean().detach(),
            }
        return output_metrics

    def _native_actor_objective(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        q: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the unmodified BFM actor objective for one canonical q."""
        actor_prompt = self._model.actor_prompt(obs, q)
        value_prompt = self._model.value_prompt(obs, q)
        dist = self._model._actor(obs, actor_prompt, self._model.cfg.actor_std)
        sampled_action = dist.sample(clip=self.cfg.train.stddev_clip)

        Qs_discriminator = self._model._critic(obs, value_prompt, sampled_action)
        _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.cfg.train.actor_pessimism_penalty)

        Qs_aux = self._model._aux_critic(obs, value_prompt, sampled_action)
        _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, self.cfg.train.actor_pessimism_penalty)

        Fs = self._model._forward_map(obs, value_prompt, sampled_action)
        Qs_fb = (Fs * value_prompt).sum(-1)
        _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)

        weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
        loss = (
            -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight
            - Q_aux.mean() * self.cfg.train.reg_coeff_aux * weight
            - Q_fb.mean()
        )
        return {
            "loss": loss,
            "action": sampled_action,
            "mean_action": dist.mean,
            "actor_prompt": actor_prompt,
            "Q_discriminator": Q_discriminator,
            "Q_aux": Q_aux,
            "Q_fb": Q_fb,
        }

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
        intact_batch: dict | None = None,
    ) -> Dict[str, torch.Tensor]:
        del action  # Scheme 4 uses native BFM objectives, never an action label.
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            intact_goal_objective = None
            intact_local_objective = None
            native_pair_loss = None
            intact_prompt_cosine = None
            intact_action_disagreement = None
            local_coef = self.cfg.train.intact_native_local_coef
            goal_coef = self.cfg.train.intact_native_goal_coef
            residual_coef = self.cfg.train.intact_native_residual_coef
            native_enabled = residual_coef > 0 and local_coef + goal_coef > 0
            diagnostic_enabled = (
                self.cfg.train.intact_grad_diagnostic_samples > 0
                and getattr(self, "_intact_grad_diagnostic_count", 0)
                < self.cfg.train.intact_grad_diagnostic_samples
            )
            # Scheme 4 preserves the complete baseline mixed-q actor objective.
            base_objective = self._native_actor_objective(obs, z)
            base_actor_loss = base_objective["loss"]
            actor_loss = base_actor_loss

            if intact_batch is not None:
                future_keys = sorted(
                    (key for key in intact_batch if key.startswith("future_observation_")),
                    key=lambda key: int(key.rsplit("_", 1)[1]),
                )
                future_observation = tree_map(
                    lambda *values: torch.cat(values, dim=0),
                    *(intact_batch[key] for key in future_keys),
                )
                intact_obs = intact_batch["observation"]
                with torch.no_grad():
                    future_b = self._model._backward_map(future_observation)
                    future_b = future_b.reshape(len(future_keys), -1, future_b.shape[-1])
                    goal_b, local_b = self._select_hindsight_prompt_embeddings(future_b)
                    q_goal = self._model.project_z(goal_b)
                    q_local = self._model.project_z(local_b)

                # Every sampled clip takes both branches. These are two calls to
                # the native BFM objective, not inverse-action MSE losses.
                intact_local_objective = self._native_actor_objective(intact_obs, q_local)
                intact_goal_objective = self._native_actor_objective(intact_obs, q_goal)
                audit_local_coef = local_coef if native_enabled else 4.0 / 7.0
                audit_goal_coef = goal_coef if native_enabled else 3.0 / 7.0
                native_pair_loss = _combine_native_actor_losses(
                    intact_local_objective["loss"],
                    intact_goal_objective["loss"],
                    audit_local_coef,
                    audit_goal_coef,
                )
                if native_enabled:
                    actor_loss = _combine_baseline_and_native_actor_losses(
                        base_actor_loss,
                        native_pair_loss,
                        residual_coef,
                    )
                intact_prompt_cosine = F.cosine_similarity(
                    intact_goal_objective["actor_prompt"],
                    intact_local_objective["actor_prompt"],
                    dim=-1,
                ).mean()
                intact_action_disagreement = F.mse_loss(
                    intact_goal_objective["mean_action"], intact_local_objective["mean_action"]
                )

            grad_diagnostics = {}
            if (
                diagnostic_enabled
                and base_objective is not None
                and intact_goal_objective is not None
                and intact_local_objective is not None
                and native_pair_loss is not None
            ):
                diagnostic_combined_loss = _combine_baseline_and_native_actor_losses(
                    base_actor_loss,
                    native_pair_loss,
                    residual_coef,
                )
                actor_parameters = tuple(parameter for parameter in self._model._actor.parameters() if parameter.requires_grad)
                loss_grads = {
                    name: torch.autograd.grad(loss, actor_parameters, retain_graph=True, allow_unused=True)
                    for name, loss in (
                        ("base", base_actor_loss),
                        ("goal", intact_goal_objective["loss"]),
                        ("local", intact_local_objective["loss"]),
                        ("pair", native_pair_loss),
                        ("combined", diagnostic_combined_loss),
                    )
                }

                def grad_dot(left, right):
                    terms = [
                        (left_grad.detach() * right_grad.detach()).sum()
                        for left_grad, right_grad in zip(left, right)
                        if left_grad is not None and right_grad is not None
                    ]
                    return torch.stack(terms).sum()

                grad_norms = {
                    name: grad_dot(grads, grads).clamp_min(0).sqrt()
                    for name, grads in loss_grads.items()
                }
                eps = torch.finfo(grad_norms["base"].dtype).eps
                grad_diagnostics = {
                    "actor_base_grad_norm": grad_norms["base"],
                    "intact_goal_grad_norm": grad_norms["goal"],
                    "intact_local_grad_norm": grad_norms["local"],
                    "intact_pair_grad_norm": grad_norms["pair"],
                    "intact_goal_to_base_grad_ratio": grad_norms["goal"] / grad_norms["base"].clamp_min(eps),
                    "intact_local_to_base_grad_ratio": grad_norms["local"] / grad_norms["base"].clamp_min(eps),
                    "intact_pair_to_base_grad_ratio": grad_norms["pair"] / grad_norms["base"].clamp_min(eps),
                    "intact_base_to_pair_scale": grad_norms["base"] / grad_norms["pair"].clamp_min(eps),
                    "intact_effective_to_base_grad_ratio": residual_coef
                    * grad_norms["pair"]
                    / grad_norms["base"].clamp_min(eps),
                    "intact_recommended_residual_coef_rho0p02": 0.02
                    * grad_norms["base"]
                    / grad_norms["pair"].clamp_min(eps),
                    "scheme4_combined_grad_norm": grad_norms["combined"],
                    "intact_goal_base_grad_cosine": grad_dot(loss_grads["goal"], loss_grads["base"])
                    / (grad_norms["goal"] * grad_norms["base"]).clamp_min(eps),
                    "intact_local_base_grad_cosine": grad_dot(loss_grads["local"], loss_grads["base"])
                    / (grad_norms["local"] * grad_norms["base"]).clamp_min(eps),
                    "intact_goal_local_grad_cosine": grad_dot(loss_grads["goal"], loss_grads["local"])
                    / (grad_norms["goal"] * grad_norms["local"]).clamp_min(eps),
                    "intact_pair_base_grad_cosine": grad_dot(loss_grads["pair"], loss_grads["base"])
                    / (grad_norms["pair"] * grad_norms["base"]).clamp_min(eps),
                    "scheme4_combined_base_grad_cosine": grad_dot(loss_grads["combined"], loss_grads["base"])
                    / (grad_norms["combined"] * grad_norms["base"]).clamp_min(eps),
                }
                self._intact_grad_diagnostic_count = getattr(self, "_intact_grad_diagnostic_count", 0) + 1
                rendered = " ".join(f"{key}={value.detach().item():.8g}" for key, value in grad_diagnostics.items())
                print(f"[INTACT-GRAD] sample={self._intact_grad_diagnostic_count} {rendered}", flush=True)

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        average_gradients(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            # Keep the standard metric keys baseline-comparable; aligned branch
            # metrics are emitted separately below.
            Q_discriminator = base_objective["Q_discriminator"]
            Q_aux = base_objective["Q_aux"]
            Q_fb = base_objective["Q_fb"]
            actor_prompt_norm = torch.norm(base_objective["actor_prompt"], dim=-1)
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "actor_base_loss": base_actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_aux": Q_aux.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
                "actor_prompt_norm": actor_prompt_norm.mean().detach(),
            }
            if intact_goal_objective is not None and intact_local_objective is not None:
                output_metrics["intact_native_goal_loss"] = intact_goal_objective["loss"].detach()
                output_metrics["intact_native_local_loss"] = intact_local_objective["loss"].detach()
                output_metrics["intact_goal_local_prompt_cosine"] = intact_prompt_cosine.detach()
                output_metrics["intact_goal_local_action_mse"] = intact_action_disagreement.detach()
                output_metrics["intact_goal_prompt_norm"] = torch.norm(
                    intact_goal_objective["actor_prompt"], dim=-1
                ).mean().detach()
                output_metrics["intact_local_prompt_norm"] = torch.norm(
                    intact_local_objective["actor_prompt"], dim=-1
                ).mean().detach()
                output_metrics["intact_native_goal_Q_fb"] = intact_goal_objective["Q_fb"].mean().detach()
                output_metrics["intact_native_local_Q_fb"] = intact_local_objective["Q_fb"].mean().detach()
                output_metrics["intact_native_pair_loss"] = native_pair_loss.detach()
                output_metrics["intact_native_scaled_loss"] = (residual_coef * native_pair_loss).detach()
                output_metrics["intact_native_residual_coef"] = torch.as_tensor(
                    residual_coef,
                    device=actor_loss.device,
                    dtype=actor_loss.dtype,
                )
            if diagnostic_enabled:
                output_metrics["actor_base_reference_loss"] = base_objective["loss"].detach()
            output_metrics.update({key: value.detach() for key, value in grad_diagnostics.items()})
        return output_metrics
