from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent


class _Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


def _objective(
    actor: _Actor,
    *,
    loss_offset: float,
    grad_scale: float,
    q_discriminator: float,
    q_aux: float,
    q_fb: float,
    actor_prompt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size = actor_prompt.shape[0]
    loss = actor.weight * grad_scale + loss_offset
    return {
        "loss": loss,
        "action": torch.full((batch_size, 1), grad_scale),
        "mean_action": torch.full((batch_size, 1), grad_scale),
        "actor_prompt": actor_prompt,
        "Q_discriminator": torch.full((batch_size, 1), q_discriminator),
        "Q_aux": torch.full((batch_size, 1), q_aux),
        "Q_fb": torch.full((batch_size, 1), q_fb),
    }


def _actor_agent(*, residual_coef: float, diagnostic_samples: int = 0) -> FBcprAuxAgent:
    agent = object.__new__(FBcprAuxAgent)
    actor = _Actor()
    agent.cfg = SimpleNamespace(
        model=SimpleNamespace(amp=False),
        train=SimpleNamespace(
            intact_native_local_coef=4.0 / 7.0,
            intact_native_goal_coef=3.0 / 7.0,
            intact_native_residual_coef=residual_coef,
            intact_grad_diagnostic_samples=diagnostic_samples,
        ),
    )
    agent._model = SimpleNamespace(
        device="cpu",
        amp_dtype=torch.float32,
        cfg=SimpleNamespace(actor_std=0.1),
        _actor=actor,
        _backward_map=lambda value: value,
        project_z=lambda value: value,
    )
    agent.actor_optimizer = torch.optim.SGD(actor.parameters(), lr=0.0)
    return agent


def _check_zero_residual_and_no_diagnostics_skip_aligned_sampling_and_objectives() -> None:
    agent = _actor_agent(residual_coef=0.0, diagnostic_samples=0)
    obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    z = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    train_buffer = SimpleNamespace(
        sample=Mock(
            return_value={
                "observation": obs,
                "action": torch.zeros(2, 1),
                "next": {
                    "observation": obs + 1.0,
                    "terminated": torch.zeros(2, 1, dtype=torch.bool),
                },
                "z": z,
            }
        )
    )
    replay_buffer = {"expert_slicer": object(), "train": train_buffer}

    agent.cfg.aux_rewards = []
    agent.cfg.aux_rewards_scaling = {}
    agent.cfg.train.batch_size = 2
    agent.cfg.train.discount = 0.99
    agent.cfg.train.grad_penalty_discriminator = 0.0
    agent.cfg.train.relabel_ratio = None
    agent.cfg.train.q_loss_coef = 0.0
    agent.cfg.train.clip_grad_norm = 0.0
    agent.cfg.train.fb_target_tau = 0.01
    agent.cfg.train.critic_target_tau = 0.01
    agent._model._obs_normalizer = torch.nn.Identity()
    agent._model._aux_reward_normalizer = torch.nn.Identity()
    agent._sample_expert_batch = Mock(
        return_value={"observation": obs + 2.0, "goal_observation": obs + 3.0}
    )
    agent._sample_intact_batch = Mock(side_effect=AssertionError("aligned batch sampled"))
    agent.encode_expert = Mock(return_value=torch.zeros_like(z))
    agent.sample_mixed_z = Mock(return_value=torch.ones_like(z))
    agent.z_buffer = SimpleNamespace(add=Mock())
    agent.update_discriminator = Mock(return_value={})
    agent.update_fb = Mock(return_value={})
    agent.update_critic = Mock(return_value={})
    agent.update_aux_critic = Mock(return_value={})
    agent._forward_map_paramlist = agent._target_forward_map_paramlist = ()
    agent._backward_map_paramlist = agent._target_backward_map_paramlist = ()
    agent._critic_map_paramlist = agent._target_critic_map_paramlist = ()
    agent._aux_critic_map_paramlist = agent._aux_target_critic_map_paramlist = ()

    objective_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def objective(current_obs: torch.Tensor, current_z: torch.Tensor):
        objective_calls.append((current_obs, current_z))
        return _objective(
            agent._model._actor,
            loss_offset=10.0,
            grad_scale=1.0,
            q_discriminator=11.0,
            q_aux=12.0,
            q_fb=13.0,
            actor_prompt=current_z,
        )

    agent._native_actor_objective = Mock(side_effect=objective)
    with (
        patch("humanoidverse.agents.fb_cpr_aux.agent._soft_update_params"),
        patch("torch.compiler.cudagraph_mark_step_begin"),
    ):
        agent.update(replay_buffer, step=1)

    agent._sample_intact_batch.assert_not_called()
    assert len(objective_calls) == 1
    assert objective_calls[0][0] is obs
    assert objective_calls[0][1] is z


def _check_base_and_aligned_objectives_receive_their_original_dataflows() -> None:
    agent = _actor_agent(residual_coef=0.02)
    obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    z = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    intact_obs = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    direct1 = torch.tensor([[21.0, 22.0], [23.0, 24.0]])
    direct8 = torch.tensor([[81.0, 82.0], [83.0, 84.0]])
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def objective(current_obs: torch.Tensor, current_q: torch.Tensor):
        calls.append((current_obs, current_q))
        call_index = len(calls) - 1
        values = (
            (10.0, 1.0, 101.0, 102.0, 103.0),
            (20.0, 2.0, 201.0, 202.0, 203.0),
            (30.0, 3.0, 301.0, 302.0, 303.0),
        )[call_index]
        return _objective(
            agent._model._actor,
            loss_offset=values[0],
            grad_scale=values[1],
            q_discriminator=values[2],
            q_aux=values[3],
            q_fb=values[4],
            actor_prompt=current_q,
        )

    agent._native_actor_objective = Mock(side_effect=objective)
    agent.update_actor(
        obs=obs,
        action=torch.full((2, 1), -999.0),
        z=z,
        clip_grad_norm=None,
        intact_batch={
            "observation": intact_obs,
            "future_observation_1": direct1,
            "future_observation_8": direct8,
        },
    )

    assert len(calls) == 3
    assert calls[0][0] is obs
    assert calls[0][1] is z
    assert calls[1][0] is intact_obs
    assert calls[2][0] is intact_obs
    torch.testing.assert_close(calls[1][1], direct1)
    torch.testing.assert_close(calls[2][1], direct8)


def _check_standard_actor_metrics_are_reported_from_the_baseline_objective() -> None:
    agent = _actor_agent(residual_coef=0.02)
    obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    z = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    intact_obs = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    calls = 0

    def objective(current_obs: torch.Tensor, current_q: torch.Tensor):
        nonlocal calls
        calls += 1
        values = (
            (10.0, 1.0, 101.0, 102.0, 103.0),
            (20.0, 2.0, 201.0, 202.0, 203.0),
            (30.0, 3.0, 301.0, 302.0, 303.0),
        )[calls - 1]
        return _objective(
            agent._model._actor,
            loss_offset=values[0],
            grad_scale=values[1],
            q_discriminator=values[2],
            q_aux=values[3],
            q_fb=values[4],
            actor_prompt=current_q,
        )

    agent._native_actor_objective = Mock(side_effect=objective)
    metrics = agent.update_actor(
        obs=obs,
        action=torch.zeros(2, 1),
        z=z,
        clip_grad_norm=None,
        intact_batch={
            "observation": intact_obs,
            "future_observation_1": torch.ones(2, 2),
            "future_observation_8": torch.full((2, 2), 8.0),
        },
    )

    torch.testing.assert_close(metrics["Q_discriminator"], torch.tensor(101.0))
    torch.testing.assert_close(metrics["Q_aux"], torch.tensor(102.0))
    torch.testing.assert_close(metrics["Q_fb"], torch.tensor(103.0))
    torch.testing.assert_close(metrics["actor_prompt_norm"], torch.tensor(5.0))
    torch.testing.assert_close(metrics["intact_native_local_Q_fb"], torch.tensor(203.0))
    torch.testing.assert_close(metrics["intact_native_goal_Q_fb"], torch.tensor(303.0))


class Scheme4DataflowTest(unittest.TestCase):
    def test_zero_residual_and_no_diagnostics_skip_aligned_work(self) -> None:
        _check_zero_residual_and_no_diagnostics_skip_aligned_sampling_and_objectives()

    def test_base_and_aligned_objectives_receive_their_original_dataflows(self) -> None:
        _check_base_and_aligned_objectives_receive_their_original_dataflows()

    def test_standard_actor_metrics_come_from_the_baseline_objective(self) -> None:
        _check_standard_actor_metrics_are_reported_from_the_baseline_objective()
