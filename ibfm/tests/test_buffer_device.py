from __future__ import annotations

import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from gymnasium import spaces

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer, TrajectoryDictBufferMultiDim, get_idxs
from humanoidverse.agents.fb.agent import FBAgent
from humanoidverse.agents.fb_cpr.agent import FBcprAgent
from humanoidverse.train import build_ufo_mjlab_config, parse_args


class BufferDeviceTest(unittest.TestCase):
    def config(self, **kwargs):
        return build_ufo_mjlab_config(
            device="cuda:0", work_dir="/tmp/ibfm_buffer_device_test",
            num_envs=1024, num_env_steps=2048, seed=4728,
            use_wandb=False, wandb_run_name=None, smoke=True, **kwargs,
        )

    def test_cpu_storage_keeps_model_and_capacity_unchanged(self):
        config = self.config(buffer_device="cpu")
        self.assertEqual(config.buffer_device, "cpu")
        self.assertEqual(config.agent.model.device, "cuda")
        self.assertEqual(config.buffer_size, 5120000)
        self.assertEqual(config.online_parallel_envs, 1024)

    def test_default_storage_still_follows_cuda_model(self):
        self.assertEqual(self.config().buffer_device, "cuda")

    def test_invalid_programmatic_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "buffer_device"):
            self.config(buffer_device="disk")

    def test_cpu_storage_is_exposed_on_cli(self):
        with patch.object(sys, "argv", ["train.py", "--buffer-device", "cpu"]):
            self.assertEqual(parse_args().buffer_device, "cpu")

    def test_cpu_replay_supports_aligned_logo_sampling(self):
        buffer = TrajectoryDictBufferMultiDim(
            capacity=32, device="cpu", n_dim=2, seq_length=8,
            end_key="truncated", output_key_t=["observation", "action", "z"],
            output_key_tp1=["observation", "terminated"],
        )
        buffer._get_idxs = get_idxs
        states = torch.arange(24).reshape(24, 1, 1).float() + torch.tensor([0, 100]).reshape(1, 2, 1)
        truncated = torch.zeros((24, 2, 1), dtype=torch.bool)
        truncated[-1] = True
        buffer.extend({
            "observation": {"state": states},
            "action": torch.zeros((24, 2, 3)), "z": torch.zeros((24, 2, 4)),
            "truncated": truncated, "terminated": torch.zeros_like(truncated),
        })
        sampled = buffer.sample(16)
        self.assertEqual(sampled["action"].device.type, "cpu")
        torch.testing.assert_close(sampled["next"]["observation"]["state"], sampled["observation"]["state"] + 1)
        agent = object.__new__(FBcprAgent)
        agent.cfg = SimpleNamespace(
            model=SimpleNamespace(seq_length=8),
            train=SimpleNamespace(batch_size=16, intact_native_goal_coef=3/7,
                                  intact_native_local_coef=4/7, intact_goal_horizon=8),
        )
        windows = agent._sample_intact_batch(buffer)
        current = windows["observation"]["state"]
        endpoint = (current // 100) * 100 + 23
        torch.testing.assert_close(windows["future_observation_8"]["state"], torch.minimum(current + 8, endpoint))
        self.assertEqual(buffer.storage["observation"]["state"].device.type, "cpu")

    def test_cpu_expert_rollout_moves_both_prompt_modes_to_model_device(self):
        observations = {"state": torch.zeros((8, 3))}

        def backward_map(obs):
            self.assertEqual(obs["state"].device.type, "meta")
            self.assertEqual(observations["state"].device.type, "cpu")
            return torch.empty((8, 4), device="meta")

        agent = object.__new__(FBAgent)
        agent._model = SimpleNamespace(device="meta", backward_map=backward_map, project_z=lambda value: value)
        expert = SimpleNamespace(
            sample=lambda *args, **kwargs: {"next": {"observation": observations}},
            sample_observation_windows=lambda **kwargs: {"goal_observation": observations},
        )
        for mode in ("mean", "direct"):
            with self.subTest(mode=mode):
                agent.cfg = SimpleNamespace(model=SimpleNamespace(seq_length=2),
                                            train=SimpleNamespace(expert_z_mode=mode, expert_z_horizon=2))
                output = agent._sample_tracking_z({"expert_slicer": expert}, 2, 4)
                self.assertEqual(tuple(output.shape), (2, 4, 4))
                self.assertEqual(output.device.type, "meta")

    def test_sequential_logo_backward_preserves_actor_gradients_and_metrics(self):
        torch.manual_seed(4728)
        config = self.config(buffer_device="cpu", disable_compile=True,
                             intact_native_goal_coef=3/7, intact_native_local_coef=4/7,
                             intact_native_residual_coef=0.005, sphere_log_mode="actor")
        agent_config = config.agent
        networks = {}
        for name in ("f", "actor", "critic", "aux_critic", "discriminator", "b"):
            network = getattr(agent_config.model.archi, name)
            overrides = {"hidden_dim": 32, "hidden_layers": 2}
            if hasattr(network, "embedding_layers"):
                overrides["embedding_layers"] = 2
            networks[name] = network.model_copy(update=overrides)
        agent_config = agent_config.model_copy(update={
            "train": agent_config.train.model_copy(update={"batch_size": 32}),
            "model": agent_config.model.model_copy(update={
                "device": "cpu", "archi": agent_config.model.archi.model_copy(update=networks),
            }),
        })
        dims = {"state": 86, "privileged_state": 485, "last_action": 29, "history_actor": 372}
        obs_space = spaces.Dict({key: spaces.Box(-float("inf"), float("inf"), (dim,)) for key, dim in dims.items()})
        sequential = agent_config.build(obs_space, 29)
        reference = copy.deepcopy(sequential)
        # Select the original joint-backward path without invoking compilation.
        reference.cfg = reference.cfg.model_copy(update={"compile": True})
        for agent in (sequential, reference):
            agent.actor_optimizer = torch.optim.SGD(agent._model._actor.parameters(), lr=0.01)
        obs = {key: torch.randn(32, dim) for key, dim in dims.items()}
        intact = {key: {name: torch.randn_like(value) for name, value in obs.items()}
                  for key in ("observation", "future_observation_1", "future_observation_8")}
        z = sequential._model.sample_z(32)
        for step in range(2):
            metrics = []
            for agent in (sequential, reference):
                torch.manual_seed(100 + step)
                metrics.append(agent.update_actor(obs, torch.zeros(32, 29), z, 1.0, intact))
            for actual, expected in zip(sequential._model._actor.parameters(), reference._model._actor.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            self.assertEqual(metrics[0].keys(), metrics[1].keys())
            for key in metrics[0]:
                torch.testing.assert_close(metrics[0][key], metrics[1][key], rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cpu_buffers_support_gpu_rollout_and_optimizer_update(self):
        torch.manual_seed(4728)
        config = self.config(buffer_device="cpu", disable_compile=True,
                             intact_native_goal_coef=3/7, intact_native_local_coef=4/7,
                             intact_native_residual_coef=0.005, sphere_log_mode="actor")
        agent_config = config.agent
        networks = {}
        for name in ("f", "actor", "critic", "aux_critic", "discriminator", "b"):
            network = getattr(agent_config.model.archi, name)
            overrides = {"hidden_dim": 32, "hidden_layers": 2}
            if hasattr(network, "embedding_layers"):
                overrides["embedding_layers"] = 2
            networks[name] = network.model_copy(update=overrides)
        agent_config = agent_config.model_copy(update={
            "train": agent_config.train.model_copy(update={"batch_size": 32, "rollout_expert_trajectories_length": 16}),
            "model": agent_config.model.model_copy(update={"archi": agent_config.model.archi.model_copy(update=networks)}),
        })
        dims = {"state": 86, "privileged_state": 485, "last_action": 29, "history_actor": 372}
        observation_space = spaces.Dict({key: spaces.Box(-float("inf"), float("inf"), (dim,)) for key, dim in dims.items()})
        agent = agent_config.build(observation_space, 29)
        replay = TrajectoryDictBufferMultiDim(
            capacity=32, device="cpu", n_dim=2, seq_length=8,
            end_key="truncated", output_key_t=["observation", "action", "z", "aux_rewards"],
            output_key_tp1=["observation", "terminated"],
        )
        replay._get_idxs = get_idxs
        truncated = torch.zeros((24, 2, 1), dtype=torch.bool)
        truncated[-1] = True
        replay.extend({
            "observation": {key: torch.randn(24, 2, dim) for key, dim in dims.items()},
            "action": torch.randn(24, 2, 29), "z": torch.randn(24, 2, 256),
            "terminated": torch.zeros_like(truncated), "truncated": truncated,
            "aux_rewards": {key: torch.rand(24, 2, 1) for key in agent_config.aux_rewards},
        })
        expert = TrajectoryDictBuffer(episodes=[{
            "observation": {key: torch.randn(32, dim) for key, dim in dims.items() if key != "history_actor"},
            "terminated": torch.zeros(32, dtype=torch.bool),
            "motion_id": torch.zeros(32, dtype=torch.long),
        }], device="cpu", seq_length=8)
        expert._get_idxs = get_idxs
        buffers = {"train": replay, "expert_slicer": expert}
        with torch.no_grad():
            context = agent.maybe_update_rollout_context(None, torch.zeros((2, 1), device="cuda"), buffers)
        self.assertEqual(context.device.type, "cuda")
        metrics = agent.update(buffers, step=1024)
        self.assertTrue(metrics)
        for name, value in metrics.items():
            self.assertTrue(torch.isfinite(value).all(), name)
            self.assertEqual(value.device.type, "cuda", name)
        self.assertEqual(replay.storage["z"].device.type, "cpu")
        self.assertEqual(expert.storage["observation"]["state"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
