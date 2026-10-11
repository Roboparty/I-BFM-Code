from __future__ import annotations

import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from gymnasium import spaces

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.fb_cpr.agent import FBcprAgent
from humanoidverse.agents.fb_cpr_aux.agent import (
    _combine_baseline_and_native_actor_losses,
    _combine_native_actor_losses,
)
from humanoidverse.agents.normalizers import TaskBatchNormNormalizerConfig
from humanoidverse.agents.presets import build_agent_preset
from humanoidverse.agents.spherical import sphere_log_map
from humanoidverse.train import build_ufo_mjlab_config, parse_args
from humanoidverse.training.workspace import _accumulate_metrics, _trajectory_output_keys


class UpdateZCliTest(unittest.TestCase):
    def _parse(self, *args: str):
        with patch.object(sys, "argv", ["train.py", *args]), patch("sys.stderr", io.StringIO()):
            return parse_args()

    def test_agent_specific_defaults_are_preserved(self) -> None:
        self.assertEqual(self._parse("--agent", "fb").update_z_every_step, 100)
        self.assertEqual(self._parse("--agent", "tldr").update_z_every_step, 10)

    def test_binary_task_batch_norm_config_is_importable_and_routes_stats(self) -> None:
        normalizer = TaskBatchNormNormalizerConfig(
            name="TaskBatchNormNormalizerConfig",
            num_tasks=2,
            task_channel=-1,
            task_threshold=0.5,
        ).build(spaces.Box(-float("inf"), float("inf"), shape=(2,))).eval()
        normalizer.running_mean.copy_(torch.tensor([[1.0, 0.0], [10.0, 1.0]]))
        normalizer.running_var.fill_(1.0)
        actual = normalizer(torch.tensor([[3.0, 0.0], [13.0, 1.0]]))
        torch.testing.assert_close(actual, torch.tensor([[2.0, 0.0], [3.0, 0.0]]), atol=2e-5, rtol=0)

    def test_disable_compile_reaches_single_gpu_config(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cuda:0",
            work_dir="/tmp/ufo_no_compile_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            disable_compile=True,
        )
        self.assertFalse(cfg.agent.compile)

    def test_tldr_cli_value_reaches_agent_config(self) -> None:
        args = self._parse("--agent", "tldr", "--update-z-every-step", "37")
        selected = build_agent_preset(
            agent=args.agent,
            device="cpu",
            compile=False,
            update_z_every_step=args.update_z_every_step,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
            wandb_project="test",
        )
        self.assertEqual(selected["agent_cfg"].train.update_z_every_step, 37)

    def test_programmatic_tldr_default_remains_ten(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_update_z_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            agent="tldr",
        )
        self.assertEqual(cfg.agent.train.update_z_every_step, 10)

    def test_native_intact_options_reach_fb_config(self) -> None:
        args = self._parse(
            "--agent",
            "fb",
            "--expert-z-mode",
            "mean",
            "--intact-native-goal-coef",
            "0.3",
            "--intact-native-local-coef",
            "0.1",
            "--intact-native-residual-coef",
            "0.02",
            "--intact-goal-horizon",
            "9",
            "--intact-grad-diagnostic-samples",
            "7",
            "--sphere-log-mode",
            "actor",
        )
        selected = build_agent_preset(
            agent=args.agent,
            device="cpu",
            compile=False,
            update_z_every_step=args.update_z_every_step,
            expert_z_mode=args.expert_z_mode,
            intact_native_goal_coef=args.intact_native_goal_coef,
            intact_native_local_coef=args.intact_native_local_coef,
            intact_native_residual_coef=args.intact_native_residual_coef,
            intact_goal_horizon=args.intact_goal_horizon,
            intact_grad_diagnostic_samples=args.intact_grad_diagnostic_samples,
            sphere_log_mode=args.sphere_log_mode,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
            wandb_project="test",
        )
        train_cfg = selected["agent_cfg"].train
        self.assertEqual(train_cfg.expert_z_mode, "mean")
        self.assertEqual(train_cfg.intact_native_goal_coef, 0.3)
        self.assertEqual(train_cfg.intact_native_local_coef, 0.1)
        self.assertEqual(train_cfg.intact_native_residual_coef, 0.02)
        self.assertEqual(train_cfg.intact_goal_horizon, 9)
        self.assertEqual(train_cfg.intact_grad_diagnostic_samples, 7)
        self.assertEqual(selected["agent_cfg"].model.sphere_log_mode, "actor")

    def test_scheme4_residual_cli_is_opt_in(self) -> None:
        args = self._parse("--agent", "fb")
        self.assertEqual(args.intact_native_residual_coef, 0.0)

    def test_scheme4_positive_residual_requires_pair_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a positive local or goal"):
            self._parse("--agent", "fb", "--intact-native-residual-coef", "0.005")

    def test_native_intact_sampler_aligns_current_local_and_goal_without_action_label(self) -> None:
        agent = object.__new__(FBcprAgent)
        agent.cfg = SimpleNamespace(
            model=SimpleNamespace(seq_length=8),
            train=SimpleNamespace(
                batch_size=1024,
                intact_native_goal_coef=1.0,
                intact_native_local_coef=1.0,
                intact_goal_horizon=8,
            ),
        )

        class CaptureBuffer:
            def sample_windows(self, **kwargs):
                return kwargs

        sampled = agent._sample_intact_batch(CaptureBuffer())
        self.assertEqual(sampled["num_sequences"], 128)
        self.assertEqual(sampled["source_length"], 8)
        self.assertTrue(sampled["clamp_to_trajectory_end"])
        self.assertEqual(sampled["windows"]["observation"], (0, 8))
        self.assertEqual(sampled["windows"]["future_observation_1"], (1, 8))
        self.assertEqual(sampled["windows"]["future_observation_8"], (8, 8))
        self.assertNotIn("action", sampled["windows"])
        self.assertTrue(all(value == "observation" for value in sampled["window_storage_keys"].values()))

    def test_window_sampler_clamps_future_frames_to_same_trajectory_endpoint(self) -> None:
        buffer = object.__new__(TrajectoryDictBufferMultiDim)
        buffer.output_key_tp1 = ("observation",)
        buffer._recompute_start_stop = False
        buffer.lengths = torch.tensor([5])
        buffer.start_idx = torch.tensor([[0, 0]])
        buffer.capacity = 8
        buffer.storage = {
            "observation": torch.arange(8).view(8, 1, 1),
            "action": (100 + torch.arange(8)).view(8, 1, 1),
        }

        with patch("torch.rand", return_value=torch.tensor([0.99])):
            sampled = buffer.sample_windows(
                num_sequences=1,
                source_length=2,
                windows={
                    "observation": (0, 2),
                    "future_observation_1": (1, 2),
                    "future_observation_8": (8, 2),
                    "action": (0, 2),
                },
                window_storage_keys={
                    "observation": "observation",
                    "future_observation_1": "observation",
                    "future_observation_8": "observation",
                    "action": "action",
                },
                clamp_to_trajectory_end=True,
            )

        torch.testing.assert_close(sampled["observation"].flatten(), torch.tensor([2, 3]))
        torch.testing.assert_close(sampled["future_observation_1"].flatten(), torch.tensor([3, 4]))
        torch.testing.assert_close(sampled["future_observation_8"].flatten(), torch.tensor([4, 4]))
        torch.testing.assert_close(sampled["action"].flatten(), torch.tensor([102, 103]))

    def test_hindsight_prompts_use_both_direct_endpoints(self) -> None:
        agent = object.__new__(FBcprAgent)
        future_b = torch.arange(3 * 2 * 4, dtype=torch.float32).view(3, 2, 4)
        goal_b, local_b = agent._select_hindsight_prompt_embeddings(future_b)
        torch.testing.assert_close(goal_b, future_b[-1])
        torch.testing.assert_close(local_b, future_b[0])

    def test_native_actor_loss_normalizes_relative_weights(self) -> None:
        combined = _combine_native_actor_losses(
            torch.tensor(6.0),
            torch.tensor(9.0),
            local_coef=1.0,
            goal_coef=2.0,
        )
        torch.testing.assert_close(combined, torch.tensor(8.0))

    def test_native_actor_loss_rejects_zero_pair_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _combine_native_actor_losses(
                torch.tensor(6.0),
                torch.tensor(9.0),
                local_coef=0.0,
                goal_coef=0.0,
            )

    def test_scheme4_keeps_baseline_and_adds_scaled_native_pair(self) -> None:
        native_pair = _combine_native_actor_losses(
            torch.tensor(7.0),
            torch.tensor(14.0),
            local_coef=4.0 / 7.0,
            goal_coef=3.0 / 7.0,
        )
        combined = _combine_baseline_and_native_actor_losses(
            torch.tensor(10.0),
            native_pair,
            residual_coef=0.005,
        )
        torch.testing.assert_close(combined, torch.tensor(10.05))

    def test_scheme4_zero_residual_is_exact_baseline(self) -> None:
        combined = _combine_baseline_and_native_actor_losses(
            torch.tensor(10.0),
            torch.tensor(99.0),
            residual_coef=0.0,
        )
        torch.testing.assert_close(combined, torch.tensor(10.0))

    def test_scheme4_gradient_coefficients_match_declared_budget(self) -> None:
        base = torch.tensor(1.0, requires_grad=True)
        local = torch.tensor(2.0, requires_grad=True)
        goal = torch.tensor(3.0, requires_grad=True)
        pair = _combine_native_actor_losses(local, goal, local_coef=4.0 / 7.0, goal_coef=3.0 / 7.0)
        loss = _combine_baseline_and_native_actor_losses(base, pair, residual_coef=0.005)
        loss.backward()
        torch.testing.assert_close(base.grad, torch.tensor(1.0))
        torch.testing.assert_close(local.grad, torch.tensor(0.005 * 4.0 / 7.0))
        torch.testing.assert_close(goal.grad, torch.tensor(0.005 * 3.0 / 7.0))

    def test_sphere_log_retains_arc_length_and_is_tangent(self) -> None:
        radius = 2.0
        base = torch.tensor([[radius, 0.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, radius, 0.0, 0.0]])
        prompt = sphere_log_map(base, target)
        torch.testing.assert_close(torch.linalg.vector_norm(prompt, dim=-1), torch.tensor([radius * torch.pi / 2]))
        torch.testing.assert_close((prompt * base).sum(dim=-1), torch.zeros(1), atol=1e-6, rtol=0.0)

    def test_sphere_log_cli_modes_reach_model_config(self) -> None:
        for mode in ("actor", "all"):
            args = self._parse("--agent", "fb", "--sphere-log-mode", mode)
            selected = build_agent_preset(
                agent="fb",
                device="cpu",
                compile=False,
                update_z_every_step=args.update_z_every_step,
                sphere_log_mode=args.sphere_log_mode,
                lr_scale=1.0,
                clip_grad_norm=0.0,
                cartwheel_aux_safe=False,
                wandb_project="test",
            )
            self.assertEqual(selected["agent_cfg"].model.sphere_log_mode, mode)

    def test_tldr_trajectory_buffer_keeps_aux_rewards(self) -> None:
        selected = build_agent_preset(
            agent="tldr",
            device="cpu",
            compile=False,
            update_z_every_step=10,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
            wandb_project="test",
        )
        self.assertIn("aux_rewards", _trajectory_output_keys(selected["agent_cfg"]))

    def test_metric_accumulation_accepts_tldr_phase_changes(self) -> None:
        totals, counts = _accumulate_metrics(
            None,
            {},
            {"tldr_te_loss": torch.tensor(2.0)},
        )
        totals, counts = _accumulate_metrics(
            totals,
            counts,
            {
                "tldr_te_loss": torch.tensor(4.0),
                "disc_wgan_gp_loss": torch.tensor(6.0),
            },
        )
        self.assertEqual(counts, {"tldr_te_loss": 2, "disc_wgan_gp_loss": 1})
        self.assertEqual((totals["tldr_te_loss"] / counts["tldr_te_loss"]).item(), 3.0)
        self.assertEqual((totals["disc_wgan_gp_loss"] / counts["disc_wgan_gp_loss"]).item(), 6.0)


if __name__ == "__main__":
    unittest.main()
