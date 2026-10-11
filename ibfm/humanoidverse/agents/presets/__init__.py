"""Agent presets used by the UFO training entrypoint."""

from __future__ import annotations

from typing import Any

from humanoidverse.agents.presets.fb import TRAIN_RUNTIME as FB_TRAIN_RUNTIME
from humanoidverse.agents.presets.fb import build_fb_agent
from humanoidverse.agents.presets.tldr import TRAIN_RUNTIME as TECH_TRAIN_RUNTIME
from humanoidverse.agents.presets.tldr import build_tldr_agent
from humanoidverse.agents.presets.tldr import build_tldr_agent as build_tech_agent


def build_agent_preset(
    *,
    agent: str,
    device: str,
    compile: bool,
    update_z_every_step: int,
    expert_z_mode: str = "mean",
    expert_z_horizon: int = 8,
    lr_scale: float,
    clip_grad_norm: float,
    cartwheel_aux_safe: bool,
    wandb_project: str,
    carry_box: bool = False,
    intact_native_goal_coef: float = 0.0,
    intact_native_local_coef: float = 0.0,
    intact_native_residual_coef: float = 0.0,
    intact_goal_horizon: int = 8,
    intact_grad_diagnostic_samples: int = 0,
    sphere_log_mode: str = "off",
    carry_aux_terms: bool = False,
    task_bn_normalizer: bool = False,
) -> dict[str, Any]:
    if agent == "fb":
        return {
            "agent_cfg": build_fb_agent(
                device=device,
                compile=compile,
                update_z_every_step=update_z_every_step,
                expert_z_mode=expert_z_mode,
                expert_z_horizon=expert_z_horizon,
                lr_scale=lr_scale,
                clip_grad_norm=clip_grad_norm,
                cartwheel_aux_safe=cartwheel_aux_safe,
                carry_box=carry_box,
                intact_native_goal_coef=intact_native_goal_coef,
                intact_native_local_coef=intact_native_local_coef,
                intact_native_residual_coef=intact_native_residual_coef,
                intact_goal_horizon=intact_goal_horizon,
                intact_grad_diagnostic_samples=intact_grad_diagnostic_samples,
                sphere_log_mode=sphere_log_mode,
                carry_aux_terms=carry_aux_terms,
                task_bn_normalizer=task_bn_normalizer,
            ),
            "wandb_group": "ufo_fb",
            "wandb_project": wandb_project,
            "train_runtime": dict(FB_TRAIN_RUNTIME),
        }
    if agent in {"tech", "tldr"}:
        return {
            "agent_cfg": build_tech_agent(
                device=device,
                compile=compile,
                update_z_every_step=update_z_every_step,
            ),
            "wandb_group": "ufo_tech",
            "wandb_project": wandb_project,
            "train_runtime": dict(TECH_TRAIN_RUNTIME),
        }
    raise ValueError(f"Unsupported agent preset: {agent}")


__all__ = ["build_agent_preset", "build_fb_agent", "build_tech_agent", "build_tldr_agent"]
