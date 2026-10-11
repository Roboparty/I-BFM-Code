import typing as tp

import pydantic
import torch
from gymnasium import spaces
from torch import nn

from .base import BaseConfig


class BatchNormNormalizerConfig(BaseConfig):
    momentum: float = 0.01

    def build(self, obs_space) -> "BatchNormNormalizer":
        return BatchNormNormalizer(obs_space, self)


class BatchNormNormalizer(nn.Module):
    def __init__(self, obs_space: spaces.Space, cfg: BatchNormNormalizerConfig):
        super().__init__()
        assert len(obs_space.shape) == 1, "BatchNormNormalizer only supports 1D observation spaces"
        self._normalizer = nn.BatchNorm1d(num_features=obs_space.shape[0], affine=False, momentum=cfg.momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalizer(x)


class TaskBatchNormNormalizerConfig(BaseConfig):
    """Binary task-specific running statistics with no affine parameters."""

    momentum: float = 0.01
    num_tasks: int = 2
    task_channel: int = -1
    task_threshold: float = 0.5

    def build(self, obs_space) -> "TaskBatchNormNormalizer":
        return TaskBatchNormNormalizer(obs_space, self)


class TaskBatchNormNormalizer(nn.Module):
    def __init__(self, obs_space: spaces.Space, cfg: TaskBatchNormNormalizerConfig):
        super().__init__()
        assert len(obs_space.shape) == 1, "TaskBatchNormNormalizer only supports 1D observation spaces"
        self.cfg = cfg
        self.eps = 1e-5
        num_features = obs_space.shape[0]
        self.register_buffer("running_mean", torch.zeros(cfg.num_tasks, num_features))
        self.register_buffer("running_var", torch.ones(cfg.num_tasks, num_features))

    def _task_ids(self, x: torch.Tensor) -> torch.Tensor:
        return (x[..., self.cfg.task_channel] > self.cfg.task_threshold).long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        task_ids = self._task_ids(x)
        output = torch.empty_like(x)
        for task_id in range(self.cfg.num_tasks):
            mask = task_ids == task_id
            if not mask.any():
                continue
            task_x = x[mask]
            if self.training:
                mean = task_x.mean(dim=0)
                var = task_x.var(dim=0, unbiased=False)
                with torch.no_grad():
                    self.running_mean[task_id] = (
                        (1 - self.cfg.momentum) * self.running_mean[task_id] + self.cfg.momentum * mean
                    )
                    self.running_var[task_id] = (
                        (1 - self.cfg.momentum) * self.running_var[task_id] + self.cfg.momentum * var
                    )
            else:
                mean = self.running_mean[task_id]
                var = self.running_var[task_id]
            output[mask] = (task_x - mean) / torch.sqrt(var + self.eps)
        return output


class IdentityNormalizerConfig(BaseConfig):
    def build(self, obs_space) -> nn.Identity:
        return nn.Identity()


AVAILABLE_NORMALIZERS = tp.Annotated[
    tp.Union[
        BatchNormNormalizerConfig,
        TaskBatchNormNormalizerConfig,
        IdentityNormalizerConfig,
    ],
    pydantic.Field(discriminator="name"),
]


class ObsNormalizerConfig(BaseConfig):
    # observation name -> normalizer
    normalizers: dict[str, AVAILABLE_NORMALIZERS] | AVAILABLE_NORMALIZERS = pydantic.Field(default_factory=dict)

    allow_mismatching_keys: bool = False

    def build(self, obs_space: spaces.Space) -> "ObsNormalizer":
        # Checking this sanity here so that imports work
        if isinstance(self.normalizers, dict) and len(self.normalizers) == 0:
            raise ValueError("ObsNormalizerConfig was initialized with no normalizers. Please provide at least one normalizer.")
        return ObsNormalizer(obs_space, self)


class ObsNormalizer(nn.Module):
    """Holder for all normalizers in dict or non-dict obs spaces."""

    def __init__(self, obs_space: spaces.Dict, cfg: ObsNormalizerConfig):
        super().__init__()
        self.cfg: ObsNormalizerConfig = cfg
        if isinstance(cfg.normalizers, dict):
            if not cfg.allow_mismatching_keys:
                if set(obs_space.keys()) != set(cfg.normalizers.keys()):
                    raise ValueError(
                        f"ObsNormalizerConfig keys {set(cfg.normalizers.keys())} do not match observation space keys {set(obs_space.keys())}. "
                        "Set allow_mismatching_keys=True to ignore this check."
                    )
            self._normalizers = nn.ModuleDict({key: cfg.normalizers[key].build(obs_space[key]) for key in cfg.normalizers.keys()})
        else:
            self._normalizers = cfg.normalizers.build(obs_space)

    def forward(self, x: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        # TODO is this is-instance check bad for performance?
        if isinstance(self.cfg.normalizers, dict):
            normalized_obs = {}
            for key in self._normalizers.keys():
                if key not in x:
                    if self.cfg.allow_mismatching_keys:
                        continue
                    else:
                        raise KeyError(f"Key '{key}' not found in the observation, but expected by normalizer.")
                tensor = x[key]
                normalized_obs[key] = self._normalizers[key](tensor)
            return normalized_obs
        else:
            return self._normalizers(x)
