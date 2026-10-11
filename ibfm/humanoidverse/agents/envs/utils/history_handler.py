import torch

from ...buffers.transition import _to_torch


class HistoryHandler:
    def __init__(self, num_envs, context_length, keys_dims, device):
        self.dims = keys_dims
        self.device = device
        self.num_envs = num_envs
        self.context_length = context_length
        self.history = {}
        self.lengths = {}

        for key in self.dims.keys():
            self.history[key] = torch.zeros(num_envs, self.context_length, self.dims[key], device=self.device)
            # we handle only the case where all the environments are reset at the same time
            self.lengths[key] = 0

    def reset(self, reset_ids):
        if len(reset_ids) == 0:
            return
        assert len(reset_ids) == self.num_envs, "reset_ids must match num_envs. We don't support partial resets."
        for key in self.history.keys():
            self.history[key][reset_ids] *= 0.0
            self.lengths[key] = 0

    def add(self, key: str, value: torch.Tensor):
        assert key in self.history.keys(), f"Key {key} not found in history"
        val = self.history[key].clone()
        self.history[key][:, 0:-1] = val[:, 1:]
        self.history[key][:, -1] = _to_torch(value, device=self.device).clone()
        self.lengths[key] += 1

    def query(self, key: str, filter_by_length: bool = True):
        assert key in self.history.keys(), f"Key {key} not found in history"
        if filter_by_length:
            return self.history[key][:, -self.lengths[key] :].clone()
        return self.history[key].clone()