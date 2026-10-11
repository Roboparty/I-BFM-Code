from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.distributed as dist


_DEFAULT_BUCKET_CAP_BYTES = 64 * 1024 * 1024
_MAX_COALESCE_TENSOR_BYTES = 1024 * 1024


def _dense_tensor_buckets(
    tensors: Iterable[torch.Tensor],
    *,
    bucket_cap_bytes: int = _DEFAULT_BUCKET_CAP_BYTES,
    max_coalesce_tensor_bytes: int = _MAX_COALESCE_TENSOR_BYTES,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    if bucket_cap_bytes <= 0:
        raise ValueError(f"bucket_cap_bytes must be positive, got {bucket_cap_bytes}.")

    buckets: list[list[torch.Tensor]] = []
    direct_tensors: list[torch.Tensor] = []
    active: dict[tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], int]] = {}
    for tensor in tensors:
        tensor_nbytes = tensor.numel() * tensor.element_size()
        if tensor.is_sparse or tensor_nbytes >= max_coalesce_tensor_bytes:
            direct_tensors.append(tensor)
            continue
        key = (tensor.device, tensor.dtype)
        bucket, bucket_nbytes = active.get(key, ([], 0))
        if bucket and bucket_nbytes + tensor_nbytes > bucket_cap_bytes:
            buckets.append(bucket)
            bucket, bucket_nbytes = [], 0
        bucket.append(tensor)
        active[key] = (bucket, bucket_nbytes + tensor_nbytes)

    buckets.extend(bucket for bucket, _ in active.values() if bucket)
    return buckets, direct_tensors


def _all_reduce_tensors_(
    tensors: Iterable[torch.Tensor],
    *,
    average: bool,
    bucket_cap_bytes: int = _DEFAULT_BUCKET_CAP_BYTES,
) -> None:
    tensor_list = list(tensors)
    if not tensor_list:
        return
    scale = 1.0 / float(dist.get_world_size()) if average else 1.0
    buckets, direct_tensors = _dense_tensor_buckets(tensor_list, bucket_cap_bytes=bucket_cap_bytes)
    with torch.no_grad():
        for bucket in buckets:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                dist.all_reduce_coalesced(bucket, op=dist.ReduceOp.SUM)
            if average:
                torch._foreach_mul_(bucket, scale)
        for tensor in direct_tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average and direct_tensors:
            torch._foreach_mul_(direct_tensors, scale)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else int(os.environ.get("WORLD_SIZE", "1"))


def barrier() -> None:
    if is_distributed():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


@torch.compiler.disable
def average_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    if not is_distributed():
        return
    _all_reduce_tensors_((param.grad for param in parameters if param.grad is not None), average=True)


@torch.compiler.disable
def broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)


@torch.compiler.disable
def sync_floating_buffers(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed():
        return
    floating_buffers = []
    for buffer in module.buffers():
        if buffer.dtype.is_floating_point or buffer.dtype.is_complex:
            floating_buffers.append(buffer.data)
        else:
            dist.broadcast(buffer.data, src=src)
    _all_reduce_tensors_(floating_buffers, average=True)


@torch.compiler.disable
def broadcast_optimizer_state(optimizer: torch.optim.Optimizer, src: int = 0) -> None:
    if not is_distributed():
        return
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                dist.broadcast(value.data, src=src)


def agent_optimizers(agent: Any) -> list[torch.optim.Optimizer]:
    names = [
        "actor_optimizer",
        "backward_optimizer",
        "forward_optimizer",
        "critic_optimizer",
        "discriminator_optimizer",
        "aux_critic_optimizer",
    ]
    return [getattr(agent, name) for name in names if hasattr(agent, name)]


def broadcast_agent_state(agent: Any, src: int = 0) -> None:
    if not is_distributed():
        return
    broadcast_module_state(agent._model, src=src)
    for optimizer in agent_optimizers(agent):
        broadcast_optimizer_state(optimizer, src=src)


@torch.compiler.disable
def broadcast_object(value: Any, src: int = 0) -> Any:
    if not is_distributed():
        return value
    objects = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


@torch.compiler.disable
def all_gather_objects(value: Any) -> list[Any]:
    """Gather a small Python payload from every distributed rank.

    Tracking evaluation returns scalar metric dictionaries, so object
    collectives keep the implementation simple and avoid padding a variable
    number of per-motion results into tensors.  The payload is intentionally
    gathered on every rank; this is supported by NCCL and the dictionaries are
    tiny compared with the model, replay buffer, and simulation state.
    """

    if not is_distributed():
        return [value]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


@torch.compiler.disable
def module_sync_report(module: torch.nn.Module, src: int = 0) -> dict[str, Any]:
    if not is_distributed():
        return {"world_size": 1, "rank": 0, "max_abs_diff_from_rank0": 0.0}

    tensors = list(module.parameters()) + [
        buffer for buffer in module.buffers() if buffer.dtype.is_floating_point or buffer.dtype.is_complex
    ]
    if tensors:
        device = tensors[0].device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_summary = torch.zeros(3, dtype=torch.float64, device=device)
    max_abs_diff = torch.zeros((), dtype=torch.float32, device=device)

    for tensor in tensors:
        detached = tensor.detach()
        local_summary[0] += detached.double().sum()
        local_summary[1] += detached.double().square().sum()
        local_summary[2] += detached.numel()
        reference = detached.clone()
        dist.broadcast(reference, src=src)
        if detached.numel() > 0:
            diff = (detached - reference).abs().max().float()
            max_abs_diff = torch.maximum(max_abs_diff, diff)

    dist.all_reduce(max_abs_diff, op=dist.ReduceOp.MAX)
    gathered = [torch.empty_like(local_summary) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_summary)
    summaries = torch.stack(gathered).cpu()
    return {
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "max_abs_diff_from_rank0": float(max_abs_diff.cpu().item()),
        "rank_param_buffer_sum": [float(x) for x in summaries[:, 0].tolist()],
        "rank_param_buffer_sqsum": [float(x) for x in summaries[:, 1].tolist()],
        "rank_param_buffer_numel": [int(x) for x in summaries[:, 2].tolist()],
    }


@torch.compiler.disable
def average_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not is_distributed():
        return dict(metrics)
    reduced: dict[str, Any] = {}
    tensor_items = []
    for key, value in metrics.items():
        if torch.is_tensor(value):
            tensor = value.detach().clone()
            if not (tensor.dtype.is_floating_point or tensor.dtype.is_complex):
                tensor = tensor.float()
            reduced[key] = tensor
            tensor_items.append(tensor)
        else:
            reduced[key] = value
    _all_reduce_tensors_(tensor_items, average=True)
    return reduced
