import math

import torch
import torch.nn.functional as F


def sphere_log_map(base: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Map target to the tangent space at base while retaining arc length."""

    if base.shape != target.shape:
        raise ValueError(f"sphere_log_map shape mismatch: base={base.shape}, target={target.shape}")
    if base.shape[-1] < 2:
        raise ValueError("sphere_log_map requires a latent dimension of at least 2")

    radius = math.sqrt(base.shape[-1])
    x = F.normalize(base, dim=-1, eps=eps)
    y = F.normalize(target, dim=-1, eps=eps)
    cosine = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    tangent = y - cosine * x
    sine = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    theta = torch.atan2(sine, cosine)

    # The logarithm is undefined at the antipode. Use a deterministic
    # orthogonal direction there so its norm still records pi * radius.
    basis_index = x.abs().argmin(dim=-1, keepdim=True)
    basis = torch.zeros_like(x).scatter_(-1, basis_index, 1.0)
    antipodal_direction = F.normalize(basis - (basis * x).sum(dim=-1, keepdim=True) * x, dim=-1, eps=eps)
    direction = tangent / sine.clamp_min(eps)
    direction = torch.where((sine <= eps) & (cosine < 0.0), antipodal_direction, direction)
    return radius * theta * direction
