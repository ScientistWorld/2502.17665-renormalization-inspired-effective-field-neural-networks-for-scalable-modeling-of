"""TorchScript probe models for the double-exchange experiment (audit only)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ExactED(nn.Module):
    """Computes E_MC exactly by diagonalizing the electron Hamiltonian inside the model."""

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        b = s.shape[0]
        h = s.shape[2]
        w = s.shape[3]
        n = h * w
        sd = s.to(torch.float64).reshape(b, 3, n)
        idx = torch.arange(n)
        r = torch.div(idx, w, rounding_mode="floor")
        c = idx % w
        right = r * w + (c + 1) % w
        down = ((r + 1) % h) * w + c
        hop = -(F.one_hot(right, n) + F.one_hot(down, n)).to(torch.float64)
        hop = (hop + hop.t()).to(torch.complex128).unsqueeze(0).expand(b, n, n)
        sx = sd[:, 0]
        sy = sd[:, 1]
        sz = sd[:, 2]
        zero = torch.zeros_like(sx)
        tl = hop + torch.diag_embed(torch.complex(-8.0 * sz + 8.3, zero))
        br = hop + torch.diag_embed(torch.complex(8.0 * sz + 8.3, zero))
        tr = torch.diag_embed(torch.complex(-8.0 * sx, 8.0 * sy))
        bl = torch.diag_embed(torch.complex(-8.0 * sx, -8.0 * sy))
        g = torch.cat([torch.cat([tl, tr], dim=2), torch.cat([bl, br], dim=2)], dim=1)
        e = torch.linalg.eigvalsh(g)
        return (-0.1 * F.softplus(-e / 0.1)).sum(dim=1).to(torch.float32)


class ConstantPerSite(nn.Module):
    """Predicts the mean training energy per site times the number of sites."""

    def __init__(self, site_mean: float):
        super().__init__()
        self.register_buffer("site_mean", torch.tensor(float(site_mean)))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return s[:, 0, 0, 0] * 0.0 + self.site_mean * (s.shape[2] * s.shape[3])


class SizeLocked(nn.Module):
    """The reference model, but only accepting 10 x 10 lattices (like a flattened MLP)."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        flat = s.reshape(s.shape[0], 300)
        return self.inner(flat.reshape(s.shape[0], 3, 10, 10))


class NaNOutput(nn.Module):
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return s[:, 0, 0, 0] * float("nan")


class WrongShape(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.inner(s)[1:]


class Zeros(nn.Module):
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return s[:, 0, 0, 0] * 0.0
