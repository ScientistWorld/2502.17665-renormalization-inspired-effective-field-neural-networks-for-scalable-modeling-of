"""TorchScript probe models for the double-exchange experiment (audit only)."""

from __future__ import annotations

from typing import List, Tuple

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


class TableEffectiveModel(nn.Module):
    """Larger translation-invariant effective model (paper Eq. 3), tables kept as code constants.

    2-body s_i.s_{i+d} for |d| <= pair_range; 4-body (s_i.s_{i+b1})(s_{i+e}.s_{i+e+b2}) for bonds
    b1, b2 with |b| <= 2 and offsets |e| <= 2. Probes how far explicit enumeration goes in budget.
    """

    def __init__(self, pairs: List[Tuple[int, int]], bonds: List[Tuple[int, int]], quads: List[Tuple[int, int, int, int]], coef: torch.Tensor):
        super().__init__()
        self.pairs = pairs
        self.bonds = bonds
        self.quads = quads
        self.register_buffer("coef", coef.to(torch.float32))

    def features(self, s: torch.Tensor) -> torch.Tensor:
        b = s.shape[0]
        sites = s.shape[2] * s.shape[3]
        feats = [torch.full([b], float(sites), dtype=s.dtype, device=s.device)]
        for dx, dy in self.pairs:
            feats.append((s * torch.roll(s, shifts=[dx, dy], dims=[2, 3])).sum(dim=[1, 2, 3]))
        bond_maps = []
        for dx, dy in self.bonds:
            bond_maps.append((s * torch.roll(s, shifts=[dx, dy], dims=[2, 3])).sum(dim=1))
        for b1, b2, ex, ey in self.quads:
            feats.append((bond_maps[b1] * torch.roll(bond_maps[b2], shifts=[ex, ey], dims=[1, 2])).sum(dim=[1, 2]))
        return torch.stack(feats, dim=1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.features(s) @ self.coef.to(s.dtype)


def large_tables(pair_range: int = 4, bond_range: int = 2, quad_range: int = 2):
    pairs = [(dx, dy) for dx in range(-pair_range, pair_range + 1) for dy in range(-pair_range, pair_range + 1) if (dx, dy) > (0, 0)]
    bonds = [(dx, dy) for dx in range(-bond_range, bond_range + 1) for dy in range(-bond_range, bond_range + 1) if (dx, dy) > (0, 0)]
    quads = []
    for b1 in range(len(bonds)):
        for b2 in range(b1, len(bonds)):
            for ex in range(-quad_range, quad_range + 1):
                for ey in range(-quad_range, quad_range + 1):
                    if b1 == b2 and (ex, ey) < (0, 0):
                        continue
                    quads.append((b1, b2, ex, ey))
    return pairs, bonds, quads
