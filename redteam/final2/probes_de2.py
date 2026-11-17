"""TorchScript probes for the double-exchange experiment added in this iteration (audit only)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

T_PHYS = 0.1
CENTER = 8.3  # spectrum of the electron matrix lies in [-3.7, 20.3]
HALF = 12.5


def chebyshev_coefficients(degree: int) -> np.ndarray:
    """Chebyshev interpolation coefficients of f(E) = -T log(1 + exp(-E/T)), E = CENTER + HALF * u."""

    n = degree + 1
    k = np.arange(n)
    u = np.cos(np.pi * (k + 0.5) / n)
    e = CENTER + HALF * u
    f = -T_PHYS * (np.log1p(np.exp(-np.abs(e) / T_PHYS)) + np.maximum(-e / T_PHYS, 0.0))
    j = np.arange(n)[:, None]
    c = (2.0 / n) * (np.cos(np.pi * j * (k[None, :] + 0.5) / n) @ f)
    c[0] /= 2.0
    return c


class ChebyshevFreeEnergy(nn.Module):
    """E_MC of the electron Hamiltonian computed without any linear-algebra solver.

    Builds the 2L^2 x 2L^2 matrix of paper Eq. 69 and evaluates tr f(H) for
    f(E) = -T log(1 + exp(-E/T)) by a Chebyshev expansion in H, using matrix products only.
    Out of bounds (DIRECTION.md): it computes the energy from the Hamiltonian. It passes
    the evaluator's op scan; this probe checks that the compute meter stops it.
    """

    def __init__(self, degree: int = 700):
        super().__init__()
        self.register_buffer("coef", torch.from_numpy(chebyshev_coefficients(degree)))

    def hamiltonian(self, s: torch.Tensor) -> torch.Tensor:
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
        return torch.cat([torch.cat([tl, tr], dim=2), torch.cat([bl, br], dim=2)], dim=1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        g = self.hamiltonian(s)
        m = g.shape[1]
        eye = torch.eye(m, dtype=torch.complex128).unsqueeze(0).expand(g.shape[0], m, m)
        x = (g - CENTER * eye) / HALF
        t0 = eye
        t1 = x
        acc = self.coef[0] * float(m) + self.coef[1] * torch.diagonal(x, dim1=1, dim2=2).sum(dim=1).real
        for k in range(2, self.coef.shape[0]):
            t2 = 2.0 * torch.matmul(x, t1) - t0
            acc = acc + self.coef[k] * torch.diagonal(t2, dim1=1, dim2=2).sum(dim=1).real
            t0 = t1
            t1 = t2
        return acc
