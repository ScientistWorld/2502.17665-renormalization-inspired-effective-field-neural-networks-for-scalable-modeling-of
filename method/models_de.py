"""Networks with O(3) symmetrization for the double-exchange energy task.

Paper, main text Fig. 3 / Fig. 4(a) and Supplementary Secs. IV-V:

* symmetrization layers T_j(S_0) = sum_{k=x,y,z} conv_{2j}(S_{0,k}) * conv_{2j+1}(S_{0,k}),
  j = 0..n, with two different bias-free filters per layer (Suppl. Eq. 55), which makes
  T_j invariant under global O(3) rotations of the spins;
* EFNN: F_1 = f_0(T_0), S_1 = g_1(T_1) * F_1, then F_j = f_{j-1}(S_{j-1}),
  S_j = g_j(T_j) * F_j for j = 2..n; every f and g is conv -> tanh -> conv;
* q: the final quasi-particle layer S_n is convolved to one channel and summed over
  the lattice to give the energy;
* DNN baseline: one symmetrization layer T_0 followed by the same conv-tanh-conv
  blocks without the S_0 injections (Suppl. Fig. 3(c)).

All convolutions use periodic (circular) padding, so a network trained on 10 x 10
lattices runs unchanged on larger ones. Implementation notes: the 2(n+1) filters of
the symmetrization layers are evaluated as one convolution with 2(n+1)C output
channels, and the n independent g_j maps as grouped convolutions (groups = n); both
are exactly the separate per-layer operations of the paper, just batched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CircConv(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, bias: bool = True, groups: int = 1):
        super().__init__()
        self.pad = kernel // 2
        self.conv = nn.Conv2d(cin, cout, kernel, padding=0, bias=bias, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pad
        if p > 0:
            x = F.pad(x, [p, p, p, p], mode="circular")
        return self.conv(x)


class Block(nn.Module):
    """conv -> tanh -> conv (the paper's f_j / g_j maps); groups > 1 runs independent maps side by side."""

    def __init__(self, channels: int, kernel: int, groups: int = 1):
        super().__init__()
        self.c1 = CircConv(channels, channels, kernel, groups=groups)
        self.c2 = CircConv(channels, channels, kernel, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c2(torch.tanh(self.c1(x)))


class Symmetrizer(nn.Module):
    """T_j(S_0) for j = 0..m-1, returned as (B, m, C, L, L)."""

    def __init__(self, m: int, channels: int, kernel: int):
        super().__init__()
        self.m = m
        self.channels = channels
        self.conv = CircConv(1, 2 * m * channels, kernel, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        b = s.shape[0]
        h = s.shape[2]
        w = s.shape[3]
        z = self.conv(s.reshape(b * 3, 1, h, w)).reshape(b, 3, self.m, 2, self.channels, h, w)
        return (z[:, :, :, 0] * z[:, :, :, 1]).sum(dim=1)


class EFNNSym(nn.Module):
    def __init__(self, channels: int = 3, layers: int = 3, kernel: int = 5):
        super().__init__()
        self.layers = layers
        self.channels = channels
        self.sym = Symmetrizer(layers + 1, channels, kernel)
        self.f = nn.ModuleList([Block(channels, kernel) for _ in range(layers)])
        self.g = Block(layers * channels, kernel, groups=layers)
        self.q = CircConv(channels, 1, kernel)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        t = self.sym(s)
        b = t.shape[0]
        h = t.shape[3]
        w = t.shape[4]
        gt = self.g(t[:, 1:].reshape(b, self.layers * self.channels, h, w)).reshape(b, self.layers, self.channels, h, w)
        x = t[:, 0]
        for i, f in enumerate(self.f):
            x = gt[:, i] * f(x)
        return self.q(x).sum(dim=[1, 2, 3])


class DNNSym(nn.Module):
    def __init__(self, channels: int = 3, layers: int = 3, kernel: int = 5):
        super().__init__()
        self.sym = Symmetrizer(1, channels, kernel)
        self.f = nn.ModuleList([Block(channels, kernel) for _ in range(layers)])
        self.q = CircConv(channels, 1, kernel)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.sym(s)[:, 0]
        for f in self.f:
            x = f(x)
        return self.q(x).sum(dim=[1, 2, 3])


class EnergyModelDE(nn.Module):
    """Network trained on standardized energies; returns raw Monte Carlo energies.

    E(S) = scale * net(S) + site_mean * (number of lattice sites), so the prediction is
    extensive and the same model applies to every lattice size.
    """

    def __init__(self, net: nn.Module, site_mean: float, scale: float):
        super().__init__()
        self.net = net
        self.register_buffer("site_mean", torch.tensor(float(site_mean), dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        sites = s.shape[2] * s.shape[3]
        return self.net(s) * self.scale + self.site_mean * sites


def build_de_model(name: str, channels: int, layers: int, kernel: int) -> nn.Module:
    table = {"efnn": EFNNSym, "dnn": DNNSym}
    if name not in table:
        raise ValueError(f"unknown model {name}; expected one of {sorted(table)}")
    return table[name](channels=channels, layers=layers, kernel=kernel)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
