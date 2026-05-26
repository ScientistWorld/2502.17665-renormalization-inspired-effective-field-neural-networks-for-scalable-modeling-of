"""EFNN and comparison networks for the three-spin experiments."""

from __future__ import annotations

import torch
from torch import nn


def mlp_block(width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))


class EFNN(nn.Module):
    """Effective Field Neural Network from the paper.

    For each field-particle layer i:
        F_i = f_{i-1}(S_{i-1})
        S_i = g_i(S_0) * F_i
    The final energy is obtained by summing a learned per-site projection.
    """

    def __init__(self, n_sites: int = 15, width: int = 105, layers: int = 2):
        super().__init__()
        self.input = nn.Linear(n_sites, width) if width != n_sites else nn.Identity()
        self.f_layers = nn.ModuleList([mlp_block(width) for _ in range(layers)])
        self.g_layers = nn.ModuleList([mlp_block(width) for _ in range(layers)])
        self.output = nn.Linear(width, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.input(x)
        s = s0
        for f_layer, g_layer in zip(self.f_layers, self.g_layers):
            field = f_layer(s)
            s = g_layer(s0) * field
        return self.output(s).squeeze(-1)


class DNN(nn.Module):
    def __init__(self, n_sites: int = 15, width: int = 105, layers: int = 2):
        super().__init__()
        parts: list[nn.Module] = [nn.Linear(n_sites, width), nn.Tanh()]
        for _ in range(max(layers - 1, 0)):
            parts.extend([nn.Linear(width, width), nn.Tanh()])
        parts.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResNet(nn.Module):
    def __init__(self, n_sites: int = 15, width: int = 105, layers: int = 2):
        super().__init__()
        self.input = nn.Linear(n_sites, width)
        self.blocks = nn.ModuleList([mlp_block(width) for _ in range(layers)])
        self.output = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.input(x))
        for block in self.blocks:
            h = h + block(h)
        return self.output(h).squeeze(-1)


class DenseNet(nn.Module):
    def __init__(self, n_sites: int = 15, width: int = 105, layers: int = 2):
        super().__init__()
        self.input = nn.Linear(n_sites, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width * (i + 1), width), nn.Tanh()) for i in range(layers)])
        self.output = nn.Linear(width * (layers + 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [torch.tanh(self.input(x))]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1)))
        return self.output(torch.cat(feats, dim=-1)).squeeze(-1)


def build_model(name: str, n_sites: int, width: int, layers: int) -> nn.Module:
    table = {"efnn": EFNN, "dnn": DNN, "resnet": ResNet, "densenet": DenseNet}
    if name not in table:
        raise ValueError(f"unknown model {name}; expected one of {sorted(table)}")
    return table[name](n_sites=n_sites, width=width, layers=layers)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
