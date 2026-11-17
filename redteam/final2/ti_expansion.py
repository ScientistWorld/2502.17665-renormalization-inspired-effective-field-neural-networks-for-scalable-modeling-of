#!/usr/bin/env python3
"""Translation-invariant local effective model (VARIANT, not a baseline; previously the frozen baseline).

The paper compares its networks with an effective spin model fitted by linear
regression:  E_eff = J_0 + sum_ij g_ij s_i.s_j + sum_ijkl g_ijkl (s_i.s_j)(s_k.s_l).
Here the expansion is translation invariant (one coefficient per relative geometry,
which the periodic lattice allows) and local, so the fitted model applies to every
lattice size:

* constant: J_0 per site;
* 2-body: s_i.s_{i+d} for every displacement d with |d_x|, |d_y| <= 3 (24 classes);
* 4-body: (s_i.s_{i+b1}) (s_{i+e}.s_{i+e+b2}) for bonds b1, b2 among the nearest and
  diagonal neighbours and offsets |e_x|, |e_y| <= 2 (202 classes).

Coefficients come from ordinary least squares on the training split. The result is
exported as a TorchScript energy model (float32 spins (N, 3, L, L) -> energies (N,)).
Until this iteration this model was the frozen ``effective_model`` baseline; it is much
stronger than the paper's own effective model (Eq. 3 restricted to 5x5 tiles, now the
baseline in ``baseline/effective_model_de.py``) and is kept here as an honest variant,
scored as a submission by ``redteam/final2/make_cases.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.double_exchange import load_split  # noqa: E402

PAIR_RANGE = 3
QUAD_RANGE = 2
BONDS = [(1, 0), (0, 1), (1, 1), (1, -1)]


def pair_table() -> list[tuple[int, int]]:
    out = []
    for dx in range(-PAIR_RANGE, PAIR_RANGE + 1):
        for dy in range(-PAIR_RANGE, PAIR_RANGE + 1):
            if (dx, dy) > (0, 0):
                out.append((dx, dy))
    return out


def quad_table() -> list[tuple[int, int, int, int]]:
    out = []
    for b1 in range(len(BONDS)):
        for b2 in range(b1, len(BONDS)):
            for ex in range(-QUAD_RANGE, QUAD_RANGE + 1):
                for ey in range(-QUAD_RANGE, QUAD_RANGE + 1):
                    if b1 == b2 and (ex, ey) < (0, 0):
                        continue
                    out.append((b1, b2, ex, ey))
    return out


class EffectiveModel(nn.Module):
    def __init__(self, coef: torch.Tensor):
        super().__init__()
        self.register_buffer("pairs", torch.tensor(pair_table(), dtype=torch.int16))
        self.register_buffer("quads", torch.tensor(quad_table(), dtype=torch.int16))
        self.register_buffer("bond_shifts", torch.tensor(BONDS, dtype=torch.int16))
        self.register_buffer("coef", coef.to(torch.float32))

    def features(self, s: torch.Tensor) -> torch.Tensor:
        b = s.shape[0]
        sites = s.shape[2] * s.shape[3]
        feats = [torch.full([b], float(sites), dtype=s.dtype, device=s.device)]
        for p in range(self.pairs.shape[0]):
            dx = int(self.pairs[p, 0])
            dy = int(self.pairs[p, 1])
            feats.append((s * torch.roll(s, shifts=[dx, dy], dims=[2, 3])).sum(dim=[1, 2, 3]))
        bonds = []
        for k in range(self.bond_shifts.shape[0]):
            dx = int(self.bond_shifts[k, 0])
            dy = int(self.bond_shifts[k, 1])
            bonds.append((s * torch.roll(s, shifts=[dx, dy], dims=[2, 3])).sum(dim=1))
        for q in range(self.quads.shape[0]):
            b1 = int(self.quads[q, 0])
            b2 = int(self.quads[q, 1])
            ex = int(self.quads[q, 2])
            ey = int(self.quads[q, 3])
            feats.append((bonds[b1] * torch.roll(bonds[b2], shifts=[ex, ey], dims=[1, 2])).sum(dim=[1, 2]))
        return torch.stack(feats, dim=1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.features(s) @ self.coef.to(s.dtype)


def design(model: EffectiveModel, x: np.ndarray, batch: int = 1000) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            out.append(model.features(torch.from_numpy(x[i : i + batch]).double()).numpy())
    return np.concatenate(out)


def relative_error(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)) / abs(float(np.mean(y))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))

    n_feat = 1 + len(pair_table()) + len(quad_table())
    probe = EffectiveModel(torch.zeros(n_feat, dtype=torch.float64))
    x, y = load_split("train")
    a = design(probe, x)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    model = EffectiveModel(torch.from_numpy(coef)).eval()
    info: dict[str, float] = {"features": float(n_feat), "train_relative_error": relative_error(y, a @ coef)}
    scripted = torch.jit.script(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(args.output))
    loaded = torch.jit.load(str(args.output), map_location="cpu").eval()
    for split in ("dev_L10", "dev_L16", "dev_L24"):
        xd, yd = load_split(split)
        with torch.no_grad():
            pred = np.concatenate([loaded(torch.from_numpy(xd[i : i + 200])).double().numpy() for i in range(0, xd.shape[0], 200)])
        info[f"{split}_relative_error"] = relative_error(yd, pred)
    args.output.with_suffix(".json").write_text(json.dumps({"model": "effective_model", "metrics": info}, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
