#!/usr/bin/env python3
"""Effective-model baseline for the double-exchange task: the paper's Eq. 3 on 5x5 tiles.

Paper (main text, Eq. 3 and the paragraph after Fig. 4):

    E_eff(S) = J_0 + sum_ij g_ij s_i.s_j + sum_ijkl g_ijkl (s_i.s_j)(s_k.s_l),

where the dot products include all pairs of spins, self-interactions included. "While the
full effective model would require 12,758,826 parameters, we reduce this to 213,201 by
restricting interactions within each 5x5 tile"; the coefficients come from linear
regression.

On the 10x10 training lattice there are four 5x5 tiles. Each tile has its own coefficients
for the 325 pairs (i <= j) of its 25 sites and for the 52,975 unordered pairs of those
pairs, plus one global constant: 1 + 4 * (325 + 52,975) = 213,201 coefficients, the
paper's count. Because s_i.s_i = 1, 31,400 of these terms duplicate others (a constant or
a two-spin term); they keep coefficient 0 and the regression runs on the 181,801 linearly
independent terms. Nothing couples spins in different tiles - that is the paper's
restriction.

Fitting. The paper fitted 2e6 * 0.8 = 1.6e6 configurations; this benchmark trains on
30,000, fewer than the number of independent terms, so plain least squares is
underdetermined. The fit is ridge regression (least squares with an L2 penalty on the
coefficients): the penalty is chosen on the last 10% of the training split from a fixed
grid, then the model is refitted on the whole split. The fit runs in the dual (kernel)
form with conjugate gradients and never forms the 30,000 x 181,801 design matrix.

The model is defined on the 10x10 lattice only, where the paper reports it (Fig. 4(b));
the evaluators score this baseline on 10x10 configurations only. The result is exported
as a TorchScript energy model (float32 spins (N, 3, 10, 10) -> energies (N,)) and, by
``scripts/baseline.sh de``, frozen into ``eval/*/baselines/de_effective_model.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.double_exchange import load_split  # noqa: E402

L = 10
TILE = 5
N_TILES = (L // TILE) ** 2  # 4
SITES = TILE * TILE  # 25
N_PAIRS = SITES * (SITES + 1) // 2  # 325 pairs i <= j (paper's parametrization)
N_QUADS = N_PAIRS * (N_PAIRS + 1) // 2  # 52,975 unordered pairs of pairs
IU_OFF = np.triu_indices(SITES, k=1)  # 300 distinct pairs i < j
IU_ALL = np.triu_indices(SITES)  # 325 pairs i <= j
NONSELF = np.nonzero(IU_ALL[0] != IU_ALL[1])[0]  # positions of the i < j pairs among the 325
LAMBDA_GRID = (1e-1, 1e0, 1e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5)
HOLDOUT_FRACTION = 0.1
INTERCEPT_SCALE = 1.0


def tile_spins_np(x: np.ndarray) -> np.ndarray:
    """(n, 3, 10, 10) spins -> (n, 4, 25, 3); tile index ty*2 + tx, site a*5 + b (row-major)."""

    n = x.shape[0]
    return x.astype(np.float64).reshape(n, 3, 2, TILE, 2, TILE).transpose(0, 2, 4, 3, 5, 1).reshape(n, N_TILES, SITES, 3)


def pair_features(x: np.ndarray, chunk: int = 4000) -> list[np.ndarray]:
    """Distinct two-spin terms s_i.s_j (i < j) of every tile: list of 4 arrays (n, 300)."""

    out = [np.empty((x.shape[0], IU_OFF[0].size)) for _ in range(N_TILES)]
    for i in range(0, x.shape[0], chunk):
        t = tile_spins_np(x[i : i + chunk])
        g = t @ t.transpose(0, 1, 3, 2)
        for k in range(N_TILES):
            out[k][i : i + chunk] = g[:, k, IU_OFF[0], IU_OFF[1]]
    return out


# ── ridge regression in the dual ─────────────────────────────────────────────
# Feature map (independent terms): [c] + per tile [x_p (300), x_p x_q (p < q), x_p^2 - 1/3].
# For dual weights a, the primal weights are: intercept c * sum(a); pair X^T a; quad
# (p < q) A_pq with A = X^T diag(a) X; quad (p = p) A_pp - sum(a) / 3.


def primal(xs: list[np.ndarray], a: np.ndarray) -> dict:
    sa = float(a.sum())
    pair, quad = [], []
    for xt in xs:
        pair.append(xt.T @ a)
        quad.append((xt * a[:, None]).T @ xt)
    return {"sa": sa, "pair": pair, "quad": quad}


def apply_primal(xs: list[np.ndarray], w: dict) -> np.ndarray:
    """Phi @ w for primal weights given in the dual-derived form above."""

    out = np.full(xs[0].shape[0], INTERCEPT_SCALE * INTERCEPT_SCALE * w["sa"])
    for xt, wp, a in zip(xs, w["pair"], w["quad"]):
        da = np.diag(a).copy()
        x2 = xt * xt
        q1 = np.einsum("ij,ij->i", xt @ a, xt)
        out += xt @ wp + 0.5 * (q1 - x2 @ da) + (x2 - 1.0 / 3.0) @ (da - w["sa"] / 3.0)
    return out


def kernel_apply(xs: list[np.ndarray], a: np.ndarray) -> np.ndarray:
    return apply_primal(xs, primal(xs, a))


def conjugate_gradient(xs: list[np.ndarray], b: np.ndarray, lam: float, x0: np.ndarray, tol: float = 1e-7, maxit: int = 400) -> tuple[np.ndarray, int, float]:
    x = x0.copy()
    r = b - (kernel_apply(xs, x) + lam * x)
    p = r.copy()
    rs = float(r @ r)
    bnorm = float(np.sqrt(b @ b))
    it = 0
    for it in range(1, maxit + 1):
        ap = kernel_apply(xs, p) + lam * p
        step = rs / float(p @ ap)
        x += step * p
        r -= step * ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol * bnorm:
            rs = rs_new
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, it, float(np.sqrt(rs) / bnorm)


def paper_parametrization(w: dict, y_mean: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Map the fitted weights onto the paper's 213,201 coefficients (J_0, 4 x 325, 4 x 52,975)."""

    j0 = y_mean + INTERCEPT_SCALE * INTERCEPT_SCALE * w["sa"]
    pair = np.zeros((N_TILES, N_PAIRS))
    quad = np.zeros((N_TILES, N_QUADS))
    qi = np.triu_indices(N_PAIRS)
    for t in range(N_TILES):
        pair[t, NONSELF] = w["pair"][t]
        a = w["quad"][t]
        diag = np.diag(a) - w["sa"] / 3.0
        full = np.zeros((N_PAIRS, N_PAIRS))
        sub = np.triu(a, k=1)  # coefficient of x_p x_q, p < q
        full[np.ix_(NONSELF, NONSELF)] = sub
        full[NONSELF, NONSELF] = diag  # coefficient of x_p^2 (the centring constant goes to J_0)
        j0 -= float(diag.sum()) / 3.0
        quad[t] = full[qi]
    return j0, pair, quad


class TileEffectiveModel(nn.Module):
    """E = J_0 + sum_tiles [ sum_{i<=j} g_ij s_i.s_j + sum_{(ij)<=(kl)} g_ijkl (s_i.s_j)(s_k.s_l) ]."""

    def __init__(self, j0: float, pair: torch.Tensor, quad: torch.Tensor):
        super().__init__()
        self.register_buffer("j0", torch.tensor([j0], dtype=torch.float32))
        self.register_buffer("pair", pair.to(torch.float32))
        self.register_buffer("quad", quad.to(torch.float32))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.shape[2] != 10 or s.shape[3] != 10:
            raise RuntimeError("the paper's tile effective model is defined on the 10x10 lattice only")
        b = s.shape[0]
        t = s.to(torch.float64).reshape(b, 3, 2, 5, 2, 5).permute(0, 2, 4, 3, 5, 1).reshape(b, 4, 25, 3)
        g = torch.matmul(t, t.transpose(2, 3)).reshape(b, 4, 625)
        iu = torch.triu_indices(25, 25)
        x = g.index_select(2, iu[0] * 25 + iu[1])
        lin = (x * self.pair.to(torch.float64)).sum(dim=2).sum(dim=1)
        qi = torch.triu_indices(325, 325)
        q = torch.zeros(4, 325 * 325, dtype=torch.float64)
        q.index_copy_(1, qi[0] * 325 + qi[1], self.quad.to(torch.float64))
        quad = (torch.matmul(x.unsqueeze(2), q.reshape(4, 325, 325)).squeeze(2) * x).sum(dim=2).sum(dim=1)
        return self.j0.to(torch.float64) + lin + quad


def relative_error(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)) / abs(float(np.mean(y))))


def predict_scripted(model, x: np.ndarray, batch: int = 250) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i : i + batch])).double().numpy() for i in range(0, x.shape[0], batch)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None, help="JSON with the penalty selection and fit diagnostics")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--train-limit", type=int, default=0, help="use only the first N training rows (smoke tests)")
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    t_start = time.time()

    x, y = load_split("train")
    if args.train_limit > 0:
        x, y = x[: args.train_limit], y[: args.train_limit]
    n = y.shape[0]
    n_hold = int(round(HOLDOUT_FRACTION * n))
    xs = pair_features(x)
    print(f"features: {n} rows, 4 tiles x 300 pair terms ({time.time() - t_start:.1f} s)", flush=True)

    # penalty selection on the last 10% of the training split
    fit_rows = slice(0, n - n_hold)
    hold_rows = slice(n - n_hold, n)
    xs_fit = [np.ascontiguousarray(xt[fit_rows]) for xt in xs]
    xs_hold = [np.ascontiguousarray(xt[hold_rows]) for xt in xs]
    y_fit, y_hold = y[fit_rows], y[hold_rows]
    mean_fit = float(np.mean(y_fit))
    alpha = np.zeros(y_fit.shape[0])
    grid = []
    for lam in sorted(LAMBDA_GRID, reverse=True):
        t0 = time.time()
        alpha, iters, resid = conjugate_gradient(xs_fit, y_fit - mean_fit, lam, alpha)
        w = primal(xs_fit, alpha)
        pred_hold = mean_fit + apply_primal(xs_hold, w)
        pred_fit = mean_fit + apply_primal(xs_fit, w)
        row = {
            "lambda": lam,
            "cg_iterations": iters,
            "cg_relative_residual": resid,
            "fit_relative_error": relative_error(y_fit, pred_fit),
            "holdout_relative_error": relative_error(y_hold, pred_hold),
            "seconds": round(time.time() - t0, 1),
        }
        grid.append(row)
        print(json.dumps(row), flush=True)
    best = min(grid, key=lambda r: r["holdout_relative_error"])
    lam = best["lambda"]
    print(f"selected lambda={lam} (holdout relative error {best['holdout_relative_error']:.6g})", flush=True)

    # final fit on the whole training split
    xs_all = [np.ascontiguousarray(xt) for xt in xs]
    mean_all = float(np.mean(y))
    alpha, iters, resid = conjugate_gradient(xs_all, y - mean_all, lam, np.zeros(n))
    w = primal(xs_all, alpha)
    pred_train = mean_all + apply_primal(xs_all, w)
    j0, pair, quad = paper_parametrization(w, mean_all)

    model = TileEffectiveModel(j0, torch.from_numpy(pair), torch.from_numpy(quad)).eval()
    scripted = torch.jit.script(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(args.output))
    loaded = torch.jit.load(str(args.output), map_location="cpu").eval()
    check = predict_scripted(loaded, x[:2000])
    gap = float(np.max(np.abs(check - pred_train[:2000])))
    if not gap < 1e-3:
        raise SystemExit(f"exported effective model disagrees with the fitted model (max gap {gap})")

    xd, yd = load_split("dev_L10")
    info = {
        "model": "effective_model (paper Eq. 3 restricted to 5x5 tiles)",
        "coefficients": int(1 + pair.size + quad.size),
        "independent_terms": int(1 + N_TILES * (IU_OFF[0].size + IU_OFF[0].size * (IU_OFF[0].size + 1) // 2)),
        "train_rows": int(n),
        "holdout_rows": int(n_hold),
        "lambda_grid": grid,
        "selected_lambda": lam,
        "final_cg_iterations": iters,
        "final_cg_relative_residual": resid,
        "export_gap": gap,
        "train_relative_error": relative_error(y, pred_train),
        "dev_L10_relative_error": relative_error(yd, predict_scripted(loaded, xd)),
        "seconds": round(time.time() - t_start, 1),
    }
    report = args.report if args.report is not None else args.output.with_suffix(".fit.json")
    report.write_text(json.dumps(info, indent=2))
    args.output.with_suffix(".json").write_text(json.dumps({"model": "effective_model", "metrics": {k: v for k, v in info.items() if k != "lambda_grid"}}, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
