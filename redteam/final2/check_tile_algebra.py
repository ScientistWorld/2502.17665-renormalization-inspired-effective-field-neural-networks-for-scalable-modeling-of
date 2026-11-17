#!/usr/bin/env python3
"""Check the tile effective model's dual/primal algebra against an explicit feature matrix.

On 60 random configurations: (1) the matrix-free kernel product equals Phi Phi^T a;
(2) conjugate gradients solve the ridge system; (3) the paper's 213,201-coefficient
parametrization (and its TorchScript export) reproduce the fitted function.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import baseline.effective_model_de as em  # noqa: E402


def phi(xs: list[np.ndarray], i: int) -> np.ndarray:
    f = [em.INTERCEPT_SCALE]
    iu = np.triu_indices(300, k=1)
    for xt in xs:
        p = xt[i]
        f += list(p)
        f += list(np.outer(p, p)[iu])
        f += list(p * p - 1.0 / 3.0)
    return np.asarray(f)


def main() -> None:
    rng = np.random.default_rng(0)
    n = 60
    v = rng.standard_normal((n, 10, 10, 3))
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    x = np.ascontiguousarray(v.transpose(0, 3, 1, 2)).astype(np.float32)
    xs = em.pair_features(x)
    P = np.stack([phi(xs, i) for i in range(n)])
    K = P @ P.T
    a = rng.standard_normal(n)
    out: dict = {"independent_features": int(P.shape[1])}
    out["kernel_apply_rel_err"] = float(np.max(np.abs(em.kernel_apply(xs, a) - K @ a)) / np.max(np.abs(K @ a)))
    y = rng.standard_normal(n)
    lam = 50.0
    al, it, res = em.conjugate_gradient(xs, y, lam, np.zeros(n), tol=1e-12, maxit=500)
    out["cg_err"] = float(np.max(np.abs(al - np.linalg.solve(K + lam * np.eye(n), y))))
    out["cg_iterations"] = it
    w = em.primal(xs, al)
    f1 = em.apply_primal(xs, w)
    out["primal_vs_explicit"] = float(np.max(np.abs(f1 - P @ (P.T @ al))))
    j0, pair, quad = em.paper_parametrization(w, 0.0)
    out["coefficients"] = int(1 + pair.size + quad.size)
    model = torch.jit.script(em.TileEffectiveModel(j0, torch.from_numpy(pair), torch.from_numpy(quad)).eval())
    with torch.no_grad():
        f2 = model(torch.from_numpy(x)).double().numpy()
    out["torchscript_vs_fit"] = float(np.max(np.abs(f2 - f1)))
    out["scale"] = float(np.max(np.abs(f1)))
    print(json.dumps(out, indent=2))
    ok = out["kernel_apply_rel_err"] < 1e-10 and out["cg_err"] < 1e-6 and out["torchscript_vs_fit"] < 1e-4 * max(1.0, out["scale"]) and out["coefficients"] == 213201
    print("TILE ALGEBRA OK" if ok else "TILE ALGEBRA FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
