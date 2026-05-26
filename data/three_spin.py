"""Data generation for the paper's 15-site three-spin energy tasks.

The paper evaluates models on
    H(S) = - sum_{i<j<k} J_ijk s_i s_j s_k
for either binary spins in {0, 1} or continuous spins in [-1, 1].
This module contains only problem data utilities, not any model code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThreeSpinConfig:
    n_sites: int = 15
    seed: int = 20250217
    train_size: int = 16000
    test_size: int = 4000


def interaction_indices(n_sites: int) -> np.ndarray:
    triples = [(i, j, k) for i in range(n_sites) for j in range(i + 1, n_sites) for k in range(j + 1, n_sites)]
    return np.asarray(triples, dtype=np.int64)


def couplings(n_sites: int = 15, seed: int = 20250217) -> np.ndarray:
    """Deterministic J_ijk values used by all methods in this gym.

    The paper states that J_ijk are constants but does not publish their values
    in the manuscript. We fix a reproducible nontrivial instance with O(1)
    couplings and keep it shared across EFNN and baselines.
    """

    rng = np.random.default_rng(seed)
    triples = interaction_indices(n_sites)
    values = rng.normal(loc=0.0, scale=1.0, size=len(triples)).astype(np.float32)
    values += 0.15 * rng.choice([-1.0, 1.0], size=len(triples)).astype(np.float32)
    return values


def energies(spins: np.ndarray, j_values: np.ndarray | None = None, n_sites: int = 15, seed: int = 20250217) -> np.ndarray:
    if j_values is None:
        j_values = couplings(n_sites=n_sites, seed=seed)
    triples = interaction_indices(n_sites)
    products = spins[:, triples[:, 0]] * spins[:, triples[:, 1]] * spins[:, triples[:, 2]]
    return -(products @ j_values).astype(np.float32)


def make_split(kind: str, cfg: ThreeSpinConfig = ThreeSpinConfig()) -> dict[str, np.ndarray]:
    if kind not in {"binary", "continuous"}:
        raise ValueError(f"unknown three-spin task kind: {kind}")

    rng = np.random.default_rng(cfg.seed + (0 if kind == "binary" else 1000))
    total = cfg.train_size + cfg.test_size
    if kind == "binary":
        spins = rng.integers(0, 2, size=(total, cfg.n_sites), dtype=np.int8).astype(np.float32)
    else:
        spins = rng.uniform(-1.0, 1.0, size=(total, cfg.n_sites)).astype(np.float32)

    j_values = couplings(n_sites=cfg.n_sites, seed=cfg.seed)
    y = energies(spins, j_values=j_values, n_sites=cfg.n_sites, seed=cfg.seed)
    return {
        "x_train": spins[: cfg.train_size],
        "y_train": y[: cfg.train_size],
        "x_test": spins[cfg.train_size :],
        "y_test": y[cfg.train_size :],
        "couplings": j_values,
        "triples": interaction_indices(cfg.n_sites),
    }


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, kind: str) -> dict[str, float]:
    err = y_pred.astype(np.float64) - y_true.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    out = {"rmse": rmse}
    if kind == "binary":
        denom = float(abs(np.mean(y_true)))
        out["relative_error"] = rmse / max(denom, 1e-12)
    else:
        out["mae"] = mae
    return out
