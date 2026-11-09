"""Public data for the 15-site binary three-spin energy task.

The energy function is a fixed (unpublished) instance of
    H(S) = - sum_{i<j<k} J_ijk s_i s_j s_k,   s_i in {0, 1},  15 sites.
Only sampled (configuration, energy) pairs are public:

* ``three_spin_binary/train.csv`` - 16,000 training rows drawn uniformly with
  replacement from all 2^15 configurations (duplicates are kept);
* ``three_spin_binary/dev.csv`` - 2,000 distinct configurations that never occur
  in the training file, for local model checking (the visible train slice).

The hidden test uses further configurations that occur in neither file.
This module contains only data utilities, not any model code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_SITES = 15
DATA_DIR = Path(__file__).resolve().parent / "three_spin_binary"


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return spins as float32 (n, 15) in {0, 1} and energies as float32 (n,)."""

    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if arr.shape[1] != N_SITES + 1:
        raise ValueError(f"{path}: expected {N_SITES + 1} columns, found {arr.shape[1]}")
    return arr[:, :N_SITES].astype(np.float32), arr[:, N_SITES].astype(np.float32)


def load_binary(data_dir: Path = DATA_DIR) -> dict[str, np.ndarray]:
    x_train, y_train = load_csv(data_dir / "train.csv")
    x_dev, y_dev = load_csv(data_dir / "dev.csv")
    return {"x_train": x_train, "y_train": y_train, "x_dev": x_dev, "y_dev": y_dev}


def configuration_codes(x: np.ndarray) -> np.ndarray:
    """Integer code of each configuration: sum_i s_i 2^i."""

    weights = (1 << np.arange(N_SITES)).astype(np.int64)
    return (np.rint(x).astype(np.int64) * weights).sum(axis=1)
