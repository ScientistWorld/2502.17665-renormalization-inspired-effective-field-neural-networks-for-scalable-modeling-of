"""Public data for the double-exchange energy task (classical spins on a square lattice).

Each configuration is an L x L periodic lattice of classical unit spins, stored as
float32 of shape (n, 3, L, L) (channels: x, y, z components). Its target is the
electronic Monte Carlo energy of that spin background (a free energy obtained by
exact diagonalization of the itinerant-electron Hamiltonian; see
``briefing/problem.md``). Spins are drawn independently and uniformly on the unit
sphere.

Spin inputs are regenerated deterministically from the public seeds below (legacy
``numpy.random.RandomState``, whose stream is frozen across numpy versions) and
checked against the SHA-256 recorded in ``double_exchange/meta.json``. Energies are
the committed CSV files in ``double_exchange/``.

* ``train``   - 30,000 training configurations on the 10 x 10 lattice;
* ``dev_L10`` - 2,000 further 10 x 10 configurations for local checking;
* ``dev_L16``, ``dev_L24`` - 300 and 100 configurations on larger lattices, for
  checking how a model trained on 10 x 10 transfers to bigger systems.

The hidden test uses further configurations of the same three sizes, drawn from
seeds that are not public. This module contains only data utilities.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "double_exchange"
TRAIN_L = 10

SPLITS: dict[str, dict] = {
    "train": {"L": 10, "n": 30000, "seed": 610_001, "file": "train_L10.csv"},
    "dev_L10": {"L": 10, "n": 2000, "seed": 620_010, "file": "dev_L10.csv"},
    "dev_L16": {"L": 16, "n": 300, "seed": 620_016, "file": "dev_L16.csv"},
    "dev_L24": {"L": 24, "n": 100, "seed": 620_024, "file": "dev_L24.csv"},
}
DEV_SPLITS = ("dev_L10", "dev_L16", "dev_L24")


def sample_spins(n: int, L: int, seed: int) -> np.ndarray:
    """n configurations of L x L unit spins, float32 (n, 3, L, L), uniform on the sphere."""

    rng = np.random.RandomState(seed)
    v = rng.standard_normal((n, L, L, 3))
    v /= np.sqrt(np.sum(v * v, axis=-1, keepdims=True))
    return np.ascontiguousarray(v.transpose(0, 3, 1, 2)).astype(np.float32)


def spins_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()


def read_energies(path: Path) -> np.ndarray:
    lines = path.read_text().split()
    if not lines or lines[0] != "energy":
        raise ValueError(f"{path}: expected a header line 'energy'")
    return np.asarray([float(v) for v in lines[1:]], dtype=np.float64)


def load_split(name: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return spins float32 (n, 3, L, L) and energies float64 (n,) of a public split."""

    spec = SPLITS[name]
    x = sample_spins(spec["n"], spec["L"], spec["seed"])
    y = read_energies(data_dir / spec["file"])
    if y.shape != (spec["n"],):
        raise ValueError(f"{name}: expected {spec['n']} energies, found {y.shape[0]}")
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        expected = json.loads(meta_path.read_text())["splits"][name]["spins_sha256"]
        if spins_sha256(x) != expected:
            raise ValueError(f"{name}: regenerated spins do not match meta.json; numpy stream changed?")
    return x, y
