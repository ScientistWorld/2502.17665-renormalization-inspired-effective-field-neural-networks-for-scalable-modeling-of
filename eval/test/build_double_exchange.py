#!/usr/bin/env python3
"""One-off builder for the double-exchange benchmark (run inside the compute container).

This file lives in ``eval/test/`` because it holds the hidden test seeds. It computes,
by exact diagonalization, the Monte Carlo energy of the paper's 2D double-exchange
model (paper Eqs. 2 and 78; Supplementary Eq. 69)

    Gamma(S) = (T_x + T_y) (x) 1_2  -  J/2 [[S^z, S^x - i S^y], [S^x + i S^y, -S^z]]  -  mu 1,
    E_MC(S)  = -T sum_j log(1 + exp(-E_j(S) / T)),     E_j = eigenvalues of Gamma(S),

with nearest-neighbour hopping t = 1 on an L x L periodic square lattice (row-major
site index r * L + s), J = 16 t, mu = -8.3 t and T = 0.1 t, as in the paper. It writes

* ``data/double_exchange/<split>.csv`` - public energies of the public splits listed in
  ``data/double_exchange.py`` (their spins regenerate from public seeds), plus
  ``data/double_exchange/meta.json`` with the spins' SHA-256 and summary statistics;
* ``eval/test/double_exchange_instance.json`` - hidden test seeds, sizes, spin
  checksums and energies for 10 x 10, 16 x 16 and 24 x 24 lattices.

Run from the workspace root: ``python eval/test/build_double_exchange.py``.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.double_exchange import DATA_DIR, SPLITS, sample_spins, spins_sha256  # noqa: E402

T_HOP = 1.0
J_HUND = 16.0
MU = -8.3
TEMPERATURE = 0.1

HIDDEN_TEST = {
    "L10": {"L": 10, "n": 4000, "seed": 1_873_406_219},
    "L16": {"L": 16, "n": 600, "seed": 1_873_406_316},
    "L24": {"L": 24, "n": 200, "seed": 1_873_406_424},
}
INSTANCE_PATH = Path(__file__).resolve().parent / "double_exchange_instance.json"
CHUNK = {10: 32, 16: 8, 24: 2}


def hopping_matrix(L: int) -> np.ndarray:
    n = L * L
    t = np.zeros((n, n))
    for r in range(L):
        for s in range(L):
            i = r * L + s
            for j in (((r + 1) % L) * L + s, r * L + (s + 1) % L):
                t[i, j] -= T_HOP
                t[j, i] -= T_HOP
    return t


def mc_energy(spins: np.ndarray) -> np.ndarray:
    """E_MC for a batch of configurations, spins (B, 3, L, L) -> (B,) float64."""

    spins = np.asarray(spins, dtype=np.float64)
    b, _, L, _ = spins.shape
    n = L * L
    sx = spins[:, 0].reshape(b, n)
    sy = spins[:, 1].reshape(b, n)
    sz = spins[:, 2].reshape(b, n)
    hop = hopping_matrix(L)
    g = np.zeros((b, 2 * n, 2 * n), dtype=np.complex128)
    g[:, :n, :n] = hop
    g[:, n:, n:] = hop
    idx = np.arange(n)
    g[:, idx, idx] += -0.5 * J_HUND * sz - MU
    g[:, n + idx, n + idx] += 0.5 * J_HUND * sz - MU
    g[:, idx, n + idx] = -0.5 * J_HUND * (sx - 1j * sy)
    g[:, n + idx, idx] = -0.5 * J_HUND * (sx + 1j * sy)
    eig = np.linalg.eigvalsh(g)
    z = -eig / TEMPERATURE
    softplus = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    return -TEMPERATURE * softplus.sum(axis=1)


def _energy_chunk(args: tuple[np.ndarray]) -> np.ndarray:
    return mc_energy(args[0])


def energies(spins: np.ndarray, workers: int) -> np.ndarray:
    L = spins.shape[2]
    step = CHUNK.get(L, 1)
    chunks = [(spins[i : i + step],) for i in range(0, spins.shape[0], step)]
    if workers <= 1:
        parts = [_energy_chunk(c) for c in chunks]
    else:
        with Pool(workers) as pool:
            parts = pool.map(_energy_chunk, chunks, chunksize=max(1, len(chunks) // (8 * workers)))
    return np.concatenate(parts)


def write_energies(path: Path, y: np.ndarray) -> None:
    path.write_text("energy\n" + "\n".join(f"{v:.8f}" for v in y) + "\n")


def summary(y: np.ndarray, L: int) -> dict:
    return {
        "L": L,
        "n": int(y.shape[0]),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "mean_per_site": float(np.mean(y) / (L * L)),
    }


def instance_digest(instance: dict) -> str:
    core = {k: instance[k] for k in ("splits", "energies")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--only-check", action="store_true", help="time a few configurations of each size and exit")
    args = p.parse_args()

    if args.only_check:
        for L in (10, 16, 24):
            x = sample_spins(4, L, 1)
            t0 = time.time()
            y = energies(x, 1)
            print(f"L={L}: {(time.time() - t0) / 4:.3f} s/config, energies {np.round(y, 4)}", flush=True)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "description": "Public double-exchange energies. Spins regenerate from the seeds in data/double_exchange.py.",
        "model": "2D double-exchange model on an L x L periodic square lattice; target = electronic Monte Carlo energy (free energy) of the spin background",
        "splits": {},
    }
    for name, spec in SPLITS.items():
        t0 = time.time()
        x = sample_spins(spec["n"], spec["L"], spec["seed"])
        y = energies(x, args.workers)
        write_energies(DATA_DIR / spec["file"], y)
        info = summary(y, spec["L"])
        info["spins_sha256"] = spins_sha256(x)
        info["file"] = spec["file"]
        meta["splits"][name] = info
        print(f"public {name}: {info} ({time.time() - t0:.1f} s)", flush=True)
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")

    instance = {
        "description": "Hidden double-exchange test sets: spins regenerate from these seeds "
        "(numpy RandomState, standard normal (n, L, L, 3) normalized to unit length, transposed to (n, 3, L, L), float32).",
        "hamiltonian": {"t": T_HOP, "J": J_HUND, "mu": MU, "T": TEMPERATURE},
        "splits": {},
        "energies": {},
    }
    for key, spec in HIDDEN_TEST.items():
        t0 = time.time()
        x = sample_spins(spec["n"], spec["L"], spec["seed"])
        y = energies(x, args.workers)
        instance["splits"][key] = {**spec, "spins_sha256": spins_sha256(x)}
        instance["energies"][key] = [float(v) for v in y]
        print(f"hidden {key}: {summary(y, spec['L'])} ({time.time() - t0:.1f} s)", flush=True)
    instance["digest"] = instance_digest(instance)
    INSTANCE_PATH.write_text(json.dumps(instance) + "\n")
    print(f"instance digest {instance['digest']}", flush=True)


if __name__ == "__main__":
    main()
