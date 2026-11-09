#!/usr/bin/env python3
"""One-off builder for the hidden three-spin benchmark instance (pure Python).

This file lives in ``eval/test/`` because it holds the hidden coupling seed: it is
the only code that can regenerate the test energies. It writes

* ``data/three_spin_binary/train.csv`` - 16,000 public training rows (paper size),
  drawn uniformly with replacement from all 2^15 binary configurations;
* ``data/three_spin_binary/dev.csv`` - 2,000 public development rows, distinct
  configurations that do not occur in the training file;
* ``eval/test/hidden_instance.json`` - the couplings and a bitmask of the held-out
  test configurations (every configuration absent from both public files).

The coupling family matches the gym's earlier public instance (J ~ N(0, 1) plus a
0.15 Rademacher offset). Seeds are tried in order and the first one whose
all-configuration energy has ``|mean| >= 0.3 * std`` is used, so the paper's
relative-error denominator (|mean energy|) is not degenerate. That criterion is
fixed before any model is trained.

Run from the workspace root: ``python3 eval/test/build_instance.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

N_SITES = 15
N_CONFIGS = 2**N_SITES
TRAIN_SIZE = 16000
DEV_SIZE = 2000
FIRST_SEED = 914_337_201
MIN_MEAN_OVER_STD = 0.3

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "three_spin_binary"
INSTANCE_PATH = Path(__file__).resolve().parent / "hidden_instance.json"


def triples() -> list[tuple[int, int, int]]:
    return [(i, j, k) for i in range(N_SITES) for j in range(i + 1, N_SITES) for k in range(j + 1, N_SITES)]


def couplings(seed: int) -> list[float]:
    rng = random.Random(seed)
    out = []
    for _ in triples():
        out.append(rng.gauss(0.0, 1.0) + 0.15 * rng.choice((-1.0, 1.0)))
    return out


def all_energies(j_values: list[float]) -> list[float]:
    """E(s) = -sum_{i<j<k} J_ijk s_i s_j s_k for every code c (s_i = bit i of c)."""

    tri = triples()
    jmap = {t: v for t, v in zip(tri, j_values)}
    energy = [0.0] * N_CONFIGS
    for code in range(1, N_CONFIGS):
        top = code.bit_length() - 1
        rest = code ^ (1 << top)
        bits = [i for i in range(top) if (rest >> i) & 1]
        delta = 0.0
        for a in range(len(bits)):
            for b in range(a + 1, len(bits)):
                delta += jmap[(bits[a], bits[b], top)]
        energy[code] = energy[rest] - delta
    return energy


def stats(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def spins(code: int) -> list[int]:
    return [(code >> i) & 1 for i in range(N_SITES)]


def write_csv(path: Path, codes: list[int], energy: list[float]) -> None:
    header = ",".join([f"s{i}" for i in range(N_SITES)] + ["energy"])
    lines = [header]
    for code in codes:
        lines.append(",".join(str(v) for v in spins(code)) + f",{energy[code]:.6f}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    seed = FIRST_SEED
    while True:
        j_values = couplings(seed)
        energy = all_energies(j_values)
        mean, std = stats(energy)
        if abs(mean) >= MIN_MEAN_OVER_STD * std:
            break
        seed += 1

    rng = random.Random(seed ^ 0x5EED)
    train_codes = [rng.randrange(N_CONFIGS) for _ in range(TRAIN_SIZE)]
    seen = set(train_codes)
    unseen = [c for c in range(N_CONFIGS) if c not in seen]
    dev_codes = sorted(rng.sample(unseen, DEV_SIZE))
    public = seen | set(dev_codes)
    test_codes = [c for c in range(N_CONFIGS) if c not in public]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(DATA_DIR / "train.csv", train_codes, energy)
    write_csv(DATA_DIR / "dev.csv", dev_codes, energy)

    mask = bytearray(N_CONFIGS // 8)
    for code in test_codes:
        mask[code // 8] |= 1 << (code % 8)
    codes_digest = hashlib.sha256(b"".join(c.to_bytes(8, "little") for c in test_codes)).hexdigest()
    test_energy = [energy[c] for c in test_codes]
    t_mean, t_std = stats(test_energy)
    instance = {
        "description": "Hidden 15-site binary three-spin instance: H(s) = -sum_{i<j<k} J_ijk s_i s_j s_k, "
        "s_i = bit i of the configuration code; couplings in lexicographic (i<j<k) order.",
        "n_sites": N_SITES,
        "seed": seed,
        "couplings": [float(v) for v in j_values],
        "test_bitmask_hex": bytes(mask).hex(),
        "n_test": len(test_codes),
        "test_codes_sha256": codes_digest,
        "train_size": TRAIN_SIZE,
        "train_distinct": len(seen),
        "dev_size": DEV_SIZE,
        "all_config_energy_mean": mean,
        "all_config_energy_std": std,
        "test_energy_mean": t_mean,
        "test_energy_std": t_std,
    }
    INSTANCE_PATH.write_text(json.dumps(instance, indent=1) + "\n")
    print(
        f"seed={seed} mean={mean:.4f} std={std:.4f} train_distinct={len(seen)} "
        f"dev={len(dev_codes)} test={len(test_codes)} test_mean={t_mean:.4f} test_std={t_std:.4f}"
    )


if __name__ == "__main__":
    main()
