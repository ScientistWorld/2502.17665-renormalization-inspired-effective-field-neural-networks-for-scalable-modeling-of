#!/usr/bin/env python3
"""Build the submission folder of every hardening case.

Each case gets ``cases/<name>/submission/`` holding exactly the files that
``run_cases.py`` copies into an empty ``outputs/`` before running the unmodified
evaluators. Honest variants submit their trained TorchScript model; shortcut probes
submit models or files built here. A probe that cannot be built is recorded in
``cases/<name>/build_error.txt`` and skipped.
"""

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.three_spin import load_binary  # noqa: E402
from method.models import build_model  # noqa: E402

H = ROOT / "redteam" / "hardening"
CASES = H / "cases"
REFERENCE = ROOT / "outputs" / "binary_efnn.pt"
FROZEN_DNN = ROOT / "eval" / "test" / "baselines" / "binary_dnn.pt"
N_SITES = 15


def fresh_submission(name: str) -> Path:
    sub = CASES / name / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    err = CASES / name / "build_error.txt"
    if err.exists():
        err.unlink()
    return sub


def save_traced(module: nn.Module, path: Path) -> None:
    example = torch.zeros(8, N_SITES)
    example[::2, ::3] = 1.0
    with torch.no_grad():
        traced = torch.jit.trace(module.eval(), example, check_trace=False)
    torch.jit.save(traced, str(path))


def triple_index() -> np.ndarray:
    return np.asarray(
        [(i, j, k) for i in range(N_SITES) for j in range(i + 1, N_SITES) for k in range(j + 1, N_SITES)], dtype=np.int64
    )


def pair_index() -> np.ndarray:
    return np.asarray([(i, j) for i in range(N_SITES) for j in range(i + 1, N_SITES)], dtype=np.int64)


class Scaled(nn.Module):
    def __init__(self, inner: nn.Module, factor: float):
        super().__init__()
        self.inner = inner
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) * self.factor


class Constant(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0] * 0.0 + self.value


class CubicEnergy(nn.Module):
    """E(x) = c0 + x @ c1 + pairs @ c2 + triples @ c3 (multilinear, degree <= 3)."""

    def __init__(self, c0: float, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray):
        super().__init__()
        tri = torch.from_numpy(triple_index())
        pair = torch.from_numpy(pair_index())
        self.register_buffer("t0", tri[:, 0].clone())
        self.register_buffer("t1", tri[:, 1].clone())
        self.register_buffer("t2", tri[:, 2].clone())
        self.register_buffer("p0", pair[:, 0].clone())
        self.register_buffer("p1", pair[:, 1].clone())
        self.register_buffer("c0", torch.tensor(float(c0)))
        self.register_buffer("c1", torch.from_numpy(c1.astype(np.float32)))
        self.register_buffer("c2", torch.from_numpy(c2.astype(np.float32)))
        self.register_buffer("c3", torch.from_numpy(c3.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tri = x[:, self.t0] * x[:, self.t1] * x[:, self.t2]
        pair = x[:, self.p0] * x[:, self.p1]
        return self.c0 + x @ self.c1 + pair @ self.c2 + tri @ self.c3


class Lookup(nn.Module):
    def __init__(self, table: np.ndarray):
        super().__init__()
        self.register_buffer("table", torch.from_numpy(table.astype(np.float32)))
        self.register_buffer("weights", (2.0 ** torch.arange(N_SITES)).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.round(x @ self.weights).long()
        return self.table[idx]


class Padded(nn.Module):
    def __init__(self, inner: nn.Module, extra: int):
        super().__init__()
        self.inner = inner
        self.register_buffer("pad", torch.zeros(extra))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + self.pad[0] * 0.0


class ListPadded(nn.Module):
    pad: List[float]

    def __init__(self, inner: nn.Module, extra: int):
        super().__init__()
        self.inner = inner
        self.pad = [0.0] * extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + self.pad[0] * 0.0


class HalfStored(nn.Module):
    """Network whose weights are stored in float16 and computed in float32."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.half()).float()


class ReadsFile(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        if x.shape[0] < 0:
            out = out + torch.from_file("eval/test/hidden_instance.json", False, 1).sum()
        return out


class DropLast(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)[:-1]


class Standardized(nn.Module):
    def __init__(self, net: nn.Module, mean: float, std: float):
        super().__init__()
        self.net = net
        self.register_buffer("y_mean", torch.tensor(float(mean)))
        self.register_buffer("y_std", torch.tensor(float(std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.y_std + self.y_mean


def multilinear_features(x: np.ndarray) -> np.ndarray:
    tri = triple_index()
    pair = pair_index()
    return np.concatenate(
        [
            np.ones((x.shape[0], 1)),
            x,
            x[:, pair[:, 0]] * x[:, pair[:, 1]],
            x[:, tri[:, 0]] * x[:, tri[:, 1]] * x[:, tri[:, 2]],
        ],
        axis=1,
    )


def old_public_couplings() -> np.ndarray:
    """Couplings of the pre-hardening public instance (seed 20250217, visible in the old data/three_spin.py)."""

    rng = np.random.default_rng(20250217)
    n = len(triple_index())
    values = rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float32)
    values += 0.15 * rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    return values


def build(name: str, fn) -> None:
    sub = fresh_submission(name)
    try:
        fn(sub)
        print(f"built {name}: {sorted(p.name for p in sub.iterdir())}", flush=True)
    except Exception:  # noqa: BLE001 - record and continue with the other probes
        (CASES / name / "build_error.txt").write_text(traceback.format_exc())
        shutil.rmtree(sub, ignore_errors=True)
        print(f"FAILED to build {name}:\n{traceback.format_exc()}", flush=True)


def main() -> None:
    torch.set_num_threads(2)
    data = load_binary()
    reference = torch.jit.load(str(REFERENCE), map_location="cpu").eval()
    y_mean = float(np.mean(data["y_train"]))
    y_std = float(np.std(data["y_train"]))

    # Honest variants: the reference itself, seed reruns, better and weaker models.
    build("reference", lambda sub: shutil.copy2(REFERENCE, sub / "binary_efnn.pt"))
    for name in ("reference_seed_20250218", "reference_seed_20250219", "better_continued", "weaker_20_epochs"):
        model = CASES / name / "model.pt"
        if model.exists() and (CASES / name / "model.json").exists():
            build(name, lambda sub, m=model: shutil.copy2(m, sub / "binary_efnn.pt"))
        else:
            print(f"skipping {name}: no trained model", flush=True)
    build("weaker_dnn_submitted", lambda sub: shutil.copy2(FROZEN_DNN, sub / "binary_submission.pt"))

    # Shortcut probes.
    def old_generator(sub: Path) -> None:
        c3 = -old_public_couplings().astype(np.float64)
        save_traced(CubicEnergy(0.0, np.zeros(N_SITES), np.zeros(len(pair_index())), c3), sub / "binary_submission.pt")

    def train_lookup(sub: Path) -> None:
        weights = (1 << np.arange(N_SITES)).astype(np.int64)
        codes = (np.rint(data["x_train"]).astype(np.int64) * weights).sum(axis=1)
        table = np.full(2**N_SITES, y_mean, dtype=np.float64)
        table[codes] = data["y_train"]
        save_traced(Lookup(table), sub / "binary_submission.pt")

    def sidecar_claims(sub: Path) -> None:
        shutil.copy2(REFERENCE, sub / "binary_efnn.pt")
        claims = {
            "metrics": {"neuron_count": 0, "param_count": 0, "relative_error": 0.0, "rmse": 0.0, "mae": 0.0, "tail_rmse": 0.0},
            "success": True,
        }
        (sub / "binary_efnn.json").write_text(json.dumps(claims))

    def move_baseline(sub: Path) -> None:
        shutil.copy2(REFERENCE, sub / "binary_efnn.pt")
        save_traced(Constant(0.0), sub / "binary_dnn.pt")

    def oversized(sub: Path) -> None:
        torch.manual_seed(0)
        net = build_model("efnn", n_sites=N_SITES, width=150, layers=3)
        save_traced(Standardized(net, y_mean, y_std), sub / "binary_submission.pt")

    def list_padded(sub: Path) -> None:
        scripted = torch.jit.script(ListPadded(reference, 40000))
        torch.jit.save(scripted, str(sub / "binary_submission.pt"))

    def polynomial(sub: Path) -> None:
        feats = multilinear_features(data["x_train"].astype(np.float64))
        coef, *_ = np.linalg.lstsq(feats, data["y_train"].astype(np.float64), rcond=None)
        n_pair = len(pair_index())
        c0, c1, c2, c3 = coef[0], coef[1 : 1 + N_SITES], coef[1 + N_SITES : 1 + N_SITES + n_pair], coef[1 + N_SITES + n_pair :]
        save_traced(CubicEnergy(c0, c1, c2, c3), sub / "binary_submission.pt")

    def not_torchscript(sub: Path) -> None:
        torch.save({"weights": torch.zeros(10)}, sub / "binary_submission.pt")

    build("shortcut_old_public_generator", old_generator)
    build("shortcut_train_lookup", train_lookup)
    build("shortcut_sidecar_claims", sidecar_claims)
    build("shortcut_scaled_0p001", lambda sub: save_traced(Scaled(reference, 0.001), sub / "binary_submission.pt"))
    build("shortcut_nan_output", lambda sub: save_traced(Scaled(reference, float("nan")), sub / "binary_submission.pt"))
    build("shortcut_catastrophic_1e6", lambda sub: save_traced(Constant(1.0e6), sub / "binary_submission.pt"))
    build("shortcut_constant_train_mean", lambda sub: save_traced(Constant(y_mean), sub / "binary_submission.pt"))
    build("shortcut_move_baseline", move_baseline)
    build("shortcut_padded_buffer", lambda sub: save_traced(Padded(reference, 10000), sub / "binary_submission.pt"))
    build("shortcut_list_attribute", list_padded)
    build("shortcut_oversized_width150", oversized)

    def half_stored(sub: Path) -> None:
        torch.manual_seed(0)
        net = HalfStored(build_model("efnn", n_sites=N_SITES, width=150, layers=3))
        save_traced(Standardized(net, y_mean, y_std), sub / "binary_submission.pt")

    def reads_file(sub: Path) -> None:
        torch.jit.save(torch.jit.script(ReadsFile(reference)), str(sub / "binary_submission.pt"))

    build("shortcut_fp16_width150", half_stored)
    build("shortcut_reads_file", reads_file)
    build("shortcut_wrong_shape", lambda sub: save_traced(DropLast(reference), sub / "binary_submission.pt"))
    build("shortcut_not_torchscript", not_torchscript)
    build("shortcut_polynomial_regression", polynomial)


if __name__ == "__main__":
    main()
