#!/usr/bin/env python3
"""Hidden-test scorer for submitted double-exchange energy models.

A submission is a TorchScript module saved as ``outputs/de_<name>.pt`` that maps float32
spins of shape (N, 3, L, L) (unit vectors, channels x, y, z, periodic L x L lattice) to
energies of shape (N,). This scorer

* regenerates the hidden test spins from seeds stored next to this file (10 x 10,
  16 x 16 and 24 x 24 lattices; never published) and reads their exact energies;
* runs every submission on all three sizes and recomputes every scored number from the
  raw predictions (no sidecar is read);
* measures the model's stored size from the archive itself, rejects model code that
  calls dense linear-algebra solvers (eigen/singular-value decompositions, inverses,
  determinants, factorizations) or runs work in other threads, and counts the
  arithmetic every forward pass performs (a deterministic count of tensor operations,
  not a timer): a submission may spend at most ``COMPUTE_BUDGET_PER_SITE`` work units
  per lattice site, which a learned surrogate needs a small fraction of and any
  diagonalization of the 2L^2 x 2L^2 electron Hamiltonian far exceeds;
* scores the frozen comparison baselines from ``eval/test/baselines`` - a file in
  ``outputs/`` that uses a baseline's name is ignored; baselines are fixed comparators
  and are not held to the submission budgets; the paper's tile effective model is
  defined on 10 x 10 lattices only and is scored there only;
* marks a run as failed when it cannot be loaded, uses a forbidden operation, exceeds
  the size or compute budget, returns the wrong shape or a non-finite value, or is no
  better than the best constant predictor (R^2 <= 0) on any lattice size. A failed run
  is scored with the worst value of every metric (``worst_metrics``), never left out.

Official entry point: ``bash scripts/evaluate_test.sh`` (no arguments).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from numpy.random import RandomState
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.test.score_three_spin_slice import RunFailure, inspect_archive  # noqa: E402

SLICE = "test"
TASK = "de"
EXPERIMENT = "double_exchange"
HERE = Path(__file__).resolve().parent
INSTANCE_PATH = HERE / "double_exchange_instance.json"
BASELINE_DIR = HERE / "baselines"
OUTPUTS_DIR = Path("outputs")
DEFAULT_OUTPUT = Path("scoring/scores_test.json")
SIZES = ("L10", "L16", "L24")
EXPECTED_COUNTS = {"L10": 4000, "L16": 600, "L24": 200}
EXPECTED_INSTANCE_DIGEST = "d43b97900bc5c497402eb96ad68bf29d6a45fcfaef43956934c1dafd453449f4"
PARAM_BUDGET = 4_000
PAYLOAD_BUDGET_BYTES = 128 * 1024
COMPUTE_BUDGET_PER_SITE = 50_000.0
BASELINE_SIZES = {"effective_model": ("L10",)}
BATCH = {10: 250, 16: 50, 24: 20}
FORBIDDEN_CODE = re.compile(
    rb"linalg_(?!vector_norm\b)\w+"
    rb"|\b(?:symeig|eig|eigh|eigvals|eigvalsh|svd|svd_lowrank|pca_lowrank|det|logdet|slogdet|inverse|pinverse"
    rb"|cholesky\w*|lu|lu_solve|lu_unpack|lu_factor\w*|qr|geqrf|orgqr|ormqr|householder_product|matrix_exp"
    rb"|matrix_power|lstsq|triangular_solve|lobpcg|from_file|_?fork|_?wait)\s*\("
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", nargs="?", help="Accepted for scheduler interface compatibility; unused")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


# ── hidden test sets ──────────────────────────────────────────────────────────


def sample_spins(n: int, L: int, seed: int) -> np.ndarray:
    rng = RandomState(seed)  # legacy generator: stream frozen across numpy versions
    v = rng.standard_normal((n, L, L, 3))
    v /= np.sqrt(np.sum(v * v, axis=-1, keepdims=True))
    return np.ascontiguousarray(v.transpose(0, 3, 1, 2)).astype(np.float32)


def load_test_sets() -> dict[str, tuple[int, np.ndarray, np.ndarray]]:
    raw = json.loads(INSTANCE_PATH.read_text())
    digest = hashlib.sha256(json.dumps({k: raw[k] for k in ("splits", "energies")}, sort_keys=True).encode()).hexdigest()
    if EXPECTED_INSTANCE_DIGEST and digest != EXPECTED_INSTANCE_DIGEST:
        raise SystemExit("hidden double-exchange instance does not match the fixed benchmark")
    sets = {}
    for key in SIZES:
        spec = raw["splits"][key]
        if spec["n"] != EXPECTED_COUNTS[key]:
            raise SystemExit(f"hidden double-exchange split {key} has {spec['n']} rows, expected {EXPECTED_COUNTS[key]}")
        x = sample_spins(spec["n"], spec["L"], spec["seed"])
        if hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest() != spec["spins_sha256"]:
            raise SystemExit(f"regenerated hidden spins for {key} do not match the instance (numpy stream changed?)")
        y = np.asarray(raw["energies"][key], dtype=np.float64)
        if y.shape != (spec["n"],):
            raise SystemExit(f"hidden double-exchange split {key} has a malformed energy list")
        sets[key] = (int(spec["L"]), x, y)
    return sets


def worst_metrics(sets: dict[str, tuple[int, np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Score of a failed run: the best constant predictor's error (R^2 = 0) at every size, full budget."""

    worst = {f"relative_error_{key}": float(np.std(y)) / abs(float(np.mean(y))) for key, (_L, _x, y) in sets.items()}
    worst["param_count"] = float(PARAM_BUDGET)
    return worst


# ── compute meter ─────────────────────────────────────────────────────────────
# Every tensor operation a forward pass dispatches is counted: matrix products and
# convolutions by their multiply-accumulates, everything else by the larger of its input
# and output element counts; wider element types count proportionally (complex128 four
# times float32). The count depends only on the operations, never on time or load.

_MATMUL_OPS = {"mm", "addmm", "_addmm_activation", "bmm", "baddbmm", "addbmm", "mv", "addmv", "dot", "vdot", "matmul"}
_CONV_OPS = {"convolution", "_convolution", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"}


class ComputeBudgetExceeded(RuntimeError):
    pass


def _flat_tensors(tree) -> list[torch.Tensor]:
    leaves, _ = tree_flatten(tree)
    return [t for t in leaves if isinstance(t, torch.Tensor)]


def _numel(ts: list[torch.Tensor]) -> int:
    return sum(int(t.numel()) for t in ts)


def _matmul_macs(name: str, args: tuple) -> int:
    ts = [a for a in args if isinstance(a, torch.Tensor)]
    if name in ("addmm", "_addmm_activation", "baddbmm", "addbmm", "addmv"):
        ts = ts[1:3]
    else:
        ts = ts[:2]
    if len(ts) < 2:
        return 0
    a, b = ts
    if name in ("dot", "vdot"):
        return int(a.numel())
    if name in ("mv", "addmv"):
        return int(a.numel())
    if name == "matmul":
        return int(a.numel()) * int(b.shape[-1]) if b.dim() > 1 else int(a.numel())
    # mm / addmm: (n,k)@(k,m); bmm / baddbmm / addbmm: (B,n,k)@(B,k,m)
    return int(a.numel()) * int(b.shape[-1])


def _conv_macs(args: tuple, kwargs: dict) -> int:
    x = args[0] if args else kwargs.get("input")
    w = args[1] if len(args) > 1 else kwargs.get("weight")
    if not isinstance(x, torch.Tensor) or not isinstance(w, torch.Tensor) or w.dim() < 3:
        return 0
    per_out = int(w.shape[1]) * int(np.prod([int(k) for k in w.shape[2:]]))
    transposed = bool(args[6]) if len(args) > 6 and isinstance(args[6], bool) else False
    if transposed:
        return int(x.numel()) * per_out
    spatial = x.dim() - 2
    stride = args[3] if len(args) > 3 else [1] * spatial
    padding = args[4] if len(args) > 4 else [0] * spatial
    dilation = args[5] if len(args) > 5 else [1] * spatial

    def at(v, i, default):
        if isinstance(v, (list, tuple)):
            return int(v[i]) if i < len(v) else int(v[-1]) if v else default
        return int(v) if isinstance(v, int) else default

    out = int(x.shape[0]) * int(w.shape[0])
    for i in range(spatial):
        k = int(w.shape[2 + i])
        size = (int(x.shape[2 + i]) + 2 * at(padding, i, 0) - at(dilation, i, 1) * (k - 1) - 1) // max(1, at(stride, i, 1)) + 1
        out *= max(size, 1)
    return out * per_out


class ComputeMeter(TorchDispatchMode):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = float(limit)
        self.total = 0.0
        self.calls = 0
        self.exceeded: str | None = None

    def _charge(self, units: float, name: str) -> None:
        self.total += units
        if self.total > self.limit:
            self.exceeded = name
            raise ComputeBudgetExceeded(f"compute budget exceeded at aten::{name}")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name.split("::")[-1]
        ins = _flat_tensors((args, kwargs))
        width = max([1.0] + [t.element_size() / 4.0 for t in ins])
        self.calls += 1
        if name in _MATMUL_OPS:
            self._charge(width * _matmul_macs(name, args), name)
        elif name in _CONV_OPS:
            self._charge(width * _conv_macs(args, kwargs), name)
        out = func(*args, **kwargs)
        outs = _flat_tensors(out)
        width = max([width] + [t.element_size() / 4.0 for t in outs])
        self._charge(width * max(_numel(ins), _numel(outs)), name)
        return out


# ── model checks and inference ────────────────────────────────────────────────


def forbidden_code(path: Path) -> list[str]:
    found: set[str] = set()
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            parts = info.filename.split("/", 1)
            rel = parts[1] if len(parts) == 2 else parts[0]
            if rel.startswith("code/") and not rel.endswith(".debug_pkl"):
                for match in FORBIDDEN_CODE.finditer(zf.read(info.filename)):
                    found.add(match.group(0).decode(errors="replace").rstrip("( "))
    return sorted(found)


def predict(
    path: Path,
    sets: dict[str, tuple[int, np.ndarray, np.ndarray]],
    budget_per_site: float | None = COMPUTE_BUDGET_PER_SITE,
    work: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Run the model on every set; with a budget, every batch is metered (work per site goes to ``work``)."""

    try:
        module = torch.jit.load(str(path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - any load error fails the run
        raise RunFailure(f"cannot load as a TorchScript module ({type(exc).__name__}: {exc})") from exc
    module.eval()
    preds = {}
    with torch.no_grad(), torch.jit.optimized_execution(False):
        for key, (L, x, _y) in sets.items():
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            step = BATCH.get(L, 10)
            chunks = []
            peak = 0.0
            for start in range(0, xt.shape[0], step):
                batch = xt[start : start + step].clone()
                n = batch.shape[0]
                meter = ComputeMeter(budget_per_site * n * L * L) if budget_per_site is not None else None
                try:
                    if meter is not None:
                        with meter:
                            out = module(batch)
                    else:
                        out = module(batch)
                except Exception as exc:  # noqa: BLE001
                    if meter is not None and meter.exceeded is not None:
                        raise RunFailure(
                            f"forward pass on {L}x{L} lattices exceeded the compute budget of {budget_per_site:.0f} work "
                            f"units per site (at aten::{meter.exceeded}, {meter.total / (n * L * L):.3g} units per site so far)"
                        ) from exc
                    raise RunFailure(f"forward pass on {L}x{L} lattices raised {type(exc).__name__}: {exc}") from exc
                if meter is not None:
                    peak = max(peak, meter.total / (n * L * L))
                if not isinstance(out, torch.Tensor):
                    raise RunFailure(f"forward returned {type(out).__name__}, expected a Tensor")
                if tuple(out.shape) not in ((n,), (n, 1)):
                    raise RunFailure(f"forward returned shape {tuple(out.shape)} for {n} inputs of size {L}x{L}; expected ({n},)")
                chunks.append(out.detach().to(torch.float64).reshape(-1).cpu().numpy())
            preds[key] = np.concatenate(chunks)
            if work is not None and budget_per_site is not None:
                work[key] = peak
    return preds


def score_model(path: Path, sets: dict[str, tuple[int, np.ndarray, np.ndarray]], kind: str = "submitted") -> tuple[dict[str, float], dict]:
    archive = inspect_archive(path)
    forbidden = sorted(set(archive["forbidden_ops"]) | set(forbidden_code(path)))
    if forbidden:
        raise RunFailure(f"model code uses forbidden operation(s) {forbidden}")
    budgets = kind != "baseline"  # eval-owned comparators are not held to the submission budgets
    if budgets and archive["payload_bytes"] > PAYLOAD_BUDGET_BYTES:
        raise RunFailure(
            f"code and pickled non-tensor attributes take {archive['payload_bytes']} bytes "
            f"(limit {PAYLOAD_BUDGET_BYTES}); numeric data must be stored in tensors"
        )
    if budgets and archive["param_count"] > PARAM_BUDGET:
        raise RunFailure(f"model stores {archive['param_count']} values, above the budget of {PARAM_BUDGET}")
    work: dict[str, float] = {}
    preds = predict(path, sets, COMPUTE_BUDGET_PER_SITE if budgets else None, work)
    metrics: dict[str, float] = {}
    details = dict(archive)
    details["forbidden_ops"] = forbidden
    for key, (L, _x, y) in sets.items():
        p = preds[key]
        if not np.all(np.isfinite(p)):
            raise RunFailure(f"{int(np.sum(~np.isfinite(p)))} non-finite predictions on {L}x{L} lattices")
        err = p - y
        rmse = float(np.sqrt(np.mean(err * err)))
        constant = float(np.std(y))
        if not rmse < constant:
            raise RunFailure(
                f"RMSE {rmse:.6g} on {L}x{L} lattices is no better than the best constant predictor ({constant:.6g}); task failed"
            )
        metrics[f"relative_error_{key}"] = rmse / abs(float(np.mean(y)))
        details[f"rmse_{key}"] = rmse
        details[f"constant_predictor_rmse_{key}"] = constant
        details[f"rows_{key}"] = int(y.shape[0])
        if key in work:
            details[f"compute_per_site_{key}"] = work[key]
    metrics["param_count"] = float(archive["param_count"])
    if budgets:
        details["compute_budget_per_site"] = COMPUTE_BUDGET_PER_SITE
    return metrics, details


def discover(outputs_dir: Path, reserved_extra: set[str] | None = None) -> tuple[list[tuple[str, Path, str]], dict[str, str]]:
    baselines = []
    for path in sorted(BASELINE_DIR.glob(f"{TASK}_*.pt")):
        name = path.stem[len(TASK) + 1 :]
        if name and "." not in name:
            baselines.append((name, path, "baseline"))
    reserved = {name for name, _, _ in baselines} | set(reserved_extra or ())
    submitted = []
    ignored: dict[str, str] = {}
    for path in sorted(outputs_dir.glob(f"{TASK}_*.pt")):
        name = path.stem[len(TASK) + 1 :]
        if not name or "." in name:
            continue
        if name in reserved:
            ignored[str(path)] = f"'{name}' is an eval-owned baseline name; the frozen baseline is scored instead"
            continue
        submitted.append((name, path, "submitted"))
    return baselines + submitted, ignored


def baseline_names() -> set[str]:
    return {p.stem[len(TASK) + 1 :] for p in BASELINE_DIR.glob(f"{TASK}_*.pt") if "." not in p.stem[len(TASK) + 1 :]}


def score_experiment(outputs_dir: Path, reserved_extra: set[str] | None = None) -> dict:
    """Score every ``outputs_dir/de_<name>.pt`` and the frozen baselines (see module docstring)."""

    sets = load_test_sets()
    entries, ignored = discover(outputs_dir, reserved_extra)
    block: dict = {
        "_meta": {
            "slice": SLICE,
            "rows": {key: int(y.shape[0]) for key, (_L, _x, y) in sets.items()},
            "param_budget": PARAM_BUDGET,
            "payload_budget_bytes": PAYLOAD_BUDGET_BYTES,
            "compute_budget_per_site": COMPUTE_BUDGET_PER_SITE,
            "failed_run_scores": worst_metrics(sets),
            "rows_are": "hidden configurations of uniformly random unit spins on periodic 10x10, 16x16 and 24x24 lattices",
            "submission_pattern": f"{outputs_dir}/{TASK}_<name>.pt",
        },
        "results": {},
        "_artifacts": {},
    }
    failures: dict[str, str] = {}
    for name, path, kind in entries:
        use = sets
        if kind == "baseline" and name in BASELINE_SIZES:
            use = {k: v for k, v in sets.items() if k in BASELINE_SIZES[name]}
        try:
            metrics, details = score_model(path, use, kind)
        except RunFailure as exc:
            failures[name] = str(exc)
            print(f"FAILED {EXPERIMENT}/{name} ({path}): {exc}", file=sys.stderr)
            if kind != "baseline":
                block["results"][name] = worst_metrics(sets)
                block["_artifacts"][name] = {"model": str(path), "kind": kind, "failed": str(exc), "scored_as": "failed run: worst value of every metric"}
            continue
        block["results"][name] = metrics
        details.update({"model": str(path), "kind": kind})
        if kind == "baseline" and name in BASELINE_SIZES:
            details["scored_sizes"] = list(BASELINE_SIZES[name])
        block["_artifacts"][name] = details
    for path, reason in ignored.items():
        print(f"warning: ignored {path}: {reason}", file=sys.stderr)
    return {
        "experiment": EXPERIMENT,
        "task": TASK,
        "block": block,
        "failures": failures,
        "ignored": ignored,
        "worst": worst_metrics(sets),
        "submitted": sorted(name for name, _, kind in entries if kind == "submitted"),
        "n_submitted": sum(1 for _, _, kind in entries if kind == "submitted"),
    }


def main() -> None:
    """Standalone use: score this experiment alone (the official entry point is score_all.py)."""

    args = parse_args()
    res = score_experiment(args.outputs_dir)
    scores: dict = {"_meta": {"slice": SLICE}, "experiments": {EXPERIMENT: res["block"]}}
    if res["failures"]:
        scores["_failures"] = {EXPERIMENT: res["failures"]}
    if res["ignored"]:
        scores["_ignored"] = res["ignored"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))
    if res["n_submitted"] == 0 or res["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
