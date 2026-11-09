#!/usr/bin/env python3
"""Visible train-slice scorer for submitted double-exchange energy models.

A submission is a TorchScript module saved as ``outputs/de_<name>.pt`` that maps float32
spins of shape (N, 3, L, L) (unit vectors, channels x, y, z, periodic L x L lattice) to
energies of shape (N,). This scorer runs every submission on the public development
configurations (``data/double_exchange``: 2,000 on 10 x 10, 300 on 16 x 16 and 100 on
24 x 24 lattices), recomputes every scored number from the raw predictions, measures
the model's stored size from the archive, rejects model code that calls dense
linear-algebra solvers, scores the frozen comparison baselines in
``eval/train/baselines``, and marks a run as failed (no metrics) when it cannot be
loaded, uses a forbidden operation, exceeds the size budget, returns the wrong shape or
a non-finite value, or is no better than the best constant predictor (R^2 <= 0) on any
lattice size. The hidden test applies the same contract to held-out configurations.

Official entry point: ``bash scripts/evaluate_train.sh`` (no arguments).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.train.score_three_spin_slice import RunFailure, inspect_archive  # noqa: E402

SLICE = "train"
TASK = "de"
EXPERIMENT = "double_exchange"
HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE / "baselines"
OUTPUTS_DIR = Path("outputs")
DEFAULT_OUTPUT = Path("scoring/scores_train.json")
SIZES = ("L10", "L16", "L24")
DEV_SPLITS = {"L10": "dev_L10", "L16": "dev_L16", "L24": "dev_L24"}
EXPECTED_COUNTS = {"L10": 2000, "L16": 300, "L24": 100}
PARAM_BUDGET = 4_000
PAYLOAD_BUDGET_BYTES = 128 * 1024
BATCH = {10: 250, 16: 50, 24: 20}
FORBIDDEN_CODE = re.compile(
    rb"linalg_(?!vector_norm\b)\w+"
    rb"|\b(?:symeig|eig|eigh|eigvals|eigvalsh|svd|svd_lowrank|pca_lowrank|det|logdet|slogdet|inverse|pinverse"
    rb"|cholesky\w*|lu|lu_solve|lu_unpack|lu_factor\w*|qr|geqrf|orgqr|ormqr|householder_product|matrix_exp"
    rb"|matrix_power|lstsq|triangular_solve|lobpcg|from_file)\s*\("
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", nargs="?", help="Accepted for scheduler interface compatibility; unused")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


# ── public development sets ───────────────────────────────────────────────────


def load_dev_sets() -> dict[str, tuple[int, np.ndarray, np.ndarray]]:
    from data.double_exchange import SPLITS, load_split

    sets = {}
    for key in SIZES:
        x, y = load_split(DEV_SPLITS[key])
        if y.shape != (EXPECTED_COUNTS[key],):
            raise SystemExit(f"public dev split {DEV_SPLITS[key]} has {y.shape[0]} rows, expected {EXPECTED_COUNTS[key]}")
        sets[key] = (int(SPLITS[DEV_SPLITS[key]]["L"]), x, y)
    return sets


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


def predict(path: Path, sets: dict[str, tuple[int, np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    try:
        module = torch.jit.load(str(path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - any load error fails the run
        raise RunFailure(f"cannot load as a TorchScript module ({type(exc).__name__}: {exc})") from exc
    module.eval()
    preds = {}
    with torch.no_grad():
        for key, (L, x, _y) in sets.items():
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            step = BATCH.get(L, 10)
            chunks = []
            for start in range(0, xt.shape[0], step):
                batch = xt[start : start + step]
                try:
                    out = module(batch)
                except Exception as exc:  # noqa: BLE001
                    raise RunFailure(f"forward pass on {L}x{L} lattices raised {type(exc).__name__}: {exc}") from exc
                if not isinstance(out, torch.Tensor):
                    raise RunFailure(f"forward returned {type(out).__name__}, expected a Tensor")
                n = batch.shape[0]
                if tuple(out.shape) not in ((n,), (n, 1)):
                    raise RunFailure(f"forward returned shape {tuple(out.shape)} for {n} inputs of size {L}x{L}; expected ({n},)")
                chunks.append(out.detach().to(torch.float64).reshape(-1).cpu().numpy())
            preds[key] = np.concatenate(chunks)
    return preds


def score_model(path: Path, sets: dict[str, tuple[int, np.ndarray, np.ndarray]]) -> tuple[dict[str, float], dict]:
    archive = inspect_archive(path)
    forbidden = sorted(set(archive["forbidden_ops"]) | set(forbidden_code(path)))
    if forbidden:
        raise RunFailure(f"model code uses forbidden operation(s) {forbidden}")
    if archive["payload_bytes"] > PAYLOAD_BUDGET_BYTES:
        raise RunFailure(
            f"code and pickled non-tensor attributes take {archive['payload_bytes']} bytes "
            f"(limit {PAYLOAD_BUDGET_BYTES}); numeric data must be stored in tensors"
        )
    if archive["param_count"] > PARAM_BUDGET:
        raise RunFailure(f"model stores {archive['param_count']} values, above the budget of {PARAM_BUDGET}")
    preds = predict(path, sets)
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
    metrics["param_count"] = float(archive["param_count"])
    return metrics, details


def discover(outputs_dir: Path) -> tuple[list[tuple[str, Path, str]], dict[str, str]]:
    baselines = []
    for path in sorted(BASELINE_DIR.glob(f"{TASK}_*.pt")):
        name = path.stem[len(TASK) + 1 :]
        if name and "." not in name:
            baselines.append((name, path, "baseline"))
    reserved = {name for name, _, _ in baselines}
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


def score_experiment(outputs_dir: Path) -> dict:
    """Score every ``outputs_dir/de_<name>.pt`` and the frozen baselines (see module docstring)."""

    sets = load_dev_sets()
    entries, ignored = discover(outputs_dir)
    block: dict = {
        "_meta": {
            "slice": SLICE,
            "rows": {key: int(y.shape[0]) for key, (_L, _x, y) in sets.items()},
            "param_budget": PARAM_BUDGET,
            "payload_budget_bytes": PAYLOAD_BUDGET_BYTES,
            "rows_are": "public dev configurations of uniformly random unit spins on periodic 10x10, 16x16 and 24x24 lattices",
            "submission_pattern": f"{outputs_dir}/{TASK}_<name>.pt",
        },
        "results": {},
        "_artifacts": {},
    }
    failures: dict[str, str] = {}
    for name, path, kind in entries:
        try:
            metrics, details = score_model(path, sets)
        except RunFailure as exc:
            failures[name] = str(exc)
            print(f"FAILED {EXPERIMENT}/{name} ({path}): {exc}", file=sys.stderr)
            continue
        block["results"][name] = metrics
        details.update({"model": str(path), "kind": kind})
        block["_artifacts"][name] = details
    for path, reason in ignored.items():
        print(f"warning: ignored {path}: {reason}", file=sys.stderr)
    return {
        "experiment": EXPERIMENT,
        "block": block,
        "failures": failures,
        "ignored": ignored,
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
