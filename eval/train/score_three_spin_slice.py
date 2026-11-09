#!/usr/bin/env python3
"""Visible train-slice scorer for submitted three-spin energy models.

A submission is a TorchScript module saved as ``outputs/binary_<name>.pt`` that maps
float32 spins of shape (N, 15) in {0, 1} to energies of shape (N,). This scorer runs
each submission on the public development rows (``data/three_spin_binary/dev.csv``:
2,000 configurations that never occur in the training file), recomputes every scored
number from the raw predictions, measures the model's stored size from the archive,
scores the frozen comparison baseline(s) in ``eval/train/baselines``, and marks a run
as failed (no metrics, exit status 1) when it cannot be loaded, returns the wrong
shape or a non-finite value, exceeds the size budget, or is no better than the best
constant predictor (R^2 <= 0). The hidden test applies the same contract to held-out
configurations.

Run with no arguments from the workspace root: ``bash scripts/evaluate_train.sh``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

SLICE = "train"
TASK = "binary"
EXPERIMENT = "three_spin_binary"
N_SITES = 15
HERE = Path(__file__).resolve().parent
DEV_CSV = HERE.parents[1] / "data" / "three_spin_binary" / "dev.csv"
BASELINE_DIR = HERE / "baselines"
OUTPUTS_DIR = Path("outputs")
DEFAULT_OUTPUT = Path("scoring/scores_train.json")

PARAM_BUDGET = 100_000
PAYLOAD_BUDGET_BYTES = 128 * 1024
FORBIDDEN_CODE_TOKENS = (b"from_file",)
BATCH = 4096

ELEMENT_BYTES = {
    "Double": 8, "Float": 4, "Half": 2, "BFloat16": 2,
    "Long": 8, "Int": 4, "Short": 2, "Char": 1, "Byte": 1, "Bool": 1,
    "ComplexDouble": 16, "ComplexFloat": 8, "ComplexHalf": 4,
    "QInt8": 1, "QUInt8": 1, "QInt32": 4, "QUInt4x2": 1, "QUInt2x4": 1,
    "Float8_e5m2": 1, "Float8_e4m3fn": 1, "Float8_e5m2fnuz": 1, "Float8_e4m3fnuz": 1,
    "UInt16": 2, "UInt32": 4, "UInt64": 8,
}
METADATA_ENTRIES = {"version", "byteorder", ".data/version", ".data/serialization_id", ".format_version", ".storage_alignment"}


class RunFailure(Exception):
    """A submission violates the evaluation contract; the run is scored as failed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", nargs="?", help="Accepted for scheduler interface compatibility; unused")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_dev_set() -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(DEV_CSV, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if arr.shape[1] != N_SITES + 1:
        raise SystemExit(f"{DEV_CSV}: expected {N_SITES + 1} columns")
    return arr[:, :N_SITES].astype(np.float32), arr[:, N_SITES]


class _Placeholder:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Placeholder()

    def __setstate__(self, state):
        pass

    def __setitem__(self, key, value):
        pass

    def append(self, value):
        pass

    def extend(self, values):
        pass


class _StorageType:
    def __init__(self, name: str):
        self.name = name


class _ArchiveUnpickler(pickle.Unpickler):
    """Reads storage records from a TorchScript pickle without executing anything."""

    def __init__(self, data: bytes, records: dict[str, tuple[str, int]]):
        super().__init__(io.BytesIO(data))
        self.records = records

    def find_class(self, module, name):
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name[: -len("Storage")])
        return _Placeholder

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            dtype = pid[1].name if isinstance(pid[1], _StorageType) else "Unknown"
            self.records[str(pid[2])] = (dtype, int(pid[4]))
        return _Placeholder()


def inspect_archive(path: Path) -> dict:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise RunFailure(f"not a TorchScript archive ({exc})") from exc
    storage_bytes = 0
    payload_bytes = 0
    numel = 0
    float32_equiv_from_records = 0
    parsed_all = True
    forbidden: list[str] = []
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/", 1)
            rel = parts[1] if len(parts) == 2 else parts[0]
            if rel.endswith(".debug_pkl") or rel.startswith("extra/") or rel in METADATA_ENTRIES:
                continue
            if rel.endswith(".pkl"):
                payload_bytes += info.file_size
                records: dict[str, tuple[str, int]] = {}
                try:
                    _ArchiveUnpickler(zf.read(info.filename), records).load()
                except Exception:
                    parsed_all = False
                for dtype, count in records.values():
                    numel += count
                    width = ELEMENT_BYTES.get(dtype, 8)
                    float32_equiv_from_records += max(count, math.ceil(count * width / 4))
                continue
            if rel.startswith("code/"):
                payload_bytes += info.file_size
                text = zf.read(info.filename)
                forbidden.extend(tok.decode() for tok in FORBIDDEN_CODE_TOKENS if tok in text)
                continue
            storage_bytes += info.file_size
    param_count = max(float32_equiv_from_records, math.ceil(storage_bytes / 4))
    return {
        "param_count": int(param_count),
        "stored_elements": int(numel),
        "storage_bytes": int(storage_bytes),
        "payload_bytes": int(payload_bytes),
        "pickles_parsed": bool(parsed_all),
        "forbidden_ops": sorted(set(forbidden)),
    }


def predict(path: Path, x: np.ndarray) -> np.ndarray:
    try:
        module = torch.jit.load(str(path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        raise RunFailure(f"cannot load as a TorchScript module ({type(exc).__name__}: {exc})") from exc
    module.eval()
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    chunks = []
    with torch.no_grad():
        for start in range(0, xt.shape[0], BATCH):
            batch = xt[start : start + BATCH]
            try:
                out = module(batch)
            except Exception as exc:  # noqa: BLE001
                raise RunFailure(f"forward pass raised {type(exc).__name__}: {exc}") from exc
            if not isinstance(out, torch.Tensor):
                raise RunFailure(f"forward returned {type(out).__name__}, expected a Tensor")
            n = batch.shape[0]
            if tuple(out.shape) not in ((n,), (n, 1)):
                raise RunFailure(f"forward returned shape {tuple(out.shape)} for {n} inputs; expected ({n},)")
            chunks.append(out.detach().to(torch.float64).reshape(-1).cpu().numpy())
    return np.concatenate(chunks)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err * err)))
    cutoff = float(np.quantile(np.abs(y_true), 0.75))
    tail = err[np.abs(y_true) >= cutoff]
    return {
        "relative_error": rmse / abs(float(np.mean(y_true))),
        "mae": float(np.mean(np.abs(err))),
        "tail_rmse": float(np.sqrt(np.mean(tail * tail))),
        "_rmse": rmse,
    }


def score_model(path: Path, x: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], dict]:
    archive = inspect_archive(path)
    if archive["forbidden_ops"]:
        raise RunFailure(f"model code uses forbidden operation(s) {archive['forbidden_ops']}")
    if archive["payload_bytes"] > PAYLOAD_BUDGET_BYTES:
        raise RunFailure(
            f"code and pickled non-tensor attributes take {archive['payload_bytes']} bytes "
            f"(limit {PAYLOAD_BUDGET_BYTES}); numeric data must be stored in tensors"
        )
    if archive["param_count"] > PARAM_BUDGET:
        raise RunFailure(f"model stores {archive['param_count']} values, above the budget of {PARAM_BUDGET}")
    y_pred = predict(path, x)
    if not np.all(np.isfinite(y_pred)):
        raise RunFailure(f"{int(np.sum(~np.isfinite(y_pred)))} non-finite predictions")
    metrics = metric_bundle(y, y_pred)
    constant_rmse = float(np.std(y))
    if not metrics["_rmse"] < constant_rmse:
        raise RunFailure(
            f"RMSE {metrics['_rmse']:.6g} is no better than the best constant predictor ({constant_rmse:.6g}); task failed"
        )
    rmse = metrics.pop("_rmse")
    metrics["param_count"] = float(archive["param_count"])
    details = dict(archive)
    details.update({"rmse": rmse, "constant_predictor_rmse": constant_rmse, "rows": int(y.shape[0])})
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


def main() -> None:
    args = parse_args()
    x, y = load_dev_set()
    entries, ignored = discover(args.outputs_dir)
    n_submitted = sum(1 for _, _, kind in entries if kind == "submitted")

    scores: dict = {
        "_meta": {
            "slice": SLICE,
            "rows": int(y.shape[0]),
            "param_budget": PARAM_BUDGET,
            "payload_budget_bytes": PAYLOAD_BUDGET_BYTES,
            "rows_are": "public dev configurations (absent from the training file)",
        },
        "experiments": {EXPERIMENT: {"_meta": {"slice": SLICE}, "results": {}, "_artifacts": {}}},
    }
    failures: dict[str, str] = {}
    exp = scores["experiments"][EXPERIMENT]
    for name, path, kind in entries:
        try:
            metrics, details = score_model(path, x, y)
        except RunFailure as exc:
            failures[name] = str(exc)
            print(f"FAILED {name} ({path}): {exc}", file=sys.stderr)
            continue
        exp["results"][name] = metrics
        details.update({"model": str(path), "kind": kind})
        exp["_artifacts"][name] = details
    if failures:
        scores["_failures"] = failures
    if ignored:
        scores["_ignored"] = ignored
        for path, reason in ignored.items():
            print(f"warning: ignored {path}: {reason}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))
    if n_submitted == 0:
        print(f"no submitted models found under {args.outputs_dir}/{TASK}_*.pt", file=sys.stderr)
        sys.exit(1)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
