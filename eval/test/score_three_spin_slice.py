#!/usr/bin/env python3
"""Hidden-test scorer for submitted three-spin energy models.

A submission is a TorchScript module saved as ``outputs/binary_<name>.pt`` that maps
float32 spins of shape (N, 15) in {0, 1} to energies of shape (N,). This scorer

* runs every submission on the held-out configurations (every 15-site configuration
  absent from both public data files) and computes their energies here, at test time,
  from the hidden instance stored next to this file;
* recomputes every scored number from the raw predictions (no sidecar is read);
* measures the model's stored size from the archive itself;
* scores the frozen comparison baseline(s) from ``eval/test/baselines`` - a file in
  ``outputs/`` that uses a baseline's name is ignored;
* floors every scored metric at the paper's reference model's own value on this slice
  (``REFERENCE_FLOOR``; raw values are kept under ``_artifacts``): the target is exactly
  a cubic polynomial of the spins, so this experiment only penalizes regressions;
* marks a run as failed (exit status 1) when it cannot be loaded, returns the wrong
  shape, returns a non-finite value, exceeds the size budget, or is no better than the
  best constant predictor (R^2 <= 0); a failed run is scored with the worst value of
  every metric (the constant predictor's errors and the full size budget).

Run with no arguments from the workspace root: ``bash scripts/evaluate_test.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

SLICE = "test"
TASK = "binary"
EXPERIMENT = "three_spin_binary"
N_SITES = 15
HERE = Path(__file__).resolve().parent
INSTANCE_PATH = HERE / "hidden_instance.json"
BASELINE_DIR = HERE / "baselines"
OUTPUTS_DIR = Path("outputs")
DEFAULT_OUTPUT = Path("scoring/scores_test.json")
EXPECTED_N_TEST = 18162
EXPECTED_TEST_SHA256 = "bf5f79e768bfd414e4d6e57feb97c1ccac58cb3cf44a8a94f642d7f3ff4a3504"
# This experiment is a non-regression check. Its target is exactly a cubic polynomial of
# the spins, so regression on enumerated spin products reproduces it to rounding error
# without learning anything that transfers. Accuracy beyond the paper's reference
# architecture therefore earns nothing: every scored metric is floored at the reference
# model's own value on this slice (raw values are kept under _artifacts). Worse accuracy
# still costs rank.
REFERENCE_FLOOR = {"relative_error": 0.029493604412623508, "mae": 0.06704212363992415, "tail_rmse": 0.0886144761785187}

PARAM_BUDGET = 100_000            # stored numeric values (float32 equivalents)
PAYLOAD_BUDGET_BYTES = 128 * 1024  # TorchScript code + pickled non-tensor attributes
FORBIDDEN_CODE_TOKENS = (b"from_file",)
BATCH = 4096

# Bytes per element of each storage type in a TorchScript archive.
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


# ── hidden instance ───────────────────────────────────────────────────────────


def interaction_indices() -> np.ndarray:
    triples = [(i, j, k) for i in range(N_SITES) for j in range(i + 1, N_SITES) for k in range(j + 1, N_SITES)]
    return np.asarray(triples, dtype=np.int64)


def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(INSTANCE_PATH.read_text())
    j_values = np.asarray(raw["couplings"], dtype=np.float64)
    triples = interaction_indices()
    if j_values.shape != (len(triples),):
        raise SystemExit("hidden instance has the wrong number of couplings")
    mask = np.frombuffer(bytes.fromhex(raw["test_bitmask_hex"]), dtype=np.uint8)
    bits = np.unpackbits(mask, bitorder="little")[: 2**N_SITES]
    codes = np.nonzero(bits)[0].astype(np.int64)
    digest = hashlib.sha256(codes.astype("<i8").tobytes()).hexdigest()
    if codes.size != EXPECTED_N_TEST or digest != EXPECTED_TEST_SHA256:
        raise SystemExit("hidden test configuration set does not match the fixed benchmark")
    x = ((codes[:, None] >> np.arange(N_SITES)) & 1).astype(np.float64)
    y = -(x[:, triples[:, 0]] * x[:, triples[:, 1]] * x[:, triples[:, 2]]) @ j_values
    return x.astype(np.float32), y


# ── model archive inspection ─────────────────────────────────────────────────


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
    """Count every stored numeric value a TorchScript model can use at run time.

    Each storage file is counted from the bytes it actually holds, using the dtype
    recorded for it in the archive's pickles: types narrower than 32 bits count one
    value per element, wider types one value per 4 bytes. Entries the model cannot
    read at run time (debug info, extra files, traced example inputs) are skipped.
    """

    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise RunFailure(f"not a TorchScript archive ({exc})") from exc
    payload_bytes = 0
    parsed_all = True
    forbidden: list[str] = []
    storage_entries: dict[str, int] = {}
    records_by_pickle: dict[str, dict[str, tuple[str, int]]] = {}
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/", 1)
            rel = parts[1] if len(parts) == 2 else parts[0]
            if (
                rel.endswith(".debug_pkl")
                or rel.startswith("extra/")
                or rel in METADATA_ENTRIES
                or rel == "traced_inputs.pkl"
                or rel.startswith("traced_inputs/")
            ):
                continue
            if rel.endswith(".pkl"):
                payload_bytes += info.file_size
                records: dict[str, tuple[str, int]] = {}
                try:
                    _ArchiveUnpickler(zf.read(info.filename), records).load()
                except Exception:
                    parsed_all = False
                records_by_pickle[rel[: -len(".pkl")]] = records
                continue
            if rel.startswith("code/"):
                payload_bytes += info.file_size
                text = zf.read(info.filename)
                forbidden.extend(tok.decode() for tok in FORBIDDEN_CODE_TOKENS if tok in text)
                continue
            storage_entries[rel] = info.file_size
    dtypes_by_key: dict[str, list[str]] = {}
    for records in records_by_pickle.values():
        for key, (dtype, _count) in records.items():
            dtypes_by_key.setdefault(key, []).append(dtype)
    param_count = 0
    elements = 0
    storage_bytes = 0
    for rel, nbytes in storage_entries.items():
        folder, _, key = rel.rpartition("/")
        if key in records_by_pickle.get(folder, {}):
            dtypes = [records_by_pickle[folder][key][0]]
        else:
            dtypes = dtypes_by_key.get(key, [])
        width = min((ELEMENT_BYTES.get(t, 1) for t in dtypes), default=4)
        count = math.ceil(nbytes / width)
        elements += count
        storage_bytes += nbytes
        param_count += max(count, math.ceil(nbytes / 4))
    return {
        "param_count": int(param_count),
        "stored_elements": int(elements),
        "storage_bytes": int(storage_bytes),
        "payload_bytes": int(payload_bytes),
        "pickles_parsed": bool(parsed_all),
        "forbidden_ops": sorted(set(forbidden)),
    }


def predict(path: Path, x: np.ndarray) -> np.ndarray:
    try:
        module = torch.jit.load(str(path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - any load error fails the run
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


# ── metrics ─────────────────────────────────────────────────────────────────────


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


def worst_metrics(y: np.ndarray) -> dict[str, float]:
    """Score of a failed run: the constant (mean) predictor's errors and the full size budget."""

    err = np.full_like(y, float(np.mean(y))) - y
    cutoff = float(np.quantile(np.abs(y), 0.75))
    tail = err[np.abs(y) >= cutoff]
    return {
        "relative_error": float(np.std(y)) / abs(float(np.mean(y))),
        "mae": float(np.mean(np.abs(err))),
        "tail_rmse": float(np.sqrt(np.mean(tail * tail))),
        "param_count": float(PARAM_BUDGET),
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
    worst = worst_metrics(y)
    raw = dict(metrics)
    for key in REFERENCE_FLOOR:
        metrics[key] = min(max(raw[key], REFERENCE_FLOOR[key]), worst[key])
    metrics["param_count"] = float(archive["param_count"])
    details = dict(archive)
    details.update({"rmse": rmse, "constant_predictor_rmse": constant_rmse, "rows": int(y.shape[0])})
    details.update({f"raw_{k}": v for k, v in raw.items()})
    details["reference_floor"] = dict(REFERENCE_FLOOR)
    return metrics, details


def baseline_names() -> set[str]:
    return {p.stem[len(TASK) + 1 :] for p in BASELINE_DIR.glob(f"{TASK}_*.pt") if "." not in p.stem[len(TASK) + 1 :]}


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


def score_experiment(outputs_dir: Path, reserved_extra: set[str] | None = None) -> dict:
    """Score every ``outputs_dir/binary_<name>.pt`` and the frozen baselines.

    Returns ``{"experiment", "block", "failures", "ignored", "n_submitted"}``; ``block``
    is this experiment's entry of the score file (``_meta``, ``results``, ``_artifacts``).
    """

    x, y = load_test_set()
    entries, ignored = discover(outputs_dir, reserved_extra)
    block: dict = {
        "_meta": {
            "slice": SLICE,
            "rows": int(y.shape[0]),
            "param_budget": PARAM_BUDGET,
            "payload_budget_bytes": PAYLOAD_BUDGET_BYTES,
            "reference_floor": dict(REFERENCE_FLOOR),
            "failed_run_scores": worst_metrics(y),
            "rows_are": "all 15-site configurations absent from the public train and dev files",
            "submission_pattern": f"{outputs_dir}/{TASK}_<name>.pt",
        },
        "results": {},
        "_artifacts": {},
    }
    failures: dict[str, str] = {}
    for name, path, kind in entries:
        try:
            metrics, details = score_model(path, x, y)
        except RunFailure as exc:
            failures[name] = str(exc)
            print(f"FAILED {EXPERIMENT}/{name} ({path}): {exc}", file=sys.stderr)
            if kind != "baseline":
                block["results"][name] = worst_metrics(y)
                block["_artifacts"][name] = {"model": str(path), "kind": kind, "failed": str(exc), "scored_as": "failed run: worst value of every metric"}
            continue
        block["results"][name] = metrics
        details.update({"model": str(path), "kind": kind})
        block["_artifacts"][name] = details
    for path, reason in ignored.items():
        print(f"warning: ignored {path}: {reason}", file=sys.stderr)
    return {
        "experiment": EXPERIMENT,
        "task": TASK,
        "block": block,
        "failures": failures,
        "ignored": ignored,
        "worst": worst_metrics(y),
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
