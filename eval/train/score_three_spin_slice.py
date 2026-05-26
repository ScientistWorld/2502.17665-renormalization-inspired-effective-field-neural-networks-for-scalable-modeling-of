#!/usr/bin/env python3
"""Score the visible train slice of saved three-spin prediction artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SLICE = "train"
SPLIT_SEED = 25021765
DEFAULT_OUTPUT = Path("scoring/scores_train.json")
DEFAULT_SCORE_SOURCE = Path("scoring/scores.json")
OUTPUTS_DIR = Path("outputs")

EXP_FOR_TASK = {
    "binary": "three_spin_binary",
    "continuous": "three_spin_continuous",
}
TASK_FOR_EXP = {v: k for k, v in EXP_FOR_TASK.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", help="NPZ file written by a method or baseline")
    parser.add_argument("--method-name", action="append", help="Score name matching the method output")
    parser.add_argument("--task", action="append", choices=["binary", "continuous"], help="Three-spin task")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--score-source",
        type=Path,
        default=DEFAULT_SCORE_SOURCE,
        help="Score file whose experiment/method keys define default discovery",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def experiments_from_scores(data: dict) -> dict:
    raw = data.get("experiments", data)
    return raw if isinstance(raw, dict) else {}


def resolve_prediction(task: str, method_name: str, outputs_dir: Path = OUTPUTS_DIR) -> Path | None:
    candidates = [
        outputs_dir / f"{task}_{method_name}.npz",
        outputs_dir / f"{method_name}_{task}.npz",
        outputs_dir / f"{method_name}.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def discover_from_score_file(score_path: Path) -> list[tuple[Path, str, str]]:
    entries: list[tuple[Path, str, str]] = []
    missing: list[str] = []
    for exp_name, exp_data in experiments_from_scores(load_json(score_path)).items():
        task = TASK_FOR_EXP.get(exp_name)
        if task is None or not isinstance(exp_data, dict):
            continue
        methods = exp_data.get("results", exp_data)
        if not isinstance(methods, dict):
            continue
        for method_name in sorted(k for k in methods if not k.startswith("_")):
            pred_path = resolve_prediction(task, method_name)
            if pred_path is None:
                missing.append(f"{exp_name}.{method_name}")
                continue
            entries.append((pred_path, method_name, task))
    if missing and entries:
        names = ", ".join(missing)
        raise SystemExit(f"missing prediction artifacts for score-source methods: {names}")
    return entries


def discover_from_outputs(outputs_dir: Path = OUTPUTS_DIR) -> list[tuple[Path, str, str]]:
    entries: list[tuple[Path, str, str]] = []
    for pred_path in sorted(outputs_dir.glob("*.npz")):
        if "_" not in pred_path.stem:
            continue
        task, method_name = pred_path.stem.split("_", 1)
        if task not in EXP_FOR_TASK or not method_name:
            continue
        entries.append((pred_path, method_name, task))
    return entries


def requested_entries(args: argparse.Namespace) -> list[tuple[Path, str, str]]:
    supplied = (args.prediction, args.method_name, args.task)
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise SystemExit("--prediction, --method-name and --task must be supplied together")
        if not (len(args.prediction) == len(args.method_name) == len(args.task)):
            raise SystemExit("--prediction, --method-name and --task must have the same length")
        return [(Path(p), m, t) for p, m, t in zip(args.prediction, args.method_name, args.task)]

    entries = discover_from_score_file(args.score_source)
    if entries:
        return entries
    entries = discover_from_outputs()
    if entries:
        return entries
    raise SystemExit("no prediction artifacts found; pass --prediction/--method-name/--task or write outputs/*.npz")


def select_slice(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    if y_true.shape[0] != y_pred.shape[0]:
        raise SystemExit(f"y_true/y_pred length mismatch: {y_true.shape[0]} != {y_pred.shape[0]}")
    n_items = int(y_true.shape[0])
    order = np.array(sorted(range(n_items), key=stable_row_key), dtype=np.int64)
    selected = np.sort(order[: n_items // 2])
    meta = {
        "slice": SLICE,
        "split_seed": SPLIT_SEED,
        "source_rows": n_items,
        "selected_rows": int(selected.size),
    }
    return y_true[selected], y_pred[selected], meta


def stable_row_key(index: int) -> int:
    payload = f"{SPLIT_SEED}:{index}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict[str, float]:
    err = y_pred.astype(np.float64) - y_true.astype(np.float64)
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err * err)))
    metrics = {
        "rmse": rmse,
        "mae": float(np.mean(abs_err)),
        "bias_abs": float(abs(np.mean(err))),
    }

    cutoff = float(np.quantile(np.abs(y_true.astype(np.float64)), 0.75))
    tail_mask = np.abs(y_true.astype(np.float64)) >= cutoff
    if bool(np.any(tail_mask)):
        tail_err = err[tail_mask]
        metrics["tail_rmse"] = float(np.sqrt(np.mean(tail_err * tail_err)))
    else:
        metrics["tail_rmse"] = rmse

    if task == "binary":
        denom = float(abs(np.mean(y_true)))
        metrics["relative_error"] = rmse / max(denom, 1e-12)
    return metrics


def sidecar_metrics(pred_path: Path) -> dict[str, float]:
    sidecar = pred_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    meta = load_json(sidecar)
    raw_metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    out: dict[str, float] = {}
    if isinstance(raw_metrics, dict) and "neuron_count" in raw_metrics:
        out["neuron_count"] = float(raw_metrics["neuron_count"])
    return out


def score_entry(pred_path: Path, method_name: str, task: str) -> tuple[str, str, dict[str, float], dict]:
    arr = np.load(pred_path)
    if "y_true" not in arr or "y_pred" not in arr:
        raise SystemExit(f"{pred_path} must contain y_true and y_pred arrays")
    y_true, y_pred, slice_meta = select_slice(arr["y_true"], arr["y_pred"])
    metrics = metric_bundle(y_true, y_pred, task)
    metrics.update(sidecar_metrics(pred_path))
    exp_name = EXP_FOR_TASK[task]
    slice_meta.update({"prediction": str(pred_path), "task": task, "method_name": method_name})
    return exp_name, method_name, metrics, slice_meta


def main() -> None:
    args = parse_args()
    scores: dict = {"_meta": {"slice": SLICE, "split_seed": SPLIT_SEED}, "experiments": {}}
    for pred_path, method_name, task in requested_entries(args):
        exp_name, method_name, metrics, slice_meta = score_entry(pred_path, method_name, task)
        exp = scores["experiments"].setdefault(exp_name, {"_meta": {"slice": SLICE}, "results": {}})
        exp["results"][method_name] = metrics
        exp.setdefault("_artifacts", {})[method_name] = slice_meta

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
