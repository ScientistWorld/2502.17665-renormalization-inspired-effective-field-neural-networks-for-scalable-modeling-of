#!/usr/bin/env python3
"""Evaluate saved prediction artifacts for the three-spin experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data.three_spin import ThreeSpinConfig, make_split, metric_bundle


EXP_FOR_TASK = {
    "binary": "three_spin_binary",
    "continuous": "three_spin_continuous",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prediction", action="append", required=True, help="NPZ file written by method or baseline")
    p.add_argument("--method-name", action="append", required=True, help="Score name matching reference.json")
    p.add_argument("--task", action="append", choices=["binary", "continuous"], required=True)
    p.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    return p.parse_args()


def official_target(task: str, arr: np.lib.npyio.NpzFile, pred_path: Path) -> np.ndarray:
    """Return evaluator-owned labels after verifying the submitted benchmark rows."""

    split = make_split(task, ThreeSpinConfig())
    expected_x = split["x_test"].astype(np.float32)
    expected_y = split["y_test"].astype(np.float32)

    if "x_test" not in arr:
        raise SystemExit(f"{pred_path} must contain x_test so the evaluator can verify the official {task} split")
    x_test = arr["x_test"].astype(np.float32)
    if x_test.shape != expected_x.shape or not np.array_equal(x_test, expected_x):
        raise SystemExit(f"{pred_path} x_test does not match the official {task} split")

    if "y_true" in arr:
        submitted_y = arr["y_true"].astype(np.float32)
        if submitted_y.shape != expected_y.shape or not np.allclose(submitted_y, expected_y, rtol=0.0, atol=1e-6):
            print(
                f"warning: {pred_path} y_true does not match evaluator-owned labels; ignoring submitted y_true",
                file=sys.stderr,
            )
    return expected_y


def main() -> None:
    args = parse_args()
    if not (len(args.prediction) == len(args.method_name) == len(args.task)):
        raise SystemExit("--prediction, --method-name and --task must have the same length")

    scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {"experiments": {}}
    for pred_path, method_name, task in zip(args.prediction, args.method_name, args.task):
        arr = np.load(pred_path)
        if "y_pred" not in arr:
            raise SystemExit(f"{pred_path} must contain a y_pred array")
        y_true = official_target(task, arr, Path(pred_path))
        y_pred = arr["y_pred"].astype(np.float32)
        if y_pred.shape != y_true.shape:
            raise SystemExit(f"{pred_path} y_pred shape {y_pred.shape} does not match expected labels {y_true.shape}")
        metrics = metric_bundle(y_true, y_pred, task)
        sidecar = Path(pred_path).with_suffix(".json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            for constraint_name in ("neuron_count",):
                if constraint_name in meta.get("metrics", {}):
                    metrics[constraint_name] = float(meta["metrics"][constraint_name])
        exp = EXP_FOR_TASK[task]
        scores["experiments"].setdefault(exp, {"results": {}})
        scores["experiments"][exp]["results"][method_name] = metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
