#!/usr/bin/env python3
"""Evaluate saved prediction artifacts for the three-spin experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.three_spin import metric_bundle


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


def main() -> None:
    args = parse_args()
    if not (len(args.prediction) == len(args.method_name) == len(args.task)):
        raise SystemExit("--prediction, --method-name and --task must have the same length")

    scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {"experiments": {}}
    for pred_path, method_name, task in zip(args.prediction, args.method_name, args.task):
        arr = np.load(pred_path)
        metrics = metric_bundle(arr["y_true"], arr["y_pred"], task)
        sidecar = Path(pred_path).with_suffix(".json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            if "neuron_count" in meta.get("metrics", {}):
                metrics["neuron_count"] = float(meta["metrics"]["neuron_count"])
        exp = EXP_FOR_TASK[task]
        scores["experiments"].setdefault(exp, {"results": {}})
        scores["experiments"][exp]["results"][method_name] = metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
