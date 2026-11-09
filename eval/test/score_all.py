#!/usr/bin/env python3
"""Official hidden test scorer: scores every experiment into one score file.

Experiments and submission files (TorchScript energy models in ``outputs/``):

* ``three_spin_binary`` - ``outputs/binary_<name>.pt`` (see score_three_spin_slice.py);
* ``double_exchange``   - ``outputs/de_<name>.pt`` (see score_double_exchange_slice.py).

Every number is recomputed from the models' raw predictions on evaluator-owned inputs;
frozen baselines in ``eval/test/baselines/`` are scored alongside. A run that fails an
experiment's contract gets no metrics in that experiment (reason under ``_failures``).
Exit status 1 when no model was submitted or any submitted model failed.

Run with no arguments from the workspace root (``bash scripts/evaluate_test.sh``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.test import score_double_exchange_slice, score_three_spin_slice  # noqa: E402

SLICE = "test"
EXPERIMENT_MODULES = (score_three_spin_slice, score_double_exchange_slice)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", nargs="?", help="Accepted for scheduler interface compatibility; unused")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_test.json"))
    args = parser.parse_args()

    scores: dict = {"_meta": {"slice": SLICE}, "experiments": {}}
    failures: dict = {}
    ignored: dict = {}
    n_submitted = 0
    for module in EXPERIMENT_MODULES:
        res = module.score_experiment(args.outputs_dir)
        scores["experiments"][res["experiment"]] = res["block"]
        if res["failures"]:
            failures[res["experiment"]] = res["failures"]
        ignored.update(res["ignored"])
        n_submitted += res["n_submitted"]
    if failures:
        scores["_failures"] = failures
    if ignored:
        scores["_ignored"] = ignored

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))
    if n_submitted == 0:
        print(f"no submitted models found under {args.outputs_dir}/ (binary_<name>.pt or de_<name>.pt)", file=sys.stderr)
        sys.exit(1)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
