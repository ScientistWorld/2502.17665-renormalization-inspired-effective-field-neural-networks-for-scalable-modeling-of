#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# Self-contained: uses only code in eval/ (never method/).
#
# Submission contract: TorchScript energy models saved in outputs/
#   outputs/binary_<name>.pt  three_spin_binary: float32 spins (N, 15) in {0, 1} -> energies (N,)
#   outputs/de_<name>.pt      double_exchange:   float32 unit spins (N, 3, L, L) -> energies (N,)
# Each is loaded, run on the public dev configurations, and scored from its raw
# predictions; the frozen comparison baselines in eval/train/baselines/ are scored
# alongside. Runs that cannot be loaded, return a wrong shape or non-finite values,
# exceed the stored-size budget, use forbidden operations, or are no better than a
# constant predictor are reported as failed (exit status 1).
#
# Writes scoring/scores.json (same numbers as the visible train slice).
#
# Usage:
#   bash scripts/evaluate.sh

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

python eval/train/score_all.py "$@" --output scoring/scores.json
