#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# Self-contained: uses only code in eval/ (never method/).
#
# Submission contract: every TorchScript energy model saved as
#   outputs/binary_<name>.pt
# (input float32 spins of shape (N, 15) in {0, 1}; output energies of shape (N,))
# is loaded, run on the public dev configurations (data/three_spin_binary/dev.csv),
# and scored from its raw predictions. The frozen comparison baseline in
# eval/train/baselines/ is scored alongside. Runs that cannot be loaded, return a
# wrong shape or non-finite values, exceed the stored-size budget, or are no better
# than a constant predictor are reported as failed (exit status 1).
#
# Writes scoring/scores.json (same numbers as the visible train slice).
#
# Usage:
#   bash scripts/evaluate.sh

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

python eval/train/score_three_spin_slice.py "$@" --output scoring/scores.json
