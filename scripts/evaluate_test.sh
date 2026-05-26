#!/bin/bash
# Evaluate saved method outputs on the held-out test slice.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

python eval/test/score_three_spin_slice.py "$@" --output scoring/scores_test.json
