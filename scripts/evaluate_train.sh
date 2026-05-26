#!/bin/bash
# Evaluate saved method outputs on the visible train slice.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

python eval/train/score_three_spin_slice.py "$@" --output scoring/scores_train.json
