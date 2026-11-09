#!/bin/bash
# Score every submitted model (outputs/binary_<name>.pt) on the visible train slice
# (public dev configurations). Takes no arguments.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

python eval/train/score_three_spin_slice.py "$@" --output scoring/scores_train.json
