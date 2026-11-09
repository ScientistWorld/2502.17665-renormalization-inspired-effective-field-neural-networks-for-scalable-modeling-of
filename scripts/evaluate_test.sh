#!/bin/bash
# Score every submitted model (outputs/binary_<name>.pt) on the held-out test slice:
# all 15-site configurations that appear in no public file, labelled at test time
# from the hidden instance in eval/test/. Takes no arguments.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

python eval/test/score_three_spin_slice.py "$@" --output scoring/scores_test.json
