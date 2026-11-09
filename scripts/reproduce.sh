#!/bin/bash
# Reproduce the reference end to end inside the compute container:
#   1. train and freeze the DNN comparison baseline (scripts/baseline.sh),
#   2. train the paper's EFNN and export outputs/binary_efnn.pt (scripts/method.sh),
#   3. score the visible train slice and, when eval/test/ is present, the hidden test.
# Data are the committed files in data/three_spin_binary/; no downloads needed.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p outputs scoring checkpoints

bash scripts/baseline.sh dnn
bash scripts/method.sh binary

bash scripts/evaluate_train.sh
if [ -f eval/test/score_three_spin_slice.py ]; then
  bash scripts/evaluate_test.sh
fi
