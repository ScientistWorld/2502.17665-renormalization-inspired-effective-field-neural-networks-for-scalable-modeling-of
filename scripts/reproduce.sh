#!/bin/bash
# Reproduce the reference end to end inside the compute container:
#   1. train and freeze the comparison baselines (scripts/baseline.sh dnn, de),
#   2. train the paper's EFNN on both tasks (scripts/method.sh binary, de),
#   3. score the visible train slice and, when eval/test/ is present, the hidden test.
# Data are the committed files in data/three_spin_binary/ and data/double_exchange/.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p outputs scoring checkpoints

bash scripts/baseline.sh dnn
bash scripts/baseline.sh de
bash scripts/method.sh binary
bash scripts/method.sh de

bash scripts/evaluate_train.sh
if [ -f eval/test/score_all.py ]; then
  bash scripts/evaluate_test.sh
fi
