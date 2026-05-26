#!/bin/bash
# Score the saved final prediction artifacts on visible train and held-out test
# evaluation slices.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
