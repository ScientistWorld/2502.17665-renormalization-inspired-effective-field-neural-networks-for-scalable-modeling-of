#!/bin/bash
# Regenerate official scores from saved prediction artifacts and validate the
# reusable benchmark contract inside the compute container.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

python eval/score_three_spin.py \
  --prediction outputs/binary_dnn.npz --method-name dnn --task binary \
  --prediction outputs/binary_efnn.npz --method-name efnn --task binary \
  --output scoring/scores.json

bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints

python validate.py --compare
