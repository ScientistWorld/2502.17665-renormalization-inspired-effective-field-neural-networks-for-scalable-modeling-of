#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

TASK="${1:-binary}"
LAYERS="${LAYERS:-2}"
WIDTH="${WIDTH:-105}"
EPOCHS="${EPOCHS:-250}"
SEED="${SEED:-20250217}"

mkdir -p checkpoints outputs

python method/train_three_spin.py \
  --task "$TASK" \
  --model efnn \
  --layers "$LAYERS" \
  --width "$WIDTH" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --output "outputs/${TASK}_efnn.npz" \
  --checkpoint "checkpoints/${TASK}_efnn.pt"

bash scripts/evaluate.sh \
  --prediction "outputs/${TASK}_efnn.npz" --method-name efnn --task "$TASK"
