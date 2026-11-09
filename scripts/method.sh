#!/bin/bash
# Run the paper's proposed method (EFNN) on the binary three-spin task.
#
# Trains on data/three_spin_binary/train.csv (training-loss model selection) and
# exports the TorchScript energy model outputs/binary_efnn.pt, then scores it on
# the visible dev slice. Defaults are the reference configuration.
#
# Usage:
#   bash scripts/method.sh            # environment variables override defaults

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

TASK="${1:-binary}"
LAYERS="${LAYERS:-3}"
WIDTH="${WIDTH:-105}"
EPOCHS="${EPOCHS:-2400}"
SEED="${SEED:-20250217}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LR="${LR:-0.001}"
LR_GAMMA="${LR_GAMMA:-0.997}"
THREADS="${THREADS:-2}"

mkdir -p checkpoints outputs

python method/train_three_spin.py \
  --task "$TASK" \
  --model efnn \
  --layers "$LAYERS" \
  --width "$WIDTH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr-gamma "$LR_GAMMA" \
  --seed "$SEED" \
  --threads "$THREADS" \
  --output "outputs/${TASK}_efnn.pt" \
  --checkpoint "checkpoints/${TASK}_efnn.pt"

bash scripts/evaluate.sh
