#!/bin/bash
# Train a comparison baseline from the paper (default: the ordinary DNN at the
# reference width) and freeze it as an eval-owned comparator:
#   eval/train/baselines/binary_<model>.pt  (and eval/test/baselines/ when present)
# Evaluators always score the frozen copy; a file in outputs/ with a baseline's
# name is ignored.
#
# Usage:
#   bash scripts/baseline.sh [dnn|resnet|densenet]

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

MODEL="${1:-dnn}"
LAYERS="${LAYERS:-2}"
WIDTH="${WIDTH:-105}"
EPOCHS="${EPOCHS:-600}"
SEED="${SEED:-20250217}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LR="${LR:-0.001}"
LR_GAMMA="${LR_GAMMA:-0.995}"
THREADS="${THREADS:-2}"

mkdir -p checkpoints/baselines

python baseline/train_three_spin.py \
  --model "$MODEL" \
  --layers "$LAYERS" \
  --width "$WIDTH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr-gamma "$LR_GAMMA" \
  --seed "$SEED" \
  --threads "$THREADS" \
  --output "checkpoints/baselines/binary_${MODEL}.pt" \
  --checkpoint "checkpoints/baselines/binary_${MODEL}_state.pt"

for dir in eval/train/baselines eval/test/baselines; do
  if [ -d "$(dirname "$dir")" ]; then
    mkdir -p "$dir"
    cp "checkpoints/baselines/binary_${MODEL}.pt" "$dir/.binary_${MODEL}.pt.tmp"
    mv "$dir/.binary_${MODEL}.pt.tmp" "$dir/binary_${MODEL}.pt"
  fi
done
echo "frozen baseline binary_${MODEL}.pt into eval/*/baselines/"
