#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

TASK="${1:-binary}"
MODEL="${2:-dnn}"
LAYERS="${LAYERS:-2}"
WIDTH="${WIDTH:-105}"
EPOCHS="${EPOCHS:-250}"
SEED="${SEED:-20250217}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.001}"
LR_GAMMA="${LR_GAMMA:-0.985}"

mkdir -p outputs

python baseline/train_three_spin.py \
  --task "$TASK" \
  --model "$MODEL" \
  --layers "$LAYERS" \
  --width "$WIDTH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr-gamma "$LR_GAMMA" \
  --seed "$SEED" \
  --output "outputs/${TASK}_${MODEL}.npz"

bash scripts/evaluate.sh \
  --prediction "outputs/${TASK}_${MODEL}.npz" --method-name "$MODEL" --task "$TASK" \
  --output "${EVAL_OUTPUT:-scoring/baseline_scores.json}"
