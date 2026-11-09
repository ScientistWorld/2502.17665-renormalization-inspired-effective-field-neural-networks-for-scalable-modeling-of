#!/bin/bash
# Train the paper's comparison baselines and freeze them as eval-owned comparators
# (eval/train/baselines/ and eval/test/baselines/). Evaluators always score the frozen
# copies; a file in outputs/ with a baseline's name is ignored.
#
# Usage:
#   bash scripts/baseline.sh dnn   # three_spin_binary: same-width DNN (2 x 105)  -> binary_dnn.pt
#   bash scripts/baseline.sh de    # double_exchange: symmetrized DNN (C=3, 3 layers) -> de_dnn.pt
#                                  #   and the paper's effective model (Eq. 3)      -> de_effective_model.pt

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

WHAT="${1:-dnn}"
SEED="${SEED:-20250217}"
THREADS="${THREADS:-2}"
mkdir -p checkpoints/baselines

freeze() {  # freeze <file name>
  for dir in eval/train/baselines eval/test/baselines; do
    if [ -d "$(dirname "$dir")" ]; then
      mkdir -p "$dir"
      cp "checkpoints/baselines/$1" "$dir/.$1.tmp"
      mv "$dir/.$1.tmp" "$dir/$1"
    fi
  done
  echo "frozen baseline $1 into eval/*/baselines/"
}

case "$WHAT" in
  dnn|resnet|densenet)
    python baseline/train_three_spin.py \
      --model "$WHAT" \
      --layers "${LAYERS:-2}" \
      --width "${WIDTH:-105}" \
      --epochs "${EPOCHS:-600}" \
      --batch-size "${BATCH_SIZE:-4096}" \
      --lr "${LR:-0.001}" \
      --lr-gamma "${LR_GAMMA:-0.995}" \
      --seed "$SEED" \
      --threads "$THREADS" \
      --output "checkpoints/baselines/binary_${WHAT}.pt" \
      --checkpoint "checkpoints/baselines/binary_${WHAT}_state.pt"
    freeze "binary_${WHAT}.pt"
    ;;
  de)
    python method/train_double_exchange.py \
      --model dnn \
      --channels "${CHANNELS:-3}" \
      --layers "${LAYERS:-3}" \
      --kernel "${KERNEL:-5}" \
      --epochs "${EPOCHS:-60}" \
      --batch-size "${BATCH_SIZE:-256}" \
      --lr "${LR:-0.002}" \
      --lr-gamma "${LR_GAMMA:-0.93}" \
      --seed "$SEED" \
      --threads "$THREADS" \
      --output checkpoints/baselines/de_dnn.pt \
      --checkpoint checkpoints/baselines/de_dnn_state.pt
    freeze de_dnn.pt
    python baseline/effective_model_de.py --threads "$THREADS" --output checkpoints/baselines/de_effective_model.pt
    freeze de_effective_model.pt
    ;;
  *)
    echo "unknown baseline '$WHAT' (expected dnn, resnet, densenet or de)" >&2
    exit 2
    ;;
esac
