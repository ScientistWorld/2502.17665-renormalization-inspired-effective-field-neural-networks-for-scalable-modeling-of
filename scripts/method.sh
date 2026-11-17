#!/bin/bash
# Run the paper's proposed method (EFNN) on one task and export its TorchScript
# energy model, then score it on the visible dev slice. Defaults are the reference
# configurations; environment variables override them.
#
# Usage:
#   bash scripts/method.sh binary   # 15-site three-spin model  -> outputs/binary_efnn.pt
#   bash scripts/method.sh de       # double-exchange model     -> outputs/de_efnn.pt
#
# double_exchange: initialization is selected on training data only. Random
# initializations train in turn for the first 20 epochs of the schedule; the first whose
# standardized training MSE is below 0.02 (it has left the plateau every run first
# settles on) continues the same 150-epoch schedule; after 24 candidates without one,
# the lowest-loss candidate continues (RESTARTS=1 trains a single initialization).

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

TASK="${1:-binary}"
SEED="${SEED:-20250217}"
# Thread count changes floating-point reduction order and hence the trained weights:
# the stored references use 2 threads for three_spin_binary and 1 for double_exchange.
THREADS_SET="${THREADS:-}"

mkdir -p checkpoints outputs

case "$TASK" in
  binary)
    python method/train_three_spin.py \
      --task binary \
      --model efnn \
      --layers "${LAYERS:-3}" \
      --width "${WIDTH:-105}" \
      --epochs "${EPOCHS:-2400}" \
      --batch-size "${BATCH_SIZE:-4096}" \
      --lr "${LR:-0.001}" \
      --lr-gamma "${LR_GAMMA:-0.997}" \
      --seed "$SEED" \
      --threads "${THREADS_SET:-2}" \
      --output outputs/binary_efnn.pt \
      --checkpoint checkpoints/binary_efnn.pt
    ;;
  de)
    python method/train_double_exchange.py \
      --model efnn \
      --channels "${CHANNELS:-3}" \
      --layers "${LAYERS:-3}" \
      --kernel "${KERNEL:-5}" \
      --epochs "${EPOCHS:-150}" \
      --batch-size "${BATCH_SIZE:-256}" \
      --lr "${LR:-0.002}" \
      --lr-gamma "${LR_GAMMA:-0.97}" \
      --restarts "${RESTARTS:-24}" \
      --select-epochs "${SELECT_EPOCHS:-20}" \
      --accept-loss "${ACCEPT_LOSS:-0.02}" \
      --seed "$SEED" \
      --threads "${THREADS_SET:-1}" \
      --output outputs/de_efnn.pt \
      --checkpoint checkpoints/de_efnn.pt
    ;;
  *)
    echo "unknown task '$TASK' (expected binary or de)" >&2
    exit 2
    ;;
esac

bash scripts/evaluate.sh
