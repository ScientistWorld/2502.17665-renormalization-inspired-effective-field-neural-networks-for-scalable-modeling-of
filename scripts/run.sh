#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p outputs scoring checkpoints

# The corrected EFNN binary artifact already exists from the previous job. If
# the workspace is fresh, regenerate it before training the comparison DNN.
if [ ! -e outputs/binary_efnn.npz ]; then
  BATCH_SIZE=4096 LR=0.002 LAYERS=2 WIDTH=105 EPOCHS=600 bash scripts/method.sh binary
fi

# Score a paper baseline on the same fixed data split. This verifies that the
# evaluator captures EFNN's structural benefit, not just an absolute error.
BATCH_SIZE=4096 LR=0.002 LAYERS=2 WIDTH=105 EPOCHS=600 bash scripts/baseline.sh binary dnn

bash scripts/evaluate.sh \
  --prediction outputs/binary_dnn.npz --method-name dnn --task binary \
  --prediction outputs/binary_efnn.npz --method-name efnn --task binary
