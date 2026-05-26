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

# Fast validation path: the latest training jobs already produced both binary
# prediction artifacts. If a fresh workspace lacks either artifact, regenerate
# the missing one before scoring.
if [ ! -e outputs/binary_efnn.npz ]; then
  BATCH_SIZE=4096 LR=0.002 LAYERS=2 WIDTH=105 EPOCHS=600 bash scripts/method.sh binary
fi

if [ ! -e outputs/binary_dnn.npz ]; then
  BATCH_SIZE=4096 LR=0.002 LAYERS=2 WIDTH=105 EPOCHS=600 bash scripts/baseline.sh binary dnn
fi

# Keep the measured DNN comparison separate because the paper figure does not
# provide exact DNN numbers for reference.json.
bash scripts/evaluate.sh \
  --prediction outputs/binary_dnn.npz --method-name dnn --task binary \
  --output scoring/baseline_scores.json

bash scripts/evaluate.sh \
  --prediction outputs/binary_efnn.npz --method-name efnn --task binary

python validate.py --compare
