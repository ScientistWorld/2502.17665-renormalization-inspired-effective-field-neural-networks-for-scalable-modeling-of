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

# Core-claim attempt with the paper's EFNN recurrence and summation output.
# The previous 20-epoch CPU smoke job established that the pipeline runs; this
# run spends the budget on the two reported three-spin EFNN settings.
LAYERS=2 WIDTH=105 EPOCHS=300 bash scripts/method.sh binary
LAYERS=3 WIDTH=150 EPOCHS=300 bash scripts/method.sh continuous

bash scripts/evaluate.sh \
  --prediction outputs/binary_efnn.npz --method-name efnn --task binary \
  --prediction outputs/continuous_efnn.npz --method-name efnn --task continuous
