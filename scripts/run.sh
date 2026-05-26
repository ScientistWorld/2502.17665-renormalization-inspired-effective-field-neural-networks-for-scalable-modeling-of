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
# CUDA PyTorch exceeded the shared cache quota, so this run stays on the
# working CPU container and spends the wall time on the binary 3-spin setting.
BATCH_SIZE=4096 LR=0.002 LAYERS=2 WIDTH=105 EPOCHS=600 bash scripts/method.sh binary

bash scripts/evaluate.sh \
  --prediction outputs/binary_efnn.npz --method-name efnn --task binary
