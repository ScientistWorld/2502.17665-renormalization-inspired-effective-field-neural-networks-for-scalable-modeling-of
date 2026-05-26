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

mkdir -p outputs scoring checkpoints

# First job: quick end-to-end validation of the EFNN recurrence, a DNN
# comparison baseline, and the evaluator that reads prediction artifacts.
LAYERS=2 WIDTH=105 EPOCHS=80 bash scripts/baseline.sh binary dnn
LAYERS=2 WIDTH=105 EPOCHS=80 bash scripts/method.sh binary
LAYERS=3 WIDTH=150 EPOCHS=80 bash scripts/method.sh continuous

bash scripts/evaluate.sh \
  --prediction outputs/binary_efnn.npz --method-name efnn --task binary \
  --prediction outputs/continuous_efnn.npz --method-name efnn --task continuous
