#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p outputs scoring checkpoints

BATCH_SIZE="${BATCH_SIZE:-4096}" LR="${LR:-0.002}" LAYERS="${LAYERS:-2}" WIDTH="${WIDTH:-105}" EPOCHS="${EPOCHS:-600}" bash scripts/baseline.sh binary dnn
BATCH_SIZE="${BATCH_SIZE:-4096}" LR="${LR:-0.002}" LAYERS="${LAYERS:-2}" WIDTH="${WIDTH:-105}" EPOCHS="${EPOCHS:-600}" bash scripts/method.sh binary

bash scripts/evaluate.sh \
  --prediction outputs/binary_dnn.npz --method-name dnn --task binary \
  --output scoring/baseline_scores.json

bash scripts/evaluate.sh \
  --prediction outputs/binary_efnn.npz --method-name efnn --task binary
