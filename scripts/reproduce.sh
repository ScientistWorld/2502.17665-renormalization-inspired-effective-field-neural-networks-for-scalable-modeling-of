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

mkdir -p outputs scoring checkpoints

LAYERS="${LAYERS:-2}" WIDTH="${WIDTH:-105}" EPOCHS="${EPOCHS:-250}" bash scripts/baseline.sh binary dnn
LAYERS="${LAYERS:-2}" WIDTH="${WIDTH:-105}" EPOCHS="${EPOCHS:-250}" bash scripts/method.sh binary
LAYERS=3 WIDTH=150 EPOCHS="${EPOCHS_CONTINUOUS:-300}" bash scripts/method.sh continuous

bash scripts/evaluate.sh \
  --prediction outputs/binary_efnn.npz --method-name efnn --task binary \
  --prediction outputs/continuous_efnn.npz --method-name efnn --task continuous
