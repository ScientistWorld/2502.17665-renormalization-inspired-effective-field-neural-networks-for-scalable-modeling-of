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

# Fast validation path: previous jobs produced binary artifacts. Regenerate
# missing outputs, then add the continuous-spin EFNN setting for broader
# coverage of the paper's first two case studies.
if [ ! -e outputs/binary_efnn.npz ] || [ "${FORCE_RERUN:-0}" = "1" ]; then
  BATCH_SIZE=4096 LR=0.001 LR_GAMMA=0.995 LAYERS=3 WIDTH=105 EPOCHS=1200 bash scripts/method.sh binary
fi

if [ ! -e outputs/binary_dnn.npz ] || [ "${FORCE_RERUN:-0}" = "1" ]; then
  BATCH_SIZE=4096 LR=0.001 LR_GAMMA=0.995 LAYERS=2 WIDTH=105 EPOCHS=600 bash scripts/baseline.sh binary dnn
fi

if [ ! -e outputs/continuous_efnn.npz ] || [ "${FORCE_RERUN:-0}" = "1" ]; then
  BATCH_SIZE=4096 LR=0.001 LR_GAMMA=0.995 LAYERS=3 WIDTH=150 EPOCHS=1200 bash scripts/method.sh continuous
fi

# Keep the measured DNN comparison separate because the paper figure does not
# provide exact DNN numbers for reference.json.
bash scripts/evaluate.sh \
  --prediction outputs/binary_dnn.npz --method-name dnn --task binary \
  --output scoring/baseline_scores.json

bash scripts/evaluate.sh \
  --prediction outputs/binary_efnn.npz --method-name efnn --task binary \
  --prediction outputs/continuous_efnn.npz --method-name efnn --task continuous

python validate.py --compare
