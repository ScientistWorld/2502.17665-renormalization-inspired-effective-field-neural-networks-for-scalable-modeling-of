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

needs_artifact_rerun() {
  local meta_path="$1"
  local min_epochs="$2"
  local expected_layers="$3"
  local expected_width="$4"

  if [ "${FORCE_RERUN:-0}" = "1" ] || [ ! -e "$meta_path" ]; then
    return 0
  fi

  python - "$meta_path" "$min_epochs" "$expected_layers" "$expected_width" <<'PY'
import json
import sys

path, min_epochs, expected_layers, expected_width = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
try:
    meta = json.load(open(path))
except Exception:
    raise SystemExit(0)

ok = (
    int(meta.get("epochs", 0)) >= min_epochs
    and int(meta.get("layers", -1)) == expected_layers
    and int(meta.get("width", -1)) == expected_width
)
raise SystemExit(1 if ok else 0)
PY
}

# Fast validation path: previous jobs produced binary artifacts. Regenerate
# missing outputs for the validated core-claim setting.
if needs_artifact_rerun outputs/binary_efnn.json 1200 3 105; then
  BATCH_SIZE=4096 LR=0.001 LR_GAMMA=0.995 LAYERS=3 WIDTH=105 EPOCHS=1200 bash scripts/method.sh binary
fi

if needs_artifact_rerun outputs/binary_dnn.json 600 2 105; then
  BATCH_SIZE=4096 LR=0.001 LR_GAMMA=0.995 LAYERS=2 WIDTH=105 EPOCHS=600 bash scripts/baseline.sh binary dnn
fi

# Keep the measured DNN comparison separate because the paper figure does not
# provide exact DNN numbers for reference.json.
bash scripts/evaluate.sh \
  --prediction outputs/binary_dnn.npz --method-name dnn --task binary \
  --output scoring/baseline_scores.json

bash scripts/evaluate.sh \
  --prediction outputs/binary_efnn.npz --method-name efnn --task binary

python validate.py --compare
