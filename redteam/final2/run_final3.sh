#!/bin/bash
# Final audit job of this iteration, round 2 (CPU): the double-exchange trainer now keeps
# the first initialization that leaves the training-loss plateau within 20 epochs (up to
# 24 tried; see method/train_double_exchange.py). Round 1 (best of 4 candidates, kept in
# round1_best_of_4/) left seed 20250219 on the plateau.
#
#   1. Retrain from the final code: the reference (scripts/method.sh de), seeds 20250218
#      and 20250219, the baselines (scripts/baseline.sh de). Candidates already trained in
#      round 1 with identical settings are reused from the progress files (their first 20
#      epochs do not depend on the acceptance rule); everything else is computed.
#   2. The better variant is retrained only if the reference weights changed.
#   3. Rebuild and rescore every case, regenerate the official score files, summarize and
#      validate.

set -uo pipefail
cd /home/user
export PYTHONPATH="/home/user"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

F=redteam/final2
C=$F/cases
ST=$F/state3
LOG=$F/run.log
mkdir -p "$C" "$ST" outputs checkpoints scoring
echo "=== round 2 start $(date -Is) nproc=$(nproc) python=$(python --version 2>&1) torch=$(python -c 'import torch;print(torch.__version__)')" | tee -a "$LOG"

run_step() {  # run_step <name> <command...>; logged, skipped when its marker exists
  local name=$1
  shift
  if [ -f "$ST/$name.done" ]; then echo "$name already done" | tee -a "$LOG"; return 0; fi
  mkdir -p "$C/$name"
  local start=$(date +%s)
  echo "=== round 2 $(date -Is)" >> "$C/$name/training.log"
  "$@" >> "$C/$name/training.log" 2>&1
  local rc=$?
  echo "$name exit=$rc seconds=$(( $(date +%s) - start ))" | tee -a "$LOG"
  if [ $rc -eq 0 ]; then touch "$ST/$name.done"; fi
  return $rc
}

ref_state_hash() {
  python - << 'PY'
import hashlib, torch
sd = torch.load("checkpoints/de_efnn.pt", map_location="cpu", weights_only=False)["model"]
h = hashlib.sha256()
for k in sorted(sd):
    h.update(k.encode()); h.update(sd[k].numpy().tobytes())
print(h.hexdigest())
PY
}

if [ ! -f "$ST/ref_hash_before" ]; then ref_state_hash > "$ST/ref_hash_before"; fi

de_reference() {
  bash scripts/method.sh de || echo "scripts/method.sh de exited non-zero (see log)"
  python -c "import json,sys; d=json.load(open('outputs/de_efnn.json')); sys.exit(0 if d.get('accept_loss') is not None else 1)"
}

de_seed() {  # de_seed <seed>
  local d="$C/de_reference_seed_$1"
  python method/train_double_exchange.py --model efnn --channels 3 --layers 3 --kernel 5 --epochs 150 --batch-size 256 \
    --lr 0.002 --lr-gamma 0.97 --restarts 24 --select-epochs 20 --accept-loss 0.02 --seed "$1" --threads 1 \
    --output "$d/model.pt" --checkpoint "$d/checkpoint.pt"
}

de_better() {  # continue the reference: 150 more epochs from lr 5e-4
  local d="$C/de_better_continued"
  ref_state_hash > "$ST/ref_hash_after"
  if cmp -s "$ST/ref_hash_before" "$ST/ref_hash_after" && [ -f "$d/model.pt" ]; then
    echo "reference weights unchanged ($(cat "$ST/ref_hash_after")): the better variant trained in round 1 continues them"
    return 0
  fi
  rm -f "$d/checkpoint.progress.pt"
  python method/train_double_exchange.py --model efnn --channels 3 --layers 3 --kernel 5 --epochs 150 --batch-size 256 \
    --lr 0.0005 --lr-gamma 0.97 --seed 20250217 --threads 1 --init-checkpoint checkpoints/de_efnn.pt \
    --output "$d/model.pt" --checkpoint "$d/checkpoint.pt"
}

run_step _de_reference de_reference
run_step de_reference_seed_20250218 de_seed 20250218
run_step de_reference_seed_20250219 de_seed 20250219
run_step _de_baselines bash scripts/baseline.sh de
run_step de_better_continued de_better
echo "=== round 2 training done $(date -Is)" | tee -a "$LOG"

if [ ! -f "$ST/cases_built" ]; then
  rm -f "$C"/*/status.json
  python $F/make_cases.py 2>&1 | tee "$F/make_cases.log"
  echo "=== cases built (exit ${PIPESTATUS[0]}) $(date -Is)" | tee -a "$LOG"
  touch "$ST/cases_built"
fi

python $F/run_cases.py 2>&1 | tee "$F/run_cases.log"
echo "=== cases scored (exit ${PIPESTATUS[0]}) $(date -Is)" | tee -a "$LOG"

ls -la outputs | tee -a "$LOG"
bash scripts/evaluate_train.sh > "$F/official_evaluate_train.log" 2>&1; echo "evaluate_train.sh exit=$?" | tee -a "$LOG"
bash scripts/evaluate_test.sh > "$F/official_evaluate_test.log" 2>&1; echo "evaluate_test.sh exit=$?" | tee -a "$LOG"
bash scripts/evaluate.sh > "$F/official_evaluate.log" 2>&1; echo "evaluate.sh exit=$?" | tee -a "$LOG"

python $F/summarize.py > "$F/summarize.log" 2>&1; echo "=== summarized (exit $?) $(date -Is)" | tee -a "$LOG"
head -45 "$F/summarize.log" | tee -a "$LOG"

python validate.py > "$F/validate.log" 2>&1; echo "validate.py exit=$?" | tee -a "$LOG"
python validate.py --post-design > "$F/validate_post_design.log" 2>&1; echo "validate.py --post-design exit=$?" | tee -a "$LOG"
echo "=== round 2 done $(date -Is)" | tee -a "$LOG"
