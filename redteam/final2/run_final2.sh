#!/bin/bash
# ROUND 1 (superseded by run_final3.sh): its trainer settings were best of 4 candidates; results in round1_best_of_4/.
# Audit job of this iteration (CPU).
#
#   1. double_exchange, from the final code: the reference EFNN (scripts/method.sh de:
#      initialization selection on training loss, seed 20250217), the frozen baselines
#      (scripts/baseline.sh de: same-size DNN with the same training, and the paper's tile
#      effective model), two more reference seeds, a better variant (continued training)
#      and a weaker one (3 epochs). three_spin_binary models are unchanged and rescored.
#   2. Build every case (make_cases.py) and score it through the unmodified
#      scripts/evaluate_train.sh and scripts/evaluate_test.sh with no arguments.
#   3. Regenerate the official score files from the reference outputs alone.
#   4. Summarize and validate.
# Every stage is resumable: finished trainings are skipped (markers in state/), unfinished
# ones resume from their progress file, scored cases are skipped.

set -uo pipefail
cd /home/user
export PYTHONPATH="/home/user"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

F=redteam/final2
C=$F/cases
ST=$F/state
LOG=$F/run.log
mkdir -p "$C" "$ST" outputs checkpoints scoring
echo "=== start $(date -Is) nproc=$(nproc) python=$(python --version 2>&1) torch=$(python -c 'import torch;print(torch.__version__)')" | tee -a "$LOG"

SLOTS=$(nproc)
if [ "$SLOTS" -gt 4 ]; then SLOTS=4; fi
if [ "$SLOTS" -lt 1 ]; then SLOTS=1; fi

launch() {  # launch <name> <command...> ; at most SLOTS concurrent trainings
  local name=$1
  shift
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n; done
  mkdir -p "$C/$name"
  (
    start=$(date +%s)
    "$@" >> "$C/$name/training.log" 2>&1
    rc=$?
    echo "$name exit=$rc seconds=$(( $(date +%s) - start ))" | tee -a "$LOG"
  ) &
}

de_reference() {
  if [ -f "$ST/reference_done" ]; then echo "reference already trained"; return 0; fi
  bash scripts/method.sh de || echo "scripts/method.sh de exited non-zero (see log)"
  python -c "import json,sys; d=json.load(open('outputs/de_efnn.json')); sys.exit(0 if d.get('selected_seed') is not None else 1)" \
    && touch "$ST/reference_done"
}

de_baselines() {
  if [ -f "$ST/baselines_done" ]; then echo "baselines already trained and frozen"; return 0; fi
  bash scripts/baseline.sh de && touch "$ST/baselines_done"
}

de_seed() {  # de_seed <seed>
  local d="$C/de_reference_seed_$1"
  if [ -f "$ST/seed_$1_done" ]; then echo "seed $1 already trained"; return 0; fi
  python method/train_double_exchange.py --model efnn --channels 3 --layers 3 --kernel 5 --epochs 150 --batch-size 256 \
    --lr 0.002 --lr-gamma 0.97 --restarts 4 --select-epochs 20 --seed "$1" --threads 1 \
    --output "$d/model.pt" --checkpoint "$d/checkpoint.pt" && touch "$ST/seed_$1_done"
}

de_better() {  # continue the reference: 150 more epochs from lr 5e-4
  local d="$C/de_better_continued"
  if [ -f "$ST/better_done" ]; then echo "better variant already trained"; return 0; fi
  while [ ! -f "$ST/reference_done" ]; do sleep 30; done
  python method/train_double_exchange.py --model efnn --channels 3 --layers 3 --kernel 5 --epochs 150 --batch-size 256 \
    --lr 0.0005 --lr-gamma 0.97 --seed 20250217 --threads 1 --init-checkpoint checkpoints/de_efnn.pt \
    --output "$d/model.pt" --checkpoint "$d/checkpoint.pt" && touch "$ST/better_done"
}

de_weaker() {
  local d="$C/de_weaker_3_epochs"
  if [ -f "$ST/weaker_done" ]; then echo "weaker already trained"; return 0; fi
  python method/train_double_exchange.py --model efnn --channels 3 --layers 3 --kernel 5 --epochs 3 --batch-size 256 \
    --lr 0.002 --lr-gamma 0.97 --restarts 1 --seed 20250217 --threads 1 \
    --output "$d/model.pt" --checkpoint "$d/checkpoint.pt" && touch "$ST/weaker_done"
}

launch _de_reference de_reference
launch _de_baselines de_baselines
launch de_reference_seed_20250218 de_seed 20250218
launch de_reference_seed_20250219 de_seed 20250219
launch de_better_continued de_better
launch de_weaker_3_epochs de_weaker
wait
echo "=== training done $(date -Is)" | tee -a "$LOG"

if [ ! -f "$ST/cases_built" ]; then
  python $F/make_cases.py 2>&1 | tee "$F/make_cases.log"
  echo "=== cases built (exit ${PIPESTATUS[0]}) $(date -Is)" | tee -a "$LOG"
  touch "$ST/cases_built"
fi

python $F/run_cases.py 2>&1 | tee -a "$F/run_cases.log"
echo "=== cases scored (exit ${PIPESTATUS[0]}) $(date -Is)" | tee -a "$LOG"

# Official score files: reference outputs alone (binary_efnn.pt, de_efnn.pt) plus the frozen baselines.
ls -la outputs | tee -a "$LOG"
bash scripts/evaluate_train.sh > "$F/official_evaluate_train.log" 2>&1; echo "evaluate_train.sh exit=$?" | tee -a "$LOG"
bash scripts/evaluate_test.sh > "$F/official_evaluate_test.log" 2>&1; echo "evaluate_test.sh exit=$?" | tee -a "$LOG"
bash scripts/evaluate.sh > "$F/official_evaluate.log" 2>&1; echo "evaluate.sh exit=$?" | tee -a "$LOG"

python $F/summarize.py > "$F/summarize.log" 2>&1; echo "=== summarized (exit $?) $(date -Is)" | tee -a "$LOG"
head -60 "$F/summarize.log" | tee -a "$LOG"

python validate.py > "$F/validate.log" 2>&1; echo "validate.py exit=$?" | tee -a "$LOG"
python validate.py --post-design > "$F/validate_post_design.log" 2>&1; echo "validate.py --post-design exit=$?" | tee -a "$LOG"
echo "=== done $(date -Is)" | tee -a "$LOG"
