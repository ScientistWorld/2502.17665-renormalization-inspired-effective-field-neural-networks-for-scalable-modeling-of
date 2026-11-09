#!/bin/bash
# Final-iteration audit (compute job, CPU).
#
#   1. double_exchange: train the reference EFNN (scripts/method.sh de) and the frozen
#      baselines (scripts/baseline.sh de: symmetrized DNN + effective model) from the
#      final code, plus honest variants: two more initialization seeds, a weaker
#      3-epoch EFNN, and a better variant that continues the reference.
#      three_spin_binary code and data are unchanged; its iteration-11 reference,
#      variants and shortcut models are rescored as they are.
#   2. Build every case's submission (make_cases.py).
#   3. Score every case through the unmodified scripts/evaluate_train.sh and
#      scripts/evaluate_test.sh with no arguments (run_cases.py).
#   4. Regenerate the official score files from the reference outputs alone.
#   5. Summarize (ranks, seed spread, paired bootstrap, probes) and validate.
#
# Every stage is resumable: finished trainings are skipped, unfinished ones resume from
# their progress file, scored cases are skipped.

set -uo pipefail
cd /home/user
export PYTHONPATH="/home/user"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export THREADS=1

F=redteam/final
C=$F/cases
LOG=$F/run.log
mkdir -p "$C" outputs checkpoints scoring
echo "=== start $(date -Is) nproc=$(nproc) python=$(python --version 2>&1) torch=$(python -c 'import torch;print(torch.__version__)')" | tee -a "$LOG"

SLOTS=$(nproc)
if [ "$SLOTS" -gt 6 ]; then SLOTS=6; fi
if [ "$SLOTS" -lt 1 ]; then SLOTS=1; fi

launch() {  # launch <name> <command...> ; at most SLOTS concurrent trainings
  local name=$1
  shift
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n; done
  mkdir -p "$C/$name"
  (
    start=$(date +%s)
    "$@" > "$C/$name/training.log" 2>&1
    rc=$?
    echo "$name exit=$rc seconds=$(( $(date +%s) - start ))" | tee -a "$LOG"
  ) &
}

de_reference() {
  if [ -f outputs/de_efnn.json ] && [ -f checkpoints/de_efnn.pt ]; then echo "reference already trained"; return 0; fi
  # method.sh ends with the visible evaluation; its exit status must not block the variants.
  bash scripts/method.sh de || echo "scripts/method.sh de exited non-zero (see log)"
  [ -f checkpoints/de_efnn.pt ]
}

de_better() {  # continue the reference: 150 more epochs from lr 5e-4
  local d="$C/de_better_continued"
  if [ -f "$d/model.json" ]; then echo "better variant already trained"; return 0; fi
  while [ ! -f checkpoints/de_efnn.pt ]; do sleep 30; done
  python method/train_double_exchange.py --model efnn --channels 3 --layers 3 --kernel 5 --epochs 150 --batch-size 256 \
    --lr 0.0005 --lr-gamma 0.97 --seed 20250217 --threads 1 --init-checkpoint checkpoints/de_efnn.pt \
    --output "$d/model.pt" --checkpoint "$d/checkpoint.pt"
}

de_seed() {  # de_seed <seed>
  local d="$C/de_reference_seed_$1"
  if [ -f "$d/model.json" ]; then echo "seed $1 already trained"; return 0; fi
  python method/train_double_exchange.py --model efnn --channels 3 --layers 3 --kernel 5 --epochs 150 --batch-size 256 \
    --lr 0.002 --lr-gamma 0.97 --seed "$1" --threads 1 --output "$d/model.pt" --checkpoint "$d/checkpoint.pt"
}

de_weaker() {
  local d="$C/de_weaker_3_epochs"
  if [ -f "$d/model.json" ]; then echo "weaker already trained"; return 0; fi
  python method/train_double_exchange.py --model efnn --channels 3 --layers 3 --kernel 5 --epochs 3 --batch-size 256 \
    --lr 0.002 --lr-gamma 0.97 --seed 20250217 --threads 1 --output "$d/model.pt" --checkpoint "$d/checkpoint.pt"
}

de_baselines() {
  if [ -f checkpoints/baselines/de_dnn.json ] && [ -f eval/test/baselines/de_dnn.pt ] && [ -f eval/train/baselines/de_dnn.pt ]; then
    echo "baselines already trained and frozen"; return 0
  fi
  bash scripts/baseline.sh de
}

launch _de_reference de_reference
launch _de_baselines de_baselines
launch de_reference_seed_20250218 de_seed 20250218
launch de_better_continued de_better
launch de_reference_seed_20250219 de_seed 20250219
launch de_weaker_3_epochs de_weaker
wait
echo "=== training done $(date -Is)" | tee -a "$LOG"

python $F/make_cases.py 2>&1 | tee "$F/make_cases.log"
echo "=== cases built (exit ${PIPESTATUS[0]}) $(date -Is)" | tee -a "$LOG"

python $F/run_cases.py 2>&1 | tee "$F/run_cases.log"
echo "=== cases scored (exit ${PIPESTATUS[0]}) $(date -Is)" | tee -a "$LOG"

# Official score files: reference outputs alone (binary_efnn.pt, de_efnn.pt) plus the frozen baselines.
ls -la outputs | tee -a "$LOG"
bash scripts/evaluate.sh > "$F/official_evaluate.log" 2>&1; echo "evaluate.sh exit=$?" | tee -a "$LOG"
bash scripts/evaluate_train.sh > "$F/official_evaluate_train.log" 2>&1; echo "evaluate_train.sh exit=$?" | tee -a "$LOG"
bash scripts/evaluate_test.sh > "$F/official_evaluate_test.log" 2>&1; echo "evaluate_test.sh exit=$?" | tee -a "$LOG"

python $F/summarize.py 2>&1 | tee "$F/summarize.log"
echo "=== summarized (exit ${PIPESTATUS[0]}) $(date -Is)" | tee -a "$LOG"

python validate.py > "$F/validate.log" 2>&1; echo "validate.py exit=$?" | tee -a "$LOG"
python validate.py --post-design > "$F/validate_post_design.log" 2>&1; echo "validate.py --post-design exit=$?" | tee -a "$LOG"
echo "=== done $(date -Is)" | tee -a "$LOG"
