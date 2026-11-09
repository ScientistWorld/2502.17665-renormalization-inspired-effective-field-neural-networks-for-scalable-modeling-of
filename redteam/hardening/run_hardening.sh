#!/bin/bash
# Hardening audit (compute job, CPU).
#
#   1. Reproduce the reference from the final code: scripts/baseline.sh (DNN,
#      frozen into eval/*/baselines) and scripts/method.sh (EFNN -> outputs/).
#   2. Train the honest variants: two more initialization seeds, a weaker
#      20-epoch EFNN, and a better variant that continues the reference.
#   3. Build shortcut probes (make_probes.py).
#   4. Score every case through the unmodified scripts/evaluate_train.sh and
#      scripts/evaluate_test.sh with no arguments (run_cases.py).
#   5. Regenerate the official score files from the reference outputs alone.
#   6. Summarize (seed spread, paired bootstrap, rank deltas) and validate.
#
# Every stage is resumable: finished trainings are skipped, unfinished ones
# resume from their progress file.

set -uo pipefail
cd /home/user
export PYTHONPATH="/home/user"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

H=redteam/hardening
C=$H/cases
LOG=$H/run.log
mkdir -p "$C" outputs checkpoints scoring
echo "=== start $(date -Is) nproc=$(nproc) python=$(python --version 2>&1) torch=$(python -c 'import torch;print(torch.__version__)')" | tee -a "$LOG"

SLOTS=$(( $(nproc) / 2 ))
if [ "$SLOTS" -lt 1 ]; then SLOTS=1; fi

launch() {  # launch <name> <command...> ; respects SLOTS concurrent jobs
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

EFNN_ARGS=(--model efnn --layers 3 --width 105 --epochs 2400 --batch-size 4096 --lr 0.001 --lr-gamma 0.997 --threads 2)

reference_then_better() {
  if [ -f outputs/binary_efnn.json ] && [ -f checkpoints/binary_efnn.pt ]; then
    echo "reference already trained"
  else
    # method.sh ends with the visible evaluation; its exit status must not block the variant.
    bash scripts/method.sh binary || echo "scripts/method.sh exited non-zero (see log)"
  fi
  [ -f checkpoints/binary_efnn.pt ] || return 1
  if [ -f "$C/better_continued/model.json" ]; then
    echo "better variant already trained"
  else
    python method/train_three_spin.py --model efnn --layers 3 --width 105 --epochs 2400 --batch-size 4096 \
      --lr 0.0001 --lr-gamma 0.999 --seed 20250217 --threads 2 \
      --init-checkpoint checkpoints/binary_efnn.pt \
      --output "$C/better_continued/model.pt" --checkpoint "$C/better_continued/checkpoint.pt" \
      > "$C/better_continued/training.log" 2>&1 || return 1
  fi
}

seed_run() {  # seed_run <seed>
  local d="$C/reference_seed_$1"
  if [ -f "$d/model.json" ]; then echo "seed $1 already trained"; return 0; fi
  python method/train_three_spin.py "${EFNN_ARGS[@]}" --seed "$1" --output "$d/model.pt" --checkpoint "$d/checkpoint.pt"
}

weaker_run() {
  local d="$C/weaker_20_epochs"
  if [ -f "$d/model.json" ]; then echo "weaker already trained"; return 0; fi
  python method/train_three_spin.py --model efnn --layers 3 --width 105 --epochs 20 --batch-size 4096 \
    --lr 0.001 --lr-gamma 0.997 --seed 20250217 --threads 2 --output "$d/model.pt" --checkpoint "$d/checkpoint.pt"
}

baseline_run() {
  if [ -f checkpoints/baselines/binary_dnn.json ] && [ -f eval/test/baselines/binary_dnn.pt ] && [ -f eval/train/baselines/binary_dnn.pt ]; then
    echo "baseline already trained and frozen"
    return 0
  fi
  bash scripts/baseline.sh dnn
}

mkdir -p "$C/better_continued"
launch _reference_and_better reference_then_better
launch _baseline_dnn baseline_run
launch reference_seed_20250218 seed_run 20250218
launch reference_seed_20250219 seed_run 20250219
launch weaker_20_epochs weaker_run
wait
echo "=== training done $(date -Is)" | tee -a "$LOG"

python $H/make_probes.py 2>&1 | tee "$H/make_probes.log"
echo "=== probes built (exit ${PIPESTATUS[0]}) $(date -Is)" | tee -a "$LOG"

python $H/run_cases.py 2>&1 | tee "$H/run_cases.log"
echo "=== cases scored (exit ${PIPESTATUS[0]}) $(date -Is)" | tee -a "$LOG"

# Official score files: reference outputs alone (outputs/binary_efnn.pt) plus the frozen baseline.
ls -la outputs | tee -a "$LOG"
bash scripts/evaluate.sh > "$H/official_evaluate.log" 2>&1; echo "evaluate.sh exit=$?" | tee -a "$LOG"
bash scripts/evaluate_train.sh > "$H/official_evaluate_train.log" 2>&1; echo "evaluate_train.sh exit=$?" | tee -a "$LOG"
bash scripts/evaluate_test.sh > "$H/official_evaluate_test.log" 2>&1; echo "evaluate_test.sh exit=$?" | tee -a "$LOG"

python $H/summarize.py 2>&1 | tee "$H/summarize.log"
echo "=== summarized (exit ${PIPESTATUS[0]}) $(date -Is)" | tee -a "$LOG"

python validate.py > "$H/validate.log" 2>&1; echo "validate.py exit=$?" | tee -a "$LOG"
python validate.py --post-design > "$H/validate_post_design.log" 2>&1; echo "validate.py --post-design exit=$?" | tee -a "$LOG"
echo "=== done $(date -Is)" | tee -a "$LOG"
