#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
for name in reference_saved reference_seed_20250217 reference_seed_20250218 reference_seed_20250219 \
  weaker_20_epochs weaker_dnn_baseline \
  shortcut_generator_answers shortcut_width_metadata shortcut_impossible_width \
  shortcut_score_files shortcut_sidecar_scores shortcut_move_baseline \
  shortcut_replace_fixed_data shortcut_forged_labels shortcut_constant_output \
  shortcut_empty_output shortcut_reduced_test_size shortcut_nan_output \
  shortcut_catastrophic_output shortcut_failing_scaled_width better_continued; do
  bash "redteam/$name/run.sh"
done
python redteam/summarize.py
python redteam/run_case.py --restore
