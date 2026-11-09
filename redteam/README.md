# Benchmark audit runs

`bash redteam/run_all.sh` runs the audit on a compute node. Each experiment has
its own `run.sh`, installs its prediction artifacts, calls the visible train
evaluator, and calls `bash scripts/evaluate_test.sh` **without arguments** in a
separate process. Neither evaluator nor the Python environment is modified.
The wrapper restores outputs, scores, and the data source after each case.

`train_variant.py` retains the reference EFNN training procedure. Initialization
seeds vary independently of the fixed dataset seed (the original trainer's
`--seed` changes both, which cannot be used for an official-split seed study).
The stronger variant continues the saved reference checkpoint with a renewed
learning rate. All model selection uses training loss.

`scores_test.json`, `status.json`, and `summary.json` in each case are actual
test results, including evaluator rejection where applicable. Generated models,
prediction arrays, logs, and temporary snapshots are ignored. The checked-in
code reconstructs them. `rank_summary.json` records a finite audit-cohort
normalization, not an assertion about unknown future attempts' normalizers.
All names starting `shortcut_` are shortcut probes, including tied probes.
