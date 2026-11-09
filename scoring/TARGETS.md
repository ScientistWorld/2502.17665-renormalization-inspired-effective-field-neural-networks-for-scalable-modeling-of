# Evaluation Targets

## Primary: three_spin_binary
Tests energy prediction for a fixed 15-site binary spin system with three-body
interactions, on configurations the submitted model has never seen.
- **Metric**: `relative_error` (lower is better): RMSE on the scored
  configurations divided by the absolute mean true energy.
- **Secondary accuracy metrics**: `mae`, `tail_rmse` (lower is better).
- **Constraint**: `param_count` — every numeric value stored in the submitted
  model file, counted by the evaluator. A model above **100,000** fails the run.
  Reported, not ranked.
- **Comparison**: the frozen same-width DNN baseline (`dnn`, owned by `eval/`)
  is scored in every run on the same configurations.
- **Failure**: a run that cannot be loaded, returns a wrong shape or non-finite
  values, exceeds the size budget, or is no better than predicting a constant
  energy (R^2 <= 0) is reported as failed with no metrics.
- **How to run**: `bash scripts/reproduce.sh` (reference + baseline), then
  `bash scripts/evaluate_train.sh` for the visible slice.

The paper's continuous-spin experiment is not a target: its reproduction did
not validate, and its paper number is kept under `_paper_reported` in
`scoring/reference.json` for context only.
