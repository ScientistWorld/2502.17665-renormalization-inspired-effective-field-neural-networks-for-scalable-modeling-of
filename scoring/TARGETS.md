# Evaluation Targets

## Primary: three_spin_binary
Tests energy prediction on a fixed binary high-order spin interaction.
- **Metric**: `relative_error` (lower is better)
- **Constraint**: `neuron_count` (lower is better for comparable error)
- **Comparison**: compare submitted predictors against the included same-width baseline on the same split.
- **How to run**: `bash scripts/reproduce.sh`

Continuous-spin artifacts are documented in `PROGRESS.md`, but they are not a
current optimization target because the reproduced RMSE is too far from the
paper value for validation.

`scoring/reference.json` still records the paper's continuous-spin headline
number so future agents have the original target available. Official
`scores.json` currently includes only experiments that passed validation.
