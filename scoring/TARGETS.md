# Evaluation Targets

## Primary: three_spin_binary
Tests energy prediction on a fixed binary high-order spin interaction.
- **Metric**: `relative_error` (lower is better)
- **Constraint**: `neuron_count` (lower is better for comparable error)
- **Comparison**: compare `efnn` against the `dnn` baseline on the same split.
- **How to run**: `bash scripts/reproduce.sh`

## Attempted Expansion: three_spin_continuous
This paper setting is implemented and can be trained manually with
`TASK=continuous` through `scripts/method.sh`, but it is not a current
optimization target because the reproduced RMSE is still too far from the
paper's reported `0.04`. Future work should restore it as a target only after
the official `scores.json` value is within validator tolerance.
