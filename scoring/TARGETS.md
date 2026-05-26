# Evaluation Targets

## Primary: three_spin_binary
Tests energy prediction on a fixed binary high-order spin interaction.
- **Metric**: `relative_error` (lower is better)
- **Constraint**: `neuron_count` (lower is better for comparable error)
- **Comparison**: compare `efnn` against the `dnn` baseline on the same split.
- **How to run**: `bash scripts/reproduce.sh`
