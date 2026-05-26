# Evaluation Targets

## Primary: three_spin_binary
Tests energy prediction on a fixed binary high-order spin interaction.
- **Metric**: `relative_error` (lower is better)
- **Constraint**: `neuron_count` (lower is better for comparable error)
- **How to run**: `bash scripts/evaluate.sh --prediction outputs/binary_efnn.npz --method-name efnn --task binary`

## Primary: three_spin_continuous
Tests energy prediction on a fixed continuous high-order spin interaction.
- **Metric**: `rmse` (lower is better)
- **Constraint**: `neuron_count` (lower is better for comparable error)
- **How to run**: `bash scripts/evaluate.sh --prediction outputs/continuous_efnn.npz --method-name efnn --task continuous`
