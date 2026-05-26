# Evaluation Targets

## Primary: three_spin_binary
Tests energy prediction on a fixed binary high-order spin interaction.
- **Metric**: `relative_error` (lower is better)
- **Constraint**: `neuron_count` (lower is better for comparable error)
- **Comparison**: compare submitted predictors against the included same-width baseline on the same split.
- **How to run**: `bash scripts/reproduce.sh`

## Secondary: three_spin_continuous
Tests the same energy-prediction objective on continuous site values.
- **Metric**: `rmse` (lower is better)
- **Constraint**: `neuron_count` (lower is better for comparable error)
- **Comparison**: improve the reproduced 150-neuron result without changing the data split or using the closed-form generator as the predictor.
