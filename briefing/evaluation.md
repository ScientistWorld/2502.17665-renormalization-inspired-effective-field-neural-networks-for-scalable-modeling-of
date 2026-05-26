# Evaluation

## Metrics

For the binary 15-site three-spin task, evaluate RMSE on the held-out test set and divide it by the absolute mean test energy to obtain relative error. Lower is better.

For the continuous 15-site three-spin task, evaluate test RMSE and MAE directly because the expected mean energy is near zero. Lower is better.

Neuron count is tracked as a constraint metric because the paper sweeps model width. Lower is better for methods with similar predictive error.

## Baselines and Targets

The paper compares against ordinary DNN, ResNet, and DenseNet architectures. The primary reference target for the binary task is relative error below roughly `5e-3` for sufficiently wide 2- or 3-layer models. The primary reference target for the continuous task is RMSE around `4e-2` for the best 3-layer, 150-neuron setting.

## Evaluation Protocol

Use the deterministic 15-site three-spin instance in `data/three_spin.py`. Train on 16,000 generated configurations and evaluate on the fixed 4,000-sample test split. Evaluation reads prediction artifacts from `outputs/*.npz`; it must not import the method implementation or regenerate predictions.
