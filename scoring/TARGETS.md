# Evaluation Targets

Rank weights: `double_exchange` 0.8, `three_spin_binary` 0.2 (`scoring/reference.json`).

## Primary: double_exchange (weight 0.8)
Energy prediction for the 2D double-exchange model (electronic Monte Carlo energy of a
classical spin background), learned from 30,000 configurations of the 10 x 10 periodic
lattice and scored on hidden configurations of three sizes.
- **Metrics** (lower is better): `relative_error_L10` (primary, coefficient 1.0),
  `relative_error_L16` (0.5), `relative_error_L24` (0.5): RMSE divided by the absolute
  mean true energy on hidden 10 x 10, 16 x 16 and 24 x 24 configurations.
- **Constraint**: `param_count` <= **4,000** stored values, measured from the model
  file. Reported, not ranked.
- **Comparisons**, frozen in `eval/` and scored in every run: `dnn` (ordinary deep
  network of the same channel count and depth) and `effective_model` (explicit
  two- and four-spin interaction expansion fitted by least squares).
- **Failure**: a run that does not load, uses a dense linear-algebra solver or reads
  files, exceeds the budget, fails on any lattice size, returns a wrong shape or
  non-finite values, or is no better than a constant on any size (R^2 <= 0) is failed.

## Secondary: three_spin_binary (weight 0.2)
Energy prediction for a fixed 15-site binary three-spin system, on every configuration
absent from the public files.
- **Metrics** (lower is better): `relative_error` (primary, 1.0), `mae` (0.25),
  `tail_rmse` (0.3).
- **Constraint**: `param_count` <= **100,000**. Reported, not ranked.
- **Comparison**: frozen same-width DNN `dnn`.
- **Failure**: same rules (load, size, shape, finite, R^2 > 0).

## Reference numbers (this benchmark)
| Experiment | Model | Stored values | Train slice (dev) | Hidden test |
|---|---|---:|---|---|
| double_exchange | Reference (paper's method, C=3, 3 layers) | 3,414 | 1.182e-3 / 8.01e-4 / 5.66e-4 | **1.204e-3 / 7.65e-4 / 6.06e-4** |
| double_exchange | `dnn` (same C and depth) | 1,596 | 2.667e-3 / 1.830e-3 / 1.070e-3 | 2.663e-3 / 1.645e-3 / 1.197e-3 |
| double_exchange | `effective_model` | 1,091 | 1.148e-3 / 6.89e-4 / 4.48e-4 | 1.166e-3 / 6.86e-4 / 4.82e-4 |
| three_spin_binary | Reference (3 layers x 105) | 95,762 | 0.0300 | **0.0295** |
| three_spin_binary | `dnn` (2 x 105) | 12,918 | 0.498 | 0.499 |

Double-exchange values are relative errors at L = 10 / 16 / 24.

The frozen `effective_model` baseline scores slightly better than the reference on
`double_exchange`. This is a measured result, not a defect: the paper's own effective
model was much weaker (about 2e-2).

Training-seed spread of the reference (three seeds, hidden test):
- double_exchange: L10 values 1.204e-3, 1.216e-3 and 1.929e-3. The third seed
  converged to a worse optimum.
- three_spin_binary: range about 0.0045.

## How to run
`bash scripts/reproduce.sh` trains the baselines and references, then runs
`bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh`.

## Not targets
The paper's continuous-spin experiment is not reproduced in this benchmark. Its paper
number is kept under `_paper_reported` in `scoring/reference.json` for context, together
with the paper's double-exchange numbers, which were obtained with 1.6 million training
configurations.
