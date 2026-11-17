# Evaluation Targets

Rank weights: `double_exchange` 0.8, `three_spin_binary` 0.2 (`scoring/reference.json`).

## Primary: double_exchange (weight 0.8)
Energy prediction for the 2D double-exchange model (electronic Monte Carlo energy of a
classical spin background), learned from 30,000 configurations of the 10 x 10 periodic
lattice and scored on hidden configurations of three sizes.
- **Metrics** (lower is better): `relative_error_L10` (primary, coefficient 1.0),
  `relative_error_L16` (0.5), `relative_error_L24` (0.5): RMSE divided by the absolute
  mean true energy on hidden 10 x 10, 16 x 16 and 24 x 24 configurations.
- **Constraints**: `param_count` <= **4,000** stored values, measured from the model
  file (reported, not ranked); at most **50,000 counted work units per lattice site** per
  prediction (deterministic operation count reported as `compute_per_site_L*`; the
  reference uses about 6,800).
- **Comparisons**, frozen in `eval/` and scored in every run: `dnn` (ordinary deep
  network of the same channel count and depth, same training) and `effective_model`
  (the paper's two- and four-spin interaction expansion restricted to 5 x 5 tiles,
  213,201 coefficients, regularized least squares; defined on 10 x 10 only, scored on
  `relative_error_L10` only).
- **Failure**: a run that does not load, uses a dense linear-algebra solver, reads files
  or forks work, exceeds a budget, fails on any lattice size, returns a wrong shape or
  non-finite values, or is no better than a constant on any size (R^2 <= 0) is failed
  and receives the worst value of every metric (the best constant predictor's error).

## Secondary: three_spin_binary (weight 0.2)
Energy prediction for a fixed 15-site binary three-spin system, on every configuration
absent from the public files.
- **Metrics** (lower is better): `relative_error` (primary, 1.0), `mae` (0.25),
  `tail_rmse` (0.3), each floored at the reference model's own value on the slice
  (non-regression check: the target is exactly a cubic polynomial of the spins, so
  accuracy beyond the reference earns nothing; worse accuracy costs rank).
- **Constraint**: `param_count` <= **100,000**. Reported, not ranked.
- **Comparison**: frozen same-width DNN `dnn`.
- **Failure**: same rules (load, size, shape, finite, R^2 > 0); a failed run, or a method
  that submits no binary model, receives the worst value of every metric.

## Reference numbers (this benchmark)
| Experiment | Model | Stored values | Train slice (dev) | Hidden test |
|---|---|---:|---|---|
| double_exchange | Reference (paper's method, C=3, 3 layers) | 3,414 | 1.182e-3 / 8.01e-4 / 5.66e-4 | **1.204e-3 / 7.65e-4 / 6.06e-4** |
| double_exchange | `dnn` (same C, depth and training) | 1,596 | 2.671e-3 / 1.832e-3 / 1.069e-3 | 2.667e-3 / 1.649e-3 / 1.199e-3 |
| double_exchange | `effective_model` (paper's, 10 x 10 only) | 213,201 | 1.151e-2 / - / - | 1.077e-2 / - / - |
| three_spin_binary | Reference (3 layers x 105) | 95,762 | 0.0300 | **0.0295** |
| three_spin_binary | `dnn` (2 x 105) | 12,918 | 0.498 | 0.499 |

Double-exchange values are relative errors at L = 10 / 16 / 24. The reference beats the
same-size DNN 2.2x and the paper's effective model 8.9x on 10 x 10, as the paper claims.

Training-seed spread of the reference (three seeds, hidden test, same training procedure):
- double_exchange: L10 1.204e-3, 1.216e-3, 1.257e-3 (range 4% of the mean); L16
  7.65e-4, 7.70e-4, 8.81e-4; L24 6.06e-4, 5.51e-4, 6.22e-4.
- three_spin_binary: range about 0.0045 (raw values; scores are floored at the reference).

## How to run
`bash scripts/reproduce.sh` trains the baselines and references, then runs
`bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh`.

## Not targets
The paper's continuous-spin experiment is not reproduced in this benchmark. Its paper
number is kept under `_paper_reported` in `scoring/reference.json` for context, together
with the paper's double-exchange numbers, which were obtained with 1.6 million training
configurations.
