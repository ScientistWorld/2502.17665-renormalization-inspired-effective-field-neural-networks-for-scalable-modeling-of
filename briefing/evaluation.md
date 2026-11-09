# Evaluation

Two experiments are scored into one score file. Rank weights (`scoring/reference.json`):
`double_exchange` 0.8, `three_spin_binary` 0.2.

## double_exchange (weight 0.8)

### Data
Public data in `data/double_exchange/`, loaded with `data.double_exchange.load_split`
(spins regenerate from public seeds; energies are the committed CSV files):

- `train`: 30,000 configurations of the 10 x 10 lattice with their energies.
- `dev_L10`, `dev_L16`, `dev_L24`: 2,000 / 300 / 100 further configurations on
  10 x 10, 16 x 16 and 24 x 24 lattices, for checking (the visible train slice).

The hidden test uses 4,000 / 600 / 200 further configurations of the same sizes, drawn
from seeds that are not public; their exact energies live only inside the evaluator.

### Submission
Save a TorchScript module as `outputs/de_<name>.pt`. It receives a float32 tensor of
shape (N, 3, L, L) (unit spin vectors, channels x, y, z, periodic lattice) and returns
energies of shape (N,). The same file is run on all three lattice sizes, so it must
accept any L. Names `dnn` and `effective_model` are reserved for frozen baselines.

### Metrics
- `relative_error_L10` (primary, lower is better, coefficient 1.0): RMSE on hidden
  10 x 10 configurations divided by the absolute mean true energy (the paper's metric).
- `relative_error_L16`, `relative_error_L24` (lower is better, coefficient 0.5 each):
  the same on 16 x 16 and 24 x 24 lattices, i.e. transfer to larger systems.
- `param_count` (constraint, not ranked): numeric values stored in the model file,
  counted by the evaluator. Budget 4,000.

### Validity
A run is failed (no metrics in this experiment) when the file does not load as
TorchScript, its code calls a dense linear-algebra solver (eigen- or singular-value
decomposition, inverse, determinant, factorization, `torch.linalg` except vector norms)
or reads files, it stores more than 4,000 values or more than 128 KiB of code plus
pickled non-tensor attributes, it raises on any lattice size, returns the wrong shape
or a non-finite value, or its RMSE on any size is no better than the best constant
prediction (R^2 <= 0).

### Baselines
Frozen inside the evaluator and scored in every run on the same configurations:
- `dnn`: an ordinary deep network with the same symmetry-invariant input layer,
  channel count and depth as the reference model (1,596 values).
- `effective_model`: an explicit spin-interaction expansion (two- and four-spin
  scalar-product terms, translation invariant, fitted by least squares; 1,091 values).

The paper reports relative errors around `1e-3` for neural networks trained on 1.6
million configurations and about `2e-2` for its effective model; this benchmark
trains on 30,000 configurations, so those numbers are context, not targets.

## three_spin_binary (weight 0.2)

### Data
A fixed 15-site system with binary spins in {0, 1} and unpublished three-body couplings.
Public data in `data/three_spin_binary/`: `train.csv` (16,000 configurations drawn
uniformly with replacement) and `dev.csv` (2,000 configurations absent from
`train.csv`). The hidden test scores every configuration in neither file (18,162),
labelled at test time.

### Submission
`outputs/binary_<name>.pt`: float32 (N, 15) in {0, 1} -> energies (N,). `dnn` is reserved.

### Metrics
- `relative_error` (primary, coefficient 1.0), `mae` (0.25), `tail_rmse` (0.3; RMSE on
  the quarter of configurations with the largest true energy magnitude), all lower is
  better.
- `param_count` (constraint, not ranked): budget 100,000.

### Validity and baseline
Failed when the file does not load, reads files, exceeds 100,000 values or 128 KiB of
code plus pickles, returns the wrong shape or non-finite values, or is no better than
a constant (R^2 <= 0). A same-width DNN (two hidden layers of 105 units) is frozen in the
evaluator. The paper's `5e-3` was obtained on its own coupling instance; context only.

## Common rules
Every number is recomputed from the submitted models' raw predictions on
evaluator-owned inputs. Self-reported numbers and sidecar files are ignored. A file in
`outputs/` that uses a reserved baseline name is ignored.

## How to Evaluate
- `bash scripts/evaluate_train.sh` (or `scripts/evaluate.sh`): visible slice (public dev
  configurations of both experiments), writes `scoring/scores_train.json`.
- `bash scripts/evaluate_test.sh`: hidden slice, same contract, writes
  `scoring/scores_test.json`. Both take no arguments and exit 1 if any submitted model
  failed.
