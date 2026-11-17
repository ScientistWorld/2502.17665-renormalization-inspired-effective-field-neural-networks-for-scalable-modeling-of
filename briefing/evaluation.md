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
A run is failed in this experiment when:
- the file does not load as TorchScript;
- its code calls a dense linear-algebra solver (eigen- or singular-value decomposition,
  inverse, determinant, factorization, `torch.linalg` except vector norms), reads files,
  or runs work in other threads (`fork`/`wait`);
- it stores more than 4,000 values or more than 128 KiB of code plus pickled non-tensor
  attributes;
- a forward pass spends more than **50,000 work units per lattice site** (the compute
  budget below);
- it raises on any lattice size, returns the wrong shape or a non-finite value;
- its RMSE on any size is no better than the best constant prediction (R^2 <= 0).

A failed run is scored, not dropped: it receives the worst value of every metric, i.e.
the best constant predictor's relative error at each size (R^2 = 0) and `param_count`
4,000. The reason is listed under `_failures`.

### Compute budget
The evaluator counts the arithmetic of every forward pass by intercepting each tensor
operation (a deterministic count, never a timer): matrix products and convolutions count
their multiply-accumulates, every other operation the larger of its input and output
element counts, and 64-bit and complex types count two to four times a 32-bit value. The
count per configuration is divided by the number of lattice sites; the per-site work of
each size is reported under `_artifacts` (`compute_per_site_L*`). For scale: the
reference model uses about 6,800 units per site, the same-size `dnn` about 2,800.

### Baselines
Frozen inside the evaluator and scored in every run on the same configurations. They
are fixed comparators, not held to the submission budgets.
- `dnn`: an ordinary deep network with the same symmetry-invariant input layer, channel
  count, depth and training procedure as the reference model (1,596 values).
- `effective_model`: the paper's explicit spin-interaction expansion (constant, all
  two-spin scalar products s_i.s_j and all products of two of them (s_i.s_j)(s_k.s_l),
  restricted to spins inside the same 5 x 5 tile of the 10 x 10 lattice; 213,201
  coefficients, fitted to the training split by regularized least squares). It is defined
  on the 10 x 10 lattice only, as in the paper, so it is scored on `relative_error_L10`
  only.

The paper reports relative errors around `1e-3` for neural networks trained on 1.6
million configurations and about `2e-2` for its effective model; this benchmark trains
on 30,000 configurations, so those numbers are context, not targets.

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

### Floor at the reference
The target of this experiment is exactly a cubic polynomial of the spins, so accuracy
beyond the paper's reference architecture can be reached by regression on enumerated
spin products, which says nothing about learned many-body modelling. Every scored metric
is therefore floored at the reference model's own value on the slice (test: relative
error 0.029494, MAE 0.067042, tail RMSE 0.088614; train: 0.029987, 0.067603, 0.085722).
A model at least as accurate as the reference scores exactly the reference; a less
accurate one scores worse. Raw values are kept under `_artifacts` (`raw_*`). This
experiment checks that a method does not break the paper's classical-spin case; it does
not reward improvements.

### Validity and baseline
Failed when the file does not load, reads files, exceeds 100,000 values or 128 KiB of
code plus pickles, returns the wrong shape or non-finite values, or is no better than
a constant (R^2 <= 0). A failed run receives the worst value of every metric (the
constant predictor's relative error, MAE and tail RMSE, and `param_count` 100,000). A
same-width DNN (two hidden layers of 105 units) is frozen in the evaluator. The paper's
`5e-3` was obtained on its own coupling instance; context only.

## Common rules
Every number is recomputed from the submitted models' raw predictions on
evaluator-owned inputs. Self-reported numbers and sidecar files are ignored. A file in
`outputs/` that uses a reserved baseline name (`dnn`, `effective_model`) is ignored in
both experiments. A method is identified by its `<name>`: a method that submits a file
to one experiment but none to the other is failed in the other and receives its worst
values there.

## How to Evaluate
- `bash scripts/evaluate_train.sh` (or `scripts/evaluate.sh`): visible slice (public dev
  configurations of both experiments), writes `scoring/scores_train.json`.
- `bash scripts/evaluate_test.sh`: hidden slice, same contract, writes
  `scoring/scores_test.json`. Both take no arguments and exit 1 if any submitted model
  failed.
