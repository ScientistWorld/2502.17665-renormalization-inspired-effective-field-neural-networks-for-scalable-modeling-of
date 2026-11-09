# Evaluation

## Task and Data

One experiment is scored: energy prediction for a fixed 15-site spin system with
binary spins in {0, 1} and three-body couplings among all site triples. The
couplings are not published. Public data (in `data/three_spin_binary/`):

- `train.csv`: 16,000 configurations drawn uniformly with replacement, with energies.
- `dev.csv`: 2,000 further configurations, none of which occur in `train.csv`.

The hidden test scores every 15-site configuration that occurs in neither file
(18,162 configurations), labelling them at test time.

## Submission

Save a TorchScript module as `outputs/binary_<name>.pt`. It receives a float32
tensor of shape (N, 15) with values in {0, 1} and returns energies of shape (N,).
Every file matching that pattern is scored; `dnn` is reserved for the frozen
baseline.

## Metrics

- `relative_error` (primary, lower is better): RMSE on the scored configurations
  divided by the absolute mean of their true energies (the paper's binary metric).
- `mae`, `tail_rmse` (lower is better): mean absolute error, and RMSE on the
  quarter of scored configurations with the largest true energy magnitude.
- `param_count` (constraint): the number of numeric values stored in the model
  file, counted by the evaluator (narrow types one per element, wide types one
  per 4 bytes). Budget 100,000. Not ranked.

## Validity

A run is reported as failed, with no metrics and exit status 1, when the file is
not a loadable TorchScript module, the output has the wrong shape, any prediction
is non-finite, the model exceeds 100,000 stored values or 128 KiB of code plus
pickled non-tensor attributes, or its RMSE is no better than that of the best
constant prediction (R^2 <= 0). Self-reported numbers and sidecar files are ignored.

## Baselines and Targets

The paper compares against ordinary DNN, ResNet, and DenseNet architectures of the
same width. A same-width DNN (two hidden layers of 105 units) is frozen inside the
evaluator and scored on the same configurations in every run. The paper reports a
relative error below about `5e-3` for its best models; that number was obtained on
the authors' own (unpublished) coupling instance, so it is context, not a target
on this instance.

## How to Evaluate

- `bash scripts/evaluate_train.sh` (or `scripts/evaluate.sh`): visible slice, the
  public dev configurations, writes `scoring/scores_train.json`.
- `bash scripts/evaluate_test.sh`: hidden slice, same contract, writes
  `scoring/scores_test.json`. Both take no arguments.
