# Scientist Constraints

## Data
- **double_exchange**: train only on split `train` of `data/double_exchange`
  (30,000 configurations of the 10 x 10 lattice, loaded with
  `data.double_exchange.load_split("train")`). The `dev_L10`, `dev_L16` and `dev_L24`
  splits are for checking (model selection) only.
- **three_spin_binary**: train only on `data/three_spin_binary/train.csv` (16,000
  samples). `dev.csv` is for checking only.
- Do not use any other source of energies. This includes energies you compute yourself
  from the electronic Hamiltonian (exact diagonalization, or any other
  electronic-structure calculation) to enlarge the training set.

## Submission
- **double_exchange**: a TorchScript module `outputs/de_<name>.pt`. It maps float32
  spins of shape (N, 3, L, L) to energies of shape (N,) and must work for every lattice
  size L. Reserved names: `dnn`, `effective_model`.
- **three_spin_binary**: a TorchScript module `outputs/binary_<name>.pt`. It maps
  float32 spins of shape (N, 15) in {0, 1} to energies (N,). Reserved name: `dnn`.
- Both must load with `torch.jit.load(path, map_location="cpu")`.

## Model Scale
- Stored numeric values (all tensors in the file: parameters, buffers and constants)
  may not exceed **4,000** for `double_exchange` and **100,000** for
  `three_spin_binary`.
- Types narrower than 32 bits count one per element; wider types count one per 4 bytes.
- TorchScript code plus pickled non-tensor attributes may take at most 128 KiB.
- **double_exchange compute budget**: a forward pass may spend at most **50,000 work
  units per lattice site**, counted by the evaluator from the tensor operations the model
  runs (multiply-accumulates of matrix products and convolutions, element counts of all
  other operations, 64-bit and complex types weighted by size). The reference uses about
  6,800. This is a count, not a timer: it does not depend on the machine.
- Exceeding any of these fails the run. Growing the model is resource scaling, not a
  methodological improvement.

## Evaluation
- Scores come only from `scripts/evaluate_train.sh` (visible slice) and the hidden test.
  Both recompute everything from the model file; sidecar JSON files and self-reported
  numbers are ignored.
- A run fails if it gives non-finite predictions, is no better than a constant on any
  lattice size, or (double_exchange) its model code calls dense linear-algebra solvers,
  runs work in other threads, or exceeds the compute budget.
- A failed run, and a method that submits to one experiment but not the other, receives
  the worst value of every metric of that experiment (the constant predictor's errors and
  the full size budget).
- **three_spin_binary** metrics are floored at the reference model's own values: being
  more accurate than the reference earns nothing there (its target is exactly a cubic
  polynomial); being less accurate costs rank.

## Compliance Checklist
- [ ] Trained only on the public training splits (dev rows used for checking only, or declared)
- [ ] No energies generated outside the provided files
- [ ] Submitted TorchScript energy models with the documented input/output shapes
- [ ] Stored values within budget (see `param_count` in the score file)
- [ ] double_exchange: compute per site within budget (see `compute_per_site_L*` under `_artifacts`)
- [ ] double_exchange: the model predicts the energy; it does not diagonalize or otherwise solve the electronic Hamiltonian (see DIRECTION.md)
- [ ] three_spin_binary: did not hard-code the energy function's form (see DIRECTION.md)
- [ ] Scores were produced by the evaluation scripts
