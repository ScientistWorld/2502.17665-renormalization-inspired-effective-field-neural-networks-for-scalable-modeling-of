# Research Direction

## Research Problem
This benchmark studies compact energy predictors for many-body spin systems with
high-order interactions. The scientific question is how to encode interaction structure
so that a model:
- fits a nontrivial energy function accurately from a fixed set of samples, at a fixed
  model scale;
- generalizes to unseen configurations;
- for the double-exchange model, transfers from the training lattice to larger
  lattices without retraining.

## Direction Scope
Future work should make the predictor more accurate on unseen configurations, more
robust across lattice sizes and across the energy range, within the same size budget
and the same training data. The main experiment (`double_exchange`, weight 0.8) is a
surrogate for an expensive quantum calculation. Neural architectures,
physics-motivated features or expansions, and hybrids of the two are all in scope, as
long as the model is learned from the provided samples.

## Allowed Changes
- Model architecture.
- Learned or physics-motivated input features, including fitted spin-interaction
  expansions and combinations with neural networks.
- Optimization, initialization, normalization, regularization and training schedule.
- Model selection on the public data.

The result must be submitted as a TorchScript energy model scored by the standard
evaluators.

## Out of Bounds
- **double_exchange**: do not compute the energy from the electronic Hamiltonian,
  either inside the model or to make extra training data. This rules out:
  - building the electron Hamiltonian and diagonalizing it, or evaluating its free
    energy by any equivalent electronic-structure calculation (for example Lanczos,
    kernel-polynomial or Chebyshev expansions of the matrix, or determinants);
  - generating labels for configurations outside the public files.

  The evaluator rejects models that call dense linear-algebra solvers. Hand-written
  equivalents are equally out of bounds, and reviewers check for them.
- **three_spin_binary**: do not build the predictor from an explicit polynomial
  expansion of the spins (for example linear or kernel regression on enumerated
  products of up to three spins, or fitting one coefficient per site triple). The
  energy function happens to be such a polynomial. Recovering its coefficients by
  regression solves this small instance with privileged knowledge of its functional
  form rather than advancing learned many-body energy prediction. The hidden test
  cannot tell such a model apart from a learned one; reviewers check for it.
- Do not use external data, train on configurations other than the public training
  files, or exceed the size budgets.
- Do not try to read evaluator files or labels from inside the model.
