# Research Direction

## Research Problem
This benchmark studies compact neural energy predictors for many-body spin
systems with high-order interactions. The scientific question is how to encode
interaction structure so that a learned model fits a nontrivial energy function
accurately from samples, at a fixed model scale, and generalizes to unseen
configurations.

## Direction Scope
Future work should stay within compact, learned neural energy prediction for the
fixed spin-energy task. Improvements should make the predictor more accurate on
unseen configurations or more robust across the energy range, within the same
size budget and the same training data.

## Allowed Changes
Model architecture, optimization, initialization, normalization, regularization,
training schedule, and model selection on the training data, provided the result
is submitted as a TorchScript energy model scored by the standard evaluators.

## Out of Bounds
- Do not build the predictor from an explicit polynomial expansion of the spins
  (for example, linear or kernel regression on enumerated products of up to three
  spins, or fitting one coefficient per site triple). The energy function happens
  to be such a polynomial; recovering its coefficients by regression solves this
  small instance with privileged knowledge of its functional form rather than
  advancing learned many-body energy prediction. The hidden test cannot tell such
  a model apart from a learned one; reviewers check for it.
- Do not use external data, train on configurations other than the public
  training file, or exceed the size budget.
- Do not try to read evaluator files or labels from inside the model.
