# Research Direction

## Research Problem
This benchmark studies compact neural energy predictors for many-body spin systems with high-order interactions. The scientific question is how to encode interaction structure so a model can fit nontrivial energy functions efficiently.

## Direction Scope
Future work should stay within compact, structured neural energy prediction for the same fixed spin-energy tasks. Improvements should make the predictor more accurate, more robust across the scored slices, or more parameter-efficient without changing the underlying generated data distribution.

## Allowed Changes
Agents may change model architecture, optimization, initialization, normalization, regularization, or training schedule, provided the output remains an energy prediction artifact that can be scored by the standard evaluator. The fixed train/test splits and metric definitions must remain unchanged.

## Out of Bounds
Do not use the closed-form generator or explicit enumeration of all cubic interaction features as the final predictor. That solves this small benchmark by using privileged knowledge of the labels rather than improving learned energy prediction. Do not use external data, train on the test split, or increase the reported width beyond the constraints just to obtain gains.
