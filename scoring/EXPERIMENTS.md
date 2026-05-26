# Experiment Overview

## three_spin_binary
Tests whether a learned energy predictor can accurately model a fixed binary many-body interaction with three-site products across all 15-site configurations sampled in the benchmark split. This is the simplest setting used to support the paper's central claim about high-order interaction modeling.

Continuous-spin artifacts are available as diagnostics and documented in
`PROGRESS.md`, but they are not part of official scoring because the reproduced
RMSE remains too far from the paper's reported value.
