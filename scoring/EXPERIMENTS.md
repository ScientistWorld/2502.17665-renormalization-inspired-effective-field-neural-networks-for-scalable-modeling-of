# Experiment Overview

## three_spin_binary
Tests whether a learned energy predictor can accurately model a fixed binary many-body interaction with three-site products across all 15-site configurations sampled in the benchmark split. This is the simplest setting used to support the paper's central claim about high-order interaction modeling.

## Attempted Expansion: three_spin_continuous
The continuous-spin setting is implemented as an attempted extension, but it is
not part of the current validated score because the reproduced error remains too
far from the paper's reported value. Treat it as a diagnostic experiment rather
than a target for comparison until it is improved.
