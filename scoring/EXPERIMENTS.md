# Experiment Overview

## three_spin_binary
Tests whether a learned energy predictor can accurately model a fixed binary many-body interaction with three-site products across all 15-site configurations sampled in the benchmark split. This is the simplest setting used to support the paper's central claim about high-order interaction modeling.

## three_spin_continuous
Tests whether the same energy-prediction problem remains tractable when each site takes a continuous value rather than a binary state. This setting checks that improvement is not limited to discrete inputs and that errors remain controlled away from the easier binary configuration space.
