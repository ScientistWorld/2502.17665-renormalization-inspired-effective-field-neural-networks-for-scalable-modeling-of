# Experiment Overview

## three_spin_binary
Tests whether a learned energy predictor can accurately model a fixed binary many-body interaction with three-site products across all 15-site configurations sampled in the benchmark split. This is the simplest setting used to support the paper's central claim about high-order interaction modeling.

## three_spin_continuous
Tests whether a learned energy predictor can model the same fixed three-body
interaction when each spin is a continuous value in `[-1, 1]` instead of a
binary state. This gives the benchmark a second input regime while preserving
the same data split and capacity constraints.
