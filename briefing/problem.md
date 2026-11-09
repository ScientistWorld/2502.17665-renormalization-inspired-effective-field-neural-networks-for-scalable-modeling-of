# Problem Definition

## Research Question

Learn accurate, scalable energy functions for classical and quantum many-body spin
systems from sampled configurations. The central challenge is to model high-order
interactions without enumerating all interaction terms, while keeping the learned
energy predictor compact enough to evaluate quickly.

This benchmark instantiates the question on a 15-site system with binary spins and
fixed (unpublished) three-body couplings between every triple of sites: a model is
trained on 16,000 sampled configurations with their energies and must predict the
energies of configurations it has never seen, within a fixed model-size budget.

## Why It Matters

Many-body simulations often require repeated energy evaluations. Exact or high-order
calculations can become prohibitively expensive as system size grows, so a successful
learned surrogate can accelerate Monte Carlo and related workflows while preserving
physically meaningful energy differences.

## Success Criteria

A successful solution predicts held-out energies with low relative error, beats an
ordinary neural-network baseline of the same width, and does not rely on a larger
model (more stored values) to obtain the gain.
