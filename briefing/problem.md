# Problem Definition

## Research Question

Learn accurate, scalable energy functions for classical and quantum many-body spin systems from sampled configurations. The central challenge is to model high-order interactions without enumerating all interaction terms, while keeping the learned energy predictor compact enough to evaluate quickly.

## Why It Matters

Many-body simulations often require repeated energy evaluations. Exact or high-order calculations can become prohibitively expensive as system size grows, so a successful learned surrogate can accelerate Monte Carlo and related workflows while preserving physically meaningful energy differences.

## Success Criteria

A successful solution predicts held-out energies with low RMSE or relative error, beats ordinary neural-network baselines at comparable scale, and does not rely on a much larger parameter count to obtain the gain.
