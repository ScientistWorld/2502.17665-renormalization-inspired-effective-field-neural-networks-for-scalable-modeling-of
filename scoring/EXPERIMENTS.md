# Experiment Overview

## double_exchange (weight 0.8)
Establishes whether a compact learned energy model can replace an expensive
quantum-mechanical calculation: the electronic Monte Carlo energy of itinerant electrons
moving in a background of classical spins. The energy is not a low-order polynomial of
the spins, and computing it exactly requires diagonalizing a matrix whose size grows
with the lattice.

Models learn from 30,000 random configurations of a 10 x 10 periodic lattice. They are
scored on hidden configurations of that size and of two larger sizes (16 x 16 and
24 x 24) without retraining, which measures whether what was learned transfers to bigger
systems.

What is measured: relative energy error at each size.

What is held fixed:
- the training data;
- a small budget on stored values, so gains come from better modelling rather than
  capacity;
- two frozen comparison models: an ordinary deep network and an explicit
  spin-interaction expansion.

Models that compute the energy by diagonalization inside the model, fail on a lattice
size, or are no better than a constant are reported as failed.

## three_spin_binary (weight 0.2)
Establishes whether a learned energy predictor can model a fixed binary many-body
interaction with three-site couplings on 15 sites, and generalize to configurations it
never saw. Models learn from 16,000 sampled configurations with their energies. The
hidden test asks for the energy of every configuration that appears in no public file,
so memorizing training rows does not help.

What is measured: the accuracy of predicted energies (relative error, mean absolute
error, and error on the highest-magnitude energies), compared with an ordinary
neural-network baseline of the same width that is frozen inside the evaluator.

What is held fixed: the training data and a budget on stored values.

This experiment carries a low weight. Its target is exactly a cubic polynomial of the
spins, so it checks the architecture comparison rather than open-ended modelling.
