# Experiment Overview

## three_spin_binary
Establishes whether a learned energy predictor can model a fixed binary
many-body interaction with three-site couplings on 15 sites, and generalize to
configurations it never saw. Models learn from 16,000 sampled configurations
with their energies; the hidden test asks for the energy of every configuration
that appears in no public file, so memorizing training rows does not help.

What is measured: the accuracy of predicted energies (relative error, mean
absolute error, and error on the highest-magnitude energies), compared with an
ordinary neural-network baseline of the same width that is frozen inside the
evaluator. What is held fixed: the training data and a budget on the number of
values the submitted model stores, so gains must come from better modelling
rather than more capacity. Runs that fail the task (no better than a constant
prediction, invalid outputs, over budget) are reported as failed.
