# Research Direction

## Research Problem
This benchmark studies compact neural energy predictors for many-body spin systems with high-order interactions. The scientific question is how to encode interaction structure so a model can fit nontrivial energy functions efficiently.

## Core Contribution
The paper's contribution is a renormalization-inspired, continued-function style neural computation for many-body energies. Future improvements should work on architectures or training methods that preserve this goal: structured high-order interaction modeling through recursive effective representations.

## Approach Scope
Approaches may modify the recursive transforms, nonlinearities, initialization, normalization, optimizer, or regularization used for compact many-body energy models. They should remain in the family of structured neural energy predictors for the same spin-energy problem.

## Out of Bounds
Do not replace the task with explicit enumeration of all cubic interaction features followed by direct linear regression as the final method. That solves this small benchmark by using privileged knowledge of the exact generator rather than improving the paper's approach family. Do not use external data or a much wider network to obtain gains.
