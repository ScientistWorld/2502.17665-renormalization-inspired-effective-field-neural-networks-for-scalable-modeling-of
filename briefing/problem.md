# Problem Definition

## Research Question

Learn accurate, scalable energy functions for classical and quantum many-body spin
systems from sampled configurations. The central challenge is to model high-order
interactions without enumerating all interaction terms, while keeping the learned
energy predictor compact enough to evaluate quickly, and to transfer what is learned
on small systems to larger ones.

The benchmark instantiates the question twice.

**Double-exchange model (main task).** Itinerant electrons hop between nearest
neighbours of an L x L periodic square lattice (hopping t = 1) and couple on every site
to a classical unit spin (Hund coupling J = 16t), at chemical potential mu = -8.3t and
temperature T = 0.1t. For a spin configuration S, the target is the electrons' Monte
Carlo energy E_MC(S) = -T sum_n log(1 + exp(-E_n(S)/T)), where E_n(S) are the
single-particle energies of the electron Hamiltonian in the background S (measured from
the chemical potential). Computing it exactly requires diagonalizing a 2L^2 x 2L^2
matrix, which is what makes Monte Carlo on large lattices expensive; a fast and
accurate surrogate removes that bottleneck. Spins are drawn independently and uniformly
on the unit sphere. A model learns from 30,000 configurations of the 10 x 10 lattice
with their energies and must predict energies of unseen configurations on 10 x 10,
16 x 16 and 24 x 24 lattices, with no retraining for the larger sizes, within a budget
of 4,000 stored numeric values.

**Three-spin model (reproduction check).** A 15-site system with binary spins in
{0, 1} and fixed (unpublished) three-body couplings between every triple of sites. A
model is trained on 16,000 sampled configurations with their energies and must predict
the energies of every configuration it has never seen, within a budget of 100,000
stored values.

## Why It Matters

Many-body simulations require repeated energy evaluations. Exact or high-order
calculations become prohibitively expensive as system size grows, so a learned
surrogate that is accurate, small, and valid on systems larger than those it was
trained on can accelerate Monte Carlo and related workflows while preserving
physically meaningful energy differences.

## Success Criteria

A successful solution predicts held-out energies with low relative error on the
training lattice size and on larger lattices, beats an ordinary neural-network
baseline of comparable size, stays within the size budget, and learns the energy from
the provided samples rather than computing it from the electronic Hamiltonian.
