# Overview

- **Paper ID:** 2502.17665
- **Title:** Renormalization-Inspired Effective Field Neural Networks for Scalable Modeling of Classical and Quantum Many-Body Systems
- **Domain:** Physics-informed machine learning / many-body systems
- **TL;DR:** The paper proposes a recursive effective-field neural architecture that learns compact energy predictors for high-order many-body interactions.

## Short Summary

The paper studies energy prediction for classical and quantum spin systems where explicit high-order interaction expansions are costly or unstable. It proposes EFNN, an architecture that recursively refines effective fields and quasi-particle representations using multiplicative input injections. The paper reports lower errors than DNN, ResNet, and DenseNet baselines on binary and continuous three-spin tasks, and strong generalization for a quantum double-exchange model.

## Key Results

- Binary 15-site three-spin task: 2- and 3-layer EFNNs with at least 105 neurons reach relative error below about `5e-3`.
- Continuous three-spin task: best reported EFNN RMSE is about `4e-2`.
- Quantum double-exchange task: a 3-layer, 3-channel EFNN trained on `10x10` lattices reaches relative error around `3.6e-4` on `40x40`.
