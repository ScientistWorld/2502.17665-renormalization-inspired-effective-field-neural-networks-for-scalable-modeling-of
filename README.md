# Renormalization-Inspired Effective Field Neural Networks for Scalable Modeling of Classical and Quantum Many-Body Systems

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2502.17665"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2502.17665-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2502.17665](https://arxiv.org/abs/2502.17665) |
| **Domain** | Physics-informed machine learning / many-body systems |
| **TL;DR** | The paper proposes a recursive effective-field neural architecture that learns compact energy predictors for high-order many-body interactions. |

## Summary

The paper studies energy prediction for classical and quantum spin systems where explicit high-order interaction expansions are costly or unstable. It proposes EFNN, an architecture that recursively refines effective fields and quasi-particle representations using multiplicative input injections. The paper reports lower errors than DNN, ResNet, and DenseNet baselines on binary and continuous three-spin tasks, and strong generalization for a quantum double-exchange model.

## Problem

Learn accurate, scalable energy functions for classical and quantum many-body spin systems from sampled configurations. The central challenge is to model high-order interactions without enumerating all interaction terms, while keeping the learned energy predictor compact enough to evaluate quickly.

## The paper's method

The paper introduces Effective Field Neural Networks (EFNNs), a recursive architecture inspired by renormalization and continued functions for learning many-body energy maps.

## Key results of the paper

- Binary 15-site three-spin task: 2- and 3-layer EFNNs with at least 105 neurons reach relative error below about `5e-3`.
- Continuous three-spin task: best reported EFNN RMSE is about `4e-2`.
- Quantum double-exchange task: a 3-layer, 3-channel EFNN trained on `10x10` lattices reaches relative error around `3.6e-4` on `40x40`.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | +100.0 |
| `gpt-5.6-sol-xhigh` | +100.0 |
| `gpt-5.6-luna-xhigh` | +100.0 |
| `gpt-5.6-terra-xhigh` | +99.9 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install git+https://github.com/zlab-princeton/scientist_world_cli
sw research 2502.17665 --mode innovate --agent claude-code
sw evaluate 2502.17665
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

