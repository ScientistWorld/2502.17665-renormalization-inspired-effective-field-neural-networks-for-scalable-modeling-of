# Paper's Method

## Key Contribution

The paper introduces Effective Field Neural Networks (EFNNs), a recursive architecture inspired by renormalization and continued functions for learning many-body energy maps.

## Approach

EFNN decomposes each layer into an effective-field transform and a quasi-particle layer. For layer `i`, it computes `F_i = f_{i-1}(S_{i-1})`, transforms the original input as `g_i(S_0)`, then forms `S_i = g_i(S_0) * F_i` by elementwise multiplication. The final energy is obtained from the final quasi-particle representation by summing or linearly projecting its elements. The functions `f` and `g` are linear-tanh-linear mappings.

The repeated injection of the original spin configuration into every quasi-particle layer is the defining computational step. It creates a nested continued-function structure that differs from DNN chains, additive ResNet skips, and DenseNet concatenation.

## Main Claims

Primary claim: EFNNs model many-body energy functions more accurately than standard DNN, ResNet, and DenseNet baselines at comparable model scale.

Secondary claims: the architecture remains effective for continuous spin inputs, can incorporate symmetry layers for quantum double-exchange energy prediction, and generalizes from smaller to larger lattices in that quantum setting.

## Implementation in this workspace

- Binary three-spin task: `method/models.py` (`EFNN`, plus `DNN`/`ResNet`/`DenseNet`
  comparisons), trained by `method/train_three_spin.py` (`bash scripts/method.sh binary`):
  3 field-particle layers of width 105, Adam with exponential decay, 2,400 epochs.
- Double-exchange task: `method/models_de.py` (`EFNNSym`: symmetrization layers
  T_j = sum_k conv(S_0,k) * conv'(S_0,k) with bias-free filters, conv-tanh-conv maps f_j
  and g_j with circular padding, q = conv to one channel then lattice sum; `DNNSym`
  without the S_0 injections), trained by `method/train_double_exchange.py`
  (`bash scripts/method.sh de`): C = 3 channels, 3 layers, 5 x 5 kernels (3,412
  parameters), batch 256, Adam lr 2e-3 decaying by 0.97 per epoch, 150 epochs,
  weights with the lowest training loss.
