| case | exits (train/test) | binary rel. err (scored) | DE rel. err L10 / L16 / L24 | binary rank | DE rank | combined rank (honest) | combined rank (all) | failure |
|---|---|---|---|---|---|---|---|---|
| _de_baselines | None/None | nan | nan / nan / nan | nan | nan | - | - | no score file |
| _de_reference | None/None | nan | nan / nan / nan | nan | nan | - | - | no score file |
| bin_better_continued | 0/0 | 0.02949 | 0.001204 / 0.0007649 / 0.0006064 | 0 | 0 | 0 | 0 |  |
| bin_reference | 0/0 | 0.02949 | 0.001204 / 0.0007649 / 0.0006064 | 0 | 0 | 0 | 0 |  |
| bin_reference_seed_20250218 | 0/0 | 0.02949 | 0.001204 / 0.0007649 / 0.0006064 | 0 | 0 | 0 | 0 |  |
| bin_reference_seed_20250219 | 0/0 | 0.03276 | 0.001204 / 0.0007649 / 0.0006064 | -0.003604 | 0 | -0.003604 | -0.001405 |  |
| bin_shortcut_move_baseline | 0/0 | 0.02949 | 0.001204 / 0.0007649 / 0.0006064 | 0 | 0 | 0 | 0 |  |
| bin_shortcut_polynomial_regression | 0/0 | 0.02949 | 0.001204 / 0.0007649 / 0.0006064 | 0 | 0 | 0 | 0 |  |
| bin_shortcut_scaled_0p001 | 1/1 | 2.808 | 0.001204 / 0.0007649 / 0.0006064 | -3.372 | 0 | -3.372 | -1.271 | RMSE 8.59637 is no better than the best constant predictor (8.1062); task failed |
| bin_shortcut_sidecar_claims | 0/0 | 0.02949 | 0.001204 / 0.0007649 / 0.0006064 | 0 | 0 | 0 | 0 |  |
| bin_weaker_20_epochs | 0/0 | 0.7247 | 0.001204 / 0.0007649 / 0.0006064 | -0.7672 | 0 | -0.7672 | -0.2989 |  |
| bin_weaker_dnn_submitted | 0/0 | 0.4987 | 0.001204 / 0.0007649 / 0.0006064 | -0.512 | 0 | -0.512 | -0.2004 |  |
| de_better_continued | 0/0 | 0.02949 | 0.001166 / 0.0007522 / 0.0005777 | 0 | 0.09178 | 0.09178 | 0.01011 |  |
| de_dnn_submitted | 0/0 | 0.02949 | 0.002663 / 0.001645 / 0.001197 | 0 | -3.242 | -3.242 | -0.3516 |  |
| de_prev_dnn_submitted | 0/0 | 0.02949 | 0.002663 / 0.001645 / 0.001197 | 0 | -3.242 | -3.242 | -0.3516 |  |
| de_prev_reference | 0/0 | 0.02949 | 0.001204 / 0.0007649 / 0.0006064 | 0 | 0 | 0 | 0 |  |
| de_prev_seed_20250218 | 0/0 | 0.02949 | 0.001216 / 0.0007698 / 0.0005511 | 0 | 0.04884 | 0.04884 | 0.006028 |  |
| de_prev_seed_20250219 | 0/0 | 0.02949 | 0.001929 / 0.001292 / 0.0008651 | 0 | -1.649 | -1.649 | -0.1785 |  |
| de_reference | 0/0 | 0.02949 | 0.001204 / 0.0007649 / 0.0006064 | 0 | 0 | 0 | 0 |  |
| de_reference_seed_20250218 | 0/0 | 0.02949 | 0.001216 / 0.0007697 / 0.0005511 | 0 | 0.04888 | 0.04888 | 0.006032 |  |
| de_reference_seed_20250219 | 0/0 | 0.02949 | 0.001974 / 0.001411 / 0.001007 | 0 | -1.984 | -1.984 | -0.2163 |  |
| de_shortcut_chebyshev_free_energy | 1/1 | 0.02949 | 0.01716 / 0.01068 / 0.007345 | 0 | -36.05 | -36.05 | -3.914 | forward pass on 10x10 lattices exceeded the compute budget of 50000 work units per site (at aten::bm |
| de_shortcut_constant_per_site | 1/1 | 0.02949 | 0.01716 / 0.01068 / 0.007345 | 0 | -36.05 | -36.05 | -3.914 | RMSE 0.877677 on 10x10 lattices is no better than the best constant predictor (0.877642); task faile |
| de_shortcut_de_only | 1/1 | 2.808 | 0.001204 / 0.0007649 / 0.0006064 | -3.372 | 0 | -3.372 | -1.271 | no outputs/binary_m.pt submitted for this experiment |
| de_shortcut_exact_diagonalization | 1/1 | 0.02949 | 0.01716 / 0.01068 / 0.007345 | 0 | -36.05 | -36.05 | -3.914 | model code uses forbidden operation(s) ['linalg_eigvalsh'] |
| de_shortcut_move_baseline | 0/0 | 0.02949 | 0.001204 / 0.0007649 / 0.0006064 | 0 | 0 | 0 | 0 |  |
| de_shortcut_nan_output | 1/1 | 0.02949 | 0.01716 / 0.01068 / 0.007345 | 0 | -36.05 | -36.05 | -3.914 | 4000 non-finite predictions on 10x10 lattices |
| de_shortcut_oversized_c9 | 1/1 | 0.02949 | 0.01716 / 0.01068 / 0.007345 | 0 | -36.05 | -36.05 | -3.914 | model stores 26436 values, above the budget of 4000 |
| de_shortcut_sidecar_claims | 0/0 | 0.02949 | 0.001204 / 0.0007649 / 0.0006064 | 0 | 0 | 0 | 0 |  |
| de_shortcut_size_locked | 1/1 | 0.02949 | 0.01716 / 0.01068 / 0.007345 | 0 | -36.05 | -36.05 | -3.914 | forward pass on 16x16 lattices raised RuntimeError: The following operation failed in the TorchScrip |
| de_shortcut_wrong_shape | 1/1 | 0.02949 | 0.01716 / 0.01068 / 0.007345 | 0 | -36.05 | -36.05 | -3.914 | forward returned shape (249,) for 250 inputs of size 10x10; expected (250,) |
| de_ti_expansion_1847 | 0/0 | 0.02949 | 0.0008759 / 0.0005518 / 0.0003756 | 0 | 0.8629 | 0.8629 | 0.09478 |  |
| de_ti_expansion_227 | 0/0 | 0.02949 | 0.001166 / 0.0006858 / 0.0004815 | 0 | 0.2696 | 0.2696 | 0.03059 |  |
| de_weaker_3_epochs | 0/0 | 0.02949 | 0.002701 / 0.001718 / 0.001284 | 0 | -3.459 | -3.459 | -0.376 |  |
