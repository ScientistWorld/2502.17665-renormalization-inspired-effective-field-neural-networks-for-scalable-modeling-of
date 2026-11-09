| case | exits (train/test) | binary rel. err | DE rel. err L10 / L16 / L24 | binary rank | DE rank | combined rank | failure |
|---|---|---|---|---|---|---|---|
| _de_baselines | None/None | nan | nan / nan / nan | nan | nan | - |  |
| _de_reference | None/None | nan | nan / nan / nan | nan | nan | - |  |
| bin_better_continued | 0/0 | 0.02101 | nan / nan / nan | 0.009165 | nan | 0.009164905560683272 |  |
| bin_reference | 0/0 | 0.02949 | nan / nan / nan | 0 | nan | 0.0 |  |
| bin_reference_seed_20250218 | 0/0 | 0.02822 | nan / nan / nan | 0.001394 | nan | 0.0013942812941000716 |  |
| bin_reference_seed_20250219 | 0/0 | 0.03276 | nan / nan / nan | -0.00359 | nan | -0.0035900682413152253 |  |
| bin_shortcut_move_baseline | 0/0 | 0.02949 | nan / nan / nan | 0 | nan | 0.0 |  |
| bin_shortcut_polynomial_regression | 0/0 | 2.798e-07 | nan / nan / nan | 0.03194 | nan | 0.03193976639949457 |  |
| bin_shortcut_scaled_0p001 | 1/1 | nan | nan / nan / nan | nan | nan | - | RMSE 8.59637 is no better than the best constant predictor (8.1062); task failed |
| bin_shortcut_sidecar_claims | 0/0 | 0.02949 | nan / nan / nan | 0 | nan | 0.0 |  |
| bin_weaker_20_epochs | 0/0 | 0.7247 | nan / nan / nan | -0.7641 | nan | -0.7641184337209604 |  |
| bin_weaker_dnn_submitted | 0/0 | 0.4987 | nan / nan / nan | -0.51 | nan | -0.5099844950376139 |  |
| de_better_continued | 0/0 | nan | 0.001166 / 0.0007522 / 0.0005777 | nan | 0.09643 | 0.09643113035530637 |  |
| de_dnn_submitted | 0/0 | nan | 0.002663 / 0.001645 / 0.001197 | nan | -3.392 | -3.392247810971929 |  |
| de_effective_large | 0/0 | nan | 0.0008759 / 0.0005518 / 0.0003756 | nan | 0.9064 | 0.9064174991628872 |  |
| de_effective_submitted | 0/0 | nan | 0.001166 / 0.0006858 / 0.0004815 | nan | 0.2862 | 0.28617102470839567 |  |
| de_reference | 0/0 | nan | 0.001204 / 0.0007649 / 0.0006064 | nan | 0 | 0.0 |  |
| de_reference_seed_20250218 | 0/0 | nan | 0.001216 / 0.0007698 / 0.0005511 | nan | 0.05317 | 0.05317371131526773 |  |
| de_reference_seed_20250219 | 0/0 | nan | 0.001929 / 0.001292 / 0.0008651 | nan | -1.725 | -1.7254696508681695 |  |
| de_shortcut_constant_per_site | 1/1 | nan | nan / nan / nan | nan | nan | - | RMSE 0.877677 on 10x10 lattices is no better than the best constant predictor (0.877642); task failed |
| de_shortcut_exact_diagonalization | 1/1 | nan | nan / nan / nan | nan | nan | - | model code uses forbidden operation(s) ['linalg_eigvalsh'] |
| de_shortcut_move_baseline | 0/0 | nan | 0.001204 / 0.0007649 / 0.0006064 | nan | 0 | 0.0 |  |
| de_shortcut_nan_output | 1/1 | nan | nan / nan / nan | nan | nan | - | 4000 non-finite predictions on 10x10 lattices |
| de_shortcut_oversized_c9 | 1/1 | nan | nan / nan / nan | nan | nan | - | model stores 26436 values, above the budget of 4000 |
| de_shortcut_sidecar_claims | 0/0 | nan | 0.001204 / 0.0007649 / 0.0006064 | nan | 0 | 0.0 |  |
| de_shortcut_size_locked | 1/1 | nan | nan / nan / nan | nan | nan | - | forward pass on 16x16 lattices raised RuntimeError: The following operation failed in the TorchScript interpre |
| de_shortcut_wrong_shape | 1/1 | nan | nan / nan / nan | nan | nan | - | forward returned shape (249,) for 250 inputs of size 10x10; expected (250,) |
| de_weaker_3_epochs | 0/0 | nan | 0.002701 / 0.001718 / 0.001284 | nan | -3.622 | -3.6221598826224337 |  |
