| case | test exit | train exit | rel. error | MAE | tail RMSE | param count | rank (honest) | rank (passing) | failure |
|---|---|---|---|---|---|---|---|---|---|
| better_continued | 0 | 0 | 0.0210142 | 0.0478501 | 0.0633103 | 95762 | 0.04582 | 0.05184 |  |
| reference | 0 | 0 | 0.0294936 | 0.0670421 | 0.0886145 | 95762 | 0 | 0 |  |
| reference_seed_20250218 | 0 | 0 | 0.0282248 | 0.0641307 | 0.08453 | 95762 | 0.006971 | 0.007887 |  |
| reference_seed_20250219 | 0 | 0 | 0.0327612 | 0.0745572 | 0.0991026 | 95762 | -0.01795 | -0.02031 |  |
| shortcut_catastrophic_1e6 | 1 | 1 | - | - | - | - | - | - | submission: RMSE 999997 is no better than the best constant predictor (8.1062); task failed |
| shortcut_constant_train_mean | 1 | 1 | - | - | - | - | - | - | submission: RMSE 8.10663 is no better than the best constant predictor (8.1062); task failed |
| shortcut_fp16_width150 | 1 | 1 | - | - | - | - | - | - | submission: model stores 190802 values, above the budget of 100000 |
| shortcut_list_attribute | 1 | 1 | - | - | - | - | - | - | submission: code and pickled non-tensor attributes take 376308 bytes (limit 131072); numeric data must be stored in tensors |
| shortcut_move_baseline | 0 | 0 | 0.0294936 | 0.0670421 | 0.0886145 | 95762 | 0 | 0 |  |
| shortcut_nan_output | 1 | 1 | - | - | - | - | - | - | submission: 18162 non-finite predictions |
| shortcut_not_torchscript | 1 | 1 | - | - | - | - | - | - | submission: cannot load as a TorchScript module (RuntimeError: PytorchStreamReader failed locating file constants.pkl: file not found. This is an internal miniz |
| shortcut_old_public_generator | 1 | 1 | - | - | - | - | - | - | submission: RMSE 12.7657 is no better than the best constant predictor (8.1062); task failed |
| shortcut_oversized_width150 | 1 | 1 | - | - | - | - | - | - | submission: model stores 190802 values, above the budget of 100000 |
| shortcut_padded_buffer | 1 | 1 | - | - | - | - | - | - | submission: model stores 105764 values, above the budget of 100000 |
| shortcut_polynomial_regression | 0 | 0 | 2.79793e-07 | 5.24354e-07 | 1.22477e-06 | 3726 | 0.1597 | 0.1807 |  |
| shortcut_reads_file | 1 | 1 | - | - | - | - | - | - | submission: model code uses forbidden operation(s) ['from_file'] |
| shortcut_scaled_0p001 | 1 | 1 | - | - | - | - | - | - | submission: RMSE 8.59637 is no better than the best constant predictor (8.1062); task failed |
| shortcut_sidecar_claims | 0 | 0 | 0.0294936 | 0.0670421 | 0.0886145 | 95762 | 0 | 0 |  |
| shortcut_train_lookup | 1 | 1 | - | - | - | - | - | - | submission: RMSE 8.10663 is no better than the best constant predictor (8.1062); task failed |
| shortcut_wrong_shape | 1 | 1 | - | - | - | - | - | - | submission: forward returned shape (4095,) for 4096 inputs; expected (4096,) |
| weaker_20_epochs | 0 | 0 | 0.724677 | 1.65042 | 2.34289 | 95762 | -3.821 | -4.323 |  |
| weaker_dnn_submitted | 0 | 0 | 0.498683 | 1.14962 | 1.50784 | 12918 | -2.55 | -2.885 |  |

Frozen baselines in the reference run: {"dnn": {"mae": 1.1496152010446494, "param_count": 12918.0, "relative_error": 0.49868271753422583, "tail_rmse": 1.5078377145951478}}
Seed spread: {"relative_error": {"values": [0.029493604412623508, 0.02822482962613103, 0.032761216644939764], "range": 0.004536387018808735, "std": 0.0019109592223115553}, "mae": {"values": [0.06704212363992415, 0.06413066253553044, 0.074557180064702], "range": 0.01042651752917155, "std": 0.004392733224265753}, "tail_rmse": {"values": [0.08861447617851868, 0.08452995556237551, 0.09910257544324107], "range": 0.014572619880865556, "std": 0.006137722931389737}, "rank_delta_honest": {"values": [0.0, 0.006971406470500353, -0.017950341206576133], "range": 0.024921747677076486}}
Paired bootstrap: {"reference_seed_20250218": {"relative_error_minus_reference": -0.001268774786492479, "bootstrap_ci95": [-0.0016922801966824286, -0.0008511092420696181], "bootstrap_sd": 0.00021972993554112764, "fraction_better_than_reference": 1.0}, "reference_seed_20250219": {"relative_error_minus_reference": 0.003267612232316256, "bootstrap_ci95": [0.002775775730194703, 0.003764856196648917], "bootstrap_sd": 0.00025635097363881413, "fraction_better_than_reference": 0.0}, "better_continued": {"relative_error_minus_reference": -0.008479410907085067, "bootstrap_ci95": [-0.00884767990971789, -0.008112493567519762], "bootstrap_sd": 0.00019002512421241393, "fraction_better_than_reference": 1.0}}
Train vs test: {"reference": {"train_relative_error": 0.029986582799004558, "test_relative_error": 0.029493604412623508}, "reference_seed_20250218": {"train_relative_error": 0.028102878591537657, "test_relative_error": 0.02822482962613103}, "reference_seed_20250219": {"train_relative_error": 0.032732000660656846, "test_relative_error": 0.032761216644939764}, "better_continued": {"train_relative_error": 0.02134912013800675, "test_relative_error": 0.02101419350553844}, "weaker_20_epochs": {"train_relative_error": 0.7312400507370979, "test_relative_error": 0.7246772899963335}, "weaker_dnn_submitted": {"train_relative_error": 0.4980023156485842, "test_relative_error": 0.49868271753422583}}
