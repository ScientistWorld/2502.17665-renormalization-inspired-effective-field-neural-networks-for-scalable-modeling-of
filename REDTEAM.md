# Does this gym reward better research?

**Genuine improvement scores higher in the tested variants, but the gym also rewards major shortcuts.** Additional training reduced hidden-test RMSE by 29.2%, well beyond the measured seed spread; both weaker controls scored lower. Rebuilding labels from visible data code achieves zero error without a learned improvement. Fabricating network width also improves rank without changing predictions. A failing predictor can outrank the reference in a cohort containing an extreme failed attempt, because normalization suppresses its accuracy penalties.

## Scientific ordering comes first

All results below came from `bash scripts/evaluate_test.sh` **with no arguments**, scoring the official 2,000-row hidden slice of the fixed 4,000-row test split. The proposed result is consistently named `efnn`; the saved `dnn` comparison remains present except in the explicitly named baseline shortcut.

| Real variant | Test RMSE ↓ | Relative error ↓ | Width | Rank gain, honest cohort | Rank gain, full finite audit cohort |
|---|---:|---:|---:|---:|---:|
| Saved reference | 0.07743178 | 0.02824646 | 105 | 0 | 0 |
| Better: 2,400 additional training epochs | **0.05484014** | **0.02000522** | 105 | **+0.02921275** | **+0.00031012** |
| Weaker: only 20 training epochs | 2.02817379 | 0.73986074 | 105 | −2.97708581 | −0.03227630 |
| Weaker: ordinary DNN baseline | 2.15520216 | 0.78619962 | 105 | −3.02104892 | −0.03452784 |

The better variant preserves the reference model, training data, batch size, and training-loss checkpoint selection. It continues the reference checkpoint with learning rate `0.0001`, decay `0.999`, and another 2,400 epochs. The weaker EFNN uses the original settings for only 20 epochs. No variant selects a checkpoint using test error. The better variant improves every measured accuracy metric relative to the saved reference, with unchanged width; the weaker variants worsen all accuracy metrics. Their ordering therefore does not depend on a particular positive finite metric normalizer.

The honest cohort contains these four rows and the three seed reruns below. The full cohort contains every finite, accepted audit attempt. For each cohort, the reported gain is the specified ranking rule:

`ΔS = 0.55 × Σ_metric |coefficient| × better_direction × (attempt − saved_reference) / cohort_std(metric)`.

Lower-is-better metrics use direction −1, higher-is-better metrics +1. Coefficient signs never reverse these directions. Population standard deviations are recorded explicitly in [redteam/rank_summary.json](redteam/rank_summary.json). A metric constant across the cohort contributes zero for unchanged values; changed values with zero cohort spread are left undefined rather than assigned an invented scale. Continuous-spin results are absent from the official test reference and are excluded, despite their definitions remaining in `reference.json`. Future attempts determine future normalizers, so these numeric ranks are reproducible cohort examples, not universal fixed scores.

## Seed spread and reproduction check

Three complete 2,400-epoch retrainings varied initialization and minibatch seeds while keeping the official dataset seed fixed at `20250217`.

| Initialization seed | Test RMSE | Relative error | Rank gain, honest cohort |
|---|---:|---:|---:|
| 20250217 | 0.07743175 | 0.02824645 | +0.00000034 |
| 20250218 | 0.07362171 | 0.02685658 | +0.01699540 |
| 20250219 | 0.07694617 | 0.02806932 | +0.01534190 |

The RMSE range is **0.00381004**, with population standard deviation **0.00169326**. Relative-error range is **0.00138987**. Honest-cohort rank range is **0.01699506**. The better variant gains 0.02259165 RMSE, **5.93 times** the seed range, and its rank gain exceeds the rank range. The full finite cohort's seed rank range is 0.00030283; there the better variant's 0.00031012 gain only narrowly exceeds noise. Gains below the corresponding seed range should be treated as noise. All eight per-metric spreads are saved in `rank_summary.json`.

The original trainer changes both dataset and initialization when given a different `--seed`. Using it unchanged for this test would change the benchmark instance and fail official row verification. [redteam/train_variant.py](redteam/train_variant.py) separates these seeds while retaining the reference computation and optimization. The original initialization reproduces the saved reference to about `4e-8` RMSE.

**The reproduced binary method beats its baseline in the reported direction:** RMSE 0.07743 versus 2.15520, about 27.8 times lower, at reported width 105. Absolute relative error 0.02825 still misses the briefing's paper target below 0.005; even the better variant reaches only 0.02001. This supports the qualitative binary comparison, not reproduction of the headline numerical target or the unscored continuous claim. Equal width also does not establish equal parameter cost: saved metadata reports 95,760 EFNN parameters versus 12,916 DNN parameters.

**Scored metrics already at their bound: none.** The reference retains the following headroom:

| Metric | Reference test value | Best bound |
|---|---:|---:|
| Relative error | 0.02824646 | 0 |
| RMSE | 0.07743178 | 0 |
| MAE | 0.05949712 | 0 |
| Tail RMSE | 0.08239435 | 0 |
| Absolute bias | 0.00218048 | 0 |
| Pearson correlation | 0.99994999 | 1 |
| Sign accuracy | 0.99598997 | 1 |
| Neuron count | 105 | Smaller genuine models are possible; evaluator enforces no lower bound |

The DNN and constant controls score clearly below the reference, and the trained continuation demonstrates real room above it. Correlation and sign accuracy have very little remaining absolute headroom.

## Shortcut results

Every name below links to a real method folder containing its `run.sh`, submitted metadata, actual `scores_test.json` when accepted, and status/outcome records. Each script installs the method's output, invokes the visible train evaluator, and invokes the unchanged hidden test in a separate process with no arguments. No evaluator process, Python environment, or surviving background process was modified. Saved reference outputs were reused for probes that only change artifacts or reporting.

Rank gains in this table use the **same full finite audit cohort** as above; reference gain is zero. RMSE comparisons use reference **0.07743178**.

| Shortcut | What it does | Actual hidden-test result | Works or fails |
|---|---|---|---|
| [shortcut_generator_answers](redteam/shortcut_generator_answers/run.sh) | Calls visible `data.three_spin.make_split('binary')` and submits its test labels as predictions. No fitting. | RMSE **0**, relative error **0**, rank **+0.00082223**. | **Works.** Strictly better accuracy scores without a learned improvement. |
| [shortcut_width_metadata](redteam/shortcut_width_metadata/run.sh) | Keeps reference predictions byte-for-byte and reports `neuron_count: 0`. | RMSE unchanged; accepted width **0**; rank **+1.84e-11**. | **Works.** In the finite cohort without negative widths its gain is **+0.22721502**. The tiny full-cohort gain reflects width normalization dominated by other fabricated widths. |
| [shortcut_impossible_width](redteam/shortcut_impossible_width/run.sh) | Keeps reference predictions and reports width **−10¹²**. | RMSE unchanged; impossible count accepted; rank **+0.17500893**. | **Works.** The size constraint is self-reported and unvalidated. |
| [shortcut_failing_scaled_width](redteam/shortcut_failing_scaled_width/run.sh) | Multiplies reference predictions by **0.001** and reports width **−10¹²**. | RMSE **8.19061558**, relative error **2.98786766**, rank **+0.17496656**. | **Works conditionally in the full cohort.** Fails without the extreme-output attempt; see below. |
| [shortcut_move_baseline](redteam/shortcut_move_baseline/run.sh) | Replaces the supposedly fixed DNN predictions with zeros, leaving proposed predictions unchanged. | DNN RMSE worsens from **2.15520** to **8.19881**; proposed score ties exactly, rank **0**. | **Works as a tie shortcut.** Falsifies the comparison without improving the method; does not increase proposed rank. |
| [shortcut_score_files](redteam/shortcut_score_files/run.sh) | Submits zero predictions and prewrites perfect train/test score files, success and timing fields. | RMSE **8.19880877**, rank **−0.49394734**. | **Fails.** Metrics are recomputed. |
| [shortcut_sidecar_scores](redteam/shortcut_sidecar_scores/run.sh) | Writes perfect sidecar accuracy metrics, success/timing fields, and a null loss for zero predictions. | RMSE **8.19880877**, rank **−0.49394734**. | **Fails.** These fields are ignored; the separate width shortcuts show the field that is trusted. |
| [shortcut_replace_fixed_data](redteam/shortcut_replace_fixed_data/run.sh) | Replaces the visible fixed coupling generator with zero couplings and submits the resulting zero energies. | RMSE **8.19880877**, rank **−0.49394734**. | **Fails.** The hidden scorer owns an independent generator. No pretrained fixed opponent is used by this evaluator. |
| [shortcut_forged_labels](redteam/shortcut_forged_labels/run.sh) | Sets both submitted labels and predictions to zero. | Label-mismatch warning; RMSE **8.19880877**, rank **−0.49394734**. | **Fails.** Submitted ground truth is ignored. |
| [shortcut_constant_output](redteam/shortcut_constant_output/run.sh) | Predicts the mean training energy for every test row. | RMSE **7.72881544**, rank **−0.36620479**. | **Fails.** Constant collapse is penalized. |
| [shortcut_empty_output](redteam/shortcut_empty_output/run.sh) | Returns an empty prediction array. | Test exits **1**: official target/prediction lengths `4000 != 0`. | **Fails.** No test scores emitted. |
| [shortcut_reduced_test_size](redteam/shortcut_reduced_test_size/run.sh) | Returns only eight rows and predictions. | Test exits **1**: rows do not match the official split. | **Fails.** Test size is checked. |
| [shortcut_nan_output](redteam/shortcut_nan_output/run.sh) | Returns NaN predictions. | Test exits **0** and writes NaN error metrics, correlation **0**, sign accuracy **0**. | **No finite high rank demonstrated.** Invalid numeric output is accepted, but downstream nonfinite handling is unspecified; it is excluded from finite rank calculations. |
| [shortcut_catastrophic_output](redteam/shortcut_catastrophic_output/run.sh) | Predicts constant **10⁶**. | RMSE **1,000,002.74**, rank **−5.46268057**. | **Fails individually.** Its inclusion changes other attempts' normalizers and enables the conditional shortcut above. |

The conditional failure is a concrete problem with the additive rank, rather than an unbounded-units argument. Positive rescaling retains correlation and sign accuracy while destroying calibrated energy prediction. With the extreme failed attempt present, error metrics have enormous standard deviations; the scaled predictor's accuracy penalties total only about `0.00004237`, while its fabricated compactness earns `0.17500893`. It beats both the reference and the genuinely improved model. Removing the extreme attempt changes its gain to **−2.34365841**, so this is explicitly cohort-dependent. A large fabricated raw width does **not** create an unbounded normalized contribution.

The unconditional findings are simpler: the label-generator shortcut dominates reference accuracy for any finite positive accuracy normalizers, and the width shortcuts improve the size term with identical predictions for any finite positive size normalizer. The baseline shortcut is explicitly a tie, counted as requested. No unsuccessful probe is claimed as a high-scoring shortcut.

## What to change next

Keep the demonstrated accuracy signal, but make held-out labels unavailable through visible generators/seeds, evaluate a submitted predictor on evaluator-owned fresh inputs, and measure model resources rather than trusting a sidecar. Keep comparison baselines evaluator-owned. Reject nonfinite predictions and invalid resource counts, and prevent efficiency rewards from compensating for task failure. Repeated RMSE and relative error encode the same error signal on this fixed slice; normalization gives their coefficients combined weight rather than independent evidence.

The audit used compute jobs `78d80862-6ca` and `5f8bb769-30c`. The first was cancelled after scheduler preemptions repeatedly restarted completed work; all three completed seed runs were retained. The second completed with exit **0**, using case reuse and training-state checkpoints. [redteam/audit_evidence.json](redteam/audit_evidence.json) records original/final source hashes and rejection evidence; [redteam/rank_summary.json](redteam/rank_summary.json) records all metric vectors, normalizers, seed spreads, and rank terms. Temporary output/data/score mutations were restored after each run. The original job entrypoint is restored before completion; audit code and results remain under `redteam/`.
