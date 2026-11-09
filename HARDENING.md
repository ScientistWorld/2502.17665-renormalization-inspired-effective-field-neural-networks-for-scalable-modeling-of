# Hardening report (iteration 11)

**Bottom line.** Better results now score higher and worse ones lower, by more than the seed spread. Every major shortcut in `REDTEAM.md` now scores as a failed run or ties the reference exactly. One major shortcut stays open: fitting a polynomial of up to three-spin products to the training data. The energy function is exactly such a polynomial, so the test cannot tell this from a learned model; the rules forbid it and the separate grader has to catch it.

To make the fixes sound, the benchmark instance changed. The old couplings had a visible seed, so the gym now uses a new hidden coupling instance. "Before" numbers below come from `REDTEAM.md`: old public instance, `.npz` prediction files, 2,000-row test slice. "After" numbers come from job `e8c9644d-093` (training and scoring) and job `d475d03e-63d` (rescoring with three added probes; honest-variant scores reproduced exactly).

Every "after" result was produced by `bash scripts/evaluate_test.sh` with **no arguments**, in a fresh process, on a fresh `outputs/` holding only that case's files. The harness is `redteam/hardening/`; per-case scores, logs and exit codes are in `redteam/hardening/cases/`, and the full summary is in `redteam/hardening/summary.{json,md}`.

## What changed

| Problem (REDTEAM.md) | Fix |
|---|---|
| Visible `data/three_spin.py` could regenerate the test labels | New hidden coupling instance. Its couplings and its list of test configurations exist only in `eval/test/hidden_instance.json`, written by `eval/test/build_instance.py`, which holds the seed. Public data are committed CSVs: `train.csv` with 16,000 rows (the paper's size, drawn with replacement) and `dev.csv` with 2,000 distinct configurations that are not in `train.csv`. The test scores **all 18,162 configurations that appear in neither file**, labelled at test time. A SHA-256 check pins the test set's size and contents. |
| Width read from a method-written sidecar | A submission is now a TorchScript model, `outputs/binary_<name>.pt`, that the evaluators **run themselves**. `param_count` is measured from the archive: every tensor storage file, counted from the bytes it actually holds, with its dtype read from the archive's pickles. Types narrower than 32 bits count one per element and wider types one per 4 bytes. Hard budget: 100,000 (reference 95,762). Code plus pickled non-tensor attributes are capped at 128 KiB. Code that calls `from_file` is rejected. Sidecar JSON is ignored. |
| A failing predictor could outrank through a non-accuracy metric | A run is **failed** (no metrics, exit status 1, reason under `_failures`) if it: cannot be loaded; returns the wrong shape; returns any non-finite value; is over the size limits; or has RMSE no better than the best constant prediction (R² ≤ 0). The ranked metrics are now only accuracy metrics on the same predictions: `relative_error` (coefficient 1.0), `mae` (0.25) and `tail_rmse` (0.3). `param_count` is reported as a gated constraint with coefficient 0. |
| Baseline predictions owned by the method | The same-width DNN is frozen in `eval/train/baselines/` and `eval/test/baselines/` and scored in every run. A file named `outputs/binary_dnn.pt` is ignored, and the reason is recorded under `_ignored`. |
| NaN accepted with exit 0 | Failed (see the row above). |
| No way to separate the instance from the initialization seed | The trainer reads the fixed data files; `--seed` now controls only initialization and minibatch order. |

**Metric and weight changes, and why:**
- `pearson_corr` (reference 0.99995) and `sign_accuracy` (0.996) are dropped. Both are saturated, and both are scale-invariant: a rescaled, failing predictor kept full credit on them.
- `rmse` is dropped. On a fixed test set it is exactly proportional to `relative_error`, so keeping it only double-counted the same signal.
- `bias_abs` is dropped. For good models it is at noise level (about 0.002 against an RMSE of about 0.08).
- `neuron_count` is replaced by the measured `param_count`.
- The binary experiment's weight goes from 0.55 to **1.0**. The continuous experiment (0.45) was never scored, so its weight was a phantom. Its paper number (0.04) stays in `scoring/reference.json` under `_paper_reported`, along with the paper's widths. The paper's binary result (EFNN `relative_error` 0.005) is unchanged.

## Do better results score higher?

Yes. Ranks use the gym's formula: Σ|coef|·direction·(x − x_ref)/std over the honest cohort, weight 1.0, metrics with nonzero coefficients only.

| Variant | Before: rel. error | Before: rank (honest cohort) | After: rel. error | After: MAE | After: tail RMSE | After: rank (honest cohort) |
|---|---:|---:|---:|---:|---:|---:|
| Reference (EFNN 3×105, 2,400 epochs, seed 20250217) | 0.028246 | 0 | **0.029494** | 0.06704 | 0.08861 | 0 |
| Better: reference + 2,400 epochs (lr 1e-4, γ 0.999) | 0.020005 | +0.0292 | **0.021014** | 0.04785 | 0.06331 | **+0.0458** |
| Seed 20250218 | 0.026857 | +0.0170 | 0.028225 | 0.06413 | 0.08453 | +0.0070 |
| Seed 20250219 | 0.028069 | +0.0153 | 0.032761 | 0.07456 | 0.09910 | −0.0180 |
| Weaker: EFNN, 20 epochs | 0.739861 | −2.977 | 0.724677 | 1.65042 | 2.34289 | −3.821 |
| Weaker: same-width DNN (2×105) | 0.786200 | −3.021 | 0.498683 | 1.14962 | 1.50784 | −2.550 |

**Noise:**
- Training-seed noise dominates. Across three seeds, relative error ranges over 0.00454 (std 0.00191); the rank range is 0.0249.
- The better variant gains **0.00848** in relative error: 1.9× the seed range and 4.4 seed standard deviations. Its rank gain (+0.046) is 1.8× the seed rank range.
- Test-row sampling noise is small. In a paired bootstrap (the same 18,162 rows for every model, 1,000 joint resamples), better minus reference is −0.00848 with 95% CI [−0.00885, −0.00811], bootstrap sd 0.00019.
- The test set is the largest possible: every unseen configuration. Within one environment the reference run is deterministic: fixed seed, 2 threads, CPU.

**Paper effect versus noise:** EFNN beats the same-width DNN 0.0295 vs 0.499, about 17× lower relative error. That gap is roughly 240 seed standard deviations.

**Train and test agree:** honest models score alike on the visible dev slice and the hidden test.

| Model | Dev rel. error | Test rel. error |
|---|---:|---:|
| Reference | 0.02999 | 0.02949 |
| Better | 0.02135 | 0.02101 |
| DNN | 0.49800 | 0.49868 |

## Shortcuts: before and after

| Shortcut | Before (REDTEAM.md) | Fix | After |
|---|---|---|---|
| Rebuild labels from the visible generator (`shortcut_generator_answers`; now probed as `shortcut_old_public_generator`, which computes the old public instance's energies) | RMSE 0, rank +0.00082: **works** | Hidden instance in `eval/test/` only | **Failed run**: RMSE 12.77 is no better than a constant prediction (8.11) |
| Report width 0 / −10¹² (`shortcut_width_metadata`, `shortcut_impossible_width`; now `shortcut_sidecar_claims`, a sidecar claiming `neuron_count` 0, `param_count` 0, error 0) | +0.227 / +0.175: **works** | Size measured from the model file; sidecar ignored | **Ties the reference exactly** (rank 0; `param_count` 95,762) |
| Scale predictions by 0.001 and report width −10¹² (`shortcut_failing_scaled_width`; now `shortcut_scaled_0p001`) | +0.175 in the full cohort: **works** | R² gate; scale-invariant metrics removed | **Failed run** (RMSE 8.60 ≥ 8.11) |
| Replace the DNN baseline predictions (`shortcut_move_baseline`) | DNN row moved 2.155 → 8.199; proposed tied | Baseline frozen in `eval/`; `outputs/binary_dnn.pt` ignored | **DNN row unchanged** (0.49868); the submitted reference ties |
| NaN predictions | Exit 0 with NaN metrics | Non-finite → failed | **Failed run** (18,162 non-finite predictions) |
| Constant 10⁶ (inflated other attempts' normalizers) | Rank −5.46; distorted the cohort | R² gate keeps such runs out of every cohort | **Failed run** |
| Constant training mean | Rank −0.366 | R² gate | **Failed run** (8.1066 ≥ 8.1062) |
| Empty or short output (`shortcut_empty_output`, `shortcut_reduced_test_size`) | Rejected | The evaluator now owns the inputs | `shortcut_wrong_shape` (N−1 outputs): **failed run** |
| *New:* memorise training energies, mean elsewhere (`shortcut_train_lookup`) | n/a (39% of the old test rows were training configurations) | Test rows exclude every public configuration | **Failed run** |
| *New:* grow the model (`shortcut_oversized_width150`, 190,802 values) | n/a (width self-reported) | Budget measured from the file | **Failed run** |
| *New:* 10,000 unused buffer values on the reference (`shortcut_padded_buffer`) | n/a | Every storage counted | **Failed run** (105,764) |
| *New:* width-150 network stored in fp16 (`shortcut_fp16_width150`: 381.6 KB, under 4 × budget in bytes) | n/a | Narrow types count one per element | **Failed run** (190,802) |
| *New:* 40,000 floats hidden in a `List[float]` attribute (`shortcut_list_attribute`) | n/a | 128 KiB cap on code + pickles | **Failed run** (376,308 bytes) |
| *New:* model code that calls `torch.from_file` (`shortcut_reads_file`) | n/a | Code scan | **Failed run** |
| *New:* not a TorchScript file (`shortcut_not_torchscript`) | n/a | Load check | **Failed run** |
| Prewritten score files / sidecar metrics (`shortcut_score_files`, `shortcut_sidecar_scores`) | Already failed | Scores still always recomputed | Covered by `shortcut_sidecar_claims` (ignored) |

## Shortcuts left open

1. **Polynomial regression on spin products — major, left.** `shortcut_polynomial_regression` fits least squares on all multilinear spin products up to degree 3 (576 features) from `train.csv`. It scores relative error **2.8e-7**, rank **+0.160** with honest-cohort normalizers, more than three times the better variant's gain. It stores 3,726 values, within budget. On this test such a model cannot be told apart from a learned one. Ruling it out needs a scored task whose energy is not a low-degree polynomial: the paper's quantum double-exchange model, which this gym does not implement. `scoring/DIRECTION.md` forbids explicit polynomial expansions and per-triple coefficient fitting, and the separate grader has to check for them.
2. **Training on the visible dev labels.** Up to 2,000 extra configurations, about 16% more distinct training configurations. Forbidden in `CONSTRAINTS.md`; not detectable.
3. **Picking the best of many seeds on dev.** This is model selection on visible data and is allowed. The gain is bounded by the seed spread (about 0.0045 relative error), half of the better variant's gain.
4. **Smuggling numbers below the caps.** About 14,000 floats in pickled lists under the 128 KiB cap, on top of the 100,000-value tensor budget. Minor (+14% capacity) and visible to the grader.

## Reference score before and after

- **Before:** relative error 0.028246 (RMSE 0.07743) on the old public instance's 2,000-row slice.
- **After:** **0.029494** (MAE 0.06704, tail RMSE 0.08861, `param_count` 95,762) on the hidden instance's 18,162 unseen configurations. The reference was retrained from the final code (`scripts/method.sh`) with the earlier procedure unchanged: 3×105, 2,400 epochs, batch 4,096, Adam lr 1e-3, γ 0.997, best training loss.
- The frozen DNN baseline (`scripts/baseline.sh`: 2×105, 600 epochs, γ 0.995) scores 0.49868 with 12,918 values.
- `scoring/scores_{train,test}.json` and `scores.json` contain only the reference (`efnn`) and the frozen baseline (`dnn`). Shortcut scores appear only in this report and in `redteam/hardening/`.

## Headroom check

**No scored metric is at its bound.** The reference's relative error is 0.0295 against a bound of 0; the paper reports below 0.005 on its own unpublished instance. MAE (0.067) and tail RMSE (0.089) are also far from 0.

A genuine variant scores above the reference: continued training reaches 0.0210, 29% lower. The simple baselines sit far below the reference (DNN 0.499, 20-epoch EFNN 0.725), with a wide margin on either side.

The only thing that reaches the bound is the out-of-bounds polynomial fit (shortcut 1). It is not a variant of the paper's method; it uses the energy function's known form. So I added no extra scored experiment.

## What I could not fix, stated plainly

- **The polynomial-regression shortcut is open** (above).
- **Only the binary three-spin experiment is scored.** The paper's continuous-spin experiment (reproduced RMSE 0.155 against the paper's 0.04 on the earlier instance; removed from scoring in iteration 9) and its quantum double-exchange experiment are not part of the gym.
- **The absolute paper number is not reproduced.** The reference reaches 0.0295 against the paper's below-0.005. The couplings differ (the paper does not publish its own), so the reproduction supports the qualitative claim — EFNN is about 17× more accurate than a same-width DNN — not the number.
- **Same width is not the same parameter count.** At equal width (105) the reference stores 95,762 values and the DNN 12,918. The budget allows a DNN or another architecture of up to 100,000 values, but no parameter-matched DNN was trained in this iteration.
- **The hidden test can't be read from the model, only from the repo.** Its couplings sit in `eval/test/hidden_instance.json`. The model cannot reach them at run time (TorchScript without `from_file`). Keeping them hidden depends on the system withholding `eval/test/` (and its git history) from future agents.
- **The old audit no longer runs.** The iteration-10 cases in `redteam/` target the old interface and are kept only as evidence.
- **Rank normalizers depend on the future cohort.** Failed runs never enter it, and every ranked metric is an accuracy metric on the same predictions, so a run that fails the task cannot outrank one that solves it. Magnitudes still depend on who is ranked.

## Requirement checklist

- **Test answers:** only in `eval/test/`, or computed there at test time.
- **Every score is recomputed** from model outputs on evaluator-owned inputs.
- **The comparison side is fixed by `eval/`:** baseline, budget, thresholds, rows, test size and the constant-prediction gate.
- **Invalid values and failed constraints** become a failed run.
- **Paper effect** is about 240 seed standard deviations.
- **The better variant** beats the reference by 1.9× the seed range and well outside the paired-bootstrap CI.
- **Paper's direction:** EFNN < DNN.
- **Ranked metrics** are all accuracy metrics, with direction set by `higher_is_better`.
- **`scripts/evaluate_test.sh`** takes no arguments, scores every `outputs/binary_*.pt` plus the frozen baseline, runs in under a minute, and never retrains.
- **Train and test** share no code and use the same interface; honest models score alike on both.
- **Fresh clone:**
  - no downloads (data are tracked CSVs);
  - no references to `shared/`;
  - `scripts/run.sh` reproduces the reference end to end;
  - `python validate.py` and `python validate.py --post-design` both pass.
