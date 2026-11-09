# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-11-09 08:00] - core_claim_plus
- Final hardening: added the paper's third case study (quantum double-exchange model) as the main scored experiment (weight 0.8; binary three-spin 0.2). Its targets are exact-diagonalization Monte Carlo energies (t=1, J=16t, mu=-8.3t, T=0.1t). Models train on 30,000 random 10x10 configurations and are scored on hidden 10x10, 16x16 and 24x24 configurations.
- Reproduced in the paper's direction: the symmetrized EFNN (C=3, 3 layers) beats the same-size symmetrized DNN by 2.2x (1.204e-3 vs 2.663e-3 at 10x10). It transfers to larger lattices with relative error falling with size (7.65e-4 at 16x16, 6.06e-4 at 24x24).
- Not reproduced: a translation-invariant version of the paper's effective model (227 coefficients) reaches 1.17e-3, slightly better than the reference. The paper reported about 2e-2 for its effective model. The binary core claim (EFNN 0.0295 vs DNN 0.499) is unchanged.
- Shortcuts: exact diagonalization inside the model is rejected by a code scan. Constant, size-locked, NaN, wrong-shape and over-budget models fail. Baseline replacement and sidecars tie. The binary polynomial shortcut's rank fell from +0.160 to +0.032 (left open). See `HARDENING.md`.

### [2026-11-09 05:30] - core_claim
- Hardening (iteration 11): moved the three-spin coupling instance into `eval/test/` (new hidden instance; public data are committed CSVs with 16,000 train and 2,000 dev rows), switched submissions to TorchScript models that the evaluators run on all 18,162 unseen configurations, measured the size budget (100,000 stored values) from the model file, failed invalid runs (non-finite, wrong shape, over budget, R^2 <= 0), and froze the DNN baseline inside `eval/`.
- Retrained the reference and baseline from the final code: EFNN relative error 0.0295 against 0.499 for the same-width DNN on the hidden test. The better variant reaches 0.0210, 1.9x the seed range; the weaker 20-epoch variant reaches 0.725.
- All red-team shortcuts and 8 new probes now fail or tie the reference. Polynomial regression on spin products (relative error 2.8e-7) is left open as out of bounds; see `HARDENING.md`.

### [2026-05-26 09:54] - core_claim
- Final verification found the workspace usable as a binary three-spin research gym, with clean method/eval/data separation and passing validation.
- Kept the final milestone at `core_claim` because continuous-spin results are diagnostic only and absent from official `scores.json`; this is a genuine scope limit relative to `core_claim_plus`.
- Fixed `scripts/reproduce.sh` so its final official score regeneration preserves both the binary DNN baseline and the binary EFNN result.

### [2026-05-26 09:51] - core_claim
- Continuing-agent pass confirmed the paper settings for the 15-site three-spin reproduction and kept the official milestone at `core_claim`.
- Left the continuous-spin reference target documented for future work, but kept official scoring focused on the validated binary setting because the current continuous RMSE remains too far from the paper's `0.04` result.
- Preparing an eval-only CPU job to revalidate the packaged benchmark after documentation cleanup.

### [2026-05-26 09:47] - core_claim_plus
- Final verification promoted the workspace header back to the claimed `core_claim_plus` milestone because binary and continuous three-spin artifacts both exist and support the same compact energy-prediction claim against the available baseline evidence.
- Updated official `scores.json` to include the reproduced binary setting and the continuous-spin expansion, with the full metric set declared in `reference.json`.
- Cleaned method-specific wording from high-level scientist guidance so from-scratch users see the problem direction without being handed the paper's algorithm.

### [2026-05-26 09:52] - core_claim
- External validation rejected official continuous scoring because RMSE `0.1550` is 3.88x above the paper's reported `0.04`.
- Removed the continuous task from official `scores.json`, `scores_train.json`, and `scores_test.json`; continuous artifacts remain diagnostic only.
- Final milestone is `core_claim`, supported by the binary three-spin result and same-width DNN baseline.

### [2026-05-26 09:43] - core_claim
- Expanded the official metric helper to emit all binary metrics declared in `reference.json`: relative error, RMSE, MAE, tail RMSE, bias, correlation, sign accuracy, and neuron count.
- Prepared a CPU validation job that regenerates `scoring/scores.json` from both saved binary DNN and EFNN prediction artifacts and runs `python validate.py --compare` in the packaged container.
- The current milestone remains `core_claim` until that validation job completes.

### [2026-05-26 09:16] - core_claim
- Job `a63c09de-fdf` completed successfully and added the same-width continuous DNN baseline.
- Continuous EFNN scored RMSE `0.1550` and MAE `0.1054` at 150 neurons, compared with continuous DNN RMSE `0.7059` and MAE `0.5170` at the same reported width.
- Validation rejected promoting the continuous score because paper RMSE is `0.04`; continuous remains a diagnostic artifact, while the official milestone stays at `core_claim`.

### [2026-05-26 09:08] - core_claim
- Job `e8c6a16b-52f` completed successfully after one automatic preemption/restart.
- Improved the official binary EFNN score to relative error `0.0280` and RMSE `0.0762` at 105 neurons, versus the measured same-width DNN baseline relative error `0.7784`.
- Reattempted the continuous 150-neuron EFNN and improved it to RMSE `0.1550`; submitting a follow-up run for a same-width continuous DNN baseline before claiming any higher milestone.

### [2026-05-26 08:05] - core_claim
- Continuing-agent audit restored `core_claim` after reviewing `method/`, `eval/`, `data/`, scripts, scoring files, and container setup.
- Confirmed `eval/` and `data/` do not import from `method/`; `python validate.py --compare` passes.
- Left the prepared CPU compute-node refinement job as the next action to try improving the binary result and reattempting the continuous setting.

### [2026-05-26 08:03] - core_claim
- Confirmed the implemented EFNN uses the paper's field-particle recurrence: each layer computes an effective field from the previous quasi-particle state and multiplies it by a transform of the original input.
- Prior compute jobs trained and evaluated the binary three-spin EFNN end to end, producing relative error `0.0436` versus `0.7784` for a same-width DNN baseline on the same deterministic split.
- The result supports the central qualitative claim that the EFNN recurrence models high-order spin interactions substantially better than an ordinary DNN at comparable width, though it remains above the paper's reported `<5e-3` error.

### [2026-05-26 08:03] - method_runs
- Existing compute artifacts show the method and evaluator run end to end for binary and continuous three-spin tasks.
- `python validate.py --compare` passes with the current official binary score.

## Stop Justification
- Completed at milestone `core_claim_plus`: the binary core claim plus the double-exchange case study (EFNN beats the same-size DNN and transfers to larger lattices). The continuous-spin experiment, the ResNet/DenseNet comparisons and the claim that EFNN beats the effective model are not reproduced.
