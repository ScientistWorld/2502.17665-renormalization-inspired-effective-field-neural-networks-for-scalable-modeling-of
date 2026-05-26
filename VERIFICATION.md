# Verification Report

## Initial Assessment
The workspace contained a working three-spin reproduction with saved binary and continuous artifacts, but the milestone files still said `core_claim` while the handoff claimed `core_claim_plus`. The method implementation matches the paper's recursive energy-prediction computation for the reproduced three-spin tasks, and evaluation is artifact-based rather than importing from `method/`.

## Issues Found
- **Critical:** External validation rejected the continuous expansion because its RMSE was 3.88x above the paper's reported value. It cannot be an official scored experiment.
- **Minor:** `scoring/scores.json` initially contained the older partial binary-only metric set.
- **Minor:** `scoring/DIRECTION.md` exposed too much of the paper's specific approach for from-scratch scientist mode.
- **Minor:** `reference.json` included inferred paper-side RMSE/MAE values that were not explicitly reported in the manuscript text.
- **Cosmetic:** `validate.py` emits a heuristic warning for train/test evaluators because they regenerate evaluator-owned labels while reading prediction artifacts. This is acceptable: the scores still depend on `outputs/*.npz`, and the scripts verify artifact rows against the fixed benchmark split.

## Fixes Applied
- Regenerated `scoring/scores.json` from saved prediction artifacts so it includes binary DNN and binary EFNN results with all declared binary metrics.
- Refreshed `scoring/scores_train.json` and `scoring/scores_test.json` so train/test slices cover the same official experiment and method keys.
- Removed continuous results from official scoring and left them documented as diagnostics.
- Rewrote `scoring/DIRECTION.md`, `scoring/TARGETS.md`, and `scoring/EXPERIMENTS.md` to keep high-level guidance method-safe while preserving the benchmark scope.
- Removed inferred reference RMSE/MAE values, leaving only manuscript-reported headline values and scale constraints.
- Updated `MILESTONES.md` to match the final milestone.

## Final Judgment
The workspace is ready as a partial research gym. From-scratch agents can use the method-agnostic problem, data, and evaluator files without seeing the implementation, and improve-mode agents can compare against the included reproduction. The main limitation is scope: official scoring covers only the validated binary three-spin task.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: official scoring covers binary three-spin only; continuous and quantum double-exchange results are not accepted scored experiments.
