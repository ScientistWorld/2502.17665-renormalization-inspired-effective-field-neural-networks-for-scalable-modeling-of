# Verification Report

## Initial Assessment
The handoff claimed `core_claim_plus`, but the workspace itself had already been downgraded to `core_claim`. The implemented method is the paper's EFNN recurrence for the 15-site three-spin tasks, and the official evaluator reads prediction artifacts without importing `method/`. The official gym score, however, only covers the binary three-spin experiment; continuous-spin artifacts exist but are documented as diagnostic because their RMSE remains far above the paper's reported `0.04`.

## Issues Found
- **Critical:** `core_claim_plus` is not defensible for the official gym because `scoring/scores.json` contains only `three_spin_binary`. The continuous run supports exploratory comparison against DNN, but it is not accepted as an official scored experiment.
- **Minor:** `scripts/reproduce.sh` ended by writing only the binary EFNN result to `scoring/scores.json`, which would drop the binary DNN baseline from the official score file after a fresh reproduction run.
- **Cosmetic:** `validate.py` reports heuristic warnings for train/test slice evaluators because they recompute evaluator-owned labels. The scripts still score submitted prediction artifacts, verify `x_test`, and do not import `method/`, so this is not a blocker.

## Fixes Applied
- Kept `MILESTONES.md` and `action.yaml` at `core_claim` to match the officially scored experiment set.
- Updated `scripts/reproduce.sh` so the final `scoring/scores.json` regeneration includes both binary DNN and binary EFNN artifacts.
- Rechecked method-agnostic files, scoring coefficients, train/test split files, download portability, and import separation.

## Final Judgment
The workspace is ready as a partial research gym. From-scratch agents can use the problem, data, constraints, and evaluator without seeing the paper method, and improve-mode agents have a validated EFNN reproduction to beat. The main limitation is scope: official scoring covers the binary three-spin task only.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: official scoring covers binary three-spin only; continuous and quantum double-exchange results are not accepted scored experiments.
