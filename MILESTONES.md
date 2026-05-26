# Reproduction Milestones

**Current: method_runs**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-05-26 05:34] - none
- Read the paper and supplementary markdown.
- Implemented the EFNN recurrence for the 15-site three-spin tasks.
- Added deterministic data generation, scoring metadata, scripts, and a first CPU job for end-to-end validation.

### [2026-05-26 05:41] - none
- Job `6a740d14-621` failed before training due to a missing workspace package import path.
- Added package markers and exported `PYTHONPATH=/home/user` from job scripts.
- Reduced the next job to a 20-epoch smoke run to verify the full train/evaluate path quickly before scaling training.

### [2026-05-26 06:26] - method_runs
- Job `44a6e3a0-171` completed successfully.
- EFNN trained and evaluated end-to-end on both 15-site binary and continuous three-spin tasks.
- Smoke-scale results were still far from the paper's reported errors, so the next run uses a corrected summation-output EFNN and longer GPU training.

### [2026-05-26 06:36] - method_runs
- CUDA PyTorch setup failed due to shared cache disk quota while extracting the wheel.
- Reverted to the working CPU PyTorch setup and configured a longer binary-task core-claim run with larger batches.

## Stop Justification
- Not stopped. A longer CPU core-claim run is being submitted.
