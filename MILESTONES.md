# Reproduction Milestones

**Current: none**

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

## Stop Justification
- Not stopped. A fixed smoke-scale reproduction job is being submitted.
