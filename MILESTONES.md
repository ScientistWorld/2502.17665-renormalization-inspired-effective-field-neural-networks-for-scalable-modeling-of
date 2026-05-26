# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-26 07:20] - core_claim
- Implemented the paper's EFNN field-particle recurrence for the 15-site three-spin Hamiltonian.
- Verified artifact-based evaluation on the binary task, with EFNN substantially outperforming the same-width DNN baseline on the fixed split.
- Preparing an additional continuous-spin EFNN run to broaden coverage beyond the minimum core claim.

### [2026-05-26 07:31] - method_runs
- Continuing-agent audit reset the active milestone to `method_runs`.
- The EFNN implementation is a real implementation of the paper's field-particle recurrence, but current official scores only include the binary experiment while `TARGETS.md` lists continuous as well.
- Updated the run script to detect stale artifacts and retrain the binary 3-layer and continuous 3-layer EFNN settings before rescoring.

### [2026-05-26 07:56] - method_runs
- Job `a0ca193f-b8f` produced improved 3-layer binary EFNN and full continuous EFNN artifacts.
- Validation failed only because the evaluator exported an extra `param_count` key not declared in `reference.json`; fixed the evaluator and submitting a validation-only rerun.
