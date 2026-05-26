# Reproduction Milestones

**Current: core_claim**

## Progress Log

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
- Not stopped; submitting a follow-up compute-node run to add the continuous DNN comparison and promote the continuous setting only if the measured baseline gap supports it.
