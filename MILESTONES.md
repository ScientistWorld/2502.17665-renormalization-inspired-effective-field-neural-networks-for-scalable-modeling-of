# Reproduction Milestones

**Current: core_claim**

## Progress Log

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
- Not stopped; submitting another compute-node run to try improving the binary score and to reattempt the continuous expansion.
