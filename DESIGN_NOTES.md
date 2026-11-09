> **Superseded (iteration 11).** The design described below (visible coupling generator,
> `.npz` prediction artifacts, sidecar `neuron_count`) was replaced by the hardened gym:
> hidden instance in `eval/test/`, TorchScript model submissions, measured size budget,
> failure gates and an eval-owned baseline. See `HARDENING.md`.

Stage 2 train/test design pass

Checked:
- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. Train calls `eval/train/` and writes `scoring/scores_train.json`; test calls `eval/test/` and writes `scoring/scores_test.json`.
- Checked the slice evaluators for hardcoded method defaults, cross-imports between `eval/train/` and `eval/test/`, and score-source discovery behavior.
- Checked `scoring/reference.json` for paper-agreement metrics and required metric metadata.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs are present.
- Ran static-only validation: `bash -n` on both evaluate scripts, `py_compile` on both slice evaluators, and JSON parsing for `scoring/reference.json`.

Changed:
- Fixed an artifact-trust shortcut in both slice evaluators. They no longer trust `y_true` from a method artifact as the scoring target. They reconstruct the official deterministic 15-site three-spin test rows inside `eval/train/` and `eval/test/`, require the artifact's `x_test` to match those rows, and score only the submitted `y_pred` against evaluator-owned targets.
- Duplicated the small official benchmark generator inside each slice evaluator instead of importing `data.three_spin`. This keeps final scoring anchored in `eval/` and prevents a future method from changing shared data code to move the target.
- Added `pearson_corr` and `sign_accuracy` to both slices for both experiments. These triangulate the regression task with ranking/shape and qualitative sign correctness, making collapsed-output or scale-gaming shortcuts easier to catch.
- Expanded `scoring/reference.json` so every nonzero metric has `higher_is_better`, `coefficient`, and `description`. Existing paper-agreement-free metrics were kept, and the compact-network constraint remains represented by `neuron_count`.

Runtime concerns for the runner iteration:
- The evaluators now require every prediction artifact to include `x_test` in the official row order. The current method writer already emits this, but alternate future methods must preserve that interface.
- The split is still a deterministic half/half row partition of the paper-style test set. It is appropriate for detecting visible-slice overfit, but the synthetic generator is visible in the benchmark, so a method that explicitly solves the known cubic generator may score well unless governance constraints disallow it.
- `neuron_count` is still read from an optional sidecar JSON. If a future artifact omits it, the metric will be missing from that method's score rather than inferred.
