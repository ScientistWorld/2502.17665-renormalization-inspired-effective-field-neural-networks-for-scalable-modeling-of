# Progress

## What Works

- Implemented the paper's EFNN computation for vector spin inputs: each layer computes an effective field from the previous quasi-particle layer, transforms the original input, multiplies them elementwise, and predicts energy from the final quasi-particle layer.
- Implemented deterministic 15-site binary and continuous three-spin data generation.
- Implemented artifact-based evaluation that reads saved predictions from `outputs/*.npz` and writes `scoring/scores.json`.
- Updated the shared metric helper so official scoring can report all binary metrics advertised in `reference.json`, not only RMSE and relative error.
- Job `44a6e3a0-171` verified the full train/evaluate loop for EFNN on both reproduced three-spin settings.
- Updated the job script to score the validated binary EFNN core-claim artifact in the official `scoring/scores.json` schema.
- Confirmed the current implementation matches the supplement's continued-function EFNN structure: linear-tanh-linear `f`/`g` transforms, recursive effective fields, multiplicative injection of `S_0`, and final summation.
- Continuing-agent audit on 2026-05-26 restored `core_claim`: the implementation is not a surrogate, evaluation is artifact-based, and structural validation passes.

## Results

- 20-epoch smoke run, binary EFNN: relative error `0.3908`, RMSE `1.0642` versus paper target relative error `< 0.005`.
- 20-epoch smoke run, continuous EFNN: RMSE `0.3781`, MAE `0.2812` versus paper target RMSE `0.04`.
- Corrected 600-epoch binary EFNN: relative error `0.0579`, RMSE `0.1576`, neuron count `105`.
- 1200-epoch 3-layer binary EFNN: relative error `0.0436`, RMSE `0.1186`, neuron count `105`.
- 1200-epoch 3-layer continuous EFNN: RMSE `0.2379`, MAE `0.1644`, neuron count `150`.
- Same-width 600-epoch DNN baseline: relative error `0.7784`, RMSE `2.1197`, neuron count `105`.
- 2400-epoch 3-layer binary EFNN from job `e8c6a16b-52f`: relative error `0.0280`, RMSE `0.0762`, neuron count `105`.
- 1800-epoch 3-layer continuous EFNN from job `e8c6a16b-52f`: RMSE `0.1550`, MAE `0.1054`, neuron count `150`.
- The latest EFNN is about 27.8x lower relative error than the same-width DNN baseline on the binary high-order interaction task. This supports the paper's core claim, though the absolute EFNN error remains above the paper's reported `<0.005` threshold.
- Official `scoring/scores.json` contains the validated binary EFNN result. The measured same-width DNN comparisons, including the continuous DNN baseline, are stored in `scoring/baseline_scores.json`.
- Final validation keeps official `scoring/scores.json` focused on the binary DNN baseline and binary EFNN result. The continuous EFNN artifact is diagnostic because its RMSE remains too far from the paper target for accepted scoring.
- Eval-only confirmation job `f0a5b497-d2b` completed successfully and passed `python validate.py --compare` inside the compute container.
- Earlier validation job `3028d0f4-657` rejected the old continuous score because RMSE `0.2379` was too far from the paper's `0.04`; the later 1800-epoch continuous artifact improved this to `0.1550`, but external validation still rejects it as too far from the paper value.
- Same-width 600-epoch continuous DNN baseline from job `a63c09de-fdf`: RMSE `0.7059`, MAE `0.5170`, neuron count `150`.
- Current milestone is `core_claim`: EFNN is much better than the same-width DNN baseline on the binary high-order interaction task, while the continuous run is retained as a diagnostic expansion rather than an official milestone.

## Remaining

- The reusable environment covers the validated binary core-claim setting.
- Higher milestones would require pushing continuous EFNN RMSE much closer to the paper's `0.04`, adding exact ResNet/DenseNet comparisons if numeric paper values can be extracted, or implementing the quantum double-exchange setting.
- Additional useful expansions: tune learning rate, batch size, and seed while preserving the EFNN recurrence; add ResNet/DenseNet baselines; implement the larger quantum double-exchange setting.

## Issues

- Job `a0ca193f-b8f` trained the 3-layer binary and continuous EFNN artifacts successfully, then failed validation because `eval/score_three_spin.py` wrote `param_count` into `scores.json` even though `reference.json` only defines `neuron_count` as the paper-reported scale constraint. The evaluator now leaves `param_count` in sidecar metadata only.
- Attempt 1 external validation rejected official continuous scoring because the reproduced RMSE `0.2379` was 5.95x above the paper's `0.04`. The later refinement improved continuous RMSE to `0.1550`, but validation still rejected the gap, so the task remains diagnostic.
- The manuscript does not specify the exact `J_ijk` values, random seed, batch size, epoch count, or learning-rate schedule for the three-spin experiments. The reproduction fixes a deterministic nontrivial `J_ijk` instance and uses Adam with exponential learning-rate decay.
- The first EFNN implementation used a learned output projection. The paper specifies `q` as summation over the final quasi-particle layer, so `method/models.py` was corrected before the core-claim attempt.
- A CUDA PyTorch setup attempt failed while extracting the wheel into the shared `uv_cache` due to disk quota. The environment was returned to the previously working CPU PyTorch setup.
- The DNN baseline job exited `-9` because Hive/Ray could not update metadata under disk quota, but the prediction artifacts and `scoring/scores.json` were written before termination.
- The clean confirmation job after that failure exited `0`, so the final score artifacts are verified.
- Job `e8c6a16b-52f` was preempted once during the first binary refinement attempt, restarted automatically, and then completed with exit code `0`.
- Job `a63c09de-fdf` completed with exit code `0`; it provided the continuous DNN comparison, but subsequent validation rejected promoting continuous EFNN to official scoring.

## Deviations from Paper

- The paper states `J_ijk` are constants but does not list them in the manuscript. This reproduction uses a fixed seeded set of Gaussian-plus-sign couplings in `data/three_spin.py`.
- The completed smoke job used 20 epochs as a fast validation run rather than the unknown full paper training schedule.
- The current binary EFNN result uses 2400 epochs with larger CPU batches and `LR_GAMMA=0.997`; this changes optimization length and schedule, not the EFNN algorithm.
