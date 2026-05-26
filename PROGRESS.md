# Progress

## What Works

- Implemented the paper's EFNN computation for vector spin inputs: each layer computes an effective field from the previous quasi-particle layer, transforms the original input, multiplies them elementwise, and predicts energy from the final quasi-particle layer.
- Implemented deterministic 15-site binary and continuous three-spin data generation.
- Implemented artifact-based evaluation that reads saved predictions from `outputs/*.npz` and writes `scoring/scores.json`.
- Job `44a6e3a0-171` verified the full train/evaluate loop for EFNN on both reproduced three-spin settings.
- Updated the job script to score the validated binary EFNN core-claim artifact in the official `scoring/scores.json` schema.
- Confirmed the current implementation matches the supplement's continued-function EFNN structure: linear-tanh-linear `f`/`g` transforms, recursive effective fields, multiplicative injection of `S_0`, and final summation.

## Results

- 20-epoch smoke run, binary EFNN: relative error `0.3908`, RMSE `1.0642` versus paper target relative error `< 0.005`.
- 20-epoch smoke run, continuous EFNN: RMSE `0.3781`, MAE `0.2812` versus paper target RMSE `0.04`.
- Corrected 600-epoch binary EFNN: relative error `0.0579`, RMSE `0.1576`, neuron count `105`.
- 1200-epoch 3-layer binary EFNN: relative error `0.0436`, RMSE `0.1186`, neuron count `105`.
- 1200-epoch 3-layer continuous EFNN: RMSE `0.2379`, MAE `0.1644`, neuron count `150`.
- Same-width 600-epoch DNN baseline: relative error `0.7784`, RMSE `2.1197`, neuron count `105`.
- EFNN is about 17.9x lower relative error than the DNN baseline on the binary high-order interaction task. This supports the paper's core claim, though the absolute EFNN error remains above the paper's reported `<0.005` threshold.
- Official `scoring/scores.json` contains only EFNN because `reference.json` can only include paper-reported numeric methods. The measured DNN result is stored in `scoring/baseline_scores.json`.
- Eval-only confirmation job `f0a5b497-d2b` completed successfully and passed `python validate.py --compare` inside the compute container.
- Validation job `3028d0f4-657` completed successfully with both EFNN artifacts, but external validation rejected the continuous score because RMSE `0.2379` is too far from the paper's `0.04`.
- Official `scoring/scores.json` now includes only the validated binary core-claim experiment; the continuous artifact remains documented as an attempted expansion.
- Current milestone is `core_claim`: EFNN is much better than the DNN baseline on the binary high-order interaction task at the same width, but the absolute error is still worse than the paper's figure.

## Remaining

- The next submitted job reruns the binary EFNN longer with the same algorithm and width, and reattempts the continuous 150-neuron EFNN as a non-official expansion artifact.
- Higher milestones would require pushing binary EFNN error closer to the paper, restoring continuous-spin scoring after its RMSE is within validator tolerance, and adding ResNet/DenseNet baselines or the quantum double-exchange setting.
- Additional useful expansions: tune learning rate, batch size, and seed while preserving the EFNN recurrence; add ResNet/DenseNet baselines; implement the larger quantum double-exchange setting.

## Issues

- Job `a0ca193f-b8f` trained the 3-layer binary and continuous EFNN artifacts successfully, then failed validation because `eval/score_three_spin.py` wrote `param_count` into `scores.json` even though `reference.json` only defines `neuron_count` as the paper-reported scale constraint. The evaluator now leaves `param_count` in sidecar metadata only.
- Attempt 1 external validation rejected official continuous scoring because the reproduced RMSE `0.2379` is 5.95x above the paper's `0.04`. The continuous task was removed from current official targets and scores rather than claiming an unsupported milestone.
- The manuscript does not specify the exact `J_ijk` values, random seed, batch size, epoch count, or learning-rate schedule for the three-spin experiments. The reproduction fixes a deterministic nontrivial `J_ijk` instance and uses Adam with exponential learning-rate decay.
- The first EFNN implementation used a learned output projection. The paper specifies `q` as summation over the final quasi-particle layer, so `method/models.py` was corrected before the core-claim attempt.
- A CUDA PyTorch setup attempt failed while extracting the wheel into the shared `uv_cache` due to disk quota. The environment was returned to the previously working CPU PyTorch setup.
- The DNN baseline job exited `-9` because Hive/Ray could not update metadata under disk quota, but the prediction artifacts and `scoring/scores.json` were written before termination.
- The clean confirmation job after that failure exited `0`, so the final score artifacts are verified.

## Deviations from Paper

- The paper states `J_ijk` are constants but does not list them in the manuscript. This reproduction uses a fixed seeded set of Gaussian-plus-sign couplings in `data/three_spin.py`.
- The completed smoke job used 20 epochs as a fast validation run rather than the unknown full paper training schedule. The current binary EFNN result uses 600 epochs with larger CPU batches.
- The current job uses slower exponential learning-rate decay (`LR_GAMMA=0.995`) than the initial run to avoid freezing the optimizer too early; this changes optimization, not the EFNN algorithm.
