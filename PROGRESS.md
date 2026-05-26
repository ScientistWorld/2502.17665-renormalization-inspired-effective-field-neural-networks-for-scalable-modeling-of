# Progress

## What Works

- Implemented the paper's EFNN computation for vector spin inputs: each layer computes an effective field from the previous quasi-particle layer, transforms the original input, multiplies them elementwise, and predicts energy from the final quasi-particle layer.
- Implemented deterministic 15-site binary and continuous three-spin data generation.
- Implemented artifact-based evaluation that reads saved predictions from `outputs/*.npz` and writes `scoring/scores.json`.
- Job `44a6e3a0-171` verified the full train/evaluate loop for EFNN on both reproduced three-spin settings.

## Results

- 20-epoch smoke run, binary EFNN: relative error `0.3908`, RMSE `1.0642` versus paper target relative error `< 0.005`.
- 20-epoch smoke run, continuous EFNN: RMSE `0.3781`, MAE `0.2812` versus paper target RMSE `0.04`.
- Corrected 600-epoch binary EFNN: relative error `0.0579`, RMSE `0.1576`, neuron count `105`.
- Same-width 600-epoch DNN baseline: relative error `0.7784`, RMSE `2.1197`, neuron count `105`.
- EFNN is about 13.4x lower relative error than the DNN baseline on the binary high-order interaction task. This supports the paper's core claim, though the absolute EFNN error remains above the paper's reported `<0.005` threshold.
- Official `scoring/scores.json` contains only EFNN because `reference.json` can only include paper-reported numeric methods. The measured DNN result is stored in `scoring/baseline_scores.json`.
- Eval-only confirmation job `f0a5b497-d2b` completed successfully and passed `python validate.py --compare` inside the compute container.

## Remaining

- Higher milestones would require pushing absolute EFNN error closer to the paper and rerunning the corrected continuous-spin experiment.
- Additional useful expansions: tune learning rate, batch size, and seed while preserving the EFNN recurrence; add ResNet/DenseNet baselines; implement the larger quantum double-exchange setting.

## Issues

- The manuscript does not specify the exact `J_ijk` values, random seed, batch size, epoch count, or learning-rate schedule for the three-spin experiments. The reproduction fixes a deterministic nontrivial `J_ijk` instance and uses Adam with exponential learning-rate decay.
- The first EFNN implementation used a learned output projection. The paper specifies `q` as summation over the final quasi-particle layer, so `method/models.py` was corrected before the core-claim attempt.
- A CUDA PyTorch setup attempt failed while extracting the wheel into the shared `uv_cache` due to disk quota. The environment was returned to the previously working CPU PyTorch setup.
- The DNN baseline job exited `-9` because Hive/Ray could not update metadata under disk quota, but the prediction artifacts and `scoring/scores.json` were written before termination.
- The clean confirmation job after that failure exited `0`, so the final score artifacts are verified.

## Deviations from Paper

- The paper states `J_ijk` are constants but does not list them in the manuscript. This reproduction uses a fixed seeded set of Gaussian-plus-sign couplings in `data/three_spin.py`.
- The completed smoke job used 20 epochs as a fast validation run rather than the unknown full paper training schedule. The current binary EFNN result uses 600 epochs with larger CPU batches.
