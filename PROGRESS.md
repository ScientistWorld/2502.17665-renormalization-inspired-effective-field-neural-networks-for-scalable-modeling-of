# Progress

## What Works

- Implemented the paper's EFNN computation for vector spin inputs: each layer computes an effective field from the previous quasi-particle layer, transforms the original input, multiplies them elementwise, and predicts energy from the final quasi-particle layer.
- Implemented deterministic 15-site binary and continuous three-spin data generation.
- Implemented artifact-based evaluation that reads saved predictions from `outputs/*.npz` and writes `scoring/scores.json`.
- Job `44a6e3a0-171` verified the full train/evaluate loop for EFNN on both reproduced three-spin settings.

## Results

- 20-epoch smoke run, binary EFNN: relative error `0.3908`, RMSE `1.0642` versus paper target relative error `< 0.005`.
- 20-epoch smoke run, continuous EFNN: RMSE `0.3781`, MAE `0.2812` versus paper target RMSE `0.04`.
- The smoke run reaches `method_runs` but not `core_claim`; errors remain too high after limited CPU training.

## Remaining

- Run the corrected summation-output EFNN longer on CPU after CUDA PyTorch installation exceeded the shared cache quota.
- If the method still underperforms, tune learning rate, batch size, and seed while preserving the EFNN recurrence.
- Add more baselines or quantum double-exchange coverage if budget remains after the core claim is demonstrated.

## Issues

- The manuscript does not specify the exact `J_ijk` values, random seed, batch size, epoch count, or learning-rate schedule for the three-spin experiments. The reproduction fixes a deterministic nontrivial `J_ijk` instance and uses Adam with exponential learning-rate decay.
- The first EFNN implementation used a learned output projection. The paper specifies `q` as summation over the final quasi-particle layer, so `method/models.py` was corrected before the core-claim attempt.
- A CUDA PyTorch setup attempt failed while extracting the wheel into the shared `uv_cache` due to disk quota. The environment was returned to the previously working CPU PyTorch setup.

## Deviations from Paper

- The paper states `J_ijk` are constants but does not list them in the manuscript. This reproduction uses a fixed seeded set of Gaussian-plus-sign couplings in `data/three_spin.py`.
- The completed smoke job used 20 epochs as a fast validation run rather than the unknown full paper training schedule. The next run increases binary-task training to 600 epochs with larger CPU batches.
