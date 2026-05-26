# Progress

## What Works

- Implemented the paper's EFNN computation for vector spin inputs: each layer computes an effective field from the previous quasi-particle layer, transforms the original input, multiplies them elementwise, and predicts energy from the final quasi-particle layer.
- Implemented deterministic 15-site binary and continuous three-spin data generation.
- Implemented artifact-based evaluation that reads saved predictions from `outputs/*.npz` and writes `scoring/scores.json`.

## Results

- No reproduced numbers yet. A CPU compute job is being submitted to train EFNN on the binary and continuous tasks and score the outputs.

## Remaining

- Inspect the first job's `scoring/scores.json`.
- If the method runs but accuracy is weak, increase epochs or tune learning rate before claiming `core_claim`.
- Add more baselines or quantum double-exchange coverage if budget remains after the core claim is demonstrated.

## Issues

- The manuscript does not specify the exact `J_ijk` values, random seed, batch size, epoch count, or learning-rate schedule for the three-spin experiments. The reproduction fixes a deterministic nontrivial `J_ijk` instance and uses Adam with exponential learning-rate decay.

## Deviations from Paper

- The paper states `J_ijk` are constants but does not list them in the manuscript. This reproduction uses a fixed seeded set of Gaussian-plus-sign couplings in `data/three_spin.py`.
- The first submitted job uses 80 epochs as a smoke-scale run rather than the unknown full paper training schedule. The architecture and data procedure remain the paper's three-spin EFNN setup.
