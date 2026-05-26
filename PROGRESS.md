# Progress

## What Works

- Implemented the paper's EFNN computation for vector spin inputs: each layer computes an effective field from the previous quasi-particle layer, transforms the original input, multiplies them elementwise, and predicts energy from the final quasi-particle layer.
- Implemented deterministic 15-site binary and continuous three-spin data generation.
- Implemented artifact-based evaluation that reads saved predictions from `outputs/*.npz` and writes `scoring/scores.json`.

## Results

- No reproduced numbers yet. Job `6a740d14-621` failed before training because `baseline/train_three_spin.py` could not import the workspace `method` package when launched as a script.
- Fixed job entrypoints to export `/home/user` on `PYTHONPATH` and added package markers for `data/`, `method/`, `eval/`, and `baseline/`.

## Remaining

- Run the smoke-scale job and inspect `scoring/scores.json`.
- If the method runs but accuracy is weak, increase epochs or tune learning rate before claiming `core_claim`.
- Add more baselines or quantum double-exchange coverage if budget remains after the core claim is demonstrated.

## Issues

- The manuscript does not specify the exact `J_ijk` values, random seed, batch size, epoch count, or learning-rate schedule for the three-spin experiments. The reproduction fixes a deterministic nontrivial `J_ijk` instance and uses Adam with exponential learning-rate decay.
- Local import smoke testing on the login node cannot import NumPy because dependencies are installed in the compute container overlay. Static validation and Python syntax compilation pass locally; runtime import will be checked in the submitted container job.

## Deviations from Paper

- The paper states `J_ijk` are constants but does not list them in the manuscript. This reproduction uses a fixed seeded set of Gaussian-plus-sign couplings in `data/three_spin.py`.
- The current submitted job uses 20 epochs as a smoke-scale run rather than the unknown full paper training schedule. The architecture and data procedure remain the paper's three-spin EFNN setup.
