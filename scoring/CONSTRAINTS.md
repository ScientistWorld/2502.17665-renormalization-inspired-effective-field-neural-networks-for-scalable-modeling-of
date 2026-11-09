# Scientist Constraints

## Data
Train only on the public files in `data/three_spin_binary/`: `train.csv`
(16,000 samples, the paper's training-set size) and, for checking, `dev.csv`
(2,000 configurations that are not in the training file). The hidden test uses
configurations that appear in neither file; they are labelled at test time.
Do not use any other source of energies.

## Submission
Save each model as a TorchScript module `outputs/binary_<name>.pt` that maps a
float32 tensor of spins with shape (N, 15) and values in {0, 1} to energies of
shape (N,). It must load with `torch.jit.load(path, map_location="cpu")`.
The name `dnn` is reserved for the frozen baseline; a file `outputs/binary_dnn.pt`
is ignored.

## Model Scale
The submitted model may store at most **100,000** numeric values (all tensors
in the file — parameters, buffers and constants; types narrower than 32 bits
count one per element, wider types one per 4 bytes). TorchScript code plus
pickled non-tensor attributes may take at most 128 KiB. Exceeding either fails
the run. Growing the model is resource scaling, not a methodological improvement.

## Evaluation
Scores come only from `scripts/evaluate_train.sh` (visible slice) and the hidden
test, which recompute everything from the model file. Sidecar JSON files and
self-reported numbers are ignored. Runs with non-finite predictions or no better
than a constant predictor are failed.

## Compliance Checklist
- [ ] Trained only on `data/three_spin_binary/train.csv` (dev rows used for checking only, or declared)
- [ ] Submitted `outputs/binary_<name>.pt` as a TorchScript energy model
- [ ] Stored values <= 100,000 (see `param_count` in the score file)
- [ ] Did not hard-code the energy function's form (see DIRECTION.md)
- [ ] Scores were produced by the evaluation scripts
