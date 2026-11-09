> **Pre-hardening audit.** The cases in this folder (except `hardening/`) were run in
> iteration 10 against the earlier public coupling instance and the `.npz`
> prediction interface. Iteration 11 replaced both (hidden instance, TorchScript
> model submissions); these scripts no longer run against the current gym and are
> kept as evidence. The current audit is `hardening/` (see `HARDENING.md`).

# Benchmark audit runs

`bash redteam/run_all.sh` runs the audit on a compute node. Each experiment has
its own `run.sh`, installs its prediction artifacts, calls the visible train
evaluator, and calls `bash scripts/evaluate_test.sh` **without arguments** in a
separate process. Neither evaluator nor the Python environment is modified.
The wrapper restores outputs, scores, and the data source after each case.

`train_variant.py` retains the reference EFNN training procedure. Initialization
seeds vary independently of the fixed dataset seed (the original trainer's
`--seed` changes both, which cannot be used for an official-split seed study).
The stronger variant continues the saved reference checkpoint with a renewed
learning rate. All model selection uses training loss.
The bundle reuses completed cases after preemption; unfinished CPU training saves
model, optimizer, schedule, best training checkpoint, and RNG states every 50
epochs. Individual case `run.sh` scripts rerun preparation and both evaluations
by default; only `REDTEAM_REUSE_COMPLETED=1` enables case reuse. To retrain a
finished model from epoch one, also remove its ignored `prediction.progress.pt`.

Artifact-reuse probes require the gym's saved reference and baseline predictions
and the saved reference checkpoint. On an environment without those artifacts,
first regenerate them with the gym's reproduction script. The actual audit used
the finished gym's artifacts, as permitted by the audit protocol. The prediction
hash and unmodified evaluator hashes are recorded with the results.

`scores_test.json`, `status.json`, and `summary.json` in each case are actual
test results, including evaluator rejection where applicable. Generated models,
prediction arrays, logs, and temporary snapshots are ignored. The checked-in
code reconstructs them. `rank_summary.json` records a finite audit-cohort
normalization, not an assertion about unknown future attempts' normalizers.
All names starting `shortcut_` are shortcut probes, including tied probes.
