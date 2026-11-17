# Hardening report (this iteration)

**Bottom line.** On the main experiment (`double_exchange`, weight 0.8), better results score higher and worse ones lower. Seed noise is now well below the effects measured:
- The reference seeds now agree: the L10 range fell from 50% of the mean to 4%.
- The paper's own effective model is now the baseline. The reference beats it 8.9× on 10×10, as the paper claims.
- Every failed run, and every method missing from an experiment, gets the worst value of each metric instead of a gap that cost nothing.
- Hand-written diagonalization that evades the op scan is now stopped by a deterministic compute budget.
- The binary experiment is now a non-regression check, so its polynomial shortcut no longer pays. Genuine binary improvements no longer pay either.

The reference model is bit-identical to the previous one; its scores did not change.

Two things are still open, stated plainly in the last sections:
- explicit translation-invariant interaction expansions (honest, learned, within budget) beat the reference EFNN;
- hand-written diagonalization in pure scalar code is not counted by the meter.

Evidence:
- Round 1, job `306abc6b-f5e`: best of 4 initializations, archived in `redteam/final2/round1_best_of_4/`.
- Round 2, job `a81fc632-0be`: the final trainer.
- Every case was scored by `bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh`, run with **no arguments** in a fresh process on an `outputs/` folder holding only that case's files.
- Every case submits one method to both experiments: its own model where it probes, the reference model in the other experiment.
- The harness is `redteam/final2/` (`run_final3.sh`, `make_cases.py`, `run_cases.py`, `summarize.py`). Per-case scores and logs are in `redteam/final2/cases/`; the summary is in `redteam/final2/summary.{json,md}`.
- `python validate.py` and `python validate.py --post-design` both pass.

The previous iteration's report is in `redteam/final/HARDENING_previous_iteration.md`.

## What changed

| Problem (source) | Fix |
|---|---|
| Reference bimodal across seeds: one seed in three 60% worse (`REVIEW.md`) | Initialization is selected on training data only. Every run first settles on a plateau (standardized training MSE ≈ 0.025, the level the same-size DNN converges to). Runs that leave it while the learning rate is high reach ≈ 0.0045; runs that leave it late stay near 0.012. Candidate initializations now train in turn for the first 20 epochs of the unchanged schedule. The first whose training loss is below 0.02 is kept and trained to the end (up to 24 tried, otherwise the lowest). The DNN comparison uses the same procedure. `method/train_double_exchange.py` |
| The baseline effective model was a stronger translation-invariant expansion, so the paper's EFNN > effective-model claim did not reproduce (`REVIEW.md`) | `baseline/effective_model_de.py` is now the paper's model: Eq. 3 restricted to 5×5 tiles, 1 + 4 × (325 + 52,975) = **213,201 coefficients**, the paper's count. It is fitted by linear least squares on the training split. 30,000 rows are fewer than its 181,801 independent terms (the paper had 1.6e6), so an L2 penalty is chosen on a 10% training holdout. The fit runs in dual form with conjugate gradients. The model is defined and scored on 10×10 only, as in the paper. The old expansion is kept as a variant (`redteam/final2/ti_expansion.py`). |
| A failed run had no metrics, and the rank skips missing values, so failing cost nothing (task rule) | Every failed run gets the **worst value of every metric**: the best constant predictor's error at each size (R² = 0) and the full size budget. A method that submits to one experiment but not the other gets worst values in the missing one. Baseline names are reserved in both experiments. `eval/{train,test}/score_*.py` |
| Hand-written diagonalization evades the op scan; the previous report left it open (task rule: limit what the method can compute per instance) | **Compute meter.** A `TorchDispatchMode` counts every tensor operation of every forward pass, deterministically and not by timer. Matrix products and convolutions count multiply-accumulates; other ops count the larger of their input and output element counts; wide or complex types are weighted by size. The budget is **50,000 units per lattice site**, checked before each matrix product runs. Reference ≈ 6,800 (7× headroom), DNN 2,800, 1,847-term expansion 7,900. Any diagonalization of the 200×200 matrix needs ≥ 1.6×10⁵ even at 10×10, and far more at 24×24. `fork`/`wait` are added to the forbidden code. Eval-owned baselines are exempt. |
| Binary: regression on enumerated cubic spin products beats the reference (+0.032 rank) and any honest binary gain (task rule: a closed form that outranks the reference is a test flaw) | `three_spin_binary` becomes a **non-regression check**. The target is exactly a cubic polynomial, so every metric is floored at the reference's own value on the slice (raw values kept as `raw_*` in `_artifacts`). Its weight stays 0.2: it still penalizes methods that break the classical case. |

Every scored number is still recomputed by the evaluator from raw predictions. Data, the hidden instance, test sizes, coefficients and weights are unchanged. Only metric descriptions in `scoring/reference.json` changed, and the paper's reported numbers are untouched.

## Do better results score higher?

Ranks use the gym rule, w_e · Σ|coef|·dir·(x − x_ref)/std. Normalizers are the std over the honest cohort: reference, 2 seeds, continued training, 3-epoch EFNN, DNN as a submission, and the two expansion variants. Hidden test values:

| Variant (double_exchange) | L10 / L16 / L24 | Rank before | Rank after |
|---|---|---:|---:|
| Reference (EFNN C=3, 3 layers) | 1.204e-3 / 7.65e-4 / 6.06e-4 | 0 | 0 |
| Better: + 150 epochs from lr 5e-4 | 1.166e-3 / 7.52e-4 / 5.78e-4 | +0.096 | **+0.094** |
| Better: translation-invariant expansion, 227 terms (previously the baseline) | 1.166e-3 / 6.86e-4 / 4.82e-4 | +0.286 | **+0.279** |
| Better: translation-invariant expansion, 1,847 terms | 8.76e-4 / 5.52e-4 / 3.76e-4 | +0.906 | **+0.885** |
| Seed 20250218 | 1.216e-3 / 7.70e-4 / 5.51e-4 | +0.053 | +0.051 |
| Seed 20250219 | before 1.929e-3 / 1.292e-3 / 8.65e-4; after 1.257e-3 / 8.81e-4 / 6.22e-4 | −1.725 | **−0.193** |
| Weaker: 3 epochs | 2.701e-3 / 1.718e-3 / 1.284e-3 | −3.622 | −3.543 |
| Weaker: same-size DNN as a submission | 2.667e-3 / 1.649e-3 / 1.199e-3 | −3.392 | −3.330 |

"Before" is the previous iteration's audit, with its cohort. Under the new evaluator, the previous iteration's seed models rescore to the same numbers (`de_prev_*` cases).

- **Paired test noise** (1,000 paired bootstrap resamples on the identical hidden inputs; difference vs reference, 95% CI):
  - continued training: L10 −3.8e-5 [−4.6, −3.1]e-5; L16 −1.3e-5 [−2.7, +0.3]e-5 (not significant); L24 −2.9e-5 [−4.7, −1.0]e-5;
  - 227-term expansion: significant at all sizes (−3.8e-5, −7.9e-5, −1.2e-4);
  - 1,847-term expansion: significant at all sizes (−3.3e-4, −2.1e-4, −2.3e-4; sd ≤ 2.7e-5).
- **Seed noise:**
  - new seeds: L10 range 5.3e-5 (std 2.2e-5); L16 range 1.2e-4; L24 range 7.1e-5; rank range 0.24;
  - previous seeds: L10 range 7.3e-4, rank range 1.78.
- **Gains relative to seed noise:**
  - the 1,847-term gain (+0.885) is 3.6× the seed rank range;
  - the 227-term gain (+0.279) is just above it;
  - continued training (+0.094) is significant on paired test inputs but inside the seed spread. It is a real but small improvement.
- **Paper effect vs seed noise:** EFNN beats the same-size DNN by 1.46e-3 at L10, 28× the seed range (2.2× / 2.2× / 2.0× at the three sizes). Every seed beats the DNN at every size.
- **Train and test agree** (dev vs hidden L10): reference 1.182e-3 vs 1.204e-3; continued 1.145e-3 vs 1.166e-3; 1,847-term 8.93e-4 vs 8.76e-4; DNN 2.671e-3 vs 2.667e-3.

**three_spin_binary**, before (raw, weight 0.2) → after (floored):
- continued training +0.0092 → 0;
- seed 20250218 +0.0014 → 0;
- seed 20250219 −0.0036 → −0.0036;
- 20 epochs −0.764 → −0.767;
- DNN −0.510 → −0.512.

Better binary models now tie the reference by design (see "left open").

## Shortcuts: before and after

All shortcuts below were rerun through the hidden test in round 2. Ranks are combined and use the honest cohort.

| Shortcut | Before | Fix | After |
|---|---|---|---|
| Polynomial regression on binary spin products (`bin_shortcut_polynomial_regression`, rel. error 2.8e-7) | **+0.032**: beats the reference and the honest binary gain | Binary floor | **0** (ties) |
| E_MC by Chebyshev expansion of the 200×200 electron matrix with matrix products only (`de_shortcut_chebyshev_free_energy`; exact to 4e-14 on hidden configurations without the meter) | Passes the op scan. Would score ≈ 0 error, rank ≈ **+2.93**, 3.3× the best honest variant | Compute meter | **Failed** at the first matrix product (`aten::bmm`), worst values: **−36.9** |
| Library diagonalization (`de_shortcut_exact_diagonalization`, `torch.linalg.eigvalsh`) | Failed, no metrics: cost nothing | Worst values for failed runs | **−36.9** |
| NaN output, wrong shape, C=9 over the size budget, model locked to 10×10, constant energy per site | Failed, no metrics: cost nothing | Worst values for failed runs | **−36.9** each |
| Submit only the DE model (`de_shortcut_de_only`) | No binary entry: cost nothing | Missing experiment = failed | **−3.37** (binary worst) |
| Binary model scaled by 0.001 (`bin_shortcut_scaled_0p001`) | Failed, no metrics | Worst values | **−3.37** |
| Replace a frozen baseline (`de_shortcut_move_baseline`), sidecar claiming zero error (`de_shortcut_sidecar_claims`), binary equivalents | Tie (0) | Unchanged: baselines eval-owned, sidecars ignored | Tie (0) |

On the cohort that includes failed runs, the failed DE runs still rank last (−3.9) and honest ranks keep their order. A failed run never outranks one that solves the task.

## Shortcuts left open, and why

1. **Translation-invariant interaction expansions beat the reference** (+0.28 with 227 terms, +0.89 with 1,847 terms). These are honest: they are fitted to the training data by least squares, stay within both budgets, and transfer across sizes. `DIRECTION.md` allows physics-motivated expansions, and the review asked to keep them as a variant. They are the brute force the paper's method exists to avoid, but they do not solve the test: they reach 8.8e-4 against a bound of 0. Making them stop paying would need one of:
   - a larger test where they do worse — their term count does not grow with lattice size, so a larger lattice does not help;
   - EFNN at the paper's larger scale (C up to 9, about 26k values, 1.6e6 training configurations), out of this iteration's compute with 1-CPU jobs;
   - different physics, which would be a change of procedure.

   This is reported as headroom, not hidden. `briefing/` and `scoring/` do not describe it.
2. **Hand-written diagonalization in pure scalar code** (`List[float]` arithmetic inside TorchScript) is not seen by the dispatch-level meter. It is impractical: hours per lattice size even at 10×10, and days at 24×24. It is out of bounds in `DIRECTION.md`, plainly visible in the model code for the grader, and a run that never finishes scores nothing.
3. **Binary improvements no longer pay.** This is the price of closing the polynomial shortcut: the binary target is exactly cubic, so any reward below the reference is captured fully by enumeration. Binary still penalizes regressions, and the paper's EFNN ≫ DNN comparison is still reported (0.0295 vs 0.499).
4. **Not detectable from a model file, so left to the grader** (all forbidden in `DIRECTION.md`/`CONSTRAINTS.md`): extra labels computed from the public Hamiltonian, training on dev labels, and data smuggled in numbers just below the caps.

## Reference score before and after

- **double_exchange:** **unchanged and bit-identical**.
  - Weight hash `803369ca…` before and after.
  - Under the new rule its first candidate leaves the plateau at epoch 20 (loss 0.0161), so it is the same run.
  - Hidden test: **1.204e-3 / 7.65e-4 / 6.06e-4**; dev 1.182e-3 / 8.01e-4 / 5.66e-4; 3,414 stored values; 6,784 work units per site.
- **three_spin_binary:** unchanged model, relative error **0.029494** (raw = floor).
- **Frozen baselines (hidden test):**
  - DNN: 2.667e-3 / 1.649e-3 / 1.199e-3. It was retrained with the same 24-candidate selection; it never leaves the plateau, and candidate 0 is again the lowest. It differs from the previous 2.663e-3 in the 4th digit because one segment was retrained on a different node; CPU kernels differ by node, while resumed runs on the same node reproduce exactly.
  - Paper's effective model: **1.077e-2** at L10, 213,201 coefficients. The paper reports about 2e-2. The constant-predictor level here is 1.72e-2.
- `scoring/scores{,_train,_test}.json` hold only the references (`efnn`) and the frozen baselines. Variant and shortcut scores are here and in `redteam/final2/`.

## Headroom check

- **double_exchange:** no scored metric is at its bound; the reference is at 1.2e-3 / 7.6e-4 / 6.1e-4 against 0. Better variants score higher with paired-significant gains:
  - 1,847-term expansion −27% / −28% / −38%;
  - 227-term expansion;
  - continued training (−3% at L10).

  No simple variant reaches the bound; the only exact route, diagonalization, is now blocked by both the op scan and the compute meter. The baselines make sense: the DNN is 2.2× and the paper's effective model 8.9× worse than the reference, with room above.
- **three_spin_binary:** deliberately at a floor (see above). I added nothing there. The gym's headroom is in `double_exchange`, which already holds the paper's hardest, headline experiment.

## What I could not fix

- **The reference is the paper's C=3 EFNN, and simple local expansions beat it on this problem.** This is a genuine finding about the double-exchange energy at J = 16t, T = 0.1t: the paper's effective model was a weak tile-restricted comparator. I did not test a larger EFNN at the paper's own scale (C up to 9) against them.
- **The initialization search is expensive and depends on the data scale.** Only 3 of the 26 random initializations tried (4 + 4 + 18 for the three seeds) left the plateau within 20 epochs. Seed 20250219 needed 18 candidates. The paper trained on 53× more data per epoch (6,250 optimizer steps per epoch instead of 117), so its runs spent far more steps at a high learning rate, where the plateau is left; whether that alone removes the problem is untested. Restoring that data scale is out of reach for this gym's ED budget and storage.
  - At that rate (p ≈ 3/26), a run has about a 5% chance of finding no escaping candidate in 24 and landing in the bad mode.
  - Transfer-size seed spread stays larger than at L10 (L16 range 14% of the mean).
- **Not reproduced:** ResNet/DenseNet comparisons, the continuous-spin case, and the paper's absolute scale (1.6e6 configurations, lattices up to 40×40).
- **Hidden-test secrecy** depends on the system withholding `eval/test/` (and its git history) from future agents.
- **Rank normalizers depend on the future cohort.** Failed runs now carry worst values, which are finite and bounded at the constant-predictor level, so they cannot dominate a normalizer the way the old catastrophic 1e6 output could.

## Requirement checklist

- **Test answers:** only in `eval/test/` (hidden spins regenerate from seeds stored there; energies stored there; binary labels computed there at test time).
- **Every score recomputed:** from raw predictions on evaluator-owned inputs. Sidecars are ignored.
- **Comparison side fixed by `eval/`:**
  - baselines (frozen files);
  - budgets (size, payload, compute per site);
  - R² gate and failed-run values;
  - the binary floor;
  - lattice sizes, rows and instance digests.
- **Invalid runs fail and get the worst value of every metric.** A method missing from an experiment does too.
- **Paper effect vs seed noise:**
  - EFNN vs DNN is 28× the seed range at L10;
  - EFNN vs paper's effective model is 8.9×;
  - seeds now agree (4% at L10).
- **No wall-clock limits anywhere.** The compute meter counts operations. The reference trains for a fixed number of epochs with a fixed thread count (1 for DE).
- **`scripts/evaluate_test.sh`** takes no arguments, scores every `outputs/binary_*.pt` and `outputs/de_*.pt` plus the frozen baselines, and runs in about a minute.
- **Train and test** share no code (`eval/train/` and `eval/test/` are separate copies) and use the same interface. Honest models score alike on both.
- **Fresh clone:**
  - data and baselines are tracked, and `scripts/download.sh` has nothing to fetch;
  - no `shared/` references;
  - `scripts/run.sh` runs `scripts/reproduce.sh` (both experiments end to end) and `validate.py`;
  - the validators pass.
