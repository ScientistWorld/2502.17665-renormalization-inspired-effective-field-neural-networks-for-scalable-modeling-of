# Hardening report (final iteration)

**Bottom line.** Better results score higher and worse ones lower.

The biggest remaining problem was the one `REDTEAM.md` and iteration 11 left open. On the only scored experiment, regression on enumerated spin products solved the test to 3e-7 relative error. No variant of that experiment can fix this:
- the binary three-spin energy *is* a cubic polynomial of the spins;
- any larger instance that can be learned at all is also solved by a polynomial-structured model.

So I added the paper's third case study, the quantum **double-exchange** model, as the main scored experiment (weight 0.8). The binary experiment drops to weight 0.2. In the new experiment:
- the energy is a free energy obtained by exact diagonalization, not a polynomial;
- models train on 10×10 lattices and are scored on hidden 10×10, 16×16 and 24×24 configurations (the paper's headline transfer claim);
- the brute force the paper's method exists to avoid, diagonalizing the Hamiltonian inside the model, is rejected by the evaluator.

Every double-exchange shortcut I built scores as a failed run or ties the reference. The binary polynomial shortcut still pays something, but its combined rank fell from **+0.160 to +0.032**. That is below every genuine double-exchange improvement measured.

One finding contradicts the paper:
- **Effective model vs EFNN.** A translation-invariant version of the paper's own effective-model baseline (explicit 2- and 4-spin terms, 227 coefficients) is slightly more accurate than the reference EFNN (C=3). The paper reported about 2e-2 for its effective model; ours reaches 1.17e-3. EFNN still beats the same-size DNN by 2.2×, as the paper claims.
- **Larger expansion.** A 1,847-term expansion beats the reference by 27%. It is far from the bound, so it counts as headroom, and the gym ranks it above the reference.

Evidence: job `1d519c60-e00` (smoke test and data build) and job `a4ea3928-732` (training from the final code; 26 cases, each scored by `bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh` with **no arguments**, in a fresh process, on an `outputs/` holding only that case's files). The harness is `redteam/final/`. Per-case scores, logs and exit codes are in `redteam/final/cases/`, the full summary in `redteam/final/summary.{json,md}`.

## What changed

| Problem | Fix |
|---|---|
| Brute-force enumeration solves the only scored test (binary three-spin, polynomial regression 2.8e-7) | **New scored experiment `double_exchange`** (paper case study 3), weight 0.8; `three_spin_binary` weight 1.0 → 0.2. Details below. |
| The paper's headline claim (train on small lattices, predict larger ones) was not tested | The double-exchange hidden test scores 10×10, 16×16 and 24×24 lattices. The same submitted model must run on all three. |
| — (new experiment) Exact diagonalization inside the model would score 2e-8 | The scorer scans TorchScript code. It fails models that call dense linear-algebra solvers (`torch.linalg.*` except vector norms, eig*, svd, inverse, det, slogdet, cholesky, lu, qr, lstsq, matrix_exp, …) or `from_file`. |
| — (new experiment) Test inputs and answers | Hidden test spins regenerate from seeds stored only in `eval/test/double_exchange_instance.json`, written by `eval/test/build_double_exchange.py` (the only ED code). Energies are stored there too. A pinned SHA-256 digest guards the instance, and the regenerated spins are checked against stored hashes. |
| — (new experiment) Comparison side | The symmetrized DNN (`dnn`) and the effective model (`effective_model`) are frozen in `eval/*/baselines/`. Files in `outputs/` with those names are ignored. |
| — (new experiment) Invalid runs | Failed (no metrics) for any of: not loadable, forbidden op, over 4,000 stored values or 128 KiB of code/pickles, raises on any lattice size, wrong shape, non-finite output, R² ≤ 0 on any size. |
| One score file for two experiments | `eval/{train,test}/score_all.py` scores both experiments (submissions `outputs/binary_<name>.pt` and `outputs/de_<name>.pt`). The binary scoring code is unchanged, only wrapped in `score_experiment()`. |

### The double-exchange experiment
- **Physics** (paper Eq. 2, Supplementary Eqs. 69 and 78): t = 1, J = 16t, μ = −8.3t, T = 0.1t, periodic L×L lattice, E_MC = −T Σ log(1 + e^(−E_n/T)). Spins are uniform on the unit sphere.
- **Physics checks:**
  - Mean E_MC at 10×10 is −51.15; the paper's Table II gives |mean| 51.65 (1% apart).
  - The numpy ED and an independent torch ED agree to 2e-8 relative.
  - Relative error falls with lattice size, as in the paper.
- **Data:**
  - public: 30,000 training configurations on 10×10, plus dev sets of 2,000 / 300 / 100 on 10×10 / 16×16 / 24×24;
  - hidden test: 4,000 / 600 / 200 configurations of the same three sizes.

  The paper used 1.6×10⁶ training configurations; this one change scales resources, not procedure. Spins regenerate from seeds (legacy `RandomState`, whose stream is frozen across numpy versions). Energies are tracked CSV files.
- **Reference** (`bash scripts/method.sh de`, `method/models_de.py`): the paper's symmetrized EFNN (Fig. 4a, Supplementary Sec. IV):
  - symmetrization layers T_j = Σ_k conv(S_0,k)⊙conv′(S_0,k);
  - conv-tanh-conv maps f_j and g_j;
  - S_j = g_j(T_j)⊙f_{j−1}(S_{j−1});
  - q = conv to one channel, then a lattice sum.

  It uses C=3 channels and 3 layers (the paper's Table II setting), 5×5 kernels with circular padding, 3,414 stored values. Training: Adam lr 2e-3 with decay 0.97 per epoch, 150 epochs, batch 256, weights with the lowest training loss.
- **Baselines** (`bash scripts/baseline.sh de`):
  - the paper's symmetrized DNN (Supplementary Fig. 3(c)), with the same C, depth and training;
  - the paper's effective model (Eq. 3) as a translation-invariant local expansion: 2-spin s_i·s_j terms up to distance 3, 4-spin (s_i·s_j)(s_k·s_l) products of short bonds, fitted by least squares. Translation invariance lets it run on every lattice size.
- **Metrics**: `relative_error_L10` (coefficient 1.0), `relative_error_L16` (0.5), `relative_error_L24` (0.5); `param_count` (constraint, coefficient 0, budget 4,000).

### Weights, and why
- `three_spin_binary` 1.0 → **0.2**. Its target is exactly a cubic polynomial of the spins. Ranking weight there mostly rewards whoever enumerates spin products, which `DIRECTION.md` forbids but the test cannot detect. It stays as the check of the paper's architecture comparison, where EFNN beats the same-width DNN 17×.
- `double_exchange` **0.8**. This is the paper's hardest and headline experiment: the abstract's "most striking" claim is lattice-size generalization. Its exact answer is expensive, it is where a surrogate matters, and brute-force enumeration does not solve it.
- Within `double_exchange`, accuracy on the training size (1.0) and on larger lattices (0.5 + 0.5) count equally.

## Do better results score higher?

Yes. Ranks use the gym's formula, computed per experiment on that experiment's honest cohort:

w_e · Σ|coef|·direction·(x − x_ref)/std

The cohorts are the reference, the better variant, two extra seeds, the weaker variant, and the frozen baselines submitted as methods. Before this iteration only the binary experiment existed, at weight 1.0.

**Before → after on `three_spin_binary`.** The models are unchanged, so the scores are identical; the rank shrinks with the weight.

| Variant | Hidden-test rel. error | Rank before (weight 1.0) | Rank after (weight 0.2) |
|---|---:|---:|---:|
| Reference (EFNN 3×105) | 0.029494 | 0 | 0 |
| Better: + 2,400 epochs | 0.021014 | +0.0458 | +0.0092 |
| Seed 20250218 / 20250219 | 0.028225 / 0.032761 | +0.0070 / −0.0180 | +0.0014 / −0.0036 |
| Weaker: 20 epochs | 0.724677 | −3.821 | −0.764 |
| Weaker: DNN | 0.498683 | −2.550 | −0.510 |

**New `double_exchange` experiment (hidden test; relative error at L = 10 / 16 / 24):**

| Variant | L10 | L16 | L24 | Rank (weight 0.8) |
|---|---:|---:|---:|---:|
| Reference EFNN (C=3, 3 layers, seed 20250217) | **1.204e-3** | **7.65e-4** | **6.06e-4** | 0 |
| Larger effective model (1,847 terms, within budget) | 8.76e-4 | 5.52e-4 | 3.76e-4 | **+0.906** |
| Frozen effective model, submitted as a method (227 terms) | 1.166e-3 | 6.86e-4 | 4.82e-4 | +0.286 |
| Better: reference + 150 epochs from lr 5e-4 | 1.166e-3 | 7.52e-4 | 5.78e-4 | +0.096 |
| Seed 20250218 | 1.216e-3 | 7.70e-4 | 5.51e-4 | +0.053 |
| Seed 20250219 (converged to a worse optimum) | 1.929e-3 | 1.292e-3 | 8.65e-4 | −1.725 |
| Weaker: 3 epochs | 2.701e-3 | 1.718e-3 | 1.284e-3 | −3.622 |
| Same-size DNN (frozen baseline, submitted as a method) | 2.663e-3 | 1.645e-3 | 1.197e-3 | −3.392 |

**Noise:**
- **Paired bootstrap** on identical hidden inputs (1,000 joint resamples), better minus reference:
  - L10: −3.8e-5, CI [−4.6e-5, −3.1e-5]
  - L16: −1.3e-5, CI [−2.7e-5, +0.3e-5] (not significant)
  - L24: −2.9e-5, CI [−4.7e-5, −1.0e-5]

  Test-sampling noise of paired differences is 0.4–4.4e-5 (bootstrap standard deviations measured for the EFNN variants; at most 2.2e-5 at L10). The larger effective model's gain (−3.3e-4 at L10) is therefore more than 10 such standard deviations; it was not bootstrapped itself. Continued training gives only a small but real gain: this reference is capacity-limited, not under-trained.
- **Training seeds** are bimodal. Two seeds land at 1.20–1.22e-3. The third converges to a worse optimum at 1.93e-3 (training loss 0.0119 vs 0.0045), so the L10 range is 7.3e-4 (std 3.4e-4). This is a property of the paper's multiplicative architecture, not of the test. A single EFNN run can therefore land well below its usual accuracy.
- **Paper effect versus noise:** EFNN beats the same-size DNN 2.2× at L10 (2.15× at L16, 1.97× at L24). The gap of 1.46e-3 is 2.0× the full seed range and 4.3 seed standard deviations, and even the bad seed beats the DNN at every size.

**Train and test agree** (dev / hidden relative error at L10):

| Model | Dev | Test |
|---|---:|---:|
| Reference | 1.182e-3 | 1.204e-3 |
| Better | 1.145e-3 | 1.166e-3 |
| Effective model | 1.148e-3 | 1.166e-3 |
| Larger effective model | 8.93e-4 | 8.76e-4 |
| DNN | 2.667e-3 | 2.663e-3 |
| Bad seed | 1.895e-3 | 1.929e-3 |

## Shortcuts: before and after

**Fixed or closed this iteration:**

| Shortcut | Before | Fix | After |
|---|---|---|---|
| Polynomial regression on the binary spin products (`bin_shortcut_polynomial_regression`, 576 features) | Rel. error 2.8e-7, rank **+0.160**: more than 3× the genuine better variant | Binary weight 1.0 → 0.2; ranking moved to `double_exchange`, which enumeration does not solve | Same error. Combined rank **+0.032**: below every genuine double-exchange gain (+0.096 to +0.906). Still positive, see "left open". |
| Exact diagonalization inside the model (`de_shortcut_exact_diagonalization`: a TorchScript model that builds the electron Hamiltonian and calls `torch.linalg.eigvalsh`) | Would score 2.1e-8 / 3.4e-8 / 2.9e-8 (computed with the scan bypassed): solves the test | Code scan for linear-algebra solvers | **Failed run** (`linalg_eigvalsh`) |
| Constant energy per site × number of sites (`de_shortcut_constant_per_site`) | n/a (new experiment) | R² gate on every size | **Failed run** (RMSE 0.877677 vs constant 0.877642) |
| Model that only works on the training lattice, like a flattened MLP (`de_shortcut_size_locked`) | n/a | Every submission runs on all three sizes | **Failed run** (raises on 16×16) |
| Over budget: EFNN C=9 (`de_shortcut_oversized_c9`, 26,436 values) | n/a | Budget 4,000, measured from the file | **Failed run** |
| NaN output / output one row short (`de_shortcut_nan_output`, `de_shortcut_wrong_shape`) | n/a | Finite and shape checks | **Failed runs** |
| Replace the frozen DNN (`de_shortcut_move_baseline`: zeros saved as `outputs/de_dnn.pt`) | n/a | Baselines frozen in `eval/`; reserved names ignored | DNN row unchanged; reference **ties** (rank 0) |
| Sidecar claiming zero error (`de_shortcut_sidecar_claims`) | n/a | Sidecars ignored | **Ties** the reference (rank 0) |

**Rescored iteration-11 fixes (binary).** These are unchanged models scored through the new combined evaluator:
- `bin_shortcut_scaled_0p001`: failed run (R² gate).
- `bin_shortcut_sidecar_claims`: ties.
- `bin_shortcut_move_baseline`: ties; the DNN row is unchanged.

The other `REDTEAM.md` shortcuts were fixed in iteration 11 and not replayed: label generator, width metadata, impossible width, NaN, catastrophic, constant, empty and short outputs (see the iteration-11 rows in `redteam/hardening/summary.md`). This iteration did not change their code paths.

## Shortcuts left open

1. **Polynomial regression on binary spin products — major, reduced, not closed.** It still earns +0.032, because a cubic target cannot be told apart from a learned model. `scoring/DIRECTION.md` forbids it and the grader has to catch it. At weight 0.2 it pays less than any real double-exchange improvement measured.
2. **Hand-written diagonalization in the double-exchange model.** Lanczos, a kernel-polynomial or Chebyshev expansion of the Hamiltonian written with plain matmuls would evade the op scan. It is forbidden in `DIRECTION.md` and `CONSTRAINTS.md`, and it is visible in the model code for the grader.
3. **Extra training labels.** The Hamiltonian is public (the paper, `_paper_reported`), so anyone can compute energies for new configurations. Forbidden; not detectable from a model file.
4. **Training on public dev labels** (both experiments): forbidden, not detectable.
5. **Smuggling numbers below the caps**: about 14,000 floats in pickled lists under 128 KiB (binary: +14% capacity; double-exchange: up to 4× the 4,000 budget). It is minor for binary. For double-exchange it is visible in the archive for the grader.

## Reference score before and after

- **three_spin_binary:** unchanged. The model is the same file and binary code and data were not modified: relative error **0.029494**, MAE 0.06704, tail RMSE 0.08861, 95,762 values. Only its weight changed (1.0 → 0.2).
- **double_exchange (new):** relative error **1.204e-3 / 7.65e-4 / 6.06e-4** at L = 10 / 16 / 24, 3,414 values. Trained from the final code with `scripts/method.sh de`.
- **Frozen baselines:** DNN 2.663e-3 / 1.645e-3 / 1.197e-3 (1,596 values); effective model 1.166e-3 / 6.86e-4 / 4.82e-4 (1,091 values).
- `scoring/scores{,_train,_test}.json` contain only the references (`efnn`) and the frozen baselines. Shortcut and variant scores are only here and in `redteam/final/`.

## Headroom check

**No scored metric is at its bound.**
- Double-exchange: the reference is at 1.2e-3 / 7.6e-4 / 6.1e-4 against a bound of 0. The R² at L10 is 0.995.
- Binary: relative error 0.0295, MAE 0.067, tail RMSE 0.089.

**A better variant scores higher in each experiment:**
- binary: continued training, −29%;
- double-exchange: continued training, −3% (paired-significant);
- double-exchange: a larger interaction expansion, −27% / −28% / −38%.

**No simple variant reaches the bound:**
- the best double-exchange result is 8.8e-4;
- the exact answer (2e-8) is only reachable by diagonalizing, which is rejected;
- on binary, only the out-of-bounds polynomial reaches the bound, which is why its weight dropped.

**Baselines make sense:**
- the DNN and the 3-epoch EFNN score 2.2× below the reference, with room above it;
- the frozen effective model scores slightly *above* the reference (see below).

**What I added:** the `double_exchange` experiment, i.e. where the paper's method meets a non-polynomial target, larger lattices, and its own effective-model baseline.

## What I could not fix, stated plainly

- **The paper's claim that EFNN beats the effective model does not reproduce here.** The paper reports about 2e-2 for its 213,201-parameter tile-restricted effective model against about 1e-3 for networks. My translation-invariant local version uses 227 coefficients and reaches 1.17e-3, slightly better than the C=3 EFNN reference. A 1,847-term version is 27% better. This is a property of the paper's comparison, which used a weak effective model, not of the gym.
  - The gym ranks such expansions above the reference. They are learned from the training data, so `DIRECTION.md` keeps them in scope.
  - Whether a larger EFNN (the paper's C up to 9, about 26k values) would beat them is untested. The 4,000 budget rules it out by design.
- **The reproduced claims:** EFNN beats a same-size DNN (2.2× on double-exchange, 17× on binary). On double-exchange it transfers from 10×10 to 16×16 and 24×24 with relative error falling with lattice size. Not reproduced or not run:
  - EFNN vs ResNet and DenseNet;
  - the continuous-spin experiment;
  - the paper's absolute numbers (1.6×10⁶ training configurations, lattices up to 40×40).
- **The reference EFNN is seed-sensitive on double-exchange** (one of three seeds converged to a 60% worse optimum). Single-run comparisons near the reference should be read with that in mind; paired test noise is small (about 1e-5).
- **Hidden-test secrecy depends on the system** withholding `eval/test/` (couplings, seeds, energies, and the ED builder) and its git history from future agents.
- **Rank normalizers depend on the future cohort.** Failed runs never enter it, and every ranked metric is an accuracy metric, so a run that fails a task cannot outrank one that solves it through other metrics.

## Requirement checklist

- **Test answers:** only in `eval/test/`, or computed there (binary energies at test time; double-exchange energies stored there, spins regenerated from hidden seeds).
- **Every score recomputed:** from raw model outputs on evaluator-owned inputs. Sidecars are ignored.
- **Comparison side fixed by `eval/`:** baselines, budgets, R² gate, lattice sizes, rows and test sizes. Instance digests are pinned.
- **Invalid runs fail:** impossible values, failed constraints and forbidden operations give a failed run.
- **Paper effect versus seeds:** EFNN vs DNN is 4.3 seed standard deviations on double-exchange and about 240 on binary.
- **Better variant:** beats the reference on paired comparisons.
- **Paper's direction:** EFNN < DNN in both experiments; relative error falls with lattice size.
- **Rank rule:** every ranked metric is an accuracy metric with `higher_is_better` set. Constraint metrics have coefficient 0 and are hard gates.
- **`scripts/evaluate_test.sh`:**
  - takes no arguments and scores every `outputs/binary_*.pt` and `outputs/de_*.pt` plus the frozen baselines;
  - runs in about a minute and never retrains.
- **Train and test:** they share no code and use the same interface; honest models score alike on both.
- **Fresh clone:**
  - data and baselines are tracked; `scripts/download.sh` has nothing to fetch;
  - no references to `shared/`;
  - `scripts/run.sh` runs `scripts/reproduce.sh` (both experiments end to end) and then `validate.py`;
  - `python validate.py` and `python validate.py --post-design` both pass (job `a4ea3928-732`).
