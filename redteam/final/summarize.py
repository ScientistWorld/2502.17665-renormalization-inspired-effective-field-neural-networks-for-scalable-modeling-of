#!/usr/bin/env python3
"""Summarize the final audit: ranks, seed spread, paired bootstrap, train/test, probes.

Rank of a case (gym rule): for each experiment e with weight w_e and each metric m with
coefficient c_m != 0,  w_e * |c_m| * dir_m * (x_m - x_m(reference)) / std_m, summed;
dir_m = +1 if higher_is_better else -1; std_m = population std of m over a cohort.
A case that submits nothing for an experiment is counted as submitting the reference
there (contribution 0); a case whose submission failed has no rank in that experiment.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

F = ROOT / "redteam" / "final"
CASES = F / "cases"
REF = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
BASELINE_NAMES = {"three_spin_binary": {"dnn"}, "double_exchange": {"dnn", "effective_model"}}
REFERENCE_CASE = {"three_spin_binary": "bin_reference", "double_exchange": "de_reference"}
HONEST = {
    "three_spin_binary": [
        "bin_reference",
        "bin_better_continued",
        "bin_reference_seed_20250218",
        "bin_reference_seed_20250219",
        "bin_weaker_20_epochs",
        "bin_weaker_dnn_submitted",
    ],
    "double_exchange": [
        "de_reference",
        "de_better_continued",
        "de_reference_seed_20250218",
        "de_reference_seed_20250219",
        "de_weaker_3_epochs",
        "de_dnn_submitted",
        "de_effective_submitted",
    ],
}
SEEDS = {e: [REFERENCE_CASE[e], f"{REFERENCE_CASE[e]}_seed_20250218", f"{REFERENCE_CASE[e]}_seed_20250219"] for e in REFERENCE_CASE}


def load_cases() -> dict:
    out = {}
    for d in sorted(p for p in CASES.iterdir() if p.is_dir()):
        rec: dict = {"status": None, "test": None, "train": None, "build_error": (d / "build_error.txt").exists()}
        if (d / "status.json").exists():
            rec["status"] = json.loads((d / "status.json").read_text())
        for s in ("train", "test"):
            f = d / f"scores_{s}.json"
            if f.exists():
                rec[s] = json.loads(f.read_text())
        out[d.name] = rec
    return out


def submitted(rec: dict, exp: str, slice_name: str = "test") -> tuple[str | None, dict | None, str | None]:
    """(method name, metrics, failure reason) of the case's own submission in an experiment."""

    scores = rec.get(slice_name)
    if not scores:
        return None, None, "no score file"
    fails = (scores.get("_failures") or {}).get(exp, {})
    results = scores["experiments"].get(exp, {}).get("results", {})
    own = {k: v for k, v in results.items() if k not in BASELINE_NAMES[exp]}
    if own:
        name = sorted(own)[0]
        return name, own[name], None
    if fails:
        name = sorted(fails)[0]
        return name, None, fails[name]
    return None, None, None


def ranked_metrics(exp: str) -> dict:
    return {m: d for m, d in REF[exp]["metrics"].items() if d.get("coefficient", 0)}


def cohort_std(cases: dict, names: list[str], exp: str) -> dict:
    stds = {}
    for m in ranked_metrics(exp):
        vals = [submitted(cases[n], exp)[1][m] for n in names if n in cases and submitted(cases[n], exp)[1]]
        stds[m] = float(np.std(vals)) if len(vals) > 1 else float("nan")
    return stds


def contribution(metrics: dict, ref: dict, stds: dict, exp: str) -> float:
    total = 0.0
    for m, d in ranked_metrics(exp).items():
        direction = 1.0 if d["higher_is_better"] else -1.0
        if stds[m] > 0:
            total += abs(d["coefficient"]) * direction * (metrics[m] - ref[m]) / stds[m]
    return REF[exp]["weight"] * total


def predictions_de(model_path: Path, sets: dict) -> dict:
    from eval.test.score_double_exchange_slice import predict

    return predict(model_path, sets)


def bootstrap(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, y.shape[0], size=(n, y.shape[0]))

    def rel(p, i):
        yy = y[i]
        return np.sqrt(np.mean((p[i] - yy) ** 2, axis=-1)) / np.abs(np.mean(yy, axis=-1))

    d = rel(pb, idx) - rel(pa, idx)
    full = float(rel(pb, np.arange(y.shape[0])) - rel(pa, np.arange(y.shape[0])))
    return {"diff": full, "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))], "sd": float(np.std(d)), "frac_lower": float(np.mean(d < 0))}


def main() -> None:
    torch.set_num_threads(2)
    cases = load_cases()
    summary: dict = {"experiments": {}, "cases": {}}
    contribs: dict[str, dict[str, dict]] = {}
    for exp in REF:
        ref_name = REFERENCE_CASE[exp]
        ref_metrics = submitted(cases[ref_name], exp)[1] if ref_name in cases else None
        if ref_metrics is None:
            summary["experiments"][exp] = {"error": f"reference case {ref_name} has no metrics"}
            continue
        honest = [n for n in HONEST[exp] if n in cases]
        passing = [n for n in cases if submitted(cases[n], exp)[1] is not None]
        std_h = cohort_std(cases, honest, exp)
        std_p = cohort_std(cases, passing, exp)
        seeds = [n for n in SEEDS[exp] if n in cases and submitted(cases[n], exp)[1]]
        spread = {}
        for m in list(ranked_metrics(exp)):
            vals = [submitted(cases[n], exp)[1][m] for n in seeds]
            spread[m] = {"values": vals, "range": float(np.ptp(vals)) if vals else None, "std": float(np.std(vals)) if vals else None}
        rank_vals = [contribution(submitted(cases[n], exp)[1], ref_metrics, std_h, exp) for n in seeds]
        spread["rank_honest"] = {"values": rank_vals, "range": float(np.ptp(rank_vals)) if rank_vals else None}
        summary["experiments"][exp] = {
            "weight": REF[exp]["weight"],
            "reference_metrics": ref_metrics,
            "honest_cohort": honest,
            "std_honest": std_h,
            "std_passing": std_p,
            "seed_spread": spread,
        }
        contribs[exp] = {}
        for n in cases:
            name, metrics, fail = submitted(cases[n], exp)
            if metrics is not None:
                contribs[exp][n] = {
                    "method": name,
                    "metrics": metrics,
                    "rank_honest": contribution(metrics, ref_metrics, std_h, exp),
                    "rank_passing": contribution(metrics, ref_metrics, std_p, exp),
                }
            elif fail is not None and name is not None:
                contribs[exp][n] = {"method": name, "failed": fail}

    # binary polynomial shortcut under the old weighting (binary weight 1.0, iteration 11)
    exp = "three_spin_binary"
    if exp in contribs and "bin_shortcut_polynomial_regression" in contribs[exp]:
        c = contribs[exp]["bin_shortcut_polynomial_regression"]
        if "rank_honest" in c:
            summary["polynomial_shortcut_old_weight_1p0_rank_honest"] = c["rank_honest"] / REF[exp]["weight"]

    for n, rec in cases.items():
        row: dict = {"build_error": rec["build_error"], "exit": {}}
        if rec["status"]:
            row["exit"] = {s: rec["status"][s]["exit"] for s in ("train", "test") if s in rec["status"]}
        total = 0.0
        ok = True
        any_sub = False
        for exp in contribs:
            c = contribs[exp].get(n)
            row[exp] = c
            if c is None:
                continue
            any_sub = True
            if "failed" in c:
                ok = False
            else:
                total += c["rank_honest"]
        row["combined_rank_honest"] = total if (ok and any_sub) else None
        tr = {}
        for exp in REF:
            _, mtr, _ = submitted(rec, exp, "train")
            _, mte, _ = submitted(rec, exp, "test")
            if mtr and mte:
                tr[exp] = {"train": {k: v for k, v in mtr.items() if k != "param_count"}, "test": {k: v for k, v in mte.items() if k != "param_count"}}
        row["train_vs_test"] = tr
        summary["cases"][n] = row

    # paired bootstrap and the exact-diagonalization probe on the hidden double-exchange sets
    try:
        from eval.test.score_double_exchange_slice import load_test_sets

        sets = load_test_sets()
        preds = {}
        for n in ["de_reference", "de_better_continued", "de_reference_seed_20250218", "de_reference_seed_20250219"]:
            p = CASES / n / "submission" / "de_efnn.pt"
            if p.exists():
                preds[n] = predictions_de(p, sets)
        boot = {}
        if "de_reference" in preds:
            for n, pr in preds.items():
                if n == "de_reference":
                    continue
                boot[n] = {k: bootstrap(sets[k][2], preds["de_reference"][k], pr[k]) for k in sets}
        summary["de_paired_bootstrap_vs_reference"] = boot
        ed = CASES / "de_shortcut_exact_diagonalization" / "submission" / "de_exact.pt"
        if ed.exists():
            t0 = time.time()
            module = torch.jit.load(str(ed), map_location="cpu").eval()
            pe = {}
            with torch.no_grad():
                for k, (L, x, _y) in sets.items():
                    step = {10: 50, 16: 10, 24: 2}[L]
                    pe[k] = np.concatenate([module(torch.from_numpy(x[i : i + step])).double().numpy() for i in range(0, x.shape[0], step)])
            summary["de_exact_diagonalization_without_defence"] = {
                f"relative_error_{k}": float(np.sqrt(np.mean((pe[k] - sets[k][2]) ** 2)) / abs(np.mean(sets[k][2]))) for k in sets
            }
            summary["de_exact_diagonalization_without_defence"]["seconds"] = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        summary["de_bootstrap_error"] = repr(exc)

    (F / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["| case | exits (train/test) | binary rel. err | DE rel. err L10 / L16 / L24 | binary rank | DE rank | combined rank | failure |", "|---|---|---|---|---|---|---|---|"]
    for n, row in summary["cases"].items():
        b = row.get("three_spin_binary") or {}
        d = row.get("double_exchange") or {}
        bm = b.get("metrics", {})
        dm = d.get("metrics", {})
        fail = b.get("failed") or d.get("failed") or ""
        lines.append(
            f"| {n} | {row['exit'].get('train')}/{row['exit'].get('test')} | "
            f"{bm.get('relative_error', float('nan')):.4g} | "
            f"{dm.get('relative_error_L10', float('nan')):.4g} / {dm.get('relative_error_L16', float('nan')):.4g} / {dm.get('relative_error_L24', float('nan')):.4g} | "
            f"{b.get('rank_honest', float('nan')):.4g} | {d.get('rank_honest', float('nan')):.4g} | "
            f"{row['combined_rank_honest'] if row['combined_rank_honest'] is not None else '-'} | {fail[:110]} |"
        )
    (F / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
