#!/usr/bin/env python3
"""Summarize this iteration's audit: ranks, seed agreement, paired bootstrap, probes.

Rank of a case (gym rule): sum over experiments e (weight w_e) and metrics m with
coefficient c_m != 0 of  w_e * |c_m| * dir_m * (x_m - x_m(reference)) / std_m,
dir_m = +1 if higher_is_better else -1, std_m = population std of m over a cohort.
Every case submits method ``m`` to both experiments (see make_cases.py); a failed or
missing submission carries the evaluator's worst values, so every case has a value in
every metric. Cohorts: ``honest`` (references, seeds, better/weaker variants, the DNN
as a submission, the expansion variants) and ``all`` (every case, failed runs included).
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

F = ROOT / "redteam" / "final2"
CASES = F / "cases"
REF = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
METHOD = "m"
REFERENCE_CASE = {"three_spin_binary": "bin_reference", "double_exchange": "de_reference"}
HONEST = {
    "three_spin_binary": [
        "bin_reference", "bin_better_continued", "bin_reference_seed_20250218", "bin_reference_seed_20250219",
        "bin_weaker_20_epochs", "bin_weaker_dnn_submitted",
    ],
    "double_exchange": [
        "de_reference", "de_reference_seed_20250218", "de_reference_seed_20250219", "de_better_continued",
        "de_weaker_3_epochs", "de_dnn_submitted", "de_ti_expansion_227", "de_ti_expansion_1847",
    ],
}
SEEDS = {
    "new": ["de_reference", "de_reference_seed_20250218", "de_reference_seed_20250219"],
    "prev": ["de_prev_reference", "de_prev_seed_20250218", "de_prev_seed_20250219"],
}
BOOTSTRAP = ["de_better_continued", "de_ti_expansion_227", "de_ti_expansion_1847", "de_reference_seed_20250218", "de_reference_seed_20250219"]


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


def own(rec: dict, exp: str, slice_name: str = "test") -> tuple[dict | None, str | None, dict]:
    scores = rec.get(slice_name)
    if not scores:
        return None, "no score file", {}
    block = scores["experiments"].get(exp, {})
    fail = (scores.get("_failures") or {}).get(exp, {}).get(METHOD)
    return block.get("results", {}).get(METHOD), fail, block.get("_artifacts", {}).get(METHOD, {})


def ranked(exp: str) -> dict:
    return {m: d for m, d in REF[exp]["metrics"].items() if d.get("coefficient", 0)}


def cohort_std(values: list[dict], exp: str) -> dict:
    return {m: float(np.std([v[m] for v in values])) if len(values) > 1 else float("nan") for m in ranked(exp)}


def contribution(metrics: dict, ref: dict, stds: dict, exp: str) -> float:
    total = 0.0
    for m, d in ranked(exp).items():
        direction = 1.0 if d["higher_is_better"] else -1.0
        if stds[m] > 0:
            total += abs(d["coefficient"]) * direction * (metrics[m] - ref[m]) / stds[m]
    return REF[exp]["weight"] * total


def rel(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)) / abs(np.mean(y)))


def bootstrap(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, y.shape[0], size=(n, y.shape[0]))

    def r(p, i):
        yy = y[i]
        return np.sqrt(np.mean((p[i] - yy) ** 2, axis=-1)) / np.abs(np.mean(yy, axis=-1))

    d = r(pb, idx) - r(pa, idx)
    return {"diff": rel(y, pb) - rel(y, pa), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))], "sd": float(np.std(d)), "frac_lower": float(np.mean(d < 0))}


def main() -> None:
    torch.set_num_threads(1)
    cases = load_cases()
    summary: dict = {"experiments": {}, "cases": {}}
    metrics_by_exp: dict = {}
    for exp in REF:
        ref_metrics = own(cases[REFERENCE_CASE[exp]], exp)[0] if REFERENCE_CASE[exp] in cases else None
        if ref_metrics is None:
            summary["experiments"][exp] = {"error": "reference case has no metrics"}
            continue
        vals = {n: own(cases[n], exp)[0] for n in cases}
        vals = {n: v for n, v in vals.items() if v is not None}
        honest = [n for n in HONEST[exp] if n in vals]
        std_h = cohort_std([vals[n] for n in honest], exp)
        std_a = cohort_std(list(vals.values()), exp)
        metrics_by_exp[exp] = (ref_metrics, vals, std_h, std_a)
        summary["experiments"][exp] = {"weight": REF[exp]["weight"], "reference_metrics": ref_metrics, "honest_cohort": honest, "std_honest": std_h, "std_all": std_a}

    # binary without the reference floor (raw values), to show what the floor removes
    exp = "three_spin_binary"
    if exp in metrics_by_exp:
        raw_vals = {}
        for n in cases:
            m, fail, art = own(cases[n], exp)
            if m is None:
                continue
            raw_vals[n] = {k: art.get(f"raw_{k}", m[k]) for k in ranked(exp)}
        ref_raw = raw_vals.get(REFERENCE_CASE[exp])
        if ref_raw:
            std_raw = cohort_std([raw_vals[n] for n in HONEST[exp] if n in raw_vals], exp)
            summary["binary_without_floor"] = {n: contribution(v, ref_raw, std_raw, exp) for n, v in raw_vals.items()}

    for n, rec in cases.items():
        row: dict = {"build_error": rec["build_error"], "exit": {}}
        if rec["status"]:
            row["exit"] = {s: rec["status"][s]["exit"] for s in ("train", "test") if s in rec["status"]}
        tot_h, tot_a, complete = 0.0, 0.0, True
        for exp, (ref_metrics, vals, std_h, std_a) in metrics_by_exp.items():
            m, fail, art = own(rec, exp)
            entry: dict = {"metrics": m, "failed": fail}
            for k in ("compute_per_site_L10", "compute_per_site_L16", "compute_per_site_L24"):
                if k in art:
                    entry[k] = art[k]
            if m is not None:
                entry["rank_honest"] = contribution(m, ref_metrics, std_h, exp)
                entry["rank_all"] = contribution(m, ref_metrics, std_a, exp)
                tot_h += entry["rank_honest"]
                tot_a += entry["rank_all"]
            else:
                complete = False
            mtr = own(rec, exp, "train")[0]
            if mtr and m:
                entry["train"] = {k: v for k, v in mtr.items() if k != "param_count"}
            row[exp] = entry
        row["combined_rank_honest"] = tot_h if complete else None
        row["combined_rank_all"] = tot_a if complete else None
        summary["cases"][n] = row

    # seed agreement
    de = "double_exchange"
    seeds: dict = {}
    for tag, names in SEEDS.items():
        rows = {n: own(cases[n], de)[0] for n in names if n in cases and own(cases[n], de)[0]}
        stat = {}
        for k in ("relative_error_L10", "relative_error_L16", "relative_error_L24"):
            v = [r[k] for r in rows.values()]
            if v:
                stat[k] = {"values": v, "mean": float(np.mean(v)), "range": float(np.ptp(v)), "std": float(np.std(v)), "range_over_mean": float(np.ptp(v) / np.mean(v))}
        if de in metrics_by_exp:
            ref_metrics, _, std_h, _ = metrics_by_exp[de]
            stat["rank_honest"] = {n: contribution(r, ref_metrics, std_h, de) for n, r in rows.items()}
        seeds[tag] = stat
    summary["de_seed_agreement"] = seeds

    # selection logs of the new runs
    sel = {}
    for name, sidecar in (
        ("de_reference", ROOT / "outputs" / "de_efnn.json"),
        ("de_reference_seed_20250218", CASES / "de_reference_seed_20250218" / "model.json"),
        ("de_reference_seed_20250219", CASES / "de_reference_seed_20250219" / "model.json"),
        ("dnn_baseline", ROOT / "checkpoints" / "baselines" / "de_dnn.json"),
    ):
        if sidecar.exists():
            d = json.loads(sidecar.read_text())
            sel[name] = {"selected_seed": d.get("selected_seed"), "selection": d.get("selection"), "metrics": d.get("metrics")}
    summary["de_initialization_selection"] = sel
    fit = ROOT / "checkpoints" / "baselines" / "de_effective_model.fit.json"
    if fit.exists():
        summary["paper_effective_model_fit"] = json.loads(fit.read_text())

    # official baselines vs reference (hidden test)
    off = ROOT / "scoring" / "scores_test.json"
    if off.exists():
        res = json.loads(off.read_text())["experiments"][de]["results"]
        if "efnn" in res:
            e = res["efnn"]
            cmp = {}
            for b in ("dnn", "effective_model"):
                if b in res:
                    cmp[b] = {k: res[b][k] / e[k] for k in res[b] if k.startswith("relative_error") and k in e}
            summary["de_baseline_over_reference"] = cmp

    # paired bootstrap and the Chebyshev probe without the meter (hidden sets)
    try:
        from eval.test.score_double_exchange_slice import load_test_sets, predict

        sets = load_test_sets()
        ref_p = CASES / "de_reference" / "submission" / "de_m.pt"
        if ref_p.exists():
            pref = predict(ref_p, sets, None)
            boot = {}
            for n in BOOTSTRAP:
                p = CASES / n / "submission" / "de_m.pt"
                if p.exists():
                    pb = predict(p, sets, None)
                    boot[n] = {k: bootstrap(sets[k][2], pref[k], pb[k]) for k in sets}
            summary["de_paired_bootstrap_vs_reference"] = boot
        cheb = CASES / "de_shortcut_chebyshev_free_energy" / "submission" / "de_m.pt"
        if cheb.exists():
            t0 = time.time()
            module = torch.jit.load(str(cheb), map_location="cpu").eval()
            out = {}
            with torch.no_grad():
                for k, n in (("L10", 8), ("L16", 1)):
                    L, x, y = sets[k]
                    p = module(torch.from_numpy(x[:n])).double().numpy()
                    out[k] = {"configs": n, "max_abs_error": float(np.max(np.abs(p - y[:n]))), "rmse_over_abs_mean_energy": float(np.sqrt(np.mean((p - y[:n]) ** 2)) / abs(np.mean(y)))}
            out["seconds"] = round(time.time() - t0, 1)
            summary["de_chebyshev_probe_without_meter"] = out
    except Exception as exc:  # noqa: BLE001
        summary["de_bootstrap_error"] = repr(exc)

    (F / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "| case | exits (train/test) | binary rel. err (scored) | DE rel. err L10 / L16 / L24 | binary rank | DE rank | combined rank (honest) | combined rank (all) | failure |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n, row in summary["cases"].items():
        b = row.get("three_spin_binary") or {}
        d = row.get("double_exchange") or {}
        bm = b.get("metrics") or {}
        dm = d.get("metrics") or {}
        fail = d.get("failed") or b.get("failed") or ""
        ch = row["combined_rank_honest"]
        ca = row["combined_rank_all"]
        lines.append(
            f"| {n} | {row['exit'].get('train')}/{row['exit'].get('test')} | {bm.get('relative_error', float('nan')):.4g} | "
            f"{dm.get('relative_error_L10', float('nan')):.4g} / {dm.get('relative_error_L16', float('nan')):.4g} / {dm.get('relative_error_L24', float('nan')):.4g} | "
            f"{b.get('rank_honest', float('nan')):.4g} | {d.get('rank_honest', float('nan')):.4g} | "
            f"{'-' if ch is None else f'{ch:.4g}'} | {'-' if ca is None else f'{ca:.4g}'} | {fail[:100]} |"
        )
    (F / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
