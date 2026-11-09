#!/usr/bin/env python3
"""Summarize the hardening cases: outcomes, seed spread, paired bootstrap, rank deltas.

Rank delta of an attempt against the reference (as the gym ranks attempts):
    dS = weight * sum_m |coef_m| * dir_m * (x_m - ref_m) / std_cohort(m)
with dir = -1 for lower-is-better metrics, population std over the stated cohort,
and only metrics with a nonzero coefficient. Failed runs carry no metrics.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
H = ROOT / "redteam" / "hardening"
CASES = H / "cases"
EXP = "three_spin_binary"
HONEST = [
    "reference",
    "reference_seed_20250218",
    "reference_seed_20250219",
    "better_continued",
    "weaker_20_epochs",
    "weaker_dnn_submitted",
]


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def submitted_entry(scores: dict | None) -> tuple[str | None, dict | None, dict | None]:
    if not scores:
        return None, None, None
    exp = scores.get("experiments", {}).get(EXP, {})
    for name, art in exp.get("_artifacts", {}).items():
        if art.get("kind") == "submitted":
            return name, exp["results"][name], art
    return None, None, None


def case_record(case: Path) -> dict:
    rec: dict = {"case": case.name}
    status = load(case / "status.json") or {}
    rec["files"] = status.get("files")
    for slice_name in ("train", "test"):
        scores = load(case / f"scores_{slice_name}.json")
        name, metrics, art = submitted_entry(scores)
        rec[slice_name] = {
            "exit": status.get(slice_name, {}).get("exit"),
            "method": name,
            "metrics": metrics,
            "artifact": art,
            "failures": (scores or {}).get("_failures"),
            "ignored": (scores or {}).get("_ignored"),
            "baselines": {
                k: v
                for k, v in (scores or {}).get("experiments", {}).get(EXP, {}).get("results", {}).items()
                if (scores or {}).get("experiments", {}).get(EXP, {}).get("_artifacts", {}).get(k, {}).get("kind") == "baseline"
            },
        }
    if (case / "build_error.txt").exists():
        rec["build_error"] = (case / "build_error.txt").read_text()[-2000:]
    meta = load(case / "model.json")
    if meta:
        rec["training"] = {k: meta.get(k) for k in ("epochs", "lr", "lr_gamma", "seed", "init_checkpoint")}
    return rec


def rank_deltas(records: dict, cohort: list[str], coefs: dict, weight: float) -> tuple[dict, dict]:
    metrics = [m for m, c in coefs.items() if c["coefficient"] != 0]
    ref = records["reference"]["test"]["metrics"]
    stds = {}
    for m in metrics:
        vals = [records[c]["test"]["metrics"][m] for c in cohort if records.get(c, {}).get("test", {}).get("metrics")]
        stds[m] = float(np.std(vals)) if len(vals) > 1 else 0.0
    out = {}
    for name, rec in records.items():
        met = rec["test"]["metrics"]
        if not met:
            out[name] = None
            continue
        total = 0.0
        for m in metrics:
            direction = 1.0 if coefs[m]["higher_is_better"] else -1.0
            diff = met[m] - ref[m]
            if stds[m] > 0:
                total += abs(coefs[m]["coefficient"]) * direction * diff / stds[m]
        out[name] = weight * total
    return out, stds


def paired_bootstrap(records: dict) -> dict:
    """Paired comparisons on identical test rows (rows resampled jointly)."""

    from eval.test import score_three_spin_slice as T

    x, y = T.load_test_set()
    preds = {}
    for name in ("reference", "reference_seed_20250218", "reference_seed_20250219", "better_continued"):
        path = CASES / name / "submission" / "binary_efnn.pt"
        if path.exists():
            preds[name] = T.predict(path, x)
    n = y.shape[0]
    ref = preds.get("reference")
    out = {}
    if ref is None:
        return out

    def rel(pred: np.ndarray, rows: np.ndarray) -> np.ndarray:
        err = pred[rows] - y[rows]
        return np.sqrt(np.mean(err * err, axis=1)) / np.abs(np.mean(y[rows], axis=1))

    def paired_diffs(pred: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)  # identical resamples for every comparison
        chunks = []
        for _ in range(10):
            rows = rng.integers(0, n, size=(100, n))
            chunks.append(rel(pred, rows) - rel(ref, rows))
        return np.concatenate(chunks)

    for name, pred in preds.items():
        if name == "reference":
            continue
        diff = paired_diffs(pred)
        point = float(np.sqrt(np.mean((pred - y) ** 2)) / abs(np.mean(y)) - np.sqrt(np.mean((ref - y) ** 2)) / abs(np.mean(y)))
        out[name] = {
            "relative_error_minus_reference": point,
            "bootstrap_ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
            "bootstrap_sd": float(np.std(diff)),
            "fraction_better_than_reference": float(np.mean(diff < 0)),
        }
    return out


def fmt(v, digits=6):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def main() -> None:
    ref_json = json.loads((ROOT / "scoring" / "reference.json").read_text())
    exp_ref = ref_json["experiments"][EXP]
    coefs = exp_ref["metrics"]
    weight = float(exp_ref["weight"])
    records = {c.name: case_record(c) for c in sorted(CASES.iterdir()) if (c / "status.json").exists()}
    if "reference" not in records or not records["reference"]["test"]["metrics"]:
        print("reference case missing or failed; cannot summarize ranks")
        (H / "summary.json").write_text(json.dumps({"records": records}, indent=2, default=str))
        return

    honest = [c for c in HONEST if c in records and records[c]["test"]["metrics"]]
    ranks_honest, stds_honest = rank_deltas(records, honest, coefs, weight)
    passing = [n for n, r in records.items() if r["test"]["metrics"]]
    ranks_all, stds_all = rank_deltas(records, passing, coefs, weight)

    seeds = [c for c in ("reference", "reference_seed_20250218", "reference_seed_20250219") if c in honest]
    seed_stats = {}
    for m in [m for m in coefs if m != "param_count"]:
        vals = [records[c]["test"]["metrics"][m] for c in seeds]
        seed_stats[m] = {"values": vals, "range": float(np.ptp(vals)), "std": float(np.std(vals))}
    seed_rank = [ranks_honest[c] for c in seeds]
    seed_stats["rank_delta_honest"] = {"values": seed_rank, "range": float(np.ptp(seed_rank))}

    try:
        bootstrap = paired_bootstrap(records)
    except Exception as exc:  # noqa: BLE001
        bootstrap = {"error": repr(exc)}

    train_vs_test = {}
    for c in honest:
        tr = records[c]["train"]["metrics"]
        te = records[c]["test"]["metrics"]
        if tr and te:
            train_vs_test[c] = {"train_relative_error": tr["relative_error"], "test_relative_error": te["relative_error"]}

    summary = {
        "rank_formula": "weight * sum_m |coef_m| * dir_m * (x_m - ref_m) / std_cohort(m), nonzero-coefficient metrics",
        "honest_cohort": honest,
        "honest_cohort_std": stds_honest,
        "rank_delta_honest_cohort": ranks_honest,
        "passing_cohort": passing,
        "passing_cohort_std": stds_all,
        "rank_delta_passing_cohort": ranks_all,
        "seed_spread": seed_stats,
        "paired_bootstrap_vs_reference": bootstrap,
        "train_vs_test": train_vs_test,
        "records": records,
    }
    (H / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = ["| case | test exit | train exit | rel. error | MAE | tail RMSE | param count | rank (honest) | rank (passing) | failure |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, rec in records.items():
        met = rec["test"]["metrics"] or {}
        fail = rec["test"]["failures"] or {}
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                rec["test"]["exit"],
                rec["train"]["exit"],
                fmt(met.get("relative_error")),
                fmt(met.get("mae")),
                fmt(met.get("tail_rmse")),
                fmt(met.get("param_count")),
                fmt(ranks_honest.get(name), 4),
                fmt(ranks_all.get(name), 4),
                "; ".join(f"{k}: {v}" for k, v in fail.items())[:160] or ("build error" if rec.get("build_error") else ""),
            )
        )
    base = records["reference"]["test"]["baselines"]
    lines.append("")
    lines.append(f"Frozen baselines in the reference run: {json.dumps(base)}")
    lines.append(f"Seed spread: {json.dumps(seed_stats)}")
    lines.append(f"Paired bootstrap: {json.dumps(bootstrap)}")
    lines.append(f"Train vs test: {json.dumps(train_vs_test)}")
    (H / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
