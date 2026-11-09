#!/usr/bin/env python3
"""Build the submission folder of every final-audit case (both experiments).

Each case gets ``cases/<name>/submission/`` holding exactly the files that
``run_cases.py`` copies into an empty ``outputs/`` before running the unmodified
``scripts/evaluate_train.sh`` and ``scripts/evaluate_test.sh``.

* ``bin_*``: three_spin_binary cases, reusing the iteration-11 models in
  ``redteam/hardening/cases/<name>/submission`` (binary code and data are unchanged).
* ``de_*``: double_exchange honest variants (trained by ``run_final.sh``) and probes
  built here. A probe that cannot be built is recorded in ``build_error.txt``.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.double_exchange import load_split  # noqa: E402
from method.models_de import EnergyModelDE, build_de_model  # noqa: E402
from probes_de import (  # noqa: E402
    ConstantPerSite,
    ExactED,
    NaNOutput,
    SizeLocked,
    TableEffectiveModel,
    WrongShape,
    Zeros,
    large_tables,
)

F = ROOT / "redteam" / "final"
CASES = F / "cases"
OLD = ROOT / "redteam" / "hardening" / "cases"
REF_DE = ROOT / "outputs" / "de_efnn.pt"
FROZEN = ROOT / "eval" / "test" / "baselines"

BINARY_CASES = [
    "reference",
    "better_continued",
    "reference_seed_20250218",
    "reference_seed_20250219",
    "weaker_20_epochs",
    "weaker_dnn_submitted",
    "shortcut_polynomial_regression",
    "shortcut_sidecar_claims",
    "shortcut_scaled_0p001",
    "shortcut_move_baseline",
]


def fresh(name: str) -> Path:
    sub = CASES / name / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    (CASES / name / "build_error.txt").unlink(missing_ok=True)
    return sub


def save_script(module: torch.nn.Module, path: Path) -> None:
    torch.jit.save(torch.jit.script(module.eval()), str(path))


def honest(name: str, model: Path) -> None:
    if not model.exists():
        raise FileNotFoundError(f"{model} missing (training did not finish)")
    shutil.copy2(model, fresh(name) / "de_efnn.pt")


def build_all() -> dict:
    ref = torch.jit.load(str(REF_DE), map_location="cpu") if REF_DE.exists() else None
    site_mean = float(np.mean(load_split("train")[1])) / 100.0

    def need_ref():
        if ref is None:
            raise FileNotFoundError("outputs/de_efnn.pt missing")
        return ref

    def large_effective(sub: Path) -> None:
        pairs, bonds, quads = large_tables()
        n_feat = 1 + len(pairs) + len(quads)
        probe = TableEffectiveModel(pairs, bonds, quads, torch.zeros(n_feat, dtype=torch.float64))
        x, y = load_split("train")
        rows = []
        with torch.no_grad():
            for i in range(0, x.shape[0], 1000):
                rows.append(probe.features(torch.from_numpy(x[i : i + 1000]).double()).numpy())
        a = np.concatenate(rows)
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        save_script(TableEffectiveModel(pairs, bonds, quads, torch.from_numpy(coef)), sub / "de_effective_large.pt")
        (CASES / "de_effective_large" / "fit.json").write_text(json.dumps({"features": n_feat}))

    builders = {
        "de_reference": lambda sub: shutil.copy2(REF_DE, sub / "de_efnn.pt"),
        "de_dnn_submitted": lambda sub: shutil.copy2(FROZEN / "de_dnn.pt", sub / "de_dnnsub.pt"),
        "de_effective_submitted": lambda sub: shutil.copy2(FROZEN / "de_effective_model.pt", sub / "de_effective.pt"),
        "de_effective_large": large_effective,
        "de_shortcut_exact_diagonalization": lambda sub: save_script(ExactED(), sub / "de_exact.pt"),
        "de_shortcut_constant_per_site": lambda sub: save_script(ConstantPerSite(site_mean), sub / "de_constant.pt"),
        "de_shortcut_size_locked": lambda sub: save_script(SizeLocked(need_ref()), sub / "de_locked.pt"),
        "de_shortcut_nan_output": lambda sub: save_script(NaNOutput(), sub / "de_nan.pt"),
        "de_shortcut_wrong_shape": lambda sub: save_script(WrongShape(need_ref()), sub / "de_wrong.pt"),
        "de_shortcut_oversized_c9": lambda sub: save_script(
            EnergyModelDE(build_de_model("efnn", 9, 3, 5), site_mean, 0.88), sub / "de_big.pt"
        ),
        "de_shortcut_move_baseline": lambda sub: (shutil.copy2(REF_DE, sub / "de_efnn.pt"), save_script(Zeros(), sub / "de_dnn.pt")),
        "de_shortcut_sidecar_claims": lambda sub: (
            shutil.copy2(REF_DE, sub / "de_efnn.pt"),
            (sub / "de_efnn.json").write_text(
                json.dumps({"metrics": {"relative_error_L10": 0.0, "relative_error_L16": 0.0, "relative_error_L24": 0.0, "param_count": 0}})
            ),
        ),
    }
    status = {}
    for name in BINARY_CASES:
        src = OLD / name / "submission"
        try:
            sub = fresh(f"bin_{name}")
            for f in sorted(src.iterdir()):
                shutil.copy2(f, sub / f.name)
            status[f"bin_{name}"] = "built"
        except Exception:  # noqa: BLE001
            (CASES / f"bin_{name}" / "build_error.txt").write_text(traceback.format_exc())
            status[f"bin_{name}"] = "FAILED"
    for name, model in (
        ("de_better_continued", CASES / "de_better_continued" / "model.pt"),
        ("de_reference_seed_20250218", CASES / "de_reference_seed_20250218" / "model.pt"),
        ("de_reference_seed_20250219", CASES / "de_reference_seed_20250219" / "model.pt"),
        ("de_weaker_3_epochs", CASES / "de_weaker_3_epochs" / "model.pt"),
    ):
        try:
            honest(name, model)
            status[name] = "built"
        except Exception:  # noqa: BLE001
            (CASES / name / "build_error.txt").write_text(traceback.format_exc())
            status[name] = "FAILED"
    for name, fn in builders.items():
        try:
            fn(fresh(name))
            status[name] = "built"
        except Exception:  # noqa: BLE001
            (CASES / name / "build_error.txt").write_text(traceback.format_exc())
            shutil.rmtree(CASES / name / "submission", ignore_errors=True)
            status[name] = "FAILED"
    return status


def main() -> None:
    torch.set_num_threads(2)
    CASES.mkdir(parents=True, exist_ok=True)
    status = build_all()
    for k, v in status.items():
        print(f"{v:6s} {k}", flush=True)


if __name__ == "__main__":
    main()
