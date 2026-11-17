#!/usr/bin/env python3
"""Build the submission folder of every audit case of this iteration.

Each case gets ``cases/<name>/submission/`` with exactly the files that ``run_cases.py``
copies into an empty ``outputs/`` before running the unmodified
``scripts/evaluate_train.sh`` and ``scripts/evaluate_test.sh``. Every case submits one
method (``m``) to both experiments - the case's own model in the experiment it probes,
the reference model in the other - so ranks are those of a complete submission.
Exceptions are stated per case (``de_shortcut_de_only`` submits one experiment only).

* ``bin_*``: three_spin_binary models of iteration 11 (code and data unchanged),
  from ``redteam/hardening/cases/<name>/submission``.
* ``de_prev_*``: the previous iteration's double-exchange reference, seeds and DNN
  (``redteam/final/cases``), rescored by the new evaluator: the "before" of the seed fix.
* ``de_*``: this iteration's references, variants and probes.

A case that cannot be built is recorded in ``build_error.txt``.
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
sys.path.insert(0, str(ROOT / "redteam" / "final"))

from data.double_exchange import load_split  # noqa: E402
from method.models_de import EnergyModelDE, build_de_model  # noqa: E402
from probes_de import NaNOutput, SizeLocked, TableEffectiveModel, WrongShape, Zeros, large_tables  # noqa: E402
from probes_de2 import ChebyshevFreeEnergy  # noqa: E402

F = ROOT / "redteam" / "final2"
CASES = F / "cases"
PREV = ROOT / "redteam" / "final" / "cases"
OLD_BIN = ROOT / "redteam" / "hardening" / "cases"
REF_DE = ROOT / "outputs" / "de_efnn.pt"
REF_BIN = ROOT / "outputs" / "binary_efnn.pt"
FROZEN = ROOT / "eval" / "test" / "baselines"

BINARY_CASES = {
    "bin_reference": "reference",
    "bin_better_continued": "better_continued",
    "bin_reference_seed_20250218": "reference_seed_20250218",
    "bin_reference_seed_20250219": "reference_seed_20250219",
    "bin_weaker_20_epochs": "weaker_20_epochs",
    "bin_weaker_dnn_submitted": "weaker_dnn_submitted",
    "bin_shortcut_polynomial_regression": "shortcut_polynomial_regression",
    "bin_shortcut_scaled_0p001": "shortcut_scaled_0p001",
    "bin_shortcut_sidecar_claims": "shortcut_sidecar_claims",
    "bin_shortcut_move_baseline": "shortcut_move_baseline",
}


def fresh(name: str) -> Path:
    sub = CASES / name / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    (CASES / name / "build_error.txt").unlink(missing_ok=True)
    return sub


def save_script(module: torch.nn.Module, path: Path) -> None:
    torch.jit.save(torch.jit.script(module.eval()), str(path))


def need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (training did not finish?)")
    return path


def with_binary_reference(sub: Path) -> None:
    shutil.copy2(need(REF_BIN), sub / "binary_m.pt")


def with_de_reference(sub: Path) -> None:
    shutil.copy2(need(REF_DE), sub / "de_m.pt")


def fit_least_squares(model_cls, args, features_fn) -> torch.nn.Module:
    x, y = load_split("train")
    rows = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 1000):
            rows.append(features_fn(torch.from_numpy(x[i : i + 1000]).double()).numpy())
    a = np.concatenate(rows)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return model_cls(*args, torch.from_numpy(coef))


def build_all() -> dict:
    site_mean = float(np.mean(load_split("train")[1])) / 100.0

    def ref_de():
        return torch.jit.load(str(need(REF_DE)), map_location="cpu")

    def ti227(sub: Path) -> None:
        import ti_expansion as ti

        n_feat = 1 + len(ti.pair_table()) + len(ti.quad_table())
        probe = ti.EffectiveModel(torch.zeros(n_feat, dtype=torch.float64))
        model = fit_least_squares(lambda c: ti.EffectiveModel(c), (), probe.features)
        save_script(model, sub / "de_m.pt")
        with_binary_reference(sub)

    def ti1847(sub: Path) -> None:
        pairs, bonds, quads = large_tables()
        n_feat = 1 + len(pairs) + len(quads)
        probe = TableEffectiveModel(pairs, bonds, quads, torch.zeros(n_feat, dtype=torch.float64))
        model = fit_least_squares(TableEffectiveModel, (pairs, bonds, quads), probe.features)
        save_script(model, sub / "de_m.pt")
        with_binary_reference(sub)
        (CASES / "de_ti_expansion_1847" / "fit.json").write_text(json.dumps({"features": n_feat}))

    def de_file(src: Path):
        def build(sub: Path) -> None:
            shutil.copy2(need(src), sub / "de_m.pt")
            with_binary_reference(sub)

        return build

    def de_module(make):
        def build(sub: Path) -> None:
            save_script(make(), sub / "de_m.pt")
            with_binary_reference(sub)

        return build

    def move_baseline(sub: Path) -> None:
        with_de_reference(sub)
        with_binary_reference(sub)
        save_script(Zeros(), sub / "de_dnn.pt")

    def sidecar(sub: Path) -> None:
        with_de_reference(sub)
        with_binary_reference(sub)
        (sub / "de_m.json").write_text(
            json.dumps({"metrics": {"relative_error_L10": 0.0, "relative_error_L16": 0.0, "relative_error_L24": 0.0, "param_count": 0}})
        )

    def de_only(sub: Path) -> None:
        with_de_reference(sub)

    builders = {
        "de_reference": de_file(REF_DE),
        "de_reference_seed_20250218": de_file(CASES / "de_reference_seed_20250218" / "model.pt"),
        "de_reference_seed_20250219": de_file(CASES / "de_reference_seed_20250219" / "model.pt"),
        "de_better_continued": de_file(CASES / "de_better_continued" / "model.pt"),
        "de_weaker_3_epochs": de_file(CASES / "de_weaker_3_epochs" / "model.pt"),
        "de_dnn_submitted": de_file(FROZEN / "de_dnn.pt"),
        "de_ti_expansion_227": ti227,
        "de_ti_expansion_1847": ti1847,
        "de_prev_reference": de_file(PREV / "de_reference" / "submission" / "de_efnn.pt"),
        "de_prev_seed_20250218": de_file(PREV / "de_reference_seed_20250218" / "submission" / "de_efnn.pt"),
        "de_prev_seed_20250219": de_file(PREV / "de_reference_seed_20250219" / "submission" / "de_efnn.pt"),
        "de_prev_dnn_submitted": de_file(PREV / "de_dnn_submitted" / "submission" / "de_dnnsub.pt"),
        "de_shortcut_exact_diagonalization": de_file(PREV / "de_shortcut_exact_diagonalization" / "submission" / "de_exact.pt"),
        "de_shortcut_chebyshev_free_energy": de_module(lambda: ChebyshevFreeEnergy(700)),
        "de_shortcut_constant_per_site": de_file(PREV / "de_shortcut_constant_per_site" / "submission" / "de_constant.pt"),
        "de_shortcut_size_locked": de_module(lambda: SizeLocked(ref_de())),
        "de_shortcut_nan_output": de_module(lambda: NaNOutput()),
        "de_shortcut_wrong_shape": de_module(lambda: WrongShape(ref_de())),
        "de_shortcut_oversized_c9": de_module(lambda: EnergyModelDE(build_de_model("efnn", 9, 3, 5), site_mean, 0.88)),
        "de_shortcut_move_baseline": move_baseline,
        "de_shortcut_sidecar_claims": sidecar,
        "de_shortcut_de_only": de_only,
    }
    status = {}
    for name, old in BINARY_CASES.items():
        try:
            sub = fresh(name)
            for f in sorted((OLD_BIN / old / "submission").iterdir()):
                target = f.name
                if f.suffix == ".pt" and f.stem not in ("binary_dnn",):
                    target = "binary_m.pt"
                elif f.suffix == ".json":
                    target = "binary_m.json"
                shutil.copy2(f, sub / target)
            with_de_reference(sub)
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
    torch.set_num_threads(1)
    CASES.mkdir(parents=True, exist_ok=True)
    status = build_all()
    for k, v in status.items():
        print(f"{v:6s} {k}", flush=True)


if __name__ == "__main__":
    main()
