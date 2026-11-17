#!/usr/bin/env python3
"""Smoke checks for this iteration's new code paths (small scale, one job).

1. The compute meter sees the operations of TorchScript models: work per site of the
   previous reference, DNN, translation-invariant expansions, and the Chebyshev
   free-energy probe; the meter's exception surfaces as a RunFailure.
2. The Chebyshev probe reproduces exact energies (so without the meter it would score).
3. The tile effective model's dual/primal algebra against an explicit feature matrix.
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

from data.double_exchange import load_split  # noqa: E402
from eval.test import score_double_exchange_slice as sde  # noqa: E402
from probes_de2 import ChebyshevFreeEnergy  # noqa: E402

OUT = ROOT / "redteam" / "final2" / "smoke"
PREV = ROOT / "redteam" / "final" / "cases"


def meter_work(module, x: torch.Tensor) -> tuple[float, int]:
    m = sde.ComputeMeter(float("inf"))
    with torch.no_grad(), torch.jit.optimized_execution(False), m:
        module(x)
    L = x.shape[2]
    return m.total / (x.shape[0] * L * L), m.calls


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"torch": torch.__version__}
    dev = {k: load_split(f"dev_{k}") for k in ("L10", "L16", "L24")}
    models = {
        "prev_reference": ROOT / "outputs" / "de_efnn.pt",
        "prev_dnn": PREV / "de_dnn_submitted" / "submission" / "de_dnnsub.pt",
        "ti_expansion_227": PREV / "de_effective_submitted" / "submission" / "de_effective.pt",
        "ti_expansion_1847": PREV / "de_effective_large" / "submission" / "de_effective_large.pt",
    }
    work: dict = {}
    for name, path in models.items():
        if not path.exists():
            work[name] = f"missing {path}"
            continue
        mod = torch.jit.load(str(path), map_location="cpu").eval()
        row = {}
        for key, (x, _y) in dev.items():
            xb = torch.from_numpy(x[: {"L10": 250, "L16": 50, "L24": 20}[key]])
            t0 = time.time()
            w, calls = meter_work(mod, xb)
            row[key] = {"work_per_site": w, "ops": calls, "seconds": round(time.time() - t0, 3)}
        work[name] = row
        print(name, json.dumps(row), flush=True)
    report["work_per_site"] = work

    # Chebyshev probe: exactness (unmetered) and metered work
    cheb = torch.jit.script(ChebyshevFreeEnergy(700).eval())
    torch.jit.save(cheb, str(OUT / "de_cheb.pt"))
    cheb = torch.jit.load(str(OUT / "de_cheb.pt"))
    exact = {}
    for key, n in (("L10", 6), ("L16", 2)):
        x, y = dev[key]
        t0 = time.time()
        with torch.no_grad():
            p = cheb(torch.from_numpy(x[:n])).double().numpy()
        exact[key] = {"max_abs_error": float(np.max(np.abs(p - y[:n]))), "relative_rmse": float(np.sqrt(np.mean((p - y[:n]) ** 2)) / abs(np.mean(y))), "seconds": round(time.time() - t0, 1)}
        print("chebyshev", key, exact[key], flush=True)
    x10 = torch.from_numpy(dev["L10"][0][:2])
    w, calls = meter_work(cheb, x10)
    exact["work_per_site_L10"] = w
    exact["ops_L10"] = calls
    report["chebyshev_probe"] = exact
    print("chebyshev work/site L10", w, calls, flush=True)

    # The meter's exception must surface as a RunFailure with the budget message.
    sets = {"L10": (10, dev["L10"][0][:250], dev["L10"][1][:250])}
    t0 = time.time()
    try:
        sde.predict(OUT / "de_cheb.pt", sets)
        report["meter_failure"] = "NOT RAISED"
    except sde.RunFailure as exc:
        report["meter_failure"] = str(exc)
    report["meter_failure_seconds"] = round(time.time() - t0, 2)
    print("meter failure:", report["meter_failure"], report["meter_failure_seconds"], flush=True)

    (OUT / "smoke_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
