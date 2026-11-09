#!/usr/bin/env python3
"""Smoke checks for the double-exchange pipeline (audit only; run inside the container).

* cross-checks the builder's numpy ED against an independent torch ED (ExactED probe);
* confirms the hidden scorer rejects that probe and accepts honest models;
* prints archive sizes of the honest models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.double_exchange import load_split  # noqa: E402
from eval.test import score_double_exchange_slice as de_test  # noqa: E402
from eval.test.score_three_spin_slice import RunFailure  # noqa: E402
from probes_de import ExactED  # noqa: E402

SMOKE = ROOT / "tmp" / "smoke"


def main() -> None:
    out = SMOKE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    ed = torch.jit.script(ExactED())
    path = SMOKE / "de_exact_ed.pt"
    torch.jit.save(ed, str(path))
    report: dict = {}
    for split, n in (("dev_L10", 64), ("dev_L16", 8)):
        x, y = load_split(split)
        with torch.no_grad():
            p = ed(torch.from_numpy(x[:n])).double().numpy()
        report[f"exact_ed_vs_labels_{split}"] = {
            "max_abs_diff": float(np.max(np.abs(p - y[:n]))),
            "relative_error": float(np.sqrt(np.mean((p - y[:n]) ** 2)) / abs(np.mean(y[:n]))),
        }
    try:
        de_test.score_model(path, {"L10": (10, *load_split("dev_L10"))})
        report["exact_ed_scored"] = "ACCEPTED (defense missing!)"
    except RunFailure as exc:
        report["exact_ed_scored"] = f"rejected: {exc}"
    for f in sorted(out.glob("*.pt")):
        report[f"archive_{f.name}"] = de_test.inspect_archive(f)
        report[f"forbidden_{f.name}"] = de_test.forbidden_code(f)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
