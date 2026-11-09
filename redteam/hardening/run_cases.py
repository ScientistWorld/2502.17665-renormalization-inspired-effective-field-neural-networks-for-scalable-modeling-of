#!/usr/bin/env python3
"""Score every hardening case through the unmodified evaluation scripts.

For each ``cases/<name>/submission/`` the reference ``outputs/`` directory is moved
aside, an empty ``outputs/`` receives exactly the case's files, and
``bash scripts/evaluate_train.sh`` and ``bash scripts/evaluate_test.sh`` run in
separate processes with no arguments. Score files, logs and exit codes are copied
into the case folder. ``outputs/`` and ``scoring/`` are restored afterwards, also
after an interruption (the next run restores first).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "redteam" / "hardening"
CASES = H / "cases"
OUT = ROOT / "outputs"
SCORING = ROOT / "scoring"
OUT_STASH = H / "_stash_outputs"
SCORING_STASH = H / "_stash_scoring"
SCORE_FILES = ("scores.json", "scores_train.json", "scores_test.json")


def restore() -> None:
    if OUT_STASH.exists():
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT_STASH.rename(OUT)
    if SCORING_STASH.exists():
        for name in SCORE_FILES:
            src = SCORING_STASH / name
            if src.exists():
                shutil.copy2(src, SCORING / name)
        shutil.rmtree(SCORING_STASH)


def run_case(case: Path) -> dict:
    OUT.mkdir()
    for f in sorted((case / "submission").iterdir()):
        shutil.copy2(f, OUT / f.name)
    status: dict = {"files": sorted(p.name for p in OUT.iterdir())}
    for slice_name, script in (("train", "scripts/evaluate_train.sh"), ("test", "scripts/evaluate_test.sh")):
        for name in SCORE_FILES:
            (SCORING / name).unlink(missing_ok=True)
        start = time.time()
        proc = subprocess.run(["bash", script], cwd=ROOT, capture_output=True, text=True)
        (case / f"evaluate_{slice_name}.log").write_text(
            proc.stdout[-30000:] + "\n--- stderr ---\n" + proc.stderr[-30000:]
        )
        produced = SCORING / f"scores_{slice_name}.json"
        target = case / f"scores_{slice_name}.json"
        if produced.exists():
            shutil.copy2(produced, target)
        elif target.exists():
            target.unlink()
        status[slice_name] = {
            "command": f"bash {script}",
            "exit": proc.returncode,
            "seconds": round(time.time() - start, 2),
            "scores_written": produced.exists(),
        }
    shutil.rmtree(OUT)
    return status


def main() -> None:
    restore()
    cases = sorted(d for d in CASES.iterdir() if (d / "submission").is_dir())
    SCORING_STASH.mkdir(parents=True, exist_ok=True)
    for name in SCORE_FILES:
        if (SCORING / name).exists():
            shutil.copy2(SCORING / name, SCORING_STASH / name)
    OUT.mkdir(exist_ok=True)
    OUT.rename(OUT_STASH)
    try:
        for case in cases:
            status = run_case(case)
            (case / "status.json").write_text(json.dumps(status, indent=2))
            print(f"{case.name}: train exit={status['train']['exit']} test exit={status['test']['exit']}", flush=True)
    finally:
        if OUT.exists():
            shutil.rmtree(OUT)
        restore()


if __name__ == "__main__":
    main()
