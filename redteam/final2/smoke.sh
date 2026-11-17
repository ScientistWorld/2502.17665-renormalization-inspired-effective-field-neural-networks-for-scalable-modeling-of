#!/bin/bash
# Smoke job for this iteration's new code (test node, < 1 hour). Results in redteam/final2/smoke/.
set -uo pipefail
cd /home/user
export PYTHONPATH="/home/user"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
S=redteam/final2/smoke
mkdir -p "$S"
LOG=$S/smoke.log
echo "=== smoke start $(date -Is) nproc=$(nproc) torch=$(python -c 'import torch;print(torch.__version__)')" | tee "$LOG"
(free -g; grep -m1 "model name" /proc/cpuinfo) 2>&1 | tee -a "$LOG"

step() {  # step <name> <command...>
  local name=$1; shift
  local t0=$(date +%s)
  "$@" > "$S/$name.log" 2>&1
  echo "$name exit=$? seconds=$(( $(date +%s) - t0 ))" | tee -a "$LOG"
}

step tile_algebra python redteam/final2/check_tile_algebra.py
step checks python redteam/final2/smoke_checks.py
step train python method/train_double_exchange.py --model efnn --epochs 4 --restarts 2 --select-epochs 2 \
  --train-limit 2000 --threads 1 --output "$S/de_smoke.pt" --checkpoint "$S/de_smoke_ckpt.pt"
step train_rerun python method/train_double_exchange.py --model efnn --epochs 4 --restarts 2 --select-epochs 2 \
  --train-limit 2000 --threads 1 --output "$S/de_smoke.pt" --checkpoint "$S/de_smoke_ckpt.pt"
step train_continue python method/train_double_exchange.py --model efnn --epochs 2 --lr 0.0005 \
  --train-limit 2000 --threads 1 --init-checkpoint "$S/de_smoke_ckpt.pt" --output "$S/de_smoke_cont.pt" --checkpoint "$S/de_smoke_cont_ckpt.pt"
step train_dnn python method/train_double_exchange.py --model dnn --epochs 3 --restarts 2 --select-epochs 1 \
  --train-limit 2000 --threads 1 --output "$S/de_smoke_dnn.pt" --checkpoint "$S/de_smoke_dnn_ckpt.pt"
step tile_fit python baseline/effective_model_de.py --train-limit 3000 --output "$S/de_tile.pt" --report "$S/de_tile.fit.json"
step tile_timing python - << 'PY'
import sys, time
sys.path.insert(0, "/home/user")
import numpy as np
import baseline.effective_model_de as em
from data.double_exchange import load_split
x, y = load_split("train")
t0 = time.time(); xs = [np.ascontiguousarray(v) for v in em.pair_features(x)]; t1 = time.time()
a = np.random.default_rng(0).standard_normal(x.shape[0])
em.kernel_apply(xs, a); t2 = time.time()
em.kernel_apply(xs, a); t3 = time.time()
print(f"features {t1 - t0:.1f} s; kernel product on {x.shape[0]} rows: {t2 - t1:.2f} s, {t3 - t2:.2f} s")
PY

# End-to-end scorers on a temporary outputs directory.
mkdir -p "$S/outputs"
cp outputs/binary_efnn.pt outputs/de_efnn.pt "$S/outputs/"
cp redteam/final2/smoke/de_cheb.pt "$S/outputs/de_cheb.pt" 2>/dev/null || true
cp redteam/final/cases/bin_shortcut_polynomial_regression/submission/binary_submission.pt "$S/outputs/binary_poly.pt"
python - << 'PY'
import sys, torch
sys.path.insert(0, "/home/user/redteam/final")
from probes_de import NaNOutput
torch.jit.save(torch.jit.script(NaNOutput()), "/home/user/redteam/final2/smoke/outputs/de_nan.pt")
PY
step score_test python eval/test/score_all.py --outputs-dir "$S/outputs" --output "$S/scores_test.json"
step score_train python eval/train/score_all.py --outputs-dir "$S/outputs" --output "$S/scores_train.json"
python - << 'PY' 2>&1 | tee -a "$LOG"
import json
for s in ("test", "train"):
    d = json.load(open(f"/home/user/redteam/final2/smoke/scores_{s}.json"))
    print(s, json.dumps({e: b["results"] for e, b in d["experiments"].items()}, indent=1))
    print(s, "failures", json.dumps(d.get("_failures"), indent=1))
    for e, b in d["experiments"].items():
        print(s, e, {k: {kk: vv for kk, vv in v.items() if kk.startswith("compute") or kk.startswith("raw_")} for k, v in b["_artifacts"].items()})
PY
echo "=== smoke done $(date -Is)" | tee -a "$LOG"
