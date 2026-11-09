#!/bin/bash
# Smoke test of the double-exchange pipeline (one-hour test node, CPU).
#   1. ED timing per lattice size, then build the public energies and the hidden test
#      instance (eval/test/build_double_exchange.py).
#   2. Fit and freeze the effective-model baseline (deterministic least squares).
#   3. Time one training epoch of EFNN (2 threads, 1 thread) and the DNN on the full
#      training split; export 1-epoch models to tmp/smoke/outputs.
#   4. Cross-check ED, the forbidden-operation defence and archive sizes.
#   5. Run both combined scorers on tmp/smoke/outputs (plus the binary reference).

set -uo pipefail
cd /home/user
export PYTHONPATH="/home/user"
S=tmp/smoke
mkdir -p $S/outputs
LOG=$S/smoke.log
echo "=== smoke start $(date -Is) nproc=$(nproc) torch=$(python -c 'import torch;print(torch.__version__, torch.get_num_threads())')" | tee -a $LOG

python eval/test/build_double_exchange.py --only-check 2>&1 | tee -a $LOG
if [ ! -f eval/test/double_exchange_instance.json ] || [ ! -f data/double_exchange/meta.json ]; then
  start=$(date +%s)
  python eval/test/build_double_exchange.py --workers "$(nproc)" 2>&1 | tee -a $LOG
  echo "build exit=${PIPESTATUS[0]} seconds=$(( $(date +%s) - start ))" | tee -a $LOG
fi

start=$(date +%s)
python baseline/effective_model_de.py --threads 2 --output checkpoints/baselines/de_effective_model.pt 2>&1 | tee -a $LOG
echo "effective model exit=${PIPESTATUS[0]} seconds=$(( $(date +%s) - start ))" | tee -a $LOG
for d in eval/train/baselines eval/test/baselines; do cp checkpoints/baselines/de_effective_model.pt $d/de_effective_model.pt; done

for cfg in "efnn 2" "efnn 1" "dnn 2"; do
  set -- $cfg
  start=$(date +%s)
  python method/train_double_exchange.py --model "$1" --epochs 1 --threads "$2" \
    --output "$S/outputs/de_smoke_${1}_t$2.pt" --progress "$S/progress_${1}_t$2.pt" 2>&1 | tee -a $LOG
  echo "train $1 threads=$2 exit=${PIPESTATUS[0]} seconds=$(( $(date +%s) - start ))" | tee -a $LOG
done
rm -f $S/outputs/*.json

python redteam/final/smoke_checks.py 2>&1 | tee -a $LOG

cp outputs/binary_efnn.pt $S/outputs/binary_efnn.pt
for slice in train test; do
  start=$(date +%s)
  python eval/$slice/score_all.py --outputs-dir $S/outputs --output $S/scores_$slice.json > $S/score_$slice.log 2>&1
  echo "score_all $slice exit=$? seconds=$(( $(date +%s) - start ))" | tee -a $LOG
  tail -5 $S/score_$slice.log | tee -a $LOG
done
python - <<'EOF' 2>&1 | tee -a tmp/smoke/smoke.log
import json
for s in ("train", "test"):
    d = json.load(open(f"tmp/smoke/scores_{s}.json"))
    for exp, blk in d["experiments"].items():
        print(s, exp, json.dumps(blk["results"], indent=None))
    print(s, "failures", d.get("_failures"))
EOF
python validate.py > $S/validate.log 2>&1; echo "validate exit=$?" | tee -a $LOG
tail -30 $S/validate.log | tee -a $LOG
echo "=== smoke done $(date -Is)" | tee -a $LOG
