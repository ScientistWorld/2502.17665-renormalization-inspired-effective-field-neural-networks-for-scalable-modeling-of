#!/bin/bash
# Score every submitted model on the hidden test slice and write scoring/scores_test.json.
# Takes no arguments. Experiments and submission files (TorchScript energy models):
#   three_spin_binary: outputs/binary_<name>.pt - all 15-site configurations absent from the
#                      public files, labelled at test time from the hidden coupling instance;
#   double_exchange:   outputs/de_<name>.pt     - hidden random spin configurations on 10x10,
#                      16x16 and 24x24 periodic lattices with exact energies.
# Frozen baselines in eval/test/baselines/ are scored alongside.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

python eval/test/score_all.py "$@" --output scoring/scores_test.json
