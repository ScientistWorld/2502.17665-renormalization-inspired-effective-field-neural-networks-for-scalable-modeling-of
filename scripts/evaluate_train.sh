#!/bin/bash
# Score every submitted model (outputs/binary_<name>.pt, outputs/de_<name>.pt) on the
# visible train slice (public dev configurations) and write scoring/scores_train.json.
# Takes no arguments.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

python eval/train/score_all.py "$@" --output scoring/scores_train.json
