#!/bin/bash
# Job entry point: reproduce the reference end to end from the committed data (train
# and freeze the baselines of both experiments, train the EFNN references, score the
# visible train slice and the hidden test) and validate the workspace.
# The final-iteration audit is redteam/final/run_final.sh; the iteration-11 audit is
# redteam/hardening/run_hardening.sh.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

bash scripts/reproduce.sh
python validate.py
