#!/bin/bash
# Job entry point: reproduce the reference end to end from the committed data (train and
# freeze the baselines of both experiments, train the EFNN references, score the visible
# train slice and the hidden test) and validate the workspace.
# This iteration's audit (cases, shortcuts, summary) is redteam/final2/run_final3.sh
# (round 1: redteam/final2/run_final2.sh).
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
bash scripts/reproduce.sh
python validate.py
