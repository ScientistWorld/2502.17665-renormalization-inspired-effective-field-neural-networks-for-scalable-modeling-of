#!/bin/bash
# Job entry point: reproduce the reference end to end from the committed data
# (train and freeze the DNN baseline, train the EFNN reference, score the visible
# train slice and the hidden test) and validate the workspace.
# The iteration-11 hardening audit is redteam/hardening/run_hardening.sh.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

bash scripts/reproduce.sh
python validate.py
