#!/bin/bash
# Job entry point. Final state: reproduce the reference end to end (scripts/reproduce.sh)
# and validate. Temporarily: this iteration's audit (redteam/final2/run_final2.sh), which
# trains the double-exchange reference and baselines from the final code and scores every case.
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
bash redteam/final2/run_final2.sh
