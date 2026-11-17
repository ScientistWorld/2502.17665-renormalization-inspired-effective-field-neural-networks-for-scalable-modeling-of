#!/bin/bash
# Job entry point. Final state: reproduce the reference end to end (scripts/reproduce.sh)
# and validate. Temporarily: this iteration's smoke checks (redteam/final2/smoke.sh).
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
bash redteam/final2/smoke.sh
