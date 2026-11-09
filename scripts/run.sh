#!/bin/bash
# Job entry point (temporary, final-iteration audit): smoke test of the new
# double-exchange experiment. Restored to the end-to-end reproduction at the end of
# the iteration.
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
bash redteam/final/smoke.sh
