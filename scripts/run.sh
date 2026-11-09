#!/bin/bash
# Job entry point (temporary, final-iteration audit): train the double-exchange
# reference, baselines and variants, score every audit case through the unmodified
# evaluators, regenerate the official score files, summarize and validate.
# Restored to the end-to-end reproduction (scripts/reproduce.sh) at the end of the iteration.
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
bash redteam/final/run_final.sh
