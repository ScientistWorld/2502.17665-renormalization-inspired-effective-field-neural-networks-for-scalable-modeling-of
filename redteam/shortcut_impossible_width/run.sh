#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python redteam/run_case.py shortcut_impossible_width
