#!/bin/bash
# Job entry point: reproduce the reference end to end from the final code
# (baseline + EFNN), train the honest variants, build and score the shortcut
# probes through the unmodified evaluators, regenerate the official score files
# from the reference outputs, and validate. See redteam/hardening/run_hardening.sh.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

bash redteam/hardening/run_hardening.sh
