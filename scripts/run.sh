#!/bin/bash
# Temporary audit job entry point; restored before the audit is finalized.
set -euo pipefail
cd /home/user
bash redteam/run_all.sh
