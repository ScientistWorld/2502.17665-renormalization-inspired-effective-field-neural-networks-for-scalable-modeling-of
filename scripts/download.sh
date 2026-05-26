#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# No external datasets or pretrained weights are needed for this reproduction.
# The benchmark data are generated deterministically by tracked source code.

set -e

cd /home/user

mkdir -p data/downloads
echo "No external downloads are required for the three-spin reproduction; data are generated deterministically from data/three_spin.py."
