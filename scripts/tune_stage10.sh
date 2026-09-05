#!/usr/bin/env bash
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
python3 -u scripts/tune_fusedce.py 2>&1 | tee nanolab/out/_tune/fusedce.log
rc=${PIPESTATUS[0]}
# Marker must land IN the log the next stage greps, not just on stdout.
echo "stage10 exit=$rc $(date -u +%FT%TZ)" | tee -a nanolab/out/_tune/fusedce.log
