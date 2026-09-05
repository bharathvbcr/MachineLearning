#!/usr/bin/env bash
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
python3 -u scripts/tune_fusedce.py 2>&1 | tee nanolab/out/_tune/fusedce.log
echo "stage10 exit=$? $(date -u +%FT%TZ)"
