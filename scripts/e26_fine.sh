set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
echo "e26 start $(date -u +%FT%TZ)"
python3 -u scripts/gpu_bundle.py --only e1_mup_basin_fine_spattn --workers 1
rc=$?
echo "--- e1_mup_basin_fine_spattn exit=$rc $(date -u +%FT%TZ)"
python3 -u scripts/gpu_bundle.py --analyse 2>&1 | tail -40
echo "e26 exit=$rc $(date -u +%FT%TZ)"
