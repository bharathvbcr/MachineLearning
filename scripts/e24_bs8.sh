set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
echo "e24 start $(date -u +%FT%TZ)"
python3 -u scripts/gpu_bundle.py --only e1_sp_bs8_tuned --workers 1
rc=$?
echo "--- e1_sp_bs8_tuned exit=$rc $(date -u +%FT%TZ)"
python3 -u scripts/gpu_bundle.py --analyse 2>&1 | tail -40
echo "e24 exit=$rc $(date -u +%FT%TZ)"
