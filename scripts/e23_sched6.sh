set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
echo "e23 start $(date -u +%FT%TZ)"
rc_any=0
for suite in e1_sp_sched6 e1_mup_sched6_spattn; do
  echo "--- $suite start $(date -u +%FT%TZ)"
  python3 -u scripts/gpu_bundle.py --only "$suite" --workers 1
  rc=$?; echo "--- $suite exit=$rc $(date -u +%FT%TZ)"
  [ "$rc" -ne 0 ] && rc_any=$rc
done
python3 -u scripts/gpu_bundle.py --analyse 2>&1 | tail -50
echo "e23 exit=$rc_any $(date -u +%FT%TZ)"
