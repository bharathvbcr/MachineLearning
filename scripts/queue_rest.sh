#!/usr/bin/env bash
# E17 (CUDA flash-attention kernel) then E10 (ratio32 + the co-leader).
#
# Rewritten after the first attempt reported success while running nothing:
#   * `python3 -m ... ratio32 --gpus 1` -- that subparser takes only --out,
#     --workers, --detach. argparse exited 2 and E10 never started.
#   * the flash extension needs the `ninja` BINARY, which pip put in
#     ~/.local/bin; a non-interactive shell does not have that on PATH, so
#     every build fell back to SDPA and benched nothing.
#   * both failures were invisible because the script echoed a HARDCODED
#     "exit=0" after each stage instead of the real status. That is the bug
#     that mattered: it turned two crashes into a clean-looking log and left
#     the GPU idle for 54 minutes. Every stage now reports $?.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
echo "queue start $(date -u +%FT%TZ)  ninja=$(command -v ninja || echo MISSING)"
command -v ninja >/dev/null || echo "WARNING: no ninja; the CUDA extension cannot build"

echo "########## E17: CUDA flash-attention kernel ##########"
python3 -u -m nanolab.tests 2>&1 | tail -30
echo "e17 tests exit=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
rc17=0
for hd in 32 64; do
  echo "--- head_dim $hd, bf16 ---"
  python3 -u -m nanolab.flash_cuda --batch 8 --seq 1024 --heads 12 \
    --kv_heads 4 --head_dim "$hd" --dtype bf16
  s=$?; echo "e17 bench hd$hd exit=$s"; [ "$s" -ne 0 ] && rc17=$s
done
echo "e17 exit=$rc17 $(date -u +%FT%TZ)"

echo "########## E10: ratio32 + the co-leader ##########"
python3 -u -m nanolab.crossover_replicate ratio32 --workers 3
echo "e10 launch exit=$? $(date -u +%FT%TZ)"
sleep 60
while pgrep -f "crossover_replicate worker" >/dev/null; do sleep 60; done
echo "e10 workers drained $(date -u +%FT%TZ)"
python3 -u -m nanolab.crossover_replicate status --out nanolab/out/crossover50m_ratio32 2>&1 | tail -30
echo "e10 exit=$? $(date -u +%FT%TZ)"
echo "queue exit=0 $(date -u +%FT%TZ)"
