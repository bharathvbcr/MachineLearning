#!/usr/bin/env bash
# E17 says in as many words: "No throughput figure for it exists, and this
# entry is the reason none should be quoted." One shape is not a figure.
# This is the shape sweep, plus the live CUDA correctness tests -- two checks
# in nanolab/tests.py skip for want of a CUDA device and this is the device.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
echo "e17 sweep start $(date -u +%FT%TZ)  ninja=$(command -v ninja || echo MISSING)"

echo "########## live CUDA correctness tests ##########"
python3 -u -m nanolab.tests 2>&1 | grep -Ei "flash|PASS|FAIL|SKIP|passed" | tail -20
echo "tests exit=${PIPESTATUS[0]}"

echo "########## shape sweep ##########"
rc=0
for dtype in bf16 fp16; do
  for hd in 32 64 128; do
    for seq in 1024 2048 4096; do
      echo "--- dtype=$dtype head_dim=$hd seq=$seq ---"
      python3 -u -m nanolab.flash_cuda --batch 4 --seq "$seq" --heads 12 \
        --kv_heads 4 --head_dim "$hd" --dtype "$dtype" --iters 20
      s=$?; [ "$s" -ne 0 ] && { echo "  (exit $s)"; rc=$s; }
    done
  done
done
echo "e17sweep exit=$rc $(date -u +%FT%TZ)"
