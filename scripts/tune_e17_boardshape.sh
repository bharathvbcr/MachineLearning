#!/usr/bin/env bash
# E17 at the shape this lab actually trains, which the first measurement was not.
#
# 2026-09-06 benched B8 / T1024 / 12 heads / 4 KV heads / head_dim 32 and found
# the mma path 1.35x on the forward but 0.95x on fwd+bwd -- so, no, for training.
# That is ONE shape and not this one. The boards run batch 32, ctx 512, 12 heads,
# head_dim 64, and n_kv_head=0 means MHA (12 KV heads), not the 4-group GQA that
# was measured. Head count, head width, sequence length and batch all differ, and
# the arithmetic intensity of the backward moves with every one of them.
#
# Waits for tune_program's own completion marker AND an idle device. Not because
# E17 needs a quiet card for correctness, but because the boards do: `workers` is
# a recorded recipe field, and a fourth process on the device would silently make
# every running board a 4-tenancy board whose elapsed_s and tok_s describe a
# recipe its recipe.json does not.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"
LOG=nanolab/out/_program.log

echo "e17_boardshape waiting for tune_program ($(date -u +%FT%TZ))"
if [ "${FOLLOWUP_NOWAIT:-0}" = "1" ]; then
  echo "  (driver already waited)"
else
deadline=$(( $(date +%s) + 20*3600 ))
while ! grep -q "^tune_program exit=" "$LOG" 2>/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "gave up waiting at $(date -u +%FT%TZ)"; exit 1; }
  sleep 120
done
quiet=0
while true; do
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
    quiet=0
  else
    quiet=$((quiet+1)); [ "$quiet" -ge 3 ] && break
  fi
  sleep 20
done
fi
echo "e17_boardshape start $(date -u +%FT%TZ)  ninja=$(command -v ninja || echo MISSING)"

rc=0
echo "=== the shape the boards train: B32 T512 12 heads MHA head_dim 64 ==="
python3 -u -m nanolab.flash_cuda --batch 32 --seq 512 --heads 12 \
  --kv_heads 12 --head_dim 64 --dtype bf16
s=$?; echo "e17bs board exit=$s"; [ "$s" -ne 0 ] && rc=$s

echo "=== and the ctx-2048 board (E15/E9), same width ==="
python3 -u -m nanolab.flash_cuda --batch 8 --seq 2048 --heads 12 \
  --kv_heads 12 --head_dim 64 --dtype bf16
s=$?; echo "e17bs ctx2048 exit=$s"; [ "$s" -ne 0 ] && rc=$s

echo "=== the 2026-09-06 shape again, as a within-session control ==="
python3 -u -m nanolab.flash_cuda --batch 8 --seq 1024 --heads 12 \
  --kv_heads 4 --head_dim 32 --dtype bf16
s=$?; echo "e17bs control exit=$s"; [ "$s" -ne 0 ] && rc=$s

echo "e17_boardshape exit=$rc $(date -u +%FT%TZ)"
