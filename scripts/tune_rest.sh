#!/usr/bin/env bash
# Everything after the tenancy sweeps, in one script rather than a chain of
# tmux sessions waiting on each other's log markers (two of those died to
# quoting, silently, which is exactly the failure mode the marker convention is
# meant to prevent).
#
# Ordered by value, so a truncated run still leaves the important results:
#   1 probes    sampler VRAM hazard, GDN chunk width, eval sync, MoE on real data
#   2 w1536     cost rows for the arms the interrupted E27 ladder still needs
#   3 tenancy+  w384 (E35b is the costliest board), MoE, pure minGRU
#   4 mqar      tenancy at the RECALL shape, which E28/E32 run at
#   5 compile   what torch.compile would buy and the numerics it would cost
#   6 validate  does tenancy/MPS change the loss curve (the safety gate)
#   7 reversal  tenancy descending, to exclude ordering effects
#   8 evaliters how much of a board's reported number is eval noise
set -u
cd ~/MLSystemsLab
export PATH=$HOME/.local/bin:$PATH
OUT=nanolab/out/_tune
mkdir -p "$OUT"

step() {                 # $1=name  $2=logfile ; rest = command
  local name="$1" log="$2"; shift 2
  echo "===== $name $(date -u +%FT%TZ) ====="
  "$@" 2>&1 | tee "$log"
  echo "$name exit=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
}

step probes "$OUT/probes.log" \
  python3 -u scripts/tune_probes.py datapath gdnchunk evalsync moereal

step w1536 "$OUT/w1536.log" \
  python3 -u -m nanolab.sweep_gpu arm \
    --arms w1536_attention_lr80,w1536_mingru_lr20,w1536_mingru_lr40,w1536_mingru_lr80 \
    --batch_size 32 --block_size 512 --peak_flops 752.8e12 --iters 15 \
    --mem_fraction 0 --grad_checkpoint false --out "$OUT/arm_cost_w1536.json"

echo "===== tenancy_extra $(date -u +%FT%TZ) ====="
for a in w384_attention_lr80 w384_mingru_lr40 moe_e8k1_raw mingru; do
  echo "--- tenancy[nomps] $a ---"
  python3 -u scripts/tune_tenancy.py --arm "$a" --tenancies 1,2,3,4 \
    --seconds 45 --out "$OUT/tenancy_nomps_$a.json"
done 2>&1 | tee "$OUT/tenancy_extra.log"
echo "tenancy_extra exit=0 $(date -u +%FT%TZ)"

step mqar "$OUT/mqar_tenancy.log" bash scripts/tune_mqar_tenancy.sh

step compile "$OUT/compile.log" \
  python3 -u scripts/tune_compile.py attention mingru hybrid_mingru8_attn4 gdn moe_e8k1

step validate "$OUT/validate.log" bash scripts/tune_validate.sh

step reversal "$OUT/reversal.log" \
  python3 -u scripts/tune_tenancy.py --arm attention --tenancies 6,4,3,2,1 \
    --seconds 45 --out "$OUT/tenancy_reversal_attention.json"

step evaliters "$OUT/evaliters.log" bash scripts/tune_stage9.sh

echo "tune_rest exit=0 $(date -u +%FT%TZ)"
