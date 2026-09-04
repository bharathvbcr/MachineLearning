#!/usr/bin/env bash
# E20: E18's depth arms matched on WALL CLOCK instead of tokens.
#
# E18 (token-matched, 50M) found looping beats its own depth control with
# disjoint intervals -- looped_attn6x2 4.2322 vs attn6 4.2960, looped_attn3x4
# 4.2742 vs attn3 4.4100 -- and that a 6-layer stack looped twice ties a real
# 12-layer stack at 34% fewer parameters.
#
# But every arm there got the same TOKENS, and they do not cost the same. At
# tenancy 3, measured tokens per wall-clock second:
#
#     attention      33458      attn6   52841
#     looped_attn6x2 34692      attn3   74156
#     looped_attn3x4 35308
#
# So under the cost basis practitioners actually pay, attn3 trains on 2.2x the
# tokens `attention` gets and 2.1x what looped_attn3x4 gets. E18's ranking says
# looping wins at equal tokens; this asks whether it survives equal wall clock.
# A reversal here is the paper's §4/§5 claim on a new axis; no reversal is a
# real result too, and a stronger one for looping.
#
# 3 workers is REQUIRED, not a preference: the budgets are sized from E18's
# tenancy-3 rates, and `wallclock_budgets` records that the first attempt at a
# wall-clock suite sized from three-to-a-GPU rates and then ran single-tenant,
# producing a 1.70x spread in the one quantity the suite holds constant. The
# stage guard refuses any other tenancy.
set -u
cd "$HOME/MLSystemsLab" || exit 1
export PATH="$HOME/.local/bin:$PATH"

# Wait on E19's LOG MARKER, never pgrep -- see scripts/overnight.sh.
deadline=$(( $(date +%s) + 8*3600 ))
while ! grep -q "^e19 exit=" nanolab/out/e19.log 2>/dev/null; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "e19 overran 8h; starting anyway"
    break
  fi
  sleep 120
done
echo "e19 done, e20 start $(date -u +%FT%TZ)"

python3 -u -m nanolab.crossover_replicate wcloop32 --workers 3
rc=$?
echo "e20 launch exit=$rc $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "e20 exit=$rc $(date -u +%FT%TZ)  LAUNCH FAILED, no jobs ran"
  exit "$rc"
fi
sleep 60
while pgrep -f "crossover_replicate worker" >/dev/null; do sleep 60; done
echo "e20 workers drained $(date -u +%FT%TZ)"

# wcboard, not `table`: this suite's arms stop at DIFFERENT token counts, so a
# token-grid table would read different amounts of training in a shared column.
# `table` refuses outright on a budget_by_arm suite, which is the correct
# behaviour and the reason this line differs from every other stage script.
python3 -u -m nanolab.crossover_replicate wcboard \
  --out nanolab/out/crossover_wcloop32 2>&1 | tail -30
python3 - <<'PYEOF'
import json
from collections import Counter
q = json.load(open("nanolab/out/crossover_wcloop32/queue.json"))
print("e20 queue:", dict(Counter(j["status"] for j in q["jobs"])),
      "of", len(q["jobs"]))
for j in q["jobs"]:
    if j["status"] != "done":
        print("   NOT DONE:", j["id"], j["status"], j.get("detail", "")[:60])
PYEOF
echo "e20 exit=$? $(date -u +%FT%TZ)"
