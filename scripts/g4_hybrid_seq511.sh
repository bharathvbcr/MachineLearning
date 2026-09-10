#!/usr/bin/env bash
# G4: the hybrids on the seq-511 recall cell -- the last open axis from E36.
#
# E36 put all three hybrids inside or above attention's Wilson interval at seq 255,
# which is the condition under which 511 was said to become interesting. The cell
# discriminates: attention itself solves only 8/10 there, against 11/15 at 255, and
# minGRU and GDN are 0/10.
#
# Ten seeds, not fifteen, to match the n already on `mqar_e16_seq511` -- the pairing
# is against that board's attention arm and a ragged n would not pair. Same cell
# parameters as the runs already there (p128, batch 128, 3000 steps, sqrt LR rule);
# block_size is derived as 4*n_pairs-1 = 511. The suite skips runs already recorded.
#
# Pre-registered reading: solve rate against attention's 8/10, paired by seed with an
# exact two-sided sign test, as at 255. Inside attention's interval -> the hybrids
# carry their recall to the longer sequence and section 6's claim extends. Below it,
# with the sign test separating -> the 255 result was sequence-local and section 6
# says so. No prediction is offered either way.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g4
stage_start g4
python3 -u -m nanolab.mqar_suite \
  --out nanolab/out/mqar_e16_seq511 --device cuda \
  --cells 128 --batch 128 --steps 3000 --seeds 10 --lr-rule sqrt \
  --arms hybrid_mingru8_attn4,hybrid_mingru_periodic,hybrid_mingru10_attn2 \
  --workers 4 --gpus 1
rc=$?; echo "g4 run exit=$rc $(date -u +%FT%TZ)"
python3 -m nanolab.mqar_suite board --out nanolab/out/mqar_e16_seq511 2>/dev/null || \
  python3 -c "
import json, collections
rows=[json.loads(l) for l in open('nanolab/out/mqar_e16_seq511/runs.jsonl')]
from nanolab.mqar_suite import arm_of
g=collections.defaultdict(list)
for r in rows: g[arm_of(r)].append(bool(r['solved']))
print('\n=== mqar_e16_seq511 solve rates (arm read from the run NAME) ===')
for a in sorted(g): print('  %-28s %2d/%2d' % (a, sum(g[a]), len(g[a])))"
echo "REMINDER: bash scripts/pull_artifacts.sh mqar_e16_seq511"
stage_end g4 "$rc"
