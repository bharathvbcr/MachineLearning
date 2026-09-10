#!/usr/bin/env bash
# G5: are Mamba-2, MLA and GDN read at their own 50M optimum, or not?
#
# `crossover50m` put Mamba-2 at +0.4348 and MLA at +0.3843 against attention, at ONE
# shared learning rate. By section 4's own argument those are not clean comparisons --
# and the paper says so. The plan of record recommended against fixing it, costing the
# fix as "roughly the ladder again, ~66 runs, ~$25". That costing assumed a 10M probe,
# which G7 and G3 have since invalidated; and it assumed the answer.
#
# So probe before re-laddering, the same way G8a precedes G9. If these families share
# attention's 50M argmin the way all three section-5 shapes did, then `crossover50m`
# read them at it already and the objection dissolves for the price of this board. If
# they do not, THAT is the finding, and the re-run is scoped by a measurement.
#
# Grows `crossover_probe50m` so the family argmins sit under one recipe, one tenancy
# and one seed with attention's and minGRU's -- the comparison is a table lookup, not
# a cross-board join. 18 jobs, n=1 at seed 1337.
#
# Pre-registered reading: per family, the argmin multiplier and whether it is interior.
# Same as attention's -> the crossover50m rows stand and section 4's caveat on them is
# narrowed to "shared LR, but at the shared optimum". Different -> the row is withdrawn
# to a caveat and the re-run is costed from the gap actually measured, not from a guess.
source "$(dirname "$0")/_stage_common.sh"
stage_wait || stage_refused g5
stage_start g5
BOARD=$(python3 -c "import json;print(','.join(json.load(open('nanolab/out/crossover_probe50m/recipe.json'))['arms']))")
G5=$(python3 -c "from nanolab.crossover_replicate import G5_ARMS; print(','.join(G5_ARMS))") || exit 1
export CROSSOVER_ARMS="$BOARD,$G5"
export CROSSOVER_JOB_PREFIX=cx32p50
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover_probe50m --workers 3 --seed 1337
rc=$?; echo "g5 launch exit=$rc $(date -u +%FT%TZ)"
python3 scripts/lr_argmin.py crossover_probe50m --compare crossover_ladder_probe
echo "REMINDER: bash scripts/pull_artifacts.sh crossover_probe50m"
stage_end g5 "$rc"
