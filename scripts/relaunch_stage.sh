#!/usr/bin/env bash
# Relaunch one stage without destroying what the last attempt recorded.
#
#   bash scripts/relaunch_stage.sh g5 g4 [PREV_HOURS]
#
# Why this exists. On 2026-09-10, g4, g5 and g6 were relaunched by hand with
# `nohup ... > nanolab/out/g4.log`, which TRUNCATES. Those three logs held the
# only copy of the 2026-09-09 cascade: g4's CUDA OOM traceback naming the four
# foreign processes and `220.00 MiB free`, and g5's 18 arms failing with
# `CUBLAS_STATUS_ALLOC_FAILED`. The findings survived only because they had
# already been quoted into docs/GAP_PLAN_2026-09-07.md section 6 an hour earlier.
# The raw logs did not. That is the same rule g8c states and follows -- the
# corrupted records are the evidence for the defect, and overwriting evidence to
# make a caveat go away is not the same as measuring it away.
#
# Appending instead of truncating would be worse, not better: `stage_wait` greps
# `^<stage> exit=` out of `nanolab/out/<stage>.log`, so a retained marker from a
# previous attempt releases the next stage the instant it starts waiting. The old
# log has to move OUT of that filename, which is what this does.
set -u
cd "$HOME/MLSystemsLab" || exit 1
NAME="${1:?usage: relaunch_stage.sh NAME [PREV] [PREV_HOURS]}"
PREV_ARG="${2:-}"
HOURS="${3:-12}"
SCRIPT=$(ls scripts/"${NAME}"_*.sh 2>/dev/null | head -1)
[ -n "$SCRIPT" ] || { echo "no scripts/${NAME}_*.sh"; exit 1; }
LOG="nanolab/out/${NAME}.log"

if [ -s "$LOG" ]; then
  KEEP="nanolab/out/${NAME}.$(date -u +%Y%m%dT%H%M%SZ).log"
  mv "$LOG" "$KEEP" || exit 1
  echo "rolled $(wc -c < "$KEEP" | tr -d ' ') bytes of the previous attempt to $KEEP"
elif [ -e "$LOG" ]; then
  echo "previous $LOG was empty; nothing to preserve"
fi

# shellcheck disable=SC2086
nohup env ${PREV_ARG:+PREV=$PREV_ARG} PREV_HOURS="$HOURS" bash "$SCRIPT" > "$LOG" 2>&1 &
echo "$NAME pid $! script=$SCRIPT prev=${PREV_ARG:-<none>} prev_hours=$HOURS"
