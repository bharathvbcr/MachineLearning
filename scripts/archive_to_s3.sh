#!/usr/bin/env bash
# Archive heavy run artifacts to S3. Run ON the GPU box, before you destroy it.
#
#   aws configure                       # you do this; I never handle keys
#   S3=s3://your-bucket/mlsystemslab bash scripts/archive_to_s3.sh
#
# Default skips ckpt.pt: that is resume state (weights + optimizer moments) for
# runs that have already finished, so it is the largest thing here and the least
# useful -- 75.4 GB against 65.6 GB for every weight you would actually reload.
# Pass KEEP_CKPT=1 if you intend to resume these runs rather than read them.
set -euo pipefail
: "${S3:?set S3=s3://bucket/prefix}"
KEEP_CKPT="${KEEP_CKPT:-0}"
SUITES="${SUITES:-crossover50m_swa32 crossover50m_swa2k mqar_e16}"

aws sts get-caller-identity >/dev/null || {
  echo "no AWS credentials on this box -- run 'aws configure' first"; exit 1; }

for suite in $SUITES; do
  d="nanolab/out/$suite"
  [ -d "$d" ] || { echo "skip $suite (absent)"; continue; }
  echo "==> $suite"
  args=(--exclude '*')
  # The JSON goes first and always: an archive of weights with no metrics.jsonl
  # or config.json is a directory of anonymous tensors.
  args+=(--include '*.json' --include '*.jsonl' --include '*.log')
  args+=(--include '*/best.pt' --include '*/final.pt')
  [ "$KEEP_CKPT" = "1" ] && args+=(--include '*/ckpt.pt')
  aws s3 sync "$d" "$S3/$suite" "${args[@]}" --only-show-errors
  echo "    done: $(aws s3 ls --recursive --summarize "$S3/$suite" | tail -2 | tr '\n' ' ')"
done

# A manifest, so the archive can be checked without downloading it.
python3 - "$S3" <<'PY' > /tmp/manifest.json
import hashlib, json, os, sys
from pathlib import Path
out = {"s3": sys.argv[1], "runs": {}}
for suite in Path("nanolab/out").iterdir():
    if not suite.is_dir():
        continue
    for run in sorted(suite.glob("*")):
        m = run / "metrics.jsonl"
        if not m.exists():
            continue
        rec = {}
        for line in m.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") == "done":
                rec = {k: r.get(k) for k in
                       ("best_val", "final_val", "tokens", "elapsed_s", "mean_tok_s")}
        sizes = {p.name: p.stat().st_size for p in run.glob("*.pt")}
        out["runs"][f"{suite.name}/{run.name}"] = {"done": rec, "weights": sizes}
print(json.dumps(out, indent=2, sort_keys=True))
PY
aws s3 cp /tmp/manifest.json "$S3/manifest.json" --only-show-errors
echo "manifest -> $S3/manifest.json  ($(python3 -c "import json;print(len(json.load(open('/tmp/manifest.json'))['runs']))") runs)"
