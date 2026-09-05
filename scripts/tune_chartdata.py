"""Emit the tenancy curves as one JSON blob for the report page's chart.

Keeps the published figure derived from the same run records as the tables, so a
number cannot be right in one and stale in the other.

    python scripts/tune_chartdata.py > docs/gpu-tuning-2026-09-05/evidence/chartdata.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TUNE = REPO / "nanolab/out/_tune"
COST = TUNE / "arm_cost_t1.json"


def series(tag: str, arm: str):
    f = TUNE / f"tenancy_{tag}_{arm}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    pts = []
    for t in sorted(d["rows"], key=int):
        r = d["rows"][t]
        pts.append({"t": int(t),
                    "tok_s": r["agg_tok_s"] if r.get("ok") else None,
                    "gb": r.get("sum_reserved_gb") if r.get("ok") else None,
                    "oom": not r.get("ok")})
    return pts


def main() -> None:
    cost = json.loads(COST.read_text())["arm"] if COST.exists() else {}
    arms = sorted({f.name.split("_", 2)[2][:-len(".json")]
                   for f in TUNE.glob("tenancy_*_*.json")
                   if not f.name.startswith("tenancy_reversal")})
    out = {"arms": [], "vram_total_gib": 94.5}
    for arm in arms:
        row = {"arm": arm,
               "mfu": cost.get(arm, {}).get("mfu"),
               "ms_per_step": cost.get(arm, {}).get("ms_per_step"),
               "nomps": series("nomps", arm),
               "mps": series("mps", arm)}
        if row["nomps"] or row["mps"]:
            out["arms"].append(row)
    rev = TUNE / "tenancy_reversal_attention.json"
    if rev.exists():
        out["reversal_attention"] = series("reversal", "attention")
    json.dump(out, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
