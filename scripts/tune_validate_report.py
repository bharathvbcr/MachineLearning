"""Compare loss trajectories across the tune_validate.sh conditions.

Reads `metrics.jsonl` from each condition directory and, per (arm, seed), diffs
the train-loss and grad-norm sequences against the a1 reference. a2 is the same
configuration as a1, so max|a1-a2| is the run-to-run noise floor: a difference
between a1 and t3/m3 only means something if it exceeds it.

    python scripts/tune_validate_report.py --root nanolab/out/_tune
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run_dir: Path) -> dict:
    tr, ev = {}, {}
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event") == "train":
            tr[r["step"]] = (r["loss"], r["grad_norm"])
        elif r.get("event") == "eval":
            ev[r["step"]] = r["val_loss"]
    return {"train": tr, "eval": ev}


def key(name: str) -> str:
    """`val<tag>_<arm>_s<seed>` -> `<arm>_s<seed>`, so conditions line up."""
    return name.split("_", 1)[1] if "_" in name else name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="nanolab/out/_tune")
    ap.add_argument("--conds", default="a1,a2,t3,m3")
    args = ap.parse_args()
    root = Path(args.root)
    conds = args.conds.split(",")

    data = {}
    for c in conds:
        d = root / f"val_{c}"
        if not d.exists():
            print(f"  (missing {d})")
            continue
        data[c] = {key(p.name): load(p) for p in sorted(d.iterdir())
                   if p.is_dir() and (p / "metrics.jsonl").exists()}

    ref = conds[0]
    if ref not in data:
        raise SystemExit(f"no reference condition {ref}")
    runs = sorted(data[ref])
    print(f"reference: {ref}   runs: {len(runs)}\n")
    print(f"{'condition':<10}{'run':<34}{'n':>5}{'max|dloss|':>12}{'max|dgnorm|':>13}{'max|dval|':>11}  verdict")
    print("-" * 96)
    summary = {}
    for c in conds[1:]:
        if c not in data:
            continue
        worst = 0.0
        for r in runs:
            a, b = data[ref].get(r), data[c].get(r)
            if not a or not b:
                print(f"{c:<10}{r:<34}  -- missing in one condition")
                continue
            steps = sorted(set(a["train"]) & set(b["train"]))
            dl = max((abs(a["train"][s][0] - b["train"][s][0]) for s in steps), default=float("nan"))
            dg = max((abs(a["train"][s][1] - b["train"][s][1]) for s in steps), default=float("nan"))
            esteps = sorted(set(a["eval"]) & set(b["eval"]))
            dv = max((abs(a["eval"][s] - b["eval"][s]) for s in esteps), default=float("nan"))
            worst = max(worst, dl)
            verdict = "IDENTICAL" if dl == 0 and dg == 0 and dv == 0 else "differs"
            print(f"{c:<10}{r:<34}{len(steps):>5}{dl:>12.3e}{dg:>13.3e}{dv:>11.3e}  {verdict}")
        summary[c] = worst
    print("-" * 96)
    noise = summary.get(conds[1])
    if noise is not None:
        print(f"\nnoise floor (a1 vs a2, same config): max|dloss| = {noise:.3e}")
        for c, w in summary.items():
            if c == conds[1]:
                continue
            tag = ("within the noise floor" if w <= noise
                   else f"EXCEEDS the noise floor by {w/max(noise,1e-30):.1f}x"
                   if noise > 0 else "differs while the control was bit-identical")
            print(f"  {c}: max|dloss| = {w:.3e}  -> {tag}")


if __name__ == "__main__":
    main()
