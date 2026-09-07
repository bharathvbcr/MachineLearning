#!/usr/bin/env python3
"""Paired-by-seed board across one or more suites, from committed run records.

    python3 scripts/paired_board.py ARM@SUITE [ARM@SUITE ...] --ref ARM@SUITE

Every arm is paired against the reference seed by seed (the five-seed grid is
shared by every suite in this repo), at every evaluation marker the two runs
share, and at ``final_val``. Prints the paired mean gap (arm minus reference,
negative = arm better), Student-t intervals at 95 / 99 / 99.9% (4 dof: the
99.9% one is the honest interval when the arm was chosen after scanning many
arms and markers), the sign count, and the mean-curve crossing tokens.

No GPU. Reads ``metrics.jsonl`` + ``config.json`` only; refuses a pair whose
recipes differ on anything but the arm (batch, block, eval_iters, token budget),
because a cross-recipe pairing is exactly the error PAPER section 4 is about.
Cross-SUITE pairing at one recipe is allowed and labelled: it is valid at this
recipe only because fresh attention arms reproduced suite 22's per seed within
+-0.001 (E18, E19b), and the label says so.

Example (the 2026-09-04 no-regret reading):
    python3 scripts/paired_board.py hybrid_mingru8_attn4@crossover50m_ratioplace32 \
        hybrid_mingru_periodic@crossover50m_ratioplace32 --ref attention@crossover50m
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanolab.crossover_replicate import _arm_from_run_dir  # noqa: E402

T4 = {"95%": 2.776445, "99%": 4.604095, "99.9%": 8.610302}   # two-sided, 4 dof
RECIPE_KEYS = ("batch_size", "block_size", "eval_iters", "max_steps", "lr_max_steps",
               "lr", "matrix_lr", "warmup_steps", "schedule", "optimizer")


def load_arm(suite: str, arm: str) -> dict:
    """{seed: {"curve": [(tokens, val)], "final": float|None, "recipe": {...}}}"""
    root = ROOT / "nanolab" / "out" / suite
    if not root.exists():
        raise SystemExit(f"no such suite: {root}")
    out = {}
    for d in sorted(root.iterdir()):
        m, c = d / "metrics.jsonl", d / "config.json"
        if not (d.is_dir() and m.exists() and c.exists()):
            continue
        if _arm_from_run_dir(d) != arm:
            continue
        cfg = json.loads(c.read_text())
        curve, final = [], None
        for line in m.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "eval" and r.get("val_loss") is not None:
                curve.append((int(r["tokens"]), float(r["val_loss"])))
            elif r.get("event") == "done":
                final = r.get("final_val")
        if curve:
            rec = {k: cfg.get(k) for k in RECIPE_KEYS}
            # Older suites recorded lr_max_steps=0 (= "decay over max_steps");
            # newer ones record the number itself. Same schedule, one spelling.
            rec["lr_max_steps"] = rec["lr_max_steps"] or rec["max_steps"]
            out[int(cfg["seed"])] = {"curve": curve, "final": final, "recipe": rec}
    if not out:
        raise SystemExit(f"no finished runs for {arm} in {suite}")
    return out


def at(curve, tokens):
    return min(curve, key=lambda e: abs(e[0] - tokens))[1]


def interval(gaps):
    m = statistics.mean(gaps)
    se = statistics.stdev(gaps) / len(gaps) ** 0.5 if len(gaps) > 1 else float("nan")
    return m, {k: (m - t * se, m + t * se) for k, t in T4.items()}


def crossings(a, b, seeds):
    """Tokens where the mean curves of a and b cross, by linear interpolation."""
    marks = sorted(set.intersection(*(set(t for t, _ in a[s]["curve"]) for s in seeds),
                                    *(set(t for t, _ in b[s]["curve"]) for s in seeds)))
    diff = [statistics.mean(at(a[s]["curve"], t) - at(b[s]["curve"], t) for s in seeds)
            for t in marks]
    out = []
    for i in range(1, len(marks)):
        if diff[i - 1] == 0 or (diff[i - 1] < 0) != (diff[i] < 0):
            t0, t1, d0, d1 = marks[i - 1], marks[i], diff[i - 1], diff[i]
            x = t0 if d1 == d0 else t0 + (t1 - t0) * (-d0) / (d1 - d0)
            out.append((x, "arm ahead" if d1 < 0 else "ref ahead"))
    return marks, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="ARM@SUITE")
    ap.add_argument("--ref", required=True, help="ARM@SUITE")
    ap.add_argument("--markers", default="4.11e6,8.21e6,12.30e6,19.68e6,32.78e6",
                    help="intermediate token markers to report (nearest saved eval)")
    a = ap.parse_args()
    ref_arm, ref_suite = a.ref.split("@")
    ref = load_arm(ref_suite, ref_arm)
    for spec in a.arms:
        arm, suite = spec.split("@")
        cur = load_arm(suite, arm)
        seeds = sorted(set(cur) & set(ref))
        if len(seeds) < 2:
            print(f"{spec}: fewer than 2 shared seeds with {a.ref}; nothing to pair")
            continue
        for s in seeds:
            if cur[s]["recipe"] != ref[s]["recipe"]:
                bad = {k: (cur[s]["recipe"][k], ref[s]["recipe"][k]) for k in RECIPE_KEYS
                       if cur[s]["recipe"][k] != ref[s]["recipe"][k]}
                raise SystemExit(f"REFUSING to pair {spec} with {a.ref}: recipes differ on {bad}")
        tag = "within-suite" if suite == ref_suite else "CROSS-SUITE (valid at this recipe only)"
        print(f"\n== {spec} minus {a.ref}   seeds {seeds}   {tag}")
        last = max(t for t, _ in cur[seeds[0]]["curve"])
        rows = [(f"{float(x)/1e6:.2f}M", float(x)) for x in a.markers.split(",")] + [("last eval", last)]
        for label, tk in rows:
            gaps = [at(cur[s]["curve"], tk) - at(ref[s]["curve"], tk) for s in seeds]
            m, iv = interval(gaps)
            print(f"  {label:>9s}  {m:+.4f}  95% [{iv['95%'][0]:+.4f},{iv['95%'][1]:+.4f}]"
                  f"  99.9% [{iv['99.9%'][0]:+.4f},{iv['99.9%'][1]:+.4f}]"
                  f"  arm better {sum(g < 0 for g in gaps)}/{len(gaps)}")
        fin = [(cur[s]["final"], ref[s]["final"]) for s in seeds]
        if all(x is not None and y is not None for x, y in fin):
            m, iv = interval([x - y for x, y in fin])
            print(f"  final_val  {m:+.4f}  95% [{iv['95%'][0]:+.4f},{iv['95%'][1]:+.4f}]"
                  f"  99.9% [{iv['99.9%'][0]:+.4f},{iv['99.9%'][1]:+.4f}]")
        else:
            print("  final_val  not recorded for every seed (older suites); curves only")
        marks, xs = crossings(cur, ref, seeds)
        print(f"  mean-curve crossings over {len(marks)} shared markers: "
              + (", ".join(f"{x/1e6:.2f}M ({who})" for x, who in xs) if xs else "none (one arm ahead throughout)"))


if __name__ == "__main__":
    main()
