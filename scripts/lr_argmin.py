#!/usr/bin/env python3
"""Where each cell's learning-rate curve bottoms out, from committed run records.

    python3 scripts/lr_argmin.py SUITE [--compare OTHER_SUITE]

A *cell* is (d_model, mixer shape) -- the thing whose learning rate is being
chosen. Shape comes from ``layer_mixers`` in each run's own config, never from
the arm name: `periodic` carries three attention layers and `bookend` two, and a
board once compared them as though placement were the only difference.

The multiplier is derived as ``lr / 6e-4`` rather than parsed out of the arm
name, so a name and an override that disagree cannot both be believed.

**Every comparison here is paired by seed.** An earlier version of this file took
the plain mean of whatever seeds each point happened to have, and G11 showed what
that costs: it added seeds 42 and 100 to one point per curve and two argmins
appeared to move. Neither had. Seed 100 is the easiest seed at every cell on
these boards and 777 the hardest -- 4.1229 against 4.1803 at one of them, a
0.0574 spread, eighteen times the rerun floor. A mean over an unbalanced seed set
therefore says more about which seeds were run than about the learning rate. So
the grid argmin is read on the seeds common to *every* point, and the margin over
the runner-up on the seeds those two share.

Three ways this refuses to name an argmin, each of which has happened:
  * fewer than three points, or a minimum at either edge -- a grid that has not
    yet found a minimum (E21's first sweep picked its top multiplier in all six
    cells; every 50M grid so far has picked its bottom one);
  * no seed common to every point, so the curve cannot be read paired at all;
  * a margin over the runner-up at or under the 0.0031 rerun floor -- a basin
    flat at this grid spacing. d1152 attention sits here: 1x beats 0.5x by 0.0064
    paired on one seed, and the two straddle the 1/width prediction.

``--compare`` prints the same table for a second suite and the shift between the
two argmins per cell, which is the horizon-transfer question when the two suites
differ only in token budget.

No GPU. Reads ``metrics.jsonl`` + ``config.json`` only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_LR = 6e-4

# Rerun-to-rerun spread at a fixed seed on this harness, measured over the
# determinism suite: 0.0014 mean, 0.0031 max. A paired difference at or under the
# max is not a difference this rig can resolve.
RERUN_FLOOR = 0.0031


# Filled in by the last cells() call: why runs were left out. A reader that
# silently drops every run on a board and reports "no finished runs" is
# indistinguishable from one that looked and found nothing.
SKIPPED: dict = {}


def cells(suite: str) -> dict:
    """{(d_model, shape): {mult: {seed: final_val}}} for one suite.

    Keyed by seed rather than appended to a list, so that no caller can average
    two points over different seed sets without first noticing they differ.

    Runs that cannot be read are counted into `SKIPPED` rather than dropped in
    silence. `crossover50m` is the case that forced this: 50 finished runs, none
    of them carrying `final_val` because the board predates the field, and this
    function reported the suite empty. Its end-of-schedule value is the last
    `eval` record -- which is what the published rows were computed from -- so
    "empty" was not just unhelpful, it was wrong.
    """
    root = ROOT / "nanolab" / "out" / suite
    if not root.exists():
        raise SystemExit(f"no such suite: {root}")
    SKIPPED.clear()
    out: dict = {}
    for d in sorted(root.iterdir()):
        m, c = d / "metrics.jsonl", d / "config.json"
        if not (d.is_dir() and m.exists() and c.exists()):
            continue
        final = None
        saw_done = False
        for line in m.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "done":
                saw_done = True
                final = r.get("final_val")
        if final is None:
            key = ("finished but no final_val (board predates the field?)"
                   if saw_done else "no done record (still running, or died)")
            SKIPPED[key] = SKIPPED.get(key, 0) + 1
            continue
        cfg = json.loads(c.read_text())
        shape = cfg.get("layer_mixers") or cfg.get("mixer", "?")
        key = (cfg.get("d_model"), shape)
        mult = cfg["lr"] / BASE_LR
        out.setdefault(key, {}).setdefault(round(mult, 4), {})[cfg.get("seed")] = float(final)
    return out


def shared_seeds(curve: dict, mults=None) -> set:
    """Seeds present at every one of `mults` (default: every point on the curve)."""
    mults = sorted(curve) if mults is None else list(mults)
    if not mults:
        return set()
    common = set(curve[mults[0]])
    for m in mults[1:]:
        common &= set(curve[m])
    return common


def paired_means(curve: dict, seeds) -> dict:
    """{mult: mean over exactly `seeds`}. Caller guarantees each mult has them."""
    return {m: statistics.mean(curve[m][s] for s in seeds) for m in curve}


def argmin(curve: dict) -> tuple[float, str]:
    """(multiplier at the lowest paired mean, verdict).

    `verdict` is the string "interior" only when the argmin is one this rig can
    defend: read on a seed set common to the whole grid, off at least three
    points, not at either edge, and ahead of the runner-up by more than the
    rerun floor. Every other outcome returns a sentence saying what is missing.
    Callers gate on ``verdict == "interior"``, so each of these fails closed.
    """
    if not curve:
        return float("nan"), "no finished runs"
    common = shared_seeds(curve)
    if not common:
        per = ", ".join(f"{m:g}x:{sorted(curve[m])}" for m in sorted(curve))
        return float("nan"), (
            "no seed is common to every point, so the grid cannot be read "
            f"paired ({per}); rerun the odd points onto the shared seeds")
    means = paired_means(curve, common)
    order = sorted(means, key=lambda k: means[k])
    best = order[0]
    n = len(common)
    tag = f"paired on {n} seed{'s' if n != 1 else ''} {sorted(common)}"
    if len(means) < 3:
        return best, f"only {len(means)} point(s); no argmin from fewer than 3 [{tag}]"
    if best == min(means):                       # lowest multiplier on the grid
        return best, f"EDGE (bottom) -- extend the grid DOWN; not an argmin [{tag}]"
    if best == max(means):
        return best, f"EDGE (top) -- extend the grid UP; not an argmin [{tag}]"

    # The runner-up decides whether the argmin is located or the basin is flat.
    # Read that one comparison on every seed those two points share, which is
    # usually more than the whole grid has in common.
    up = order[1]
    pair = shared_seeds(curve, (best, up))
    deltas = [curve[up][s] - curve[best][s] for s in sorted(pair)]
    margin = statistics.mean(deltas)
    wins = sum(1 for d in deltas if d > 0)
    pair_tag = (f"{best:g}x beats {up:g}x by {margin:.4f} on {wins}/{len(deltas)} "
                f"of the {len(deltas)} seeds they share")
    if margin <= RERUN_FLOOR:
        return best, (f"FLAT -- {pair_tag}, at or under the {RERUN_FLOOR} rerun "
                      f"floor; the argmin is not resolved at this grid spacing [{tag}]")
    return best, "interior"


def show(suite: str) -> dict:
    data = cells(suite)
    skipped = dict(SKIPPED)
    print(f"\n=== {suite} ===")
    for why, n in sorted(skipped.items()):
        print(f"  NOTE: {n} run(s) not read -- {why}")
    if not data:
        if skipped:
            print("  REFUSING to report this suite as empty: every run was skipped "
                  "for the reason(s) above, which is not the same as finding nothing.")
        else:
            print("  (no finished runs)")
        return {}
    picks = {}
    for key in sorted(data, key=lambda k: (k[0] or 0, str(k[1]))):
        curve = data[key]
        best, verdict = argmin(curve)
        picks[key] = (best, verdict)
        common = shared_seeds(curve)
        pts = "  ".join(
            f"{m:g}x:"
            + (f"{statistics.mean(curve[m][s] for s in common):.4f}"
               if common else "----")
            + ("*" if m == best else "")
            + f"(n{len(curve[m])})"
            for m in sorted(curve))
        print(f"  d{key[0]} {str(key[1])[:44]:46s}")
        print(f"      {pts}")
        dropped = sorted({s for m in curve for s in curve[m]} - common)
        if dropped:
            print(f"      means paired on {sorted(common)}; "
                  f"{len(dropped)} seed(s) {dropped} not at every point, held out")
        print(f"      argmin {best:g}x -- {verdict}")
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite")
    ap.add_argument("--compare", help="second suite to read the same way")
    a = ap.parse_args()
    picks = show(a.suite)
    if not a.compare:
        return
    other = show(a.compare)
    shared = sorted(set(picks) & set(other), key=lambda k: (k[0] or 0, str(k[1])))
    print(f"\n=== argmin shift: {a.suite} vs {a.compare} ===")
    if not shared:
        print("  no cell appears in both suites")
        return
    for key in shared:
        b1, v1 = picks[key]
        b2, v2 = other[key]
        note = "" if v1 == v2 == "interior" else \
            "   [at least one side is not a located argmin; shift is a lower bound]"
        print(f"  d{key[0]} {str(key[1])[:38]:40s} {b2:g}x -> {b1:g}x"
              f"  ({'no move' if b1 == b2 else f'x{b1 / b2:g}'}){note}")


if __name__ == "__main__":
    main()
