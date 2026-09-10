#!/usr/bin/env python3
"""Decide the re-tuned ladder from the MEASURED 50M argmin, not from an assumption.

    python3 scripts/g9_plan.py            # print the plan
    python3 scripts/g9_plan.py --arms     # emit the arm list, or exit 2 with a reason

The ladder spent 50M at learning rates probed over 610 steps, and G7/G3 showed every
arm wants less at 3051. Re-running it needs a multiplier, and picking one by hand is
the error being repaired -- so this reads the probe boards and derives it.

Rules, all of which can REFUSE rather than guess:

  * One board per cell, never two. Pooling an arm's LR curve across boards would
    reintroduce the cross-recipe join this whole program is about, so a cell is read
    from whichever board carries the most points for it and the choice is printed.
  * An argmin at the edge is not an argmin (E21's rule, which it learned by running
    off the top and which every 50M grid since has broken at the bottom). A cell whose
    minimum sits at either end of its grid refuses, and names the direction to extend.
  * Every width is planned from its OWN probe. The first draft let w1536 inherit a
    multiplier when the probed widths agreed -- and then the probe came back with d384
    wanting 2x and d768 wanting 1x. The 50M argmin is arm-invariant but NOT
    width-invariant, so there was nothing to inherit; w1536 got its own probe instead.
    A width with no probe is not planned at all, and says so.
"""
from __future__ import annotations

import argparse
import json
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("lr_argmin", ROOT / "scripts" / "lr_argmin.py")
lr_argmin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lr_argmin)

from nanolab.crossover_replicate import ARMS, LADDER_WIDTHS  # noqa: E402

# Where each cell is read from. w384/w768 come from the 50M probe; w1152 from the
# ladder board, whose low side (G8b) and high side (G3) together span the same grid.
SOURCES = {384: "crossover_probe50m", 768: "crossover_probe50m",
           1152: "crossover_ladder50m", 1536: "crossover_probe50m_w1536"}
PROBED = tuple(sorted(SOURCES))
MIXERS = ("attention", "mingru")


def resolved(arm) -> dict:
    """The model an arm actually trains: base Config, then its own overrides."""
    from nanolab.config import Config
    cfg = Config()
    out = {k: getattr(cfg, k) for k in
           ("d_model", "n_head", "head_dim", "lr", "matrix_lr", "mingru_expand")}
    out.update(dict(arm.overrides or ()))
    out["mixer"] = arm.mixer
    out["layer_mixers"] = arm.layer_mixers
    return out


def canonical(name: str, on_board: tuple[str, ...] = ()) -> str:
    """Prefer an equivalent arm the target board already carries.

    `w768_attention_lr10` and the bare `attention` arm resolve to the SAME model --
    d_model 768, lr 6e-4 -- because the ladder's w768 overrides restate the base
    Config. The 50M probe put d768's argmin at exactly 1x, so planning the generated
    name would have queued fifteen jobs reproducing runs already on the board at n=5.

    Only swaps when the board actually has the twin, so a board that does not carry
    the base arm keeps the generated name and its own naming convention.
    """
    spec = {a.name: a for a in ARMS}
    if name not in spec:
        return name
    target = resolved(spec[name])
    for other in on_board:
        if other != name and other in spec and resolved(spec[other]) == target:
            return other
    return name


def mult_name(m: float) -> str:
    return str(float(m)).replace(".", "")


def collect() -> tuple[dict, list[str]]:
    """{(width, mixer): (mult, verdict, n_points, board)} plus refusal reasons."""
    got, refused = {}, []
    cache = {}
    for w in PROBED:
        board = SOURCES[w]
        if board not in cache:
            try:
                cache[board] = lr_argmin.cells(board)
            except SystemExit as e:
                refused.append(str(e))
                cache[board] = {}
        for mx in MIXERS:
            curve = cache[board].get((w, mx))
            if not curve:
                refused.append(f"no finished runs for d{w} {mx} in {board}")
                continue
            best, verdict = lr_argmin.argmin(curve)
            got[(w, mx)] = (best, verdict, len(curve), board)
            if verdict != "interior":
                refused.append(f"d{w} {mx}: {verdict}")
    return got, refused


# Section 5's cells: the two hybrid shapes and the attention arm they are compared
# against, all at d_model 768. Read from the same probe, under the same edge rule --
# whether these three agree on a multiplier is what decides whether a shared-LR board
# was ever reading them from different distances.
HYBRID_CELLS = {
    "attention": "attention",
    "hybrid_mingru8_attn4": "mingru*8,attention*4",
    "hybrid_mingru_periodic":
        "mingru*3,attention,mingru*3,attention,mingru*3,attention",
}


def plan_hybrids(as_arms: bool) -> None:
    try:
        data = lr_argmin.cells("crossover_probe50m")
    except SystemExit as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        raise SystemExit(2)
    known = {x.name for x in ARMS}
    try:
        rec = json.loads((ROOT / "nanolab" / "out" / "crossover50m_ratioplace32"
                          / "recipe.json").read_text())
        on_board = tuple(rec.get("arms", ()))
    except (OSError, ValueError):
        on_board = ()
    picks, refused, arms = {}, [], []
    for label, shape in HYBRID_CELLS.items():
        curve = data.get((768, shape))
        if not curve:
            refused.append(f"no finished runs for d768 {label} in crossover_probe50m")
            continue
        best, verdict = lr_argmin.argmin(curve)
        picks[label] = best
        if verdict != "interior":
            refused.append(f"d768 {label}: {verdict}")
            continue
        name = f"w768_{label}_lr{mult_name(best)}"
        if name not in known:
            refused.append(f"not in the arm registry: {name}")
            continue
        arms.append(canonical(name, on_board))
    if refused:
        print("REFUSING to plan section 5 at its argmin:\n  " + "\n  ".join(refused),
              file=sys.stderr)
        raise SystemExit(2)
    if as_arms:
        print(",".join(arms))
        return
    print("=== section 5 at each arm's own 50M argmin ===")
    for label, m in sorted(picks.items()):
        print(f"  d768 {label:<24s} {m:g}x")
    same = len(set(picks.values())) == 1
    verdict = ("a shared LR was never reading them from different distances"
               if same else
               "a shared-LR board WAS reading them from different distances")
    print(f"  all three agree: {same} -- {verdict}")
    print("  " + "\n  ".join(arms))


def plan_single_width(board: str, width: int, as_arms: bool) -> None:
    """Argmins for one width read from one board -- G6's rung, and any future one.

    Same edge rule as everything else: a width that has to be probed separately gets
    no dispensation from it, and refusing costs one probe where not refusing costs a
    rung.
    """
    try:
        data = lr_argmin.cells(board)
    except SystemExit as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        raise SystemExit(2)
    known = {x.name for x in ARMS}
    arms, refused, picks = [], [], {}
    for mx in MIXERS:
        curve = data.get((width, mx))
        if not curve:
            refused.append(f"no finished runs for d{width} {mx} in {board}")
            continue
        best, verdict = lr_argmin.argmin(curve)
        picks[mx] = best
        if verdict != "interior":
            refused.append(f"d{width} {mx}: {verdict}")
            continue
        name = f"w{width}_{mx}_lr{mult_name(best)}"
        if name in known:
            arms.append(name)
        else:
            refused.append(f"not in the arm registry: {name}")
    if refused:
        print(f"REFUSING to plan the d{width} rung:\n  " + "\n  ".join(refused),
              file=sys.stderr)
        raise SystemExit(2)
    if as_arms:
        print(",".join(arms))
        return
    print(f"=== d{width} at its own measured 50M argmin ({board}) ===")
    for mx, m in sorted(picks.items()):
        print(f"  {mx:<10s} {m:g}x")
    print("  " + "\n  ".join(arms))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", action="store_true",
                    help="print a comma-separated arm list and nothing else")
    ap.add_argument("--hybrids", action="store_true",
                    help="plan section 5's w768 cells instead of the width ladder")
    ap.add_argument("--width", type=int,
                    help="plan one width from --board instead of the whole ladder")
    ap.add_argument("--board", help="probe board for --width")
    a = ap.parse_args()
    if a.width:
        if not a.board:
            print("--width needs --board", file=sys.stderr)
            raise SystemExit(2)
        return plan_single_width(a.board, a.width, a.arms)
    if a.hybrids:
        return plan_hybrids(a.arms)
    got, refused = collect()

    if not a.arms:
        print("=== measured 50M argmins ===")
        for k in sorted(got):
            best, verdict, n, board = got[k]
            print(f"  d{k[0]:<5d} {k[1]:<10s} {best:g}x  ({n} points, {board})  {verdict}")

    if refused:
        msg = "REFUSING to plan the re-tuned ladder:\n  " + "\n  ".join(refused)
        msg += ("\n  Fix the grid before spending n=5 on it -- an argmin at the edge is "
                "not an argmin, and this is the exact error being repaired.")
        print(msg, file=sys.stderr)
        raise SystemExit(2)

    # Every width from its own probe. Nothing is inherited: the 50M argmin is not
    # width-invariant (d384 wants 2x, d768 1x), so an inherited multiplier would be
    # a guess wearing a measurement's clothes.
    plan, notes = {}, []
    for mx in MIXERS:
        picks = {w: got[(w, mx)][0] for w in PROBED}
        for w in PROBED:
            plan[(w, mx)] = picks[w]
        if len(set(picks.values())) > 1:
            notes.append(f"  {mx}: the argmin MOVES with width at 50M -- "
                         + ", ".join(f"d{w} {m:g}x" for w, m in sorted(picks.items())))
    for w in sorted(set(LADDER_WIDTHS) - set(PROBED)):
        notes.append(f"  d{w}: no probe, so NOT planned -- give it one rather than "
                     "letting it inherit")

    known = {x.name for x in ARMS}
    arms, missing = [], []
    for (w, mx), m in sorted(plan.items()):
        name = f"w{w}_{mx}_lr{mult_name(m)}"
        (arms if name in known else missing).append(name)
    if missing:
        print(f"REFUSING: not in the arm registry: {missing}", file=sys.stderr)
        raise SystemExit(2)

    if a.arms:
        print(",".join(arms))
        return
    print("\n=== plan ===")
    for n in notes:
        print(n)
    print(f"  {len(arms)} arms x 5 seeds; already-done jobs are skipped by the launcher")
    print("  " + "\n  ".join(arms))
    planned = set(PROBED) | {1536}
    assert planned <= set(LADDER_WIDTHS), "SOURCES names a width the ladder does not have"
    for w in sorted(set(LADDER_WIDTHS) - planned):
        print(f"  NOTE d{w} is a ladder width with no probe and is NOT planned here; "
              "give it its own probe (see G6) rather than letting it inherit.")


if __name__ == "__main__":
    main()
