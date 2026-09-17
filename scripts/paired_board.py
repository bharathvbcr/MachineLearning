#!/usr/bin/env python3
"""Paired-by-seed board across one or more suites, from committed run records.

    python3 scripts/paired_board.py ARM@SUITE [ARM@SUITE ...] --ref ARM@SUITE

Every arm is paired against the reference seed by seed, at markers the two runs
actually share, and at ``final_val``. Prints the paired mean gap (arm minus
reference, negative = arm better), Student-t intervals at 95 / 99 / 99.9% (the
99.9% one is the honest interval when the arm was chosen after scanning many
arms and markers), the sign count, and the mean-curve crossing tokens.

No GPU. Reads ``metrics.jsonl`` + ``config.json`` only; refuses a pair whose
recipes differ on anything but the arm, because a cross-recipe pairing is
exactly the error PAPER section 4 is about.

"The arm" includes the fields the REGISTRY declares for it. An arm registered
with an ``lr`` override is "that shape at that learning rate" -- the LR is what
distinguishes it from its siblings, not drift between two runs meant to match --
so a difference on a field either arm declares is waved through and named in the
header. A difference on a field NEITHER arm declares still refuses, which is the
accidental mismatch this guard exists to catch. Without this, a board run at each
arm's own measured optimum cannot be read at all: the comparison it exists to
make is the one the guard rejects.
Cross-SUITE pairing at one recipe is allowed and labelled: it is valid at this
recipe only because fresh attention arms reproduced suite 22's per seed within
+-0.001 (E18, E19b), and the label says so.

Four things this used to get wrong, all of them silently
--------------------------------------------------------
* **The interval was always the 4-dof one.** ``T4`` was a hardcoded row applied
  at any sample size: exact at five seeds, 35.5% too NARROW at three, and 36%
  too wide at thirty-two. ``nanolab.paired_stats`` computes the quantile now.
* **The recipe guard read ten fields of a hundred-odd.** It could not see
  ``compile``, ``dtype``, ``dataset``, ``grad_accum``, ``fused_ce``,
  ``weight_decay`` or any architecture flag -- so the manuscript's
  "only the budget changes" 50M-to-200M pairing passed a guard that never looked
  at ``compile``, which differs. ``nanolab.run_identity`` inverts the list: every
  field counts unless it is named as bookkeeping.
* **Duplicate runs resolved by sort order.** Two directories for one (arm, seed)
  meant the last one read won, with nothing printed.
* **Markers were snapped per curve, independently.** ``at(cur, 12.3e6)`` and
  ``at(ref, 12.3e6)`` each took their own nearest evaluation, so a value at
  token 100 could be differenced against one at 200. Markers are now resolved
  once, on the grid the arms share, and the token actually used is printed.

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
from nanolab.crossover_replicate import ARMS, _arm_from_run_dir  # noqa: E402
from nanolab.paired_stats import paired_interval  # noqa: E402
from nanolab.run_identity import compare, expand_declared  # noqa: E402

CONFIDENCES = ("95%", "99%", "99.9%")
_CONF_VALUE = {"95%": 0.95, "99%": 0.99, "99.9%": 0.999}

# How far a requested marker may be snapped onto the shared grid before the row
# stops describing the budget that was asked for. The saved cadence is ~0.8M
# tokens against markers 4M apart, so a snap worth more than a tenth of the
# request means the grid does not contain the point.
MARKER_SNAP_TOLERANCE = 0.10


def declared_keys(*arm_names: str) -> set[str]:
    """Config keys the registry sets as part of these arms' identity.

    Read from ``ARMS``, the canonical owner, rather than from the run configs --
    a config cannot say whether a value was chosen for the arm or drifted into
    it, and that distinction is the whole point of the guard. An arm the registry
    does not know contributes nothing, so an unregistered name cannot widen the
    exemption.
    """
    spec = {a.name: a for a in ARMS}
    out: set[str] = set()
    for name in arm_names:
        arm = spec.get(name)
        if arm is None:
            continue
        out |= {k for k, _ in (arm.overrides or ())}
        # `overrides` carries only the EXTRA settings. The registry also gives
        # every arm a `mixer` and a `layer_mixers`, and those are the arm --
        # `hybrid_mingru8_attn4` IS "mingru*8,attention*4". Reading `overrides`
        # alone made the guard refuse this file's own documented example, which
        # is the tell that the exemption was being read from the wrong half of
        # the registry entry. The old ten-key RECIPE_KEYS never noticed because
        # it did not compare the mixer fields at all.
        out |= {"mixer", "layer_mixers"}
    # `n_kv_head` follows from `n_head`, `head_dim` from `d_model`/`n_head`, and
    # so on. An arm that declares the source declares the consequence; refusing
    # a pairing on the consequence names a symptom instead of the cause.
    return expand_declared(out)


def load_arm(suite: str, arm: str) -> dict:
    """{seed: {"curve": [(tokens, val)], "final": float|None, "cfg": {...}}}"""
    root = ROOT / "nanolab" / "out" / suite
    if not root.exists():
        raise SystemExit(f"no such suite: {root}")
    out: dict = {}
    where: dict = {}
    for d in sorted(root.iterdir()):
        m, c = d / "metrics.jsonl", d / "config.json"
        if not (d.is_dir() and m.exists() and c.exists()):
            continue
        if _arm_from_run_dir(d) != arm:
            continue
        cfg = json.loads(c.read_text())
        curve, final, dones = [], None, 0
        for line in m.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "eval" and r.get("val_loss") is not None:
                curve.append((int(r["tokens"]), float(r["val_loss"])))
            elif r.get("event") == "done":
                dones += 1
                final = r.get("final_val")
        if not curve:
            continue
        # The token axis is what every marker and crossing in this file is
        # read against, and before D1 a resumed run restarted `tokens` at 0
        # while `step` carried on -- so the axis went backwards mid-curve and
        # `at()` interpolated across the fold without complaining. That is
        # how a crossing analysis once found no crossing at all. Refuse the
        # run rather than return a curve that looks fine and is not: a check
        # that cannot run must not report what a passing check reports.
        fold = next((i for i in range(1, len(curve))
                     if curve[i][0] <= curve[i - 1][0]), None)
        if fold is not None:
            raise SystemExit(
                f"REFUSING {d.name}: its token axis goes backwards at eval "
                f"{fold} ({curve[fold - 1][0]} -> {curve[fold][0]}). This is "
                "a pre-D1 resumed run; re-read it keyed on `step`, or rerun "
                "it on the fixed harness.")
        # A run with evaluations but no terminal record did not finish. It used
        # to load anyway, with `final=None` -- indistinguishable from an older
        # board that finished but predates the field. An unfinished run's curve
        # is a prefix, and a prefix silently shortens every marker row it enters.
        if dones == 0:
            raise SystemExit(
                f"REFUSING {d.name}: {len(curve)} evaluations and no `done` "
                f"record, so it is still running or it died. Its curve is a "
                f"prefix, not a short run. Exclude it explicitly or rerun it.")
        if dones > 1:
            raise SystemExit(
                f"REFUSING {d.name}: {dones} `done` records in one metrics "
                f"file. Which one is the run's terminal value is not decided "
                f"by reading the last line -- that is a resume or a collision, "
                f"and it needs resolving in the record, not here.")
        seed = int(cfg["seed"])
        if seed in out:
            raise SystemExit(
                f"REFUSING {suite}: two runs claim arm {arm!r} seed {seed} -- "
                f"{where[seed]} and {d.name}. Taking whichever sorted last is "
                f"how a duplicate becomes a result. Name the one you mean.")
        where[seed] = d.name
        out[seed] = {"curve": curve, "final": final, "cfg": cfg}
    if not out:
        raise SystemExit(f"no finished runs for {arm} in {suite}")
    return out


def shared_markers(*arms_and_seeds) -> list[int]:
    """Token markers every (arm, seed) in the argument list recorded.

    Every difference this file prints is read here. Taking each curve's own
    nearest point instead is what let a value at one budget be subtracted from a
    value at another.
    """
    sets = []
    for arm, seeds in arms_and_seeds:
        for s in seeds:
            sets.append({t for t, _ in arm[s]["curve"]})
    return sorted(set.intersection(*sets)) if sets else []


def resolve_marker(marks: list[int], requested: float) -> tuple[int, float]:
    """(nearest shared marker, |snap| as a fraction of the request)."""
    tok = min(marks, key=lambda t: abs(t - requested))
    rel = abs(tok - requested) / requested if requested else 0.0
    return tok, rel


def at(curve, tokens):
    """The value this curve recorded AT ``tokens``. Exact, never nearest.

    Callers pass a marker taken from `shared_markers`, so the point exists. A
    miss is a bug in the caller and says so, rather than quietly returning the
    nearest neighbour -- which is the behaviour this function used to have and
    the reason two budgets could be differenced.
    """
    for t, v in curve:
        if t == tokens:
            return v
    raise KeyError(f"no evaluation at token {tokens}; "
                   f"curve spans {curve[0][0]}..{curve[-1][0]}")


def interval(gaps):
    """(mean, {confidence: (lo, hi)}) at the dof this sample actually has."""
    m = statistics.fmean(gaps)
    return m, {c: paired_interval(gaps, _CONF_VALUE[c])[1:] for c in CONFIDENCES}


def guard(cur: dict, ref: dict, seeds, arm_name: str, ref_name: str,
          spec: str, ref_spec: str) -> tuple[set, dict]:
    """Refuse a pairing whose recipes differ on anything neither arm declares.

    The canonical owner of that decision. It used to live inside ``main`` here,
    which meant every OTHER caller of ``load_arm`` -- ``crossing_token.py``, the
    paper's estimator -- got the curves with no guard at all and never knew.

    Returns (fields differing by design, fields at least one run never recorded).
    """
    declared = declared_keys(arm_name, ref_name)
    by_design: set[str] = set()
    unknown: dict[str, int] = {}
    for s in seeds:
        cmp = compare(cur[s]["cfg"], ref[s]["cfg"], declared=declared)
        by_design |= set(cmp.by_design)
        for k in cmp.unknown:
            unknown[k] = unknown.get(k, 0) + 1
        if cmp.differ:
            raise SystemExit(
                f"REFUSING to pair {spec} with {ref_spec} at seed {s}: recipes "
                f"differ on {dict(sorted(cmp.differ.items()))}")
    return by_design, unknown


def crossings(a, b, seeds, marks):
    """Tokens where the mean curves of a and b cross, by linear interpolation."""
    diff = [statistics.fmean([at(a[s]["curve"], t) - at(b[s]["curve"], t)
                              for s in seeds]) for t in marks]
    out = []
    for i in range(1, len(marks)):
        if diff[i - 1] == 0 or (diff[i - 1] < 0) != (diff[i] < 0):
            t0, t1, d0, d1 = marks[i - 1], marks[i], diff[i - 1], diff[i]
            x = t0 if d1 == d0 else t0 + (t1 - t0) * (-d0) / (d1 - d0)
            out.append((x, "arm ahead" if d1 < 0 else "ref ahead"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="ARM@SUITE")
    ap.add_argument("--ref", required=True, help="ARM@SUITE")
    ap.add_argument("--markers", default="4.11e6,8.21e6,12.30e6,19.68e6,32.78e6",
                    help="intermediate token markers to report (nearest SHARED eval)")
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
        by_design, unknown = guard(cur, ref, seeds, arm, ref_arm, spec, a.ref)
        tag = ("within-suite" if suite == ref_suite
               else "CROSS-SUITE (valid at this recipe only)")
        if by_design:
            # Named, never silent: a reader must not have to check the registry to
            # learn that the two arms were trained at different learning rates.
            tag += ", arms differ BY DESIGN on " + ", ".join(sorted(by_design))
        print(f"\n== {spec} minus {a.ref}   seeds {seeds}   {tag}")
        if unknown:
            print(f"   NOT CHECKED: {', '.join(sorted(unknown))} -- at least one "
                  f"run never recorded these, so the recipes match on every field "
                  f"both wrote down, which is not the same as matching.")

        marks = shared_markers((cur, seeds), (ref, seeds))
        if not marks:
            print("   no token marker is shared by every run; nothing to compare")
            continue
        rows = [(f"{float(x) / 1e6:.2f}M", float(x)) for x in a.markers.split(",")]
        rows.append(("last eval", float(marks[-1])))
        for label, want in rows:
            tok, rel = resolve_marker(marks, want)
            gaps = [at(cur[s]["curve"], tok) - at(ref[s]["curve"], tok) for s in seeds]
            m, iv = interval(gaps)
            note = ""
            if rel > MARKER_SNAP_TOLERANCE:
                note = f"  << asked {want / 1e6:.2f}M, nearest shared is {tok / 1e6:.2f}M"
            print(f"  {label:>9s} @{tok / 1e6:7.2f}M  {m:+.4f}"
                  f"  95% [{iv['95%'][0]:+.4f},{iv['95%'][1]:+.4f}]"
                  f"  99.9% [{iv['99.9%'][0]:+.4f},{iv['99.9%'][1]:+.4f}]"
                  f"  arm better {sum(g < 0 for g in gaps)}/{len(gaps)}{note}")
        fin = [(cur[s]["final"], ref[s]["final"]) for s in seeds]
        if all(x is not None and y is not None for x, y in fin):
            m, iv = interval([x - y for x, y in fin])
            print(f"  final_val            {m:+.4f}"
                  f"  95% [{iv['95%'][0]:+.4f},{iv['95%'][1]:+.4f}]"
                  f"  99.9% [{iv['99.9%'][0]:+.4f},{iv['99.9%'][1]:+.4f}]")
        else:
            print("  final_val  not recorded for every seed (older suites); curves only")
        xs = crossings(cur, ref, seeds, marks)
        print(f"  mean-curve crossings over {len(marks)} shared markers: "
              + (", ".join(f"{x / 1e6:.2f}M ({who})" for x, who in xs)
                 if xs else "none (one arm ahead throughout)"))


if __name__ == "__main__":
    main()
