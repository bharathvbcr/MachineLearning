#!/usr/bin/env python3
"""The token budget at which the reference arm overtakes the arm, per seed.

    python3 scripts/crossing_token.py ARM@SUITE --ref ARM@SUITE

This is the paper's headline estimator and it had no script. Section 3.2's
crossing tokens and their intervals were produced by hand, which is how "10.70M
[10.24, 11.16]" and the per-seed range beside it came to live only in a table.

The crossing is read **per seed** and then averaged, not read off the mean curve.
Those are different numbers: the mean-curve crossing has no interval, and
`paired_board.py` prints it precisely because it is cheap. Averaging the per-seed
crossings gives the Student-t interval the paper quotes, and the per-seed spread
is what says whether the estimate means anything.

Curves are loaded through `paired_board.load_arm`, so the token-axis guard
applies here too: a run whose `tokens` restarted mid-curve is refused rather than
interpolated across (D1).

Two ways this refuses, both of which the paper hit:
  * a seed with no crossing, or with the sign flipping more than once late --
    an average over "the crossing" of curves that do not each have one is not a
    crossing token;
  * arms that TIE at the end. Interpolating a sign change between curves whose
    difference is near zero divides by that difference. At ctx 2048 the paper
    measured 31.94M with an interval of [14.19, 49.70] this way and declined to
    report it. Below `--min-separation` (default 0.05 nats) this says so.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib.util                                            # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "paired_board", ROOT / "scripts" / "paired_board.py")
paired_board = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paired_board)

T = {2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 9: 2.262157}
RERUN_FLOOR = 0.0031


def seed_crossing(arm_curve, ref_curve, marks):
    """Tokens where (arm - ref) last turns positive: the reference overtaking.

    `marks` is the intersection of the two curves' own token grids, so
    `paired_board.at` -- which snaps to the nearest recorded eval rather than
    interpolating -- returns each curve's exact value there. The interpolation
    is between adjacent marks, to place the zero of the difference.

    Returns (token, n_late_flips).
    """
    at = paired_board.at
    diff = [at(arm_curve, t) - at(ref_curve, t) for t in marks]
    flips = []
    for i in range(1, len(marks)):
        if diff[i - 1] < 0 <= diff[i]:
            t0, t1, d0, d1 = marks[i - 1], marks[i], diff[i - 1], diff[i]
            flips.append(t0 if d1 == d0 else t0 + (t1 - t0) * (-d0) / (d1 - d0))
    return (flips[-1] if flips else None), len(flips)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", help="ARM@SUITE -- the arm that leads early")
    ap.add_argument("--ref", required=True, help="ARM@SUITE -- the arm that overtakes")
    ap.add_argument("--min-separation", type=float, default=0.05,
                    help="refuse if the arms differ by less than this at final_val")
    a = ap.parse_args()

    arm_name, arm_suite = a.arm.split("@")
    ref_name, ref_suite = a.ref.split("@")
    cur = paired_board.load_arm(arm_suite, arm_name)
    ref = paired_board.load_arm(ref_suite, ref_name)
    seeds = sorted(set(cur) & set(ref))
    if len(seeds) < 2:
        raise SystemExit(f"fewer than 2 shared seeds between {a.arm} and {a.ref}")

    missing = [s for s in seeds if cur[s]["final"] is None or ref[s]["final"] is None]
    if missing:
        raise SystemExit(f"seeds {missing} have no `done` record; rerun or exclude them")
    gaps = [cur[s]["final"] - ref[s]["final"] for s in seeds]
    sep = statistics.mean(gaps)
    marks = sorted(set.intersection(
        *(set(t for t, _ in cur[s]["curve"]) for s in seeds),
        *(set(t for t, _ in ref[s]["curve"]) for s in seeds)))

    print(f"\n== {a.arm}  overtaken by  {a.ref}")
    print(f"   seeds {seeds}   {len(marks)} shared markers")
    print(f"   separation at final_val: {sep:+.4f} nats "
          f"({'ref better' if sep > 0 else 'arm better'})")
    if abs(sep) < a.min_separation:
        print(f"   REFUSING: the arms differ by {abs(sep):.4f} < {a.min_separation} at the "
              f"end.\n   A crossing is interpolated through the difference between two "
              f"curves;\n   near zero that division is unstable and the interval is "
              f"meaningless.\n   The paper's ctx-2048 cell reads 31.94M [14.19, 49.70] "
              f"this way and is not reported.")
        raise SystemExit(3)

    per_seed, refused = {}, []
    for s in seeds:
        x, n = seed_crossing(cur[s]["curve"], ref[s]["curve"], marks)
        if x is None:
            refused.append(f"seed {s}: no crossing on the shared markers")
        elif n > 1:
            refused.append(f"seed {s}: {n} crossings; which one is 'the' crossing is "
                           f"not defined")
        else:
            per_seed[s] = x
    for r in refused:
        print(f"   {r}")
    if len(per_seed) < 2:
        raise SystemExit("fewer than 2 seeds produced a single clean crossing")
    if refused:
        print(f"   REFUSING: {len(refused)} of {len(seeds)} seeds have no single "
              f"crossing.\n   A mean over the seeds that happen to cross is not the "
              f"crossing token.")
        raise SystemExit(3)

    vals = [per_seed[s] for s in seeds]
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals)
    t = T.get(len(vals) - 1)
    if t is None:
        raise SystemExit(f"no t value for {len(vals) - 1} dof; add it")
    half = t * sd / len(vals) ** 0.5
    each = "  ".join(f"{v / 1e6:.2f}" for v in vals)
    print(f"   crossing token: {mean / 1e6:.2f}M  "
          f"95% [{(mean - half) / 1e6:.2f}, {(mean + half) / 1e6:.2f}]  "
          f"(n={len(vals)})")
    print(f"   per seed: {each}   range {min(vals) / 1e6:.2f}-{max(vals) / 1e6:.2f}M")


if __name__ == "__main__":
    main()
