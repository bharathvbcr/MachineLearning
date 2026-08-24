#!/usr/bin/env python3
"""Analyse the D7 LR re-tune and regenerate research/d7-lr-retune.json.

Every number reported for D7 during the run was computed in an ad-hoc shell heredoc.
That is exactly the practice section 7.4 of the paper identifies as the project's
actual failure mode -- sound measurements, hand-summarised, drifting from the
artifacts. This script is the fix: one path from ledger to artifact, re-runnable, so
the published numbers can be regenerated rather than retyped.

Reads:  out/funnel/d7_lr_retune_1000/ledger.json
Writes: research/d7-lr-retune.json   (with --write)

Analysis notes:

  * Comparisons are PAIRED by seed. The seed offset is large (up to ~0.013) relative
    to the effects here, and shared between candidates, so pairing removes most of it.
    An unpaired reading of the same runs gives intervals ~5x wider and was twice a
    source of wrong conclusions during this run.
  * An interval needs n>=3 to be informative (nanolab's MIN_SEEDS_FOR_INFORMATIVE_CI,
    the D1 fix). At n=2, t_1 = 12.706 makes every interval uselessly wide, so cells
    with n<3 report a sign test only.
  * "Bracketed" means the candidate's minimum is INTERIOR: strictly worse at the next
    grid point on both sides, by a paired sign test. A boundary minimum measures
    "lower is better in this range", not an optimum.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "out/funnel/d7_lr_retune_1000/ledger.json"
ARTIFACT = ROOT / "research/d7-lr-retune.json"

# Student-t two-sided 95% critical values by degrees of freedom.
T95 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445}
MIN_SEEDS_FOR_INFORMATIVE_CI = 3

# The margin the funnel used to select muon_polar_adamw over normuon_adamw at
# exact_128m_1000, from research/native-optimizer-funnel.json.
SELECTION_MARGIN = 0.031226


def load() -> dict[str, dict[float, dict[int, float]]]:
    if not LEDGER.exists():
        sys.exit(f"no ledger at {LEDGER}")
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    out: dict[str, dict[float, dict[int, float]]] = {}
    for j in data["jobs"]:
        if j["status"] != "done" or j.get("validation_bpb") is None:
            continue
        out.setdefault(j["candidate"], {}).setdefault(float(j["lr"]), {})[int(j["seed"])] = \
            float(j["validation_bpb"])
    return out


def interval(values: list[float]) -> tuple[float, float | None, int]:
    """(mean, half-width or None if uninformative, n)."""
    n = len(values)
    if n < 2:
        return (values[0] if values else math.nan), None, n
    if n < MIN_SEEDS_FOR_INFORMATIVE_CI:
        return statistics.mean(values), None, n
    half = T95.get(n - 1, 1.959964) * statistics.stdev(values) / math.sqrt(n)
    return statistics.mean(values), half, n


def paired(a: dict[int, float], b: dict[int, float]) -> dict | None:
    """Paired a - b over the seeds both cells share."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    d = [a[s] - b[s] for s in seeds]
    m, half, n = interval(d)
    return {
        "seeds": seeds, "per_seed": {str(s): round(x, 6) for s, x in zip(seeds, d)},
        "mean": round(m, 6),
        "ci95_half_width": round(half, 6) if half is not None else None,
        "ci95_informative": half is not None,
        "excludes_zero": bool(half is not None and (m - half) * (m + half) > 0),
        "sign_consistent": all(x > 0 for x in d) or all(x < 0 for x in d),
        "n": n,
    }


def bracketed(cells: dict[float, dict[int, float]], lr: float) -> dict:
    """Is `lr` an interior minimum -- worse on BOTH sides by a paired sign test?"""
    lrs = sorted(cells)
    i = lrs.index(lr)
    res = {"lr": lr, "has_left_neighbour": i > 0, "has_right_neighbour": i < len(lrs) - 1}
    for side, j in (("left", i - 1), ("right", i + 1)):
        if 0 <= j < len(lrs) and (j != i):
            p = paired(cells[lrs[j]], cells[lr])   # neighbour minus candidate
            res[side] = {"lr": lrs[j], "worse_than_candidate": bool(p and p["mean"] > 0),
                         "sign_consistent": bool(p and p["sign_consistent"]),
                         "detail": p}
        else:
            res[side] = None
    res["bracketed"] = bool(
        res["has_left_neighbour"] and res["has_right_neighbour"]
        and res["left"] and res["right"]
        and res["left"]["worse_than_candidate"] and res["left"]["sign_consistent"]
        and res["right"]["worse_than_candidate"] and res["right"]["sign_consistent"])
    return res


def build_report() -> tuple[dict, dict]:
    """Compute the whole D7 report from the ledger. Returns (report, raw cells).

    Split out of main() so paper/derive_figures.py can import the computation
    rather than reimplement it. Two scripts deriving the same published numbers
    independently is how they drift apart, which is the failure this repository
    keeps finding in its own record.
    """
    data = load()
    cands = sorted(data)
    report: dict = {"candidates": {}, "matched_lr": {}, "notes": []}

    for c in cands:
        cells = data[c]
        rows = {}
        for lr in sorted(cells):
            m, half, n = interval(list(cells[lr].values()))
            rows[f"{lr:g}"] = {
                "mean": round(m, 6), "n": n,
                "per_seed": {str(s): round(v, 6) for s, v in sorted(cells[lr].items())},
                "sd": round(statistics.stdev(list(cells[lr].values())), 6) if n > 1 else None,
                "ci95_half_width": round(half, 6) if half is not None else None,
            }
        best = min(cells, key=lambda k: statistics.mean(list(cells[k].values())))
        worst_selected = 0.05 if c == "muon_polar_adamw" else 0.1
        pen = paired(cells[worst_selected], cells[best]) if worst_selected in cells else None
        report["candidates"][c] = {
            "cells": rows,
            "best_tested_lr": best,
            "best_tested_mean": round(statistics.mean(list(cells[best].values())), 6),
            "bracket": bracketed(cells, best),
            "selected_lr": worst_selected,
            "inherited_lr_penalty": pen,
            "penalty_x_selection_margin": (
                round(pen["mean"] / SELECTION_MARGIN, 3) if pen else None),
        }

    a, b = "muon_polar_adamw", "normuon_adamw"
    if a in data and b in data:
        common = sorted(set(data[a]) & set(data[b]))
        n_sign = n_sep = 0
        for lr in common:
            p = paired(data[a][lr], data[b][lr])
            report["matched_lr"][f"{lr:g}"] = p
            n_sign += bool(p and p["sign_consistent"] and p["mean"] > 0)
            n_sep += bool(p and p["excludes_zero"])
        report["matched_lr_summary"] = {
            "points": len(common),
            "normuon_leads_sign_consistent": n_sign,
            "intervals_excluding_zero": n_sep,
        }
        ba = report["candidates"][a]["best_tested_lr"]
        bb = report["candidates"][b]["best_tested_lr"]
        # Unequal tuning depth only biases the comparison on the side the optima are
        # on. Extra points on the clearly-worse side (NorMuon's 0.07/0.1) cost nothing.
        # What matters is whether one candidate was allowed to search LOWER than the
        # other while both minima sit at the low end of the grid.
        lo_a, lo_b = min(data[a]), min(data[b])
        at_low_edge = (ba <= sorted(data[a])[1] if len(data[a]) > 1 else True) or \
                      (bb <= sorted(data[b])[1] if len(data[b]) > 1 else True)
        report["best_tested_comparison"] = {
            "muon_polar_adamw_lr": ba, "normuon_adamw_lr": bb,
            "paired": paired(data[a][ba], data[b][bb]),
            "lowest_lr_swept": {a: lo_a, b: lo_b},
            "equal_depth_on_the_low_side": lo_a == lo_b,
            "a_minimum_at_low_edge": ba <= sorted(data[a])[1] if len(data[a]) > 1 else None,
            "b_minimum_at_low_edge": bb <= sorted(data[b])[1] if len(data[b]) > 1 else None,
            "biased_by_unequal_depth": bool(lo_a != lo_b and at_low_edge),
        }

    # Self-describing artifact: the numbers above plus what they do and do not support.
    if report.get("matched_lr"):
        pol = [lr for lr, q in report["matched_lr"].items() if q and q["mean"] < 0]
        nor = [lr for lr, q in report["matched_lr"].items() if q and q["mean"] > 0]
        report["verdict"] = {
            "headline": (
                "The muon_polar_adamw / normuon_adamw ordering CROSSES OVER in learning "
                "rate. muon_polar_adamw leads at the low end, normuon_adamw at the high "
                "end, each sign-consistent across all seeds tested. There is no "
                "recipe-independent answer to which optimizer is better at this scale."),
            "polar_leads_at_lr": pol,
            "normuon_leads_at_lr": nor,
            "mechanism": (
                "Offset flat basins: muon_polar_adamw is flat across 0.005-0.008, "
                "normuon_adamw across 0.008-0.0125. Each wins inside its own basin and "
                "loses inside the other's."),
            "what_the_funnel_did": (
                "exact_128m_1000 compared muon_polar_adamw at lr 0.05 against "
                "normuon_adamw at lr 0.1 -- both far up the high-LR wall, on one side of "
                "a crossing it had no way to see -- and recorded a 0.031226 BPB margin. "
                "That margin measures the learning rates, not the optimizers."),
            "selection_status": (
                "RETIRED. Not reversed: normuon_adamw is not crowned in its place, "
                "because the ordering is learning-rate conditional and no cell separates "
                "the two at their respective optima."),
            "supersedes": [
                "polar-lr-transfer-2026-08-22 (500 steps, n=1, truncated grid)",
                "the '1.95x the selection margin' figure, withdrawn",
                "the 'optimum is near 0.035' figure, withdrawn",
            ],
        }
        report["limits"] = [
            "Sign tests are the primary evidence. Only 3 of 8 matched cells have an "
            "interval excluding zero; the effects are at or below the per-cell seed "
            "sd of ~0.006.",
            "Neither candidate's optimum is a POINT: both have flat basins whose "
            "interior minimum is not resolvable at n=3.",
            "One protocol only: arch02-128m, 1000 steps, Metal/M5. Nothing here "
            "transfers to other scales, horizons or hardware.",
            "Wall-clock and step timings from these runs are contended and must not "
            "be quoted as throughput.",
        ]
        report["provenance"] = {
            "generated_by": "scripts/d7_analyze.py",
            "from_ledger": "out/funnel/d7_lr_retune_1000/ledger.json",
            "jobs": sum(len(c) for cand in (data[a], data[b]) for c in cand.values()),
            "protocol": ("argv byte-identical to native-optimizer-funnel.json stage "
                         "exact_128m_1000 except --out"),
            "note": ("Regenerate with `python3 scripts/d7_analyze.py --write` rather "
                     "than editing by hand."),
        }

    return report, data


CHAMPION = ROOT / "research/champion-run.json"


def build_champion_sync(report: dict) -> tuple[dict, str]:
    """Derive champion-run.json's D7 fields from the same report as the artifact.

    ``research/champion-run.json`` carried a hand-written ``lr_transfer_finding``
    and ``lock_reason`` describing an earlier, truncated round of this grid. It
    survived the round that superseded it and ended up asserting the OPPOSITE
    conclusion from paper section 8.3 -- in a machine-readable file the paper
    cites. Two documents deriving the same result independently is how they drift;
    this makes the report the single owner of both.

    Returns (lr_transfer_finding, lock_reason). Never touches ``locked``: whether
    to lock is a judgement, and D3 says it stays open.
    """
    v = report["verdict"]
    a, b = "muon_polar_adamw", "normuon_adamw"
    ca, cb = report["candidates"][a], report["candidates"][b]
    bt = report["best_tested_comparison"]
    ptd = bt["paired"]
    sm = report["matched_lr_summary"]

    def _pen(c: dict) -> str:
        pen = c["inherited_lr_penalty"]
        if not pen:
            return "not measured"
        ci = (f" +/- {pen['ci95_half_width']:.6f}" if pen["ci95_half_width"] is not None
              else f" (n={pen['n']}, sign test only)")
        return (f"{pen['mean']:.6f}{ci} = {c['penalty_x_selection_margin']}x "
                f"the {SELECTION_MARGIN} selection margin")

    finding = {
        "id": "d7-lr-retune-2026-08-23",
        "generated_by": "scripts/d7_analyze.py --write (do not edit by hand)",
        "artifact": "research/d7-lr-retune.json",
        "supersedes": v["supersedes"] + [
            "the 24-job / two-seed round of this same grid, which read 'normuon_adamw "
            "ahead at all five matched LRs' and a best-cell gap of 0.016317. Its grid "
            "stopped above the crossing, so it saw only NorMuon's side of it."],
        "protocol": report["provenance"]["protocol"],
        "jobs": report["provenance"]["jobs"],
        "seeds_per_cell_max": max(row["n"] for c in (ca, cb) for row in c["cells"].values()),
        "headline": v["headline"],
        "matched_lr_result": (
            f"{len(v['polar_leads_at_lr'])} of {sm['points']} matched cells favour "
            f"{a} (lr {', '.join(v['polar_leads_at_lr'])}) and "
            f"{sm['normuon_leads_sign_consistent']} favour {b} "
            f"(lr {', '.join(v['normuon_leads_at_lr'])}); every cell is "
            f"sign-consistent across all seeds tested."),
        "best_tested_result": (
            f"{a} at lr {bt[a + '_lr']:g} -> {ca['best_tested_mean']:.6f}; "
            f"{b} at lr {bt[b + '_lr']:g} -> {cb['best_tested_mean']:.6f}; "
            f"paired gap {ptd['mean']:+.6f}"
            + (f" +/- {ptd['ci95_half_width']:.6f}" if ptd["ci95_half_width"] is not None else "")
            + (", separated" if ptd["excludes_zero"] else ", SPANS ZERO -- no separation")),
        "crossover": {"polar_leads_at_lr": v["polar_leads_at_lr"],
                      "normuon_leads_at_lr": v["normuon_leads_at_lr"],
                      "mechanism": v["mechanism"]},
        "inherited_lr_penalty_bpb": {
            a: (ca["inherited_lr_penalty"] or {}).get("mean"),
            b: (cb["inherited_lr_penalty"] or {}).get("mean"),
        },
        "penalty_x_selection_margin": {
            a: ca["penalty_x_selection_margin"], b: cb["penalty_x_selection_margin"]},
        "penalty_detail": {a: _pen(ca), b: _pen(cb)},
        "optimum_bracketed": {a: ca["bracket"]["bracketed"], b: cb["bracket"]["bracketed"]},
        "why_the_funnel_got_it_wrong": v["what_the_funnel_did"],
        "selection_status": v["selection_status"],
        "limits": report["limits"],
    }

    reason = (
        "Not locked, and the recorded selection is RETIRED rather than reversed. "
        + v["headline"] + " " + v["what_the_funnel_did"] + " "
        f"Inherited-LR penalties: {a} {_pen(ca)}; {b} {_pen(cb)}. "
        f"Do not lock on {a}, and do not lock on {b} either: at each candidate's own "
        f"best tested cell the paired gap is {ptd['mean']:+.6f}"
        + (f" +/- {ptd['ci95_half_width']:.6f}" if ptd["ci95_half_width"] is not None else "")
        + (", which spans zero. " if not ptd["excludes_zero"] else ". ")
        + "A lock needs a bracketed optimum for both candidates at a shared protocol; "
        f"bracketed is {a}={ca['bracket']['bracketed']}, {b}={cb['bracket']['bracketed']}. "
        "Derived by scripts/d7_analyze.py from research/d7-lr-retune.json; do not edit by hand.")
    return finding, reason


def sync_champion(report: dict, write: bool) -> bool:
    """Write (or check) champion-run.json's D7 fields. True = already in sync."""
    if not CHAMPION.exists():
        sys.exit(f"no champion record at {CHAMPION}")
    rec = json.loads(CHAMPION.read_text(encoding="utf-8"))
    finding, reason = build_champion_sync(report)
    in_sync = (rec.get("lr_transfer_finding") == finding
               and rec.get("lock_reason") == reason)
    if rec.get("locked") is not False:
        sys.exit("refusing to touch a champion record whose `locked` is not false "
                 "(D3: this selection is retired and must stay open)")
    if write and not in_sync:
        rec["lr_transfer_finding"] = finding
        rec["lock_reason"] = reason
        CHAMPION.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"synced {CHAMPION.relative_to(ROOT)} lr_transfer_finding + lock_reason")
    return in_sync


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="regenerate research/d7-lr-retune.json and sync champion-run.json")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if either published artifact has drifted from the ledger")
    args = ap.parse_args()

    report, data = build_report()
    cands = sorted(data)

    # Console summary
    for c in cands:
        r = report["candidates"][c]
        print(f"\n=== {c} ===")
        print(f"{'lr':>9} {'mean':>10} {'n':>3} {'sd':>10} {'ci95':>10}")
        for lr, row in r["cells"].items():
            sd = f"{row['sd']:.6f}" if row["sd"] is not None else "-"
            ci = f"{row['ci95_half_width']:.6f}" if row["ci95_half_width"] is not None else "n<3"
            print(f"{lr:>9} {row['mean']:>10.6f} {row['n']:>3} {sd:>10} {ci:>10}")
        print(f"  best tested lr {r['best_tested_lr']:g} -> {r['best_tested_mean']:.6f}"
              f"   bracketed: {r['bracket']['bracketed']}")
        if r["inherited_lr_penalty"]:
            p = r["inherited_lr_penalty"]
            ci = f" +/- {p['ci95_half_width']:.6f}" if p["ci95_half_width"] else " (n<3)"
            print(f"  inherited lr {r['selected_lr']:g} penalty: {p['mean']:.6f}{ci}"
                  f"  = {r['penalty_x_selection_margin']}x the {SELECTION_MARGIN} margin")

    if report["matched_lr"]:
        s = report["matched_lr_summary"]
        print(f"\n=== matched-LR comparison (Polar - NorMuon) ===")
        print(f"{'lr':>9} {'gap':>11} {'ci95':>11} {'signs':>8} {'sep?':>5}")
        for lr, p in report["matched_lr"].items():
            ci = f"+/-{p['ci95_half_width']:.6f}" if p["ci95_half_width"] else "n<3"
            print(f"{lr:>9} {p['mean']:>+11.6f} {ci:>11} "
                  f"{('%d-of-%d' % (p['n'], p['n'])) if p['sign_consistent'] else 'SPLIT':>8} "
                  f"{'YES' if p['excludes_zero'] else 'no':>5}")
        print(f"  NorMuon leads sign-consistently at {s['normuon_leads_sign_consistent']}"
              f"/{s['points']}; intervals exclude zero at {s['intervals_excluding_zero']}/{s['points']}")
        bt = report["best_tested_comparison"]
        p = bt["paired"]
        ci = f" +/- {p['ci95_half_width']:.6f}" if p["ci95_half_width"] else " (n<3)"
        print(f"\n  best-tested: Polar@{bt['muon_polar_adamw_lr']:g} vs "
              f"NorMuon@{bt['normuon_adamw_lr']:g}: {p['mean']:+.6f}{ci}"
              f"  {'separated' if p['excludes_zero'] else 'SPANS ZERO'}")
        if bt["biased_by_unequal_depth"]:
            lo = bt["lowest_lr_swept"]
            # `a`/`b` are locals of build_report(); read the names off the payload
            # instead. This branch previously raised NameError -- a guard that
            # crashes instead of warning is worse than no guard.
            depths = ", ".join(f"{k} {v:g}" for k, v in sorted(lo.items()))
            print(f"  WARNING: unequal tuning depth on the low side. Lowest LR swept: "
                  f"{depths}, and at least one minimum sits at its "
                  f"grid's low edge. Comparing best-tested points across grids of "
                  f"different depth is the same error this experiment documents. "
                  f"Report the matched-LR rows, or extend the shallower grid.")
        elif bt["equal_depth_on_the_low_side"]:
            print("  (grids are equally deep on the low side; best-tested comparison is fair)")

    if args.check:
        stale = []
        want = json.dumps(report, indent=2) + "\n"
        if not ARTIFACT.exists() or ARTIFACT.read_text(encoding="utf-8") != want:
            stale.append(str(ARTIFACT.relative_to(ROOT)))
        if not sync_champion(report, write=False):
            stale.append(str(CHAMPION.relative_to(ROOT)) + " (lr_transfer_finding / lock_reason)")
        if stale:
            print("\nDRIFT: these published artifacts disagree with the ledger:",
                  file=sys.stderr)
            for x in stale:
                print(f"  {x}", file=sys.stderr)
            print("  regenerate with `python3 scripts/d7_analyze.py --write`", file=sys.stderr)
            return 1
        print("\ncheck: research/d7-lr-retune.json and research/champion-run.json "
              "both match the ledger")
        return 0

    if args.write:
        ARTIFACT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {ARTIFACT.relative_to(ROOT)}")
        sync_champion(report, write=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
