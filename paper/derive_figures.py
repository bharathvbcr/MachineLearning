#!/usr/bin/env python3
"""Re-derive every number in the manuscript from the run artifacts.

    python3 paper/derive_figures.py            # print the table
    python3 paper/derive_figures.py --json     # machine-readable
    python3 paper/derive_figures.py --check    # verify the manuscript matches

The manuscript this feeds was reconstructed after its source file was lost:
`PAPER_2026-08_Recipe_Dependent_Rankings.md` was removed from the working tree
and stripped from every commit by a history rewrite. What survived is the thing
that actually matters — `nanolab/out/crossover*/` and `research/`, the runs
themselves.

So the prose was rewritten around numbers this script computes, rather than
numbers anybody remembered. That ordering is the point: a figure in the
manuscript that this script cannot reproduce is a figure that should not be in
the manuscript, and `--check` is what says so.

`--check` enforces that in both directions:

  * every figure this script derives must appear in the manuscript, so the
    prose cannot silently fall behind a re-run; and
  * every figure the manuscript states must either be one this script derives
    or carry a documented reason in STATED_NOT_DERIVED, so a stale or invented
    number cannot sit in prose the check never reads.

Until 2026-08-23 only the first direction existed, and it covered five figures
against the manuscript's forty-four -- an unqualified OK over an 11% sample.
Both the coverage and the second direction were added after that audit.

One claim from the pre-loss version did not survive that test. See MIXER_BOARD
below.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "nanolab" / "out"
MANUSCRIPT = ROOT / "PAPER_2026-08_Recipe_Dependent_Rankings.md"

# How far apart the ten board arms' horizons may sit before the board stops
# being a ranking "at 50M tokens". The two eval cadences in crossover50m differ
# by 1.6%; the stale-artifact bug that hid an arm at 40.16M was a 20% gap. Any
# spread that large is a measurement difference, not a rounding detail.
HORIZON_SPREAD_LIMIT_PCT = 5.0

sys.path.insert(0, str(ROOT))
# The one 95% multiplier in this repository, table and fallback included.  This
# script used to carry its own 1.96 for the matched-batch board, which is the
# same normal-quantile defect §6 of the manuscript records the funnel fixing --
# committed in the script that checks the manuscript. Import, do not re-derive.
from nanolab.native_funnel import _t_critical_95  # noqa: E402


# ----------------------------------------------------------------- loading


def eval_curve(run_dir: Path):
    """(tokens, val_loss) at each eval for one run."""
    xs, ys = [], []
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return xs, ys
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") != "eval":
                continue
            v = r.get("val_loss", r.get("val", r.get("loss")))
            if v is not None and r.get("tokens") is not None:
                xs.append(r["tokens"])
                ys.append(v)
    return xs, ys


def arm(out_dir: Path, prefix: str):
    """Mean curve across seeds for one arm, truncated to the common grid."""
    if not out_dir.is_dir():
        return None
    runs = sorted(
        d for d in os.listdir(out_dir)
        if d.startswith(prefix + "_s") and (out_dir / d).is_dir()
    )
    curves = [eval_curve(out_dir / r) for r in runs]
    pairs = [(r, c) for r, c in zip(runs, curves) if c[0]]
    if not pairs:
        return None
    n = min(len(c[0]) for _, c in pairs)
    return {
        "n": len(pairs),
        "seeds": sorted(int(r.split("_s")[-1]) for r, _ in pairs),
        "tokens": pairs[0][1][0][:n],
        "mean": [st.fmean([c[1][i] for _, c in pairs]) for i in range(n)],
    }


def crossings(a, b):
    """Where a's mean curve crosses b's, by linear interpolation.

    Negative difference means `a` is ahead, since these are losses.
    """
    diff = [x - y for x, y in zip(a["mean"], b["mean"])]
    out = []
    for i in range(1, len(diff)):
        if (diff[i - 1] < 0) != (diff[i] < 0):
            x0, x1 = a["tokens"][i - 1], a["tokens"][i]
            y0, y1 = diff[i - 1], diff[i]
            out.append({
                "tokens": x0 + (x1 - x0) * (0 - y0) / (y1 - y0),
                "winner": "attention" if diff[i] < 0 else "minGRU",
            })
    return out


# ------------------------------------------------------------- the figures


def headline():
    """The 50M-token replication: two crossings where a single seed saw one."""
    d = json.loads((OUT / "crossover50m" / "summary.json").read_text())
    A, M = d["arms"]["attention"], d["arms"]["mingru"]
    cx = crossings(A, M)
    gaps = [b - a for a, b in zip(A["tokens"], A["tokens"][1:])]
    return {
        "n": A["n"],
        "final_tokens": A["tokens"][-1],
        # §2 states the grid spacing to justify how precisely a crossing can be
        # located; it is a property of the eval grid, so derive it, don't type it.
        "grid_spacing": st.fmean(gaps),
        "grid_uniform": len(set(gaps)) == 1,
        "crossings": cx,
        "attention_final": A["mean"][-1],
        "attention_ci": [A["lo"][-1], A["hi"][-1]],
        "mingru_final": M["mean"][-1],
        "mingru_ci": [M["lo"][-1], M["hi"][-1]],
        "advantage": A["mean"][-1] - M["mean"][-1],
    }


CONTROLS = [
    ("crossover8m_bs8", "cx8", "batch 8, the original batch size"),
    ("crossover20m_locked", "cx20", "20M budget, cosine schedule truncated with it"),
    ("crossover20m_matched_lr", "cx20h", "20M budget, 50M cosine horizon kept"),
]


def controls():
    rows = []
    for run, prefix, label in CONTROLS:
        p = OUT / run
        A, M = arm(p, f"{prefix}_attention"), arm(p, f"{prefix}_mingru")
        if not A or not M:
            rows.append({"run": run, "label": label, "error": "arms missing"})
            continue
        rows.append({
            "run": run,
            "label": label,
            "n": A["n"],
            "through_tokens": A["tokens"][-1],
            "crossings": crossings(A, M),
        })
    return rows


def mixer_board():
    """The ten-arm ranking at 50M tokens.

    MIXER_BOARD — the one claim that did not survive reconstruction.

    The pre-loss manuscript reported this board as placing attention first at
    4.222 [4.204, 4.240] "but statistically tied with a 10xminGRU + 2xattention
    hybrid at 4.232 [4.210, 4.254]". The first half reproduces exactly. The
    second does not: the runner-up computes to 4.275 [4.245, 4.305] from these
    runs, and its interval is disjoint from attention's.

    The 4.232 figure could not be located in any surviving run. The matched
    batch-32 sweep (crossover50m_matched32) does place a minGRU hybrid on top,
    at 4.214 [4.207, 4.221] -- but that sweep has no attention arm at all, so it
    cannot support a claim about a tie with attention.

    The honest reading is that attention wins this board outright at this
    recipe, and that a paper arguing rankings are properties of the measurement
    had a ranking claim of its own that depended on which condition was
    tabulated. That is stated in the manuscript rather than quietly corrected.
    """
    d = json.loads((OUT / "crossover50m" / "summary.json").read_text())
    rows = sorted(
        ({"arm": name,
          "final": a["mean"][-1],
          "ci": [a["lo"][-1], a["hi"][-1]],
          "n": a["n"],
          # Each arm's own last evaluated token count. Carried per row because
          # this board once listed an arm measured at 40.16M beside nine at
          # 49.99M and read as a single-horizon ranking: the seeds had run on
          # two eval grids, and the aligner that built summary.json took their
          # exact intersection. A board that does not carry its horizon cannot
          # show you that.
          "horizon": a["tokens"][-1]}
         for name, a in d["arms"].items()),
        key=lambda r: r["final"],
    )
    tied = rows[1]["ci"][0] <= rows[0]["ci"][1] if len(rows) > 1 else False
    horizons = [r["horizon"] for r in rows]
    lo, hi = min(horizons), max(horizons)
    return {
        "rows": rows,
        "top_two_overlap": tied,
        "horizon_min": lo,
        "horizon_max": hi,
        "horizon_spread_pct": 100.0 * (hi - lo) / hi if hi else 0.0,
    }


def matched_batch_board():
    """The batch-32 sweep, which has no attention arm."""
    p = OUT / "crossover50m_matched32"
    if not p.is_dir():
        return None
    arms = sorted({d.rsplit("_s", 1)[0] for d in os.listdir(p)
                   if "_s" in d and (p / d).is_dir()})
    rows = []
    for a in arms:
        finals = []
        for d in sorted(os.listdir(p)):
            if not d.startswith(a + "_s"):
                continue
            mp = p / d / "metrics.jsonl"
            if not mp.exists():
                continue
            last = None
            for line in mp.open():
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    pass
            if last and last.get("event") == "done" and last.get("best_val") is not None:
                finals.append(last["best_val"])
        if len(finals) >= 2:
            n = len(finals)
            m, sd = st.fmean(finals), st.stdev(finals)
            h = _t_critical_95(n - 1) * sd / math.sqrt(n)
            rows.append({"arm": a.replace("cx32_", ""), "final": m,
                         "ci": [m - h, m + h], "n": n, "df": n - 1})
    rows.sort(key=lambda r: r["final"])
    return {"rows": rows, "has_attention_arm": any("attention" == r["arm"] for r in rows)}


def optimizer_funnel():
    """Scale reversal in the Apple-silicon optimizer funnel."""
    f = ROOT / "research" / "native-optimizer-funnel.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())

    def rank(stage, agg="mean"):
        per = {}
        for j in d["jobs"]:
            if j.get("stage") != stage:
                continue
            v = (j.get("result") or {}).get("validation_bpb")
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            per.setdefault(j["candidate"], []).append(v)
        if agg == "min":
            return sorted(((c, min(v)) for c, v in per.items()), key=lambda x: x[1])
        return sorted(((c, st.fmean(v)) for c, v in per.items()), key=lambda x: x[1])

    ran = {j["candidate"] for j in d["jobs"]}
    blocked = set(d.get("blocked_candidates") or {})
    # The screen ranks each candidate by its own lowest finite BPB -- the
    # funnel's stated advancement rule, not the mean across learning rates.
    screen = rank("lr_sweep_16m", agg="min")
    at16 = rank("advance_1000")
    at128 = rank("exact_128m_500")
    champ = rank("exact_128m_1000")
    pos = lambda rows, name: next(  # noqa: E731
        (i + 1 for i, (c, _) in enumerate(rows) if c == name), None)

    return {
        "candidates_total": len(ran | blocked),
        "candidates_ran": len(ran),
        "blocked": sorted(blocked),
        "screen_rank_polar": pos(screen, "muon_polar_adamw"),
        "screen_rank_mona": pos(screen, "mona_adamw"),
        "rank16_mona": pos(at16, "mona_adamw"),
        "bpb16_mona": dict(at16).get("mona_adamw"),
        "rank128_mona": pos(at128, "mona_adamw"),
        "n128": len(at128),
        "bpb128_mona": dict(at128).get("mona_adamw"),
        "champion": (d.get("champion") or {}).get("candidate"),
        "champion_bpb": dict(champ).get("muon_polar_adamw"),
        # The stage is named `exact_128m`, where "exact" is the bit-exactness
        # resume gate, not the parameter count -- which is 128.37M, not 128M.
        # §6 stated it as "an exact 128M-parameter model" until this was read
        # off the gate artifact instead of off the stage name.
        "exact_128m_params": ((d.get("champion") or {})
                              .get("winner_exact_gate_evidence") or {})
                             .get("parameter_count"),
        "ci_correction": any("Student-t" in json.dumps(c)
                             for c in (d.get("corrections") or [])),
    }


def collect():
    return {
        "headline": headline(),
        "controls": controls(),
        "mixer_board": mixer_board(),
        "matched_batch_board": matched_batch_board(),
        "optimizer_funnel": optimizer_funnel(),
    }


# ------------------------------------------------------------------ output


def render(f):
    h = f["headline"]
    print("=" * 72)
    print("HEADLINE — 50M tokens, GH200")
    print("=" * 72)
    print(f"  n = {h['n']} seeds, through {h['final_tokens']/1e6:.2f}M tokens")
    for c in h["crossings"]:
        print(f"  crossing at {c['tokens']/1e6:.3f}M tokens — {c['winner']} takes the lead")
    print(f"  final: attention {h['attention_final']:.4f} "
          f"[{h['attention_ci'][0]:.4f}, {h['attention_ci'][1]:.4f}]")
    print(f"         minGRU    {h['mingru_final']:.4f} "
          f"[{h['mingru_ci'][0]:.4f}, {h['mingru_ci'][1]:.4f}]")
    print(f"  advantage {h['advantage']:+.3f}")

    print()
    print("=" * 72)
    print("CONTROLS — one confound at a time")
    print("=" * 72)
    for c in f["controls"]:
        if "error" in c:
            print(f"  {c['label']}: {c['error']}")
            continue
        print(f"  {c['label']}")
        print(f"    n={c['n']}, through {c['through_tokens']/1e6:.2f}M tokens")
        if not c["crossings"]:
            print("    NO CROSSING")
        for x in c["crossings"]:
            print(f"    {x['tokens']/1e6:.2f}M — {x['winner']} takes the lead")

    b = f["mixer_board"]
    print()
    print("=" * 72)
    print("MIXER BOARD — ten arms at 50M tokens")
    print("=" * 72)
    for r in b["rows"]:
        print(f"  {r['arm']:<24}{r['final']:7.3f}  [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
              f"   @ {r['horizon']/1e6:.2f}M")
    print(f"  top two intervals overlap: {b['top_two_overlap']}")
    print(f"  horizons: {b['horizon_min']/1e6:.2f}M-{b['horizon_max']/1e6:.2f}M "
          f"({b['horizon_spread_pct']:.1f}% spread)")
    if b["horizon_spread_pct"] > HORIZON_SPREAD_LIMIT_PCT:
        print(f"  !! WARNING: arms differ by more than "
              f"{HORIZON_SPREAD_LIMIT_PCT}% in horizon; this is not a "
              f"single-horizon ranking and must not be reported as one")

    mb = f["matched_batch_board"]
    if mb:
        print()
        print(f"  matched batch-32 sweep: {len(mb['rows'])} arms, "
              f"attention arm present: {mb['has_attention_arm']}")
        if mb["rows"]:
            r = mb["rows"][0]
            print(f"    best: {r['arm']} {r['final']:.3f} "
                  f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")

    o = f["optimizer_funnel"]
    if o:
        print()
        print("=" * 72)
        print("OPTIMIZER FUNNEL — the second axis")
        print("=" * 72)
        print(f"  {o['candidates_total']} candidates ({o['candidates_ran']} ran, "
              f"{len(o['blocked'])} blocked: {', '.join(o['blocked'])})")
        print(f"  MONA: rank {o['rank16_mona']} at 16M ({o['bpb16_mona']:.6f} BPB), "
              f"rank {o['rank128_mona']} of {o['n128']} at 128M ({o['bpb128_mona']:.6f})")
        print(f"  Polar Express: rank {o['screen_rank_polar']} in the LR screen "
              f"-> champion ({o['champion_bpb']:.6f} BPB)")
        print(f"  recorded champion: {o['champion']}")
        print(f"  two-seed Student-t interval correction recorded: {o['ci_correction']}")


def expected_figures(f):
    """(token, description) for every figure this script derives.

    This list is the contract `--check` enforces, so it must stay exhaustive.
    It used to hold five entries -- two crossings, the advantage and the top
    two board rows -- while the manuscript stated forty-four decimal figures
    and the docstring above promised all of them were re-derived. A check that
    examines 11% of its subject and prints an unqualified OK is worse than no
    check: it certifies the 89% it never read. Anything rendered by `render()`
    belongs here.
    """
    h, c, b = f["headline"], f["controls"], f["mixer_board"]
    mb, o = f["matched_batch_board"], f["optimizer_funnel"]
    cx = h["crossings"]

    want = [
        (f"{cx[0]['tokens']/1e6:.2f}M", "headline: early crossing"),
        (f"{cx[1]['tokens']/1e6:.2f}M", "headline: late crossing"),
        (f"{h['advantage']:.3f}".lstrip("-"), "headline: final advantage"),
        (f"{h['final_tokens']/1e6:.2f}M", "headline: final token count"),
        (f"{h['grid_spacing']/1e6:.2f}M", "protocol: eval grid spacing"),
        (f"{h['attention_final']:.3f}", "headline: attention final"),
        (f"{h['attention_ci'][0]:.3f}", "headline: attention CI low"),
        (f"{h['attention_ci'][1]:.3f}", "headline: attention CI high"),
        (f"{h['mingru_final']:.3f}", "headline: minGRU final"),
        (f"{h['mingru_ci'][0]:.3f}", "headline: minGRU CI low"),
        (f"{h['mingru_ci'][1]:.3f}", "headline: minGRU CI high"),
        (f"n = {h['n']}", "headline: seed count"),
    ]

    late = {}
    for row in c:
        if "error" in row:
            continue
        want.append((f"{row['through_tokens']/1e6:.2f}M",
                     f"control {row['run']}: horizon"))
        for x in row["crossings"]:
            want.append((f"{x['tokens']/1e6:.2f}M",
                         f"control {row['run']}: crossing"))
        if row["crossings"]:
            late[row["run"]] = row["crossings"][-1]["tokens"]

    # §4's load-bearing claim: telling the cosine schedule about the truncation
    # is the only difference between these two rows, and it moves the late
    # crossing by this much. Subtract it here rather than trusting the prose.
    if {"crossover20m_locked", "crossover20m_matched_lr"} <= late.keys():
        shift = late["crossover20m_locked"] - late["crossover20m_matched_lr"]
        want.append((f"{shift/1e6:.2f}M", "controls: schedule-truncation shift"))

    for i, row in enumerate(b["rows"], 1):
        want.append((f"{row['final']:.3f}", f"board rank {i}: mean"))
        want.append((f"{row['ci'][0]:.3f}", f"board rank {i}: CI low"))
        want.append((f"{row['ci'][1]:.3f}", f"board rank {i}: CI high"))

    # The board spans two eval cadences. Both ends and the spread are stated in
    # §5 so the table cannot be read as a single-horizon ranking again.
    want += [
        (f"{b['horizon_min']/1e6:.2f}M", "board: shortest arm horizon"),
        (f"{b['horizon_max']/1e6:.2f}M", "board: longest arm horizon"),
        (f"{b['horizon_spread_pct']:.1f}", "board: horizon spread, percent"),
    ]

    if mb and mb["rows"]:
        r = mb["rows"][0]
        want.append((f"{r['final']:.3f}", "matched batch-32: leader mean"))
        want.append((f"{r['ci'][0]:.3f}", "matched batch-32: leader CI low"))
        want.append((f"{r['ci'][1]:.3f}", "matched batch-32: leader CI high"))

    if o:
        want += [
            (str(o["candidates_total"]), "funnel: candidates enrolled"),
            (str(o["candidates_ran"]), "funnel: candidates that ran"),
            (f"{o['bpb16_mona']:.6f}", "funnel: MONA BPB at 16M"),
            (f"{o['bpb128_mona']:.6f}", "funnel: MONA BPB at 128M"),
            (f"{o['champion_bpb']:.6f}", "funnel: champion BPB"),
        ]
        want += [(name, f"funnel: blocked candidate {name}")
                 for name in o["blocked"]]
        if o.get("exact_128m_params"):
            p = o["exact_128m_params"]
            want.append((f"{p:,}", "funnel: exact_128m parameter count"))
            want.append((f"{p/1e6:.2f}M", "funnel: exact_128m params, millions"))

    # De-duplicate while preserving order; several figures repeat by design
    # (the headline finals are also board rows 1 and 7).
    seen, out = set(), []
    for token, why in want:
        if token not in seen:
            seen.add(token)
            out.append((token, why))
    return out


# Figures the manuscript states that are deliberately not derived from the
# runs. Every entry carries its reason, and anything numeric in the text that
# is neither derived nor listed here fails the check -- that is what stops a
# stale figure surviving in prose the script never reads.
STATED_NOT_DERIVED = {
    "4.0": "CC BY 4.0 licence identifier, not a measurement",
    "6.6M": "the original single-seed claim under test, quoted from §1",
    "7.4M": "the original single-seed claim under test, quoted from §1",
    "1.96": "normal quantile, named in §5 as the wrong multiplier",
    "1.960": "normal quantile, named in §6 as the wrong multiplier",
    "2.776": "Student-t multiplier at df=4, named in §5",
    "12.706": "Student-t multiplier at df=1, named in §6",
    "6.5": "ratio 12.706/1.960, the §6 understatement factor",
    "1.42": "ratio 2.776/1.960, the §5 understatement factor",
    "4.207": "superseded normal-quantile bound, quoted in §5 as superseded",
    "4.221": "superseded normal-quantile bound, quoted in §5 as superseded",
    "4.232": "pre-loss figure §5 corrects; appears in no surviving run",
    "4.210": "pre-loss figure §5 corrects; appears in no surviving run",
    "4.254": "pre-loss figure §5 corrects; appears in no surviving run",
    "4.759": "stale Mamba-2 mean from the pre-regeneration summary.json, §5",
    "4.680": "stale Mamba-2 CI bound from the pre-regeneration summary.json, §5",
    "4.837": "stale Mamba-2 CI bound from the pre-regeneration summary.json, §5",
    "40.16M": "horizon the stale Mamba-2 figure was actually measured at, §5",
    "0.37M": "128.37M minus the 128M an earlier draft claimed, §6",
}

_FIGURE_RE = re.compile(r"(?<![\w.])\d+\.\d+M?(?![\w])")


def check(f):
    """Both directions: derived figures appear, stated figures are accounted for."""
    if not MANUSCRIPT.exists():
        print(f"FAIL: {MANUSCRIPT.name} does not exist")
        return 1
    body = MANUSCRIPT.read_text()

    wanted = expected_figures(f)
    missing = [(token, why) for token, why in wanted if token not in body]

    # Reverse direction. The docstring above promises that a figure the script
    # cannot reproduce is caught; for years it only checked that derived
    # figures were present, which never reads the prose's own numbers.
    derived = {token for token, _ in wanted}
    derived |= {t[:-1] for t in derived if t.endswith("M")}
    stated = sorted(set(_FIGURE_RE.findall(body)))
    unaccounted = [s for s in stated
                   if s not in derived and s not in STATED_NOT_DERIVED]

    if missing or unaccounted:
        if missing:
            print(f"FAIL: {len(missing)} of {len(wanted)} derived figures are "
                  f"absent from {MANUSCRIPT.name} (the prose has drifted):")
            for token, why in missing:
                print(f"  - {why}: derived {token!r}, not found in the text")
        if unaccounted:
            print(f"FAIL: {len(unaccounted)} figure(s) stated in "
                  f"{MANUSCRIPT.name} are neither derived from the runs nor "
                  f"listed in STATED_NOT_DERIVED with a reason:")
            for s in unaccounted:
                print(f"  - {s!r}")
        return 1

    print(f"OK: all {len(wanted)} derived figures appear in {MANUSCRIPT.name}, "
          f"and all {len(stated)} figures it states are accounted for "
          f"({len(STATED_NOT_DERIVED)} by documented exception)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not OUT.is_dir():
        print(f"FAIL: {OUT} does not exist; the run artifacts are what this reads")
        return 2

    f = collect()
    if args.json:
        print(json.dumps(f, indent=2))
        return 0
    if args.check:
        return check(f)
    render(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
