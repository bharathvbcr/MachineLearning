#!/usr/bin/env python3
"""E11 phase 1: re-index the suite-26 board on wall clock instead of tokens.

PAPER 4.5 ranks ten arms at a matched TOKEN budget. Practitioners do not buy
tokens, they buy time, and 6.4 already shows hybrid throughput reorders with batch
size. So "the best hybrid ties attention" is a claim measured on one cost basis.

This re-reads each arm's committed loss curve at the token count it would have
reached in the wall clock the SLOWEST arm needed for the full budget, and prints
the two boards side by side. No GPU: it reads the committed metrics.jsonl only.

LIMITS, carried into the output rather than left here:
  * Suite 22/26 throughputs are CONTENDED (2 jobs/GPU, gap E6). The re-indexing is
    only valid within-suite at the same concurrency; it is not a machine spec.
  * tok_s is the median over a run's train records, so eval pauses are excluded.
    Wall clock per arm is therefore compute time, not elapsed time.
  * An arm is re-read at the largest eval marker at or before its budget. Markers
    are 200-step spaced, so the re-index has that granularity and no more.

    python3 scripts/e11_wallclock_board.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanolab.crossover_replicate import mean_ci, measured_rate_by_arm  # noqa: E402

SUITES = ("nanolab/out/crossover50m", "nanolab/out/crossover50m_matched32",
          "nanolab/out/crossover50m_ratio32")


def load() -> dict:
    """{arm: {seed: {"rate": tok/s, "curve": [(tokens, val)...]}}}"""
    out: dict = defaultdict(dict)
    for suite in SUITES:
        for cfgp in sorted((ROOT / suite).glob("*/config.json")):
            mp = cfgp.with_name("metrics.jsonl")
            if not mp.exists():
                continue
            try:
                cfg = json.loads(cfgp.read_text())
            except json.JSONDecodeError:
                continue
            if cfg.get("batch_size") != 32 or cfg.get("block_size") != 512:
                continue
            arm = cfgp.parent.name.split("_s")[0].split("_", 1)[-1]
            rates, curve = [], []
            for line in mp.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "train" and r.get("tok_s"):
                    rates.append(float(r["tok_s"]))
                elif r.get("event") == "eval" and r.get("tokens") is not None \
                        and r.get("val_loss") is not None:
                    curve.append((int(r["tokens"]), float(r["val_loss"])))
            if rates and curve:
                out[arm][int(cfg["seed"])] = {
                    "rate": st.median(rates), "curve": sorted(curve)}
    return out


def at_or_before(curve, tokens):
    ok = [(t, v) for t, v in curve if t <= tokens]
    return ok[-1] if ok else None


def main() -> int:
    data = load()
    if not data:
        print("no committed runs found", file=sys.stderr)
        return 1
    arms = {a: s for a, s in data.items() if len(s) >= 3}
    budget = max(t for s in arms.values() for r in s.values()
                 for t, _ in r["curve"])
    # The wall-clock budget is the FASTEST arm's time to finish the token budget.
    #
    # Anchoring on the slowest arm instead makes this a tautology: every faster arm
    # would overshoot the budget in that time, get clamped to it, and the two boards
    # come out identical by construction. The question is "at equal TIME, who is
    # ahead", and at equal time the fast arm has seen MORE data while the slow arm
    # has seen less. So T must be small enough that no arm runs past its measured
    # curve -- which is exactly the fastest arm's completion time.
    # ONE owner for "this arm's throughput": phase 2 sizes its wall-clock budgets
    # from the same function. A second median here drifted from it -- keying runs
    # by seed silently dropped an arm's suite-22 runs when suite 26 reused the
    # seed -- and phase 2's wall clock would then not be phase 1's wall clock.
    rate = {a: r for a, r in measured_rate_by_arm().items() if a in arms}
    fastest, top = max(rate.items(), key=lambda kv: kv[1])
    wall = budget / top

    print(f"budget {budget:,} tokens")
    print(f"wall-clock budget = {wall/60:.1f} min = what the FASTEST arm ({fastest}, "
          f"{top:,.0f} tok/s) needs")
    print("to finish it. Every arm is then read at the tokens IT reaches in that "
          "time.\n")
    print(f"{'arm':<24}{'n':>3}{'tok/s':>9}{'token-matched (95% t)':>26}"
          f"{'wall-matched (95% t)':>26}{'rank':>10}")

    rows = []
    for arm, seeds in arms.items():
        reach = int(rate[arm] * wall)
        # Keyed BY SEED, not appended in iteration order. The sign test below is a
        # paired comparison, and pairing two arms by list position silently pairs
        # different seeds whenever the two arms' runs were globbed in a different
        # order -- which they are, when an arm appears in more than one suite.
        tok_by, wall_by = {}, {}
        for sd, r in seeds.items():
            end = at_or_before(r["curve"], budget)
            got = at_or_before(r["curve"], min(reach, budget))
            if end:
                tok_by[sd] = end[1]
            if got:
                wall_by[sd] = got[1]
        tok_vals = [tok_by[k] for k in sorted(tok_by)]
        wall_vals = [wall_by[k] for k in sorted(wall_by)]
        if not tok_vals or not wall_vals:
            continue
        rows.append({"arm": arm, "rate": rate[arm], "reach": min(reach, budget),
                     "tok": mean_ci(tok_vals), "wall": mean_ci(wall_vals),
                     "n": len(tok_vals), "per_seed_wall": wall_by,
                     "per_seed_tok": tok_by, "seeds": sorted(seeds)})

    tok_rank = {r["arm"]: i for i, r in
                enumerate(sorted(rows, key=lambda r: r["tok"][0]), 1)}
    wall_rank = {r["arm"]: i for i, r in
                 enumerate(sorted(rows, key=lambda r: r["wall"][0]), 1)}
    for r in sorted(rows, key=lambda r: r["tok"][0]):
        a = r["arm"]
        move = wall_rank[a] - tok_rank[a]
        flag = f"{tok_rank[a]}->{wall_rank[a]}" + ("*" if move else "")
        tm = f"{r['tok'][0]:.4f} [{r['tok'][1]:.4f}, {r['tok'][2]:.4f}]"
        wm = f"{r['wall'][0]:.4f} [{r['wall'][1]:.4f}, {r['wall'][2]:.4f}]"
        print(f"  {a:<22}{r['n']:>3}{r['rate']:>9,.0f}{tm:>26}{wm:>26}{flag:>10}")

    # A rank change is only a finding if it survives the seeds. Every adjacent
    # pair that swapped is checked two ways: do the intervals separate, and does
    # the winner win on every seed? PAPER 5.3 is the precedent -- its champion
    # rested on a sign-consistent 2-of-2, not on separated intervals, and saying
    # which one carries a result is the difference between a ranking and a mean.
    by_arm = {r["arm"]: r for r in rows}
    order_tok = [r["arm"] for r in sorted(rows, key=lambda r: r["tok"][0])]
    order_wall = [r["arm"] for r in sorted(rows, key=lambda r: r["wall"][0])]
    swaps = []
    for i, a in enumerate(order_wall):
        for b in order_wall[i + 1:]:
            if order_tok.index(a) > order_tok.index(b):
                swaps.append((a, b))
    if swaps:
        print(f"\n  {len(swaps)} pair(s) actually swapped order. Each, checked "
              "against the seeds:")
        print(f"    {'pair':<46}{'gap':>9}{'sep?':>6}{'sign':>9}")
        for a, b in swaps:
            ra, rb = by_arm[a], by_arm[b]
            gap = rb["wall"][0] - ra["wall"][0]
            sep = ra["wall"][2] < rb["wall"][1]
            shared = sorted(set(ra["per_seed_wall"]) & set(rb["per_seed_wall"]))
            wins = sum(1 for sd in shared
                       if ra["per_seed_wall"][sd] < rb["per_seed_wall"][sd])
            n = len(shared)
            print(f"    {a + ' < ' + b:<46}{gap:>+9.4f}"
                  f"{('yes' if sep else 'NO'):>6}{f'{wins}/{n}':>9}")
        print("    sep? = 95% t intervals do not overlap. sign = seeds where the")
        print("    wall-matched winner is ahead. A swap with sep=NO and sign<n/n is")
        print("    inside the noise and must not be reported as a reordering.")

    moved = [a for a in tok_rank if tok_rank[a] != wall_rank[a]]
    print()
    if moved:
        print(f"  {len(moved)} of {len(rows)} arms change rank on the wall-clock "
              "basis:")
        print("    " + ", ".join(sorted(moved)))
        head = sorted(rows, key=lambda r: r["tok"][0])[:2]
        gap_tok = head[1]["tok"][0] - head[0]["tok"][0]
        gap_wall = head[1]["wall"][0] - head[0]["wall"][0]
        print(f"  Section 4.5's headline pair ({head[0]['arm']} vs {head[1]['arm']}) "
              f"goes from")
        print(f"  {gap_tok:+.4f} token-matched to {gap_wall:+.4f} wall-matched -- the "
              f"tie is {abs(gap_wall/gap_tok):.0f}x wider on time.")
        print("  Cost basis is a recipe axis. Phase 2 (true matched-wall-clock runs)")
        print("  is worth pricing.")
    else:
        print(f"  No arm changes rank. The board is stable across the two cost")
        print("  bases, and E11 phase 2 is NOT worth its GPU time.")
    print("\n  THE LIMIT THAT MATTERS -- this re-index is BIASED AGAINST SLOW ARMS.")
    print("  Every committed run anneals a cosine over the FULL 50M budget. Reading")
    print("  an arm at 17.8M tokens therefore reads a run that was scheduled for 50M")
    print("  and stopped early, with its learning rate still high and un-annealed. A")
    print("  true 17.8M-budget run would anneal over 17.8M and finish BETTER. PAPER")
    print("  4.3 measures exactly this: truncating a run versus re-scheduling it moved")
    print("  the crossing by 2.2M tokens, 18% of its location.")
    print("  So the reordering above is an UPPER BOUND on the real effect, and the")
    print("  slower an arm is, the more it is penalised. That is precisely why phase 2")
    print("  runs each arm with its own cosine instead of re-reading these curves.")
    print("\n  Other limits: suite 22/26 throughputs are contended (2 jobs/GPU, gap")
    print("  E6), so this is valid within-suite at that concurrency and is not a")
    print("  machine spec. tok_s excludes eval pauses, so 'wall clock' here is compute")
    print("  time. Re-index granularity is one eval marker (200 steps).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
