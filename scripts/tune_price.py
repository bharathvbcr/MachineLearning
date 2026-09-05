"""Price the E28-E35 program from MEASURED per-arm rates, not estimates.

The handoff table's GPU-hours came from elapsed times of past suites divided by
their tenancy. This recomputes them from `nanolab.sweep_gpu arm` rows plus the
eval cadence the recipe actually implies, so a board's cost is
    train:  max_steps            x ms_per_step
    eval:   n_evals x eval_iters x fwd_ms
and the two are reported separately, because `eval_iters` is a recipe field
(changing it moves every board's markers) while tenancy is not.

    python scripts/tune_price.py --cost nanolab/out/_tune/arm_cost_t1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nanolab.crossover_replicate import scale_to_token_budget  # noqa: E402

RATE = 2.29  # $/hour, Lambda GH200 on-demand

# (board, arms, seeds, token budget) -- from the handoff's section 5 table.
BOARDS = [
    ("E29 moe raw router", ["attention", "moe_e1k1_raw", "moe_e4k1_raw", "moe_e8k1_raw"], 5, 50_000_000),
    ("E30a attention into ratioplace32", ["attention"], 5, 50_000_000),
    ("E30b parity hybrids", ["attention", "hybrid_mingru8_attn4_x1", "hybrid_mingru_periodic_x1"], 5, 50_000_000),
    ("E31 tie x value-residual", ["attention", "attention_untied", "attention_novr", "attention_untied_novr"], 5, 50_000_000),
    ("E33 depth for width", ["attention", "attn6_w512", "attn6_w576", "w384_attention_lr10"], 5, 50_000_000),
    ("E34 copy-loss probe", ["attention", "mingru", "hybrid_mingru8_attn4"], 5, 50_000_000),
    ("E35 w384 ladder rung 1", ["w384_attention_lr80", "w384_mingru_lr40",
                                "w384_hybrid_mingru8_attn4_lr40", "w384_hybrid_mingru8_attn4_lr40"], 5, 200_000_000),
    ("E35b w384 ladder rung 2", ["w384_attention_lr80", "w384_mingru_lr40",
                                 "w384_hybrid_mingru8_attn4_lr40", "w384_hybrid_mingru8_attn4_lr40"], 5, 800_000_000),
]


def job_seconds(row: dict, budget: int, eval_iters: int = 20) -> tuple[float, float]:
    s = scale_to_token_budget(32, 512, token_budget=budget)
    train = s["max_steps"] * row["ms_per_step"] / 1000
    n_evals = s["max_steps"] // s["eval_interval"]
    ev = n_evals * eval_iters * row["fwd_ms"] / 1000
    return train, ev


def arm_speedup(arm: str, use_measured: bool, allow_mps: bool, fallback: float) -> float:
    """Best measured aggregate-throughput multiplier for this arm, or fallback."""
    if not use_measured:
        return fallback
    best = 1.0
    tags = ["nomps"] + (["mps"] if allow_mps else [])
    for tag in tags:
        f = REPO / f"nanolab/out/_tune/tenancy_{tag}_{arm}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        ok = {int(k): v["agg_tok_s"] for k, v in d["rows"].items() if v.get("ok")}
        if 1 in ok and ok:
            best = max(best, max(ok.values()) / ok[1])
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", default="nanolab/out/_tune/arm_cost_t1.json")
    ap.add_argument("--tenancy-speedup", type=float, default=1.0,
                    help="fallback multiplier for arms with no measured tenancy "
                         "curve (1.0 = serial)")
    ap.add_argument("--use-measured-tenancy", action="store_true",
                    help="read scripts/tune_tenancy.py output per arm and apply "
                         "each arm's own best speed-up instead of one global "
                         "number -- the sign differs by arm, so a single "
                         "multiplier is wrong for any mixed board")
    ap.add_argument("--mps", action="store_true",
                    help="with --use-measured-tenancy, allow the MPS rows to win")
    ap.add_argument("--startup-s", type=float, default=35.0,
                    help="per-job process start: import, model build, corpus to GPU")
    args = ap.parse_args()

    cost = json.loads(Path(REPO / args.cost).read_text())["arm"]
    tot_h = tot_train = tot_eval = 0.0
    print(f"{'board':<36}{'jobs':>5}{'train h':>9}{'eval h':>8}{'start h':>8}{'GPU-h':>8}{'$':>8}")
    print("-" * 82)
    for name, arms, seeds, budget in BOARDS:
        tr = ev = 0.0
        missing = [a for a in arms if a not in cost or not cost[a].get("ok")]
        if missing:
            print(f"{name:<36}  -- no measured row for {missing}")
            continue
        sped = 0.0
        for a in arms:
            t, e = job_seconds(cost[a], budget)
            tr += t * seeds
            ev += e * seeds
            # Each arm carries its own tenancy multiplier: attention loses at
            # tenancy>1 without MPS while gdn gains 1.53x, so one board-wide
            # number would be wrong for any board mixing the two.
            sped += (t + e) * seeds / arm_speedup(
                a, args.use_measured_tenancy, args.mps, args.tenancy_speedup)
        n_jobs = len(arms) * seeds
        st = n_jobs * args.startup_s
        h = (sped + st) / 3600
        tot_h += h
        tot_train += tr / 3600
        tot_eval += ev / 3600
        print(f"{name:<36}{n_jobs:>5}{tr/3600:>9.2f}{ev/3600:>8.2f}{st/3600:>8.2f}"
              f"{h:>8.2f}{h*RATE:>8.2f}")
    print("-" * 82)
    print(f"{'TOTAL (E29-E35b, 50M/200M/800M boards)':<36}{'':>5}{tot_train:>9.2f}"
          f"{tot_eval:>8.2f}{'':>8}{tot_h:>8.2f}{tot_h*RATE:>8.2f}")
    print(f"\n  eval is {tot_eval/(tot_train+tot_eval)*100:.0f}% of compute "
          f"(eval_iters=20 is a recipe field: changing it moves every board's markers)")
    if args.use_measured_tenancy:
        print(f"  tenancy: each arm's own measured best"
              f"{' (MPS rows allowed)' if args.mps else ' (no MPS)'}")
    else:
        print(f"  tenancy speedup applied: {args.tenancy_speedup:.2f}x")
    print(f"  excludes E28/E32 (recall grid: different shape, priced separately)")


if __name__ == "__main__":
    main()
