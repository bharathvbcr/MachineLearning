#!/usr/bin/env python3
"""Alternate the tessl and torch/MLX attention lanes so clock drift cancels.

Same protocol as `paired_cross_runtime.py`, for the attention configs. Drift is
15-20% on this hardware, so a paired sweep is what separates a real ratio from
a scheduling artifact -- and it stays necessary as the gap narrows, even where
today's ratios are far outside any drift band.

Every reported ratio covers the whole requested config set or the run fails.

  python3 bench/attn_paired.py --rounds 5 --out bench/results/attn_ladder.json
"""
import argparse, json, math, os, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
RUST_BIN = os.path.join(CRATE, "target", "release", "bench_flash_attn")
PY_LANE = os.path.join(HERE, "flash_attn_torch_mlx.py")

sys.path.insert(0, HERE)
from flash_attn_torch_mlx import BY_LABEL  # noqa: E402


def _run(cmd, env, what, key):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{what} exited {r.returncode}:\n{r.stderr.strip()}")
    try:
        rows = json.loads(r.stdout)
    except ValueError as exc:
        raise SystemExit(f"{what}: stdout is not JSON ({exc}):\n{r.stdout[:400]}")
    out = {}
    for x in rows:
        ms = x["median_ms"]
        if not (isinstance(ms, (int, float)) and math.isfinite(ms) and ms > 0):
            raise SystemExit(f"{what}: {x['cfg']} reported median_ms={ms}")
        out[(x["cfg"], key(x))] = ms
    return out


def run_rust(cfgs, iters, warmup):
    env = dict(os.environ, BENCH_ATTN_CFGS=",".join(cfgs),
               BENCH_ITERS=str(iters), BENCH_WARMUP=str(warmup))
    # The binary emits one row per implementation, so the key has to be the
    # runtime tag; keying every row "tessl" silently kept only the last.
    return _run([RUST_BIN], env, "bench_flash_attn", lambda x: x["runtime"])


def run_py(cfgs, iters, warmup, lanes):
    return _run([sys.executable, PY_LANE, "--cfgs", ",".join(cfgs), "--lanes", lanes,
                 "--iters", str(iters), "--warmup", str(warmup)],
                dict(os.environ), "flash_attn_torch_mlx.py", lambda x: x["runtime"])


def missing_coverage(per_round, cfgs, lanes, tessl_key="tessl"):
    """(cfg, lane) pairs absent from at least one round."""
    gaps = set()
    for c in cfgs:
        for a, b in per_round:
            if (c, tessl_key) not in a:
                gaps.add(f"{c}/{tessl_key}")
            for ln in lanes:
                if (c, ln) not in b:
                    gaps.add(f"{c}/{ln}")
    return sorted(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lanes", default="torch,mlx")
    ap.add_argument("--impl", default="tessl-decode",
                    choices=["tessl", "tessl-tiled", "tessl-decode", "tessl-rows"],
                    help="which tessl implementation to compare")
    ap.add_argument("--cfgs", default="all")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.rounds < 1:
        raise SystemExit(f"--rounds must be >= 1, got {args.rounds}")
    if args.iters < 1:
        raise SystemExit(f"--iters must be >= 1, got {args.iters}")
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be >= 0, got {args.warmup}")
    if not os.path.exists(RUST_BIN):
        raise SystemExit(f"{RUST_BIN} not built. "
                         "cargo build --release --bin bench_flash_attn")
    cfgs = list(BY_LABEL) if args.cfgs == "all" else [x.strip() for x in args.cfgs.split(",")]
    bad = [x for x in cfgs if x not in BY_LABEL]
    if bad:
        raise SystemExit(f"--cfgs: unknown {bad}; expected from {list(BY_LABEL)}")
    if args.impl == "tessl-decode":  # noqa: SIM102
        # FlashDecoding is defined only at Tq == 1. Narrowing is announced
        # rather than silent: a geomean over a quietly reduced config set is
        # the same defect this driver refuses elsewhere.
        decode_only = [c for c in cfgs if BY_LABEL[c]["tq"] == 1]
        dropped = [c for c in cfgs if c not in decode_only]
        if not decode_only:
            raise SystemExit(
                f"--impl tessl-decode needs at least one Tq == 1 config; "
                f"{cfgs} are all prefill")
        if dropped:
            print(f"--impl tessl-decode: skipping {len(dropped)} prefill config(s) "
                  f"{dropped} -- the decode path is Tq == 1 only",
                  file=sys.stderr)
        cfgs = decode_only

    sel = [x.strip() for x in args.lanes.split(",") if x.strip()]
    bad = [x for x in sel if x not in ("torch", "mlx")]
    if bad:
        raise SystemExit(f"--lanes: unknown {bad}; expected torch and/or mlx")
    lane_keys = [{"torch": "torch-mps", "mlx": "mlx"}[x] for x in sel]

    per_round = []
    for r in range(args.rounds):
        if r % 2 == 0:
            a = run_rust(cfgs, args.iters, args.warmup)
            b = run_py(cfgs, args.iters, args.warmup, args.lanes)
        else:
            b = run_py(cfgs, args.iters, args.warmup, args.lanes)
            a = run_rust(cfgs, args.iters, args.warmup)
        per_round.append((a, b))
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr)

    gaps = missing_coverage(per_round, cfgs, lane_keys, args.impl)
    if gaps:
        raise SystemExit(
            f"{len(gaps)} (config, lane) pair(s) absent from at least one round -- "
            f"{gaps[:6]}. Refusing to report ratios over part of the set.")

    doc = dict(rounds=args.rounds, iters=args.iters, warmup=args.warmup,
               cfgs=cfgs, lanes=lane_keys, tessl_impl=args.impl, comparisons=[])
    for ln in lane_keys:
        rows = []
        for c in cfgs:
            # >1 means tessl is SLOWER. Stated in the header rather than left to
            # the reader, because the GEMM sweep's ratios point the other way.
            rr = [a[(c, args.impl)] / b[(c, ln)] for a, b in per_round]
            rows.append((c, statistics.median(rr), min(rr), max(rr)))
        meds = [m for _, m, _, _ in rows]
        if any(m <= 0 or not math.isfinite(m) for m in meds):
            raise SystemExit(f"{ln}: non-positive median ratio in {meds}")
        g = math.exp(sum(math.log(m) for m in meds) / len(meds))
        print(f"\n{args.impl} / {ln} wall clock   ({args.rounds} alternating rounds, "
              f"{len(rows)}/{len(cfgs)} configs).  >1 means tessl is slower.")
        print(f"  {'config':<24}{'median':>9}{'min':>9}{'max':>9}")
        for c, med, lo, hi in rows:
            print(f"  {c:<24}{med:>8.1f}x{lo:>8.1f}x{hi:>8.1f}x")
        worst = max(rows, key=lambda r: r[1])
        print(f"  {'GEOMEAN':<24}{g:>8.1f}x   worst {worst[1]:.1f}x at {worst[0]}")
        doc["comparisons"].append(dict(
            lane=ln, tessl_impl=args.impl, geomean_tessl_over_other=g,
            configs_covered=len(rows), configs_requested=len(cfgs),
            worst_config=worst[0], worst_median=worst[1],
            best_median=min(meds),
            per_config=[dict(cfg=c, median=m, min=lo, max=hi) for c, m, lo, hi in rows]))

    if args.out:
        with open(args.out, "w") as f:
            f.write(json.dumps(doc, indent=2))
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
