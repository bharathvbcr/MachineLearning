#!/usr/bin/env python3
"""Alternate the tessl and PyTorch/MLX GEMM lanes so clock drift cancels.

Running the Rust sweep once and the Python sweep once -- minutes apart, in
separate processes -- puts all the drift between them into the ratio. Two such
runs of the identical benchmark disagreed by 16-21% on the torch lane alone,
which is larger than most of the differences being reported.

This alternates the lanes round by round and reports the median of the
per-round ratios, plus the observed spread, so a claim can be checked against
its own noise floor instead of resting on a single ordering.

Every reported geomean covers the **whole** requested ladder or the run fails.
A geomean over the three shapes that happened to report used to print exactly
like a geomean over all eight.

  python3 bench/paired_cross_runtime.py --lanes torch,mlx \\
      --rounds 5 --out bench/results/gemm_speed_ladder.json
"""
import argparse, json, math, os, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
# Resolved from this file, not the caller's working directory. The sibling
# audit script already had to learn this.
RUST_BIN = os.path.join(CRATE, "target", "release", "bench_gemm_sweep")
PY_SWEEP = os.path.join(HERE, "gemm_sweep_mlx.py")

DEFAULT_SHAPES = ("512x512x512,1024x1024x1024,2048x2048x2048,4096x4096x4096,"
                  "2048x768x768,8192x3072x768,8192x768x3072,4096x4096x1024")

# (tessl lane, comparison lane, label). The comparison lane names which
# external runtime and dtype must be selected for the pair to be measurable.
PAIRS = [
    ("tensorops-f32", "mps-f32", "torch", "f32", "tessl f32-exact vs torch f32"),
    ("tensorops-tf32", "mps-f32", "torch", "f32", "tessl tf32 vs torch f32"),
    ("tensorops-bf16", "mps-bf16", "torch", "bf16", "tessl bf16 vs torch bf16"),
    ("tensorops-f32", "mlx-f32", "mlx", "f32", "tessl f32-exact vs MLX f32"),
    ("tensorops-tf32", "mlx-f32", "mlx", "f32", "tessl tf32 vs MLX f32"),
    ("tensorops-bf16", "mlx-bf16", "mlx", "bf16", "tessl bf16 vs MLX bf16"),
]


def missing_coverage(per_round, labels, mine, theirs):
    """(shape, lane) pairs absent from at least one round.

    Formed before any ratio is, because a shape missing from one side is a
    measurement that did not happen -- and a geomean that quietly drops it is
    indistinguishable from one that covered the whole ladder.
    """
    gaps = set()
    for s in labels:
        for rd in per_round:
            if (s, mine) not in rd[0]:
                gaps.add(f"{s}/{mine}")
            if (s, theirs) not in rd[1]:
                gaps.add(f"{s}/{theirs}")
    return sorted(gaps)


def _run(cmd, env, what):
    """Surface the child's stderr. `check=True` raised a CalledProcessError
    whose message omitted the one thing that explains the failure."""
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{what} exited {r.returncode}:\n{r.stderr.strip()}")
    try:
        rows = json.loads(r.stdout)
    except ValueError as exc:
        raise SystemExit(f"{what}: stdout is not JSON ({exc}):\n{r.stdout[:400]}")
    out = {}
    for x in rows:
        g = x["gflops"]
        if not (isinstance(g, (int, float)) and math.isfinite(g) and g > 0):
            raise SystemExit(f"{what}: {x['shape']}/{x['backend']} reported gflops={g}")
        out[(x["shape"], x["backend"])] = g
    return out


def run_rust(shapes, iters, warmup):
    env = dict(os.environ, BENCH_SHAPES=shapes, BENCH_ITERS=str(iters),
               BENCH_WARMUP=str(warmup))
    return _run([RUST_BIN], env, "bench_gemm_sweep")


def run_py(shapes, iters, warmup, lanes, dtypes):
    env = dict(os.environ, BENCH_SHAPES=shapes)
    return _run([sys.executable, PY_SWEEP, "--lanes", lanes, "--dtypes", dtypes,
                 "--iters", str(iters), "--warmup", str(warmup)], env,
                "gemm_sweep_mlx.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lanes", default="torch")
    ap.add_argument("--dtypes", default="f32,bf16")
    ap.add_argument("--shapes", default=DEFAULT_SHAPES)
    ap.add_argument("--out")
    args = ap.parse_args()

    # A zero-round run used to print nothing and exit 0, which reads as a clean
    # sweep rather than as a sweep that never happened.
    if args.rounds < 1:
        raise SystemExit(f"--rounds must be >= 1, got {args.rounds}")
    if args.iters < 1:
        raise SystemExit(f"--iters must be >= 1, got {args.iters}")
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be >= 0, got {args.warmup}")
    if not os.path.exists(RUST_BIN):
        raise SystemExit(f"{RUST_BIN} not built. "
                         "cargo build --release --bin bench_gemm_sweep")

    labels = [s.strip() for s in args.shapes.split(",") if s.strip()]
    for s in labels:
        d = s.split("x")
        if len(d) != 3 or not all(p.isdigit() and int(p) > 0 for p in d):
            raise SystemExit(f"--shapes entry must be MxNxK with positive dims, got {s!r}")
    if len(set(labels)) != len(labels):
        raise SystemExit(f"--shapes contains duplicates: {labels}")

    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    dtypes = [x.strip() for x in args.dtypes.split(",") if x.strip()]
    pairs = [p for p in PAIRS if p[2] in lanes and p[3] in dtypes]
    if not pairs:
        raise SystemExit(
            f"--lanes {lanes} x --dtypes {dtypes} selects no comparison pair; "
            f"available: {sorted({(p[2], p[3]) for p in PAIRS})}")

    per_round = []
    for r in range(args.rounds):
        # Alternate which lane goes first so a warm-up asymmetry cannot favour
        # the same side every round.
        if r % 2 == 0:
            a = run_rust(args.shapes, args.iters, args.warmup)
            b = run_py(args.shapes, args.iters, args.warmup, args.lanes, args.dtypes)
        else:
            b = run_py(args.shapes, args.iters, args.warmup, args.lanes, args.dtypes)
            a = run_rust(args.shapes, args.iters, args.warmup)
        per_round.append((a, b))
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr)

    # Coverage is checked before any ratio is formed. A shape missing from one
    # side is a measurement that did not happen, and a geomean that quietly
    # drops it is indistinguishable from one that covered everything.
    for mine, theirs, _, _, label in pairs:
        missing = missing_coverage(per_round, labels, mine, theirs)
        if missing:
            raise SystemExit(
                f"{label}: {len(missing)} (shape, lane) pair(s) absent from at "
                f"least one round -- {missing[:6]}. Refusing to report a geomean "
                "over part of the ladder as if it covered all of it.")

    doc = dict(rounds=args.rounds, iters=args.iters, warmup=args.warmup,
               shapes=labels, lanes=lanes, dtypes=dtypes, comparisons=[], peak_gflops=[])

    for mine, theirs, _, _, label in pairs:
        rows = []
        for s in labels:
            rr = [a[(s, mine)] / b[(s, theirs)] for a, b in per_round]
            rows.append((s, statistics.median(rr), min(rr), max(rr)))
        meds = [m for _, m, _, _ in rows]
        if any(m <= 0 or not math.isfinite(m) for m in meds):
            raise SystemExit(f"{label}: non-positive median ratio in {meds}")
        g = math.exp(sum(math.log(m) for m in meds) / len(meds))

        print(f"\n{label}   ({args.rounds} alternating rounds, "
              f"{len(rows)}/{len(labels)} shapes)")
        print(f"  {'shape':<16}{'median':>9}{'min':>9}{'max':>9}")
        for s, med, lo, hi in rows:
            print(f"  {s:<16}{med:>8.2f}x{lo:>8.2f}x{hi:>8.2f}x")
        worst = min(rows, key=lambda r: r[2])
        print(f"  {'GEOMEAN of medians':<16}{g:>8.2f}x   worst (shape, round) "
              f"{worst[2]:.2f}x at {worst[0]}")

        doc["comparisons"].append(dict(
            label=label, tessl_lane=mine, other_lane=theirs,
            geomean=g, shapes_covered=len(rows), shapes_requested=len(labels),
            worst_shape=min(rows, key=lambda r: r[1])[0],
            worst_median=min(meds), best_median=max(meds),
            worst_single_round=worst[2], worst_single_round_shape=worst[0],
            per_shape=[dict(shape=s, median=m, min=lo, max=hi)
                       for s, m, lo, hi in rows]))

    # Peak throughput per lane, over every shape and round actually measured.
    lane_names = sorted({b for rd in per_round for (_, b) in rd[0]} |
                        {b for rd in per_round for (_, b) in rd[1]})
    print(f"\n{'lane':<18}{'peak GFLOP/s':>14}  at")
    for lane in lane_names:
        best, at = 0.0, None
        for a, b in per_round:
            for src in (a, b):
                for s in labels:
                    v = src.get((s, lane))
                    if v is not None and v > best:
                        best, at = v, s
        if at is None:
            continue
        doc["peak_gflops"].append(dict(lane=lane, peak_gflops=best, shape=at))
        print(f"{lane:<18}{best:>14.0f}  {at}")

    if args.out:
        with open(args.out, "w") as f:
            f.write(json.dumps(doc, indent=2))
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
