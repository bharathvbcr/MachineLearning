#!/usr/bin/env python3
"""Adversarial tests for the parity scorer in `gemm_sweep_mlx.py`.

Every case here fabricates a dump on disk and asserts the scorer's response.
No GPU and no `bench_gemm_sweep` run is needed, so this is runnable anywhere --
including the hosted CI runners that cannot reach MPP TensorOps.

The bar each case defends is the same one: a check that could not run must
never report the same result as a check that ran and passed. Every case below
corresponds to a way the previous harness returned a clean-looking report over
a dump it had not actually verified.

  python3 bench/test_parity_harness.py
"""
import json, os, shutil, sys, tempfile
import numpy as np

HERE_BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_BENCH)
from gemm_sweep_mlx import parity  # noqa: E402

M, N, K = 32, 24, 16
LANES = ["tensorops-f32", "simdgroup-f32", "tensorops-bf16", "tensorops-tf32"]
FAILURES = []


def build_dump(root, seeds=2, lanes=LANES, err_scale=None):
    """A structurally valid dump. `err_scale[i]` perturbs seed i's results so a
    test can pin down which seed the aggregate reports."""
    os.makedirs(root, exist_ok=True)
    names = []
    for i in range(seeds):
        sd = os.path.join(root, f"seed_{i:02}")
        os.makedirs(sd, exist_ok=True)
        names.append(f"seed_{i:02}")
        rng = np.random.default_rng(1000 + i)
        a = rng.uniform(-1, 1, (M, K)).astype(np.float32)
        b = rng.uniform(-1, 1, (K, N)).astype(np.float32)
        np.save(os.path.join(sd, "parity_a.npy"), a)
        np.save(os.path.join(sd, "parity_b.npy"), b)
        c = (a.astype(np.float64) @ b.astype(np.float64))
        if err_scale is not None:
            # Relative to the reference, so the perturbation scales with each
            # element's own error budget instead of blowing it on small ones.
            c = c * (1.0 + err_scale[i])
        for lane in lanes:
            np.save(os.path.join(sd, f"parity_c_{lane}.npy"), c.astype(np.float32))
    with open(os.path.join(root, "parity_manifest.json"), "w") as f:
        json.dump(dict(shape="synthetic", dist="uniform", m=M, n=N, k=K,
                       lanes=lanes, seeds=names), f)
    return root


def edit_manifest(root, **kw):
    path = os.path.join(root, "parity_manifest.json")
    with open(path) as f:
        man = json.load(f)
    man.update(kw)
    with open(path, "w") as f:
        json.dump(man, f)


def check(name, fn, expect_reject, needle=None):
    """`expect_reject` -- the scorer must refuse. Otherwise it must succeed."""
    try:
        rows = fn()
    except SystemExit as exc:
        if not expect_reject:
            FAILURES.append(f"{name}: rejected a valid dump -- {exc}")
            print(f"  FAIL  {name}: unexpected rejection: {exc}")
            return None
        msg = str(exc)
        if needle and needle.lower() not in msg.lower():
            FAILURES.append(f"{name}: rejected, but message lacks {needle!r}: {msg}")
            print(f"  FAIL  {name}: message lacks {needle!r}")
            return None
        print(f"  ok    {name}: rejected -- {msg.splitlines()[0][:88]}")
        return None
    except Exception as exc:  # a raw traceback is not a usable diagnosis
        FAILURES.append(f"{name}: raised {type(exc).__name__} instead of SystemExit: {exc}")
        print(f"  FAIL  {name}: raw {type(exc).__name__}: {exc}")
        return None
    if expect_reject:
        FAILURES.append(f"{name}: ACCEPTED a dump it should have refused")
        print(f"  FAIL  {name}: accepted")
        return None
    print(f"  ok    {name}: accepted")
    return rows


# (label, argv tail, env overrides, expected substring in the error).
# Every one of these either panicked or silently did the wrong thing before.
CLI_CASES = [
    ("--dump-parity with no directory", ["--dump-parity"], {}, "requires a directory"),
    ("--dump-parity followed by a flag", ["--dump-parity", "--x"], {}, "requires a directory"),
    ("BENCH_ITERS=0", [], {"BENCH_ITERS": "0"}, "below the minimum"),
    ("BENCH_ITERS=abc", [], {"BENCH_ITERS": "abc"}, "not a non-negative integer"),
    ("BENCH_ITERS=-1", [], {"BENCH_ITERS": "-1"}, "not a non-negative integer"),
    ("BENCH_WARMUP=xyz", [], {"BENCH_WARMUP": "xyz"}, "not a non-negative integer"),
    ("BENCH_SHAPES=garbage", [], {"BENCH_SHAPES": "garbage"}, "must be MxNxK"),
    ("BENCH_SHAPES=64x64", [], {"BENCH_SHAPES": "64x64"}, "must be MxNxK"),
    ("BENCH_SHAPES with a zero dim", [], {"BENCH_SHAPES": "0x64x64"}, "zero dimension"),
    ("BENCH_SHAPES=64xAx64", [], {"BENCH_SHAPES": "64xAx64"}, "not an integer"),
    ("BENCH_PARITY_SEEDS=0", ["--dump-parity", "{tmp}"], {"BENCH_PARITY_SEEDS": "0"},
     "below the minimum"),
    ("BENCH_PARITY_SEEDS=abc", ["--dump-parity", "{tmp}"], {"BENCH_PARITY_SEEDS": "abc"},
     "not a non-negative integer"),
    ("BENCH_SHAPES alongside --dump-parity", ["--dump-parity", "{tmp}"],
     {"BENCH_SHAPES": "64x64x64"}, "does not apply"),
    ("BENCH_PARITY_SHAPE=nonsense", ["--dump-parity", "{tmp}"],
     {"BENCH_PARITY_SHAPE": "nonsense"}, "not a ladder label"),
    ("BENCH_PARITY_SHAPE=512x512x512 (dims, not a label)", ["--dump-parity", "{tmp}"],
     {"BENCH_PARITY_SHAPE": "512x512x512"}, "not a ladder label"),
    ("BENCH_PARITY_SHAPE empty", ["--dump-parity", "{tmp}"],
     {"BENCH_PARITY_SHAPE": ""}, "not a ladder label"),
    ("BENCH_PARITY_DIST=gaussian", ["--dump-parity", "{tmp}"],
     {"BENCH_PARITY_DIST": "gaussian"}, "not a distribution"),
    ("BENCH_PARITY_DIST empty", ["--dump-parity", "{tmp}"],
     {"BENCH_PARITY_DIST": ""}, "not a distribution"),
]


def cli_contract():
    """Argument and environment handling in `bench_gemm_sweep`.

    Needs the built binary and a working Metal runtime (main() initialises the
    GPU before it parses anything), so this section announces a skip rather
    than passing vacuously where neither is available.
    """
    import subprocess
    print("\n-- bench_gemm_sweep argument / environment contract --")
    binary = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "target", "release", "bench_gemm_sweep")
    if not os.path.exists(binary):
        print(f"  SKIP  {binary} not built "
              "(cargo build --release --bin bench_gemm_sweep); CLI contract unverified")
        return
    probe = subprocess.run([binary, "--dump-parity"], capture_output=True, text=True)
    if "panicked" not in probe.stderr and "requires a directory" not in probe.stderr:
        print(f"  SKIP  no usable Metal runtime: {probe.stderr.strip()[:90]}")
        return

    tmp = tempfile.mkdtemp(prefix="parity-cli-")
    try:
        for label, argv, env, needle in CLI_CASES:
            argv = [x.replace("{tmp}", tmp) for x in argv]
            e = dict(os.environ, **env)
            # A stale value from the caller's shell would silently change the case.
            for var in ("BENCH_ITERS", "BENCH_WARMUP", "BENCH_SHAPES",
                        "BENCH_PARITY_SEEDS", "BENCH_PARITY_SHAPE",
                        "BENCH_PARITY_DIST"):
                if var not in env:
                    e.pop(var, None)
            r = subprocess.run([binary] + argv, env=e, capture_output=True, text=True)
            out = r.stderr + r.stdout
            if "panicked at" in out or r.returncode < 0:
                FAILURES.append(f"CLI {label}: panicked instead of erroring cleanly")
                print(f"  FAIL  {label}: panic")
            elif r.returncode == 0:
                FAILURES.append(f"CLI {label}: exited 0, accepting bad input")
                print(f"  FAIL  {label}: accepted (exit 0)")
            elif needle.lower() not in out.lower():
                FAILURES.append(f"CLI {label}: error lacks {needle!r}: {out.strip()[:120]}")
                print(f"  FAIL  {label}: message lacks {needle!r}")
            else:
                print(f"  ok    {label}: exit {r.returncode}, names the cause")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ladder_merge():
    """`parity_ladder.merge` -- the cross-shape aggregation.

    A per-shape sweep introduces a second way to under-report: a lane that
    covers only part of the ladder, averaged into one number that reads as a
    full sweep. Same rule as the seed axis, one level up.
    """
    from parity_ladder import merge
    print("\n-- ladder merge (cross-shape x distribution aggregation) --")
    cells = [("square_512", "uniform"), ("square_1024", "uniform"),
             ("mlp_down", "heavy_tail")]
    labels = [c[0] for c in cells]

    def row(lane, cell, rel, dtype="f32"):
        return dict(lane=lane, shape=cell[0], dist=cell[1], out_dtype=dtype, seeds=2,
                    max_rel_err=rel, median_rel_err=rel, min_rel_err=rel,
                    max_abs_err=rel, mean_abs_err=rel, per_seed_rel_err=[rel],
                    max_elem_rel_err=rel, max_higham_ratio=rel, per_seed_higham=[rel])

    full = [row("L", c, r) for c, r in zip(cells, [1e-6, 5e-6, 9e-6])]
    out = check("full grid merges", lambda: merge(full, cells), expect_reject=False)

    check("bare labels instead of (shape, dist) pairs",
          lambda: merge(full, labels), True, "would be split into characters")
    if out:
        got = out[0]
        if got["worst_shape"] != "mlp_down" or abs(got["worst_rel_err"] - 9e-6) > 1e-18:
            FAILURES.append(f"merge picked {got['worst_shape']} @ {got['worst_rel_err']}")
            print(f"  FAIL  worst is {got['worst_shape']} @ {got['worst_rel_err']:.2e}")
        else:
            print(f"  ok    worst across ladder = {got['worst_rel_err']:.1e} "
                  f"at {got['worst_shape']}")

    check("lane absent from one cell", lambda: merge(full[:2], cells),
          True, "whole grid")

    partial = full[:2] + [dict(lane="L", shape="mlp_down", dist="heavy_tail",
                               skipped="mlx missing")]
    check("lane scored on some cells, skipped on others",
          lambda: merge(partial, cells), True, "whole grid")

    allskip = [dict(lane="L", shape=c[0], dist=c[1], skipped="mlx missing")
               for c in cells]
    out = check("lane skipped on the whole grid", lambda: merge(allskip, cells),
                expect_reject=False)
    if out and "skipped" not in out[0]:
        FAILURES.append("an all-skipped lane was reported as scored")
        print("  FAIL  all-skipped lane reported as scored")
    elif out:
        print("  ok    all-skipped lane reported as skipped, not scored")

    mixed = [row("L", cells[0], 1e-6, "f32"), row("L", cells[1], 1e-6, "bf16"),
             row("L", cells[2], 1e-6, "f32")]
    check("output dtype varies across cells", lambda: merge(mixed, cells),
          True, "dtype varies")


def speed_coverage():
    """`paired_cross_runtime.missing_coverage` -- the timing sweep's version of
    the same rule. A geomean over three of eight shapes used to print exactly
    like a geomean over eight."""
    from paired_cross_runtime import missing_coverage
    print("\n-- paired timing sweep: ladder coverage --")
    labels = ["512x512x512", "1024x1024x1024", "2048x2048x2048"]
    full_a = {(s, "tensorops-f32"): 1.0 for s in labels}
    full_b = {(s, "mps-f32"): 1.0 for s in labels}

    rounds = [(dict(full_a), dict(full_b)) for _ in range(3)]
    gaps = missing_coverage(rounds, labels, "tensorops-f32", "mps-f32")
    if gaps:
        FAILURES.append(f"complete rounds reported gaps: {gaps}")
        print(f"  FAIL  complete rounds reported gaps: {gaps}")
    else:
        print("  ok    complete coverage reports no gaps")

    # torch MPS dropping out mid-sweep: one round returns nothing for a shape.
    holed = [(dict(full_a), dict(full_b)) for _ in range(3)]
    del holed[1][1][("1024x1024x1024", "mps-f32")]
    gaps = missing_coverage(holed, labels, "tensorops-f32", "mps-f32")
    if gaps != ["1024x1024x1024/mps-f32"]:
        FAILURES.append(f"a hole in one round was not caught: {gaps}")
        print(f"  FAIL  hole in one round not caught: {gaps}")
    else:
        print(f"  ok    a shape missing from one round is caught: {gaps}")

    # The comparison runtime absent entirely -- every shape, every round.
    gaps = missing_coverage([(dict(full_a), {}) for _ in range(3)], labels,
                            "tensorops-f32", "mps-f32")
    if len(gaps) != len(labels):
        FAILURES.append(f"a fully absent lane was not fully reported: {gaps}")
        print(f"  FAIL  fully absent lane: {gaps}")
    else:
        print(f"  ok    a fully absent comparison lane names all {len(gaps)} shapes")


def attn_paired_contract():
    """`attn_paired` -- the same coverage rule, plus the batching guard.

    The batching guard exists because this driver once set BENCH_ATTN_BATCHED
    for the Python lane and not the Rust one, compared tessl at batch=1 against
    MLX at batch=32, and printed a perfectly plausible 11.6x. Both lanes now
    echo the batch they ran and the driver refuses the ratio; that refusal is
    what is tested here.
    """
    import types
    import attn_paired
    print("\n-- attn_paired: coverage rule and batching guard --")

    cfgs = ["swa128_decode_1k", "swa128_decode_4k"]
    a = {(c, "tessl-decode"): 1.0 for c in cfgs}
    b = {(c, "mlx"): 1.0 for c in cfgs}
    rounds = [(dict(a), dict(b)) for _ in range(3)]
    gaps = attn_paired.missing_coverage(rounds, cfgs, ["mlx"], "tessl-decode")
    if gaps:
        FAILURES.append(f"attn complete rounds reported gaps: {gaps}")
        print(f"  FAIL  complete rounds reported gaps: {gaps}")
    else:
        print("  ok    complete coverage reports no gaps")

    holed = [(dict(a), dict(b)) for _ in range(3)]
    del holed[2][1][("swa128_decode_4k", "mlx")]
    gaps = attn_paired.missing_coverage(holed, cfgs, ["mlx"], "tessl-decode")
    if gaps != ["swa128_decode_4k/mlx"]:
        FAILURES.append(f"attn hole in one round not caught: {gaps}")
        print(f"  FAIL  hole in one round not caught: {gaps}")
    else:
        print(f"  ok    a config missing from one round is caught: {gaps}")

    def with_rows(rows):
        def fake_run(cmd, env=None, **kw):
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
        return fake_run

    real_run = attn_paired.subprocess.run
    try:
        # A lane that ran at a different batch than the driver asked for.
        attn_paired.subprocess.run = with_rows(
            [dict(cfg="swa128_decode_1k", runtime="mlx", median_ms=1.0, batched=1)])
        try:
            attn_paired._run(["x"], {}, "fake lane", lambda x: x["runtime"], 32)
        except SystemExit as exc:
            if "different batching" in str(exc):
                print("  ok    a lane at the wrong batch refuses to form a ratio")
            else:
                FAILURES.append(f"batch mismatch raised the wrong error: {exc}")
                print(f"  FAIL  wrong error: {exc}")
        else:
            FAILURES.append("a batch mismatch was accepted")
            print("  FAIL  batch mismatch accepted")

        # A lane that reports no batch at all cannot be confirmed, and an
        # unconfirmable check must not pass like a confirmed one.
        attn_paired.subprocess.run = with_rows(
            [dict(cfg="swa128_decode_1k", runtime="mlx", median_ms=1.0)])
        try:
            attn_paired._run(["x"], {}, "fake lane", lambda x: x["runtime"], 32)
        except SystemExit as exc:
            if "no 'batched' field" in str(exc):
                print("  ok    a lane that reports no batch is refused, not trusted")
            else:
                FAILURES.append(f"missing batch field raised the wrong error: {exc}")
                print(f"  FAIL  wrong error: {exc}")
        else:
            FAILURES.append("a lane reporting no batch was accepted")
            print("  FAIL  missing batch field accepted")

        # A non-finite or zero median is a failed measurement, not a fast one.
        for bad in (0.0, float("nan"), -1.0):
            attn_paired.subprocess.run = with_rows(
                [dict(cfg="swa128_decode_1k", runtime="mlx", median_ms=bad, batched=1)])
            try:
                attn_paired._run(["x"], {}, "fake lane", lambda x: x["runtime"], 1)
            except SystemExit:
                pass
            else:
                FAILURES.append(f"median_ms={bad} was accepted as a timing")
                print(f"  FAIL  median_ms={bad} accepted")
                break
        else:
            print("  ok    zero, NaN and negative medians are all refused")
    finally:
        attn_paired.subprocess.run = real_run


ATTN_CLI_CASES = [
    ("--dump-parity with no directory", ["--dump-parity"], {}, "requires a directory"),
    ("BENCH_ITERS=0", [], {"BENCH_ITERS": "0"}, "below the minimum"),
    ("BENCH_ITERS=abc", [], {"BENCH_ITERS": "abc"}, "not a non-negative integer"),
    ("BENCH_ATTN_DIST=gaussian", [], {"BENCH_ATTN_DIST": "gaussian"}, "not a distribution"),
    ("BENCH_ATTN_CFGS=nope", [], {"BENCH_ATTN_CFGS": "nope"}, "not a configuration"),
    ("BENCH_ATTN_CFGS=swa128_prefill_5120", [],
     {"BENCH_ATTN_CFGS": "swa128_prefill_5120"}, "not a configuration"),
    # A parity artifact must describe the shipping configuration. Under a
    # tuning override it would describe a kernel no caller reaches, which is
    # exactly how the forced-decode lane came to be dumped at chunk 256 while
    # the library shipped 128.
    ("--dump-parity with BENCH_ATTN_DECODE_CHUNK", ["--dump-parity", "/tmp/attn-refused"],
     {"BENCH_ATTN_DECODE_CHUNK": "64"}, "shipping configuration"),
    ("--dump-parity with BENCH_ATTN_DECODE_R", ["--dump-parity", "/tmp/attn-refused"],
     {"BENCH_ATTN_DECODE_R": "32"}, "shipping configuration"),
    ("--dump-parity with BENCH_ATTN_ROWS_R", ["--dump-parity", "/tmp/attn-refused"],
     {"BENCH_ATTN_ROWS_R": "32"}, "shipping configuration"),
    ("--dump-parity with TESSL_ATTN_TILED", ["--dump-parity", "/tmp/attn-refused"],
     {"TESSL_ATTN_TILED": "1"}, "shipping configuration"),
]


def attention_semantics():
    """The masking rule and GQA mapping the Python lane transcribes.

    If these were wrong, torch and MLX would be wrong in exactly the same way
    as each other and a speed ratio would still look plausible -- which is why
    the parity run scores the comparison runtimes against the f64 reference too,
    and why the rule itself is checked here against hand-computed cases.
    """
    from flash_attn_torch_mlx import keep_mask, CFGS, BY_LABEL, live_pairs
    print("\n-- attention masking rule and config mirror --")

    # Causal prefill from position 0: row t keeps exactly t+1 keys.
    c = dict(tq=6, tkv=6, window=-1, q_off=0, kv_off=0)
    m = keep_mask(c)
    if list(m.sum(axis=1)) == [1, 2, 3, 4, 5, 6]:
        print("  ok    global causal: row t keeps t+1 keys")
    else:
        FAILURES.append(f"causal row counts wrong: {list(m.sum(axis=1))}")
        print(f"  FAIL  causal row counts: {list(m.sum(axis=1))}")

    # Sliding window w: row t keeps min(t+1, w).
    c = dict(tq=6, tkv=6, window=3, q_off=0, kv_off=0)
    m = keep_mask(c)
    if list(m.sum(axis=1)) == [1, 2, 3, 3, 3, 3]:
        print("  ok    sliding window: row t keeps min(t+1, w) keys")
    else:
        FAILURES.append(f"window row counts wrong: {list(m.sum(axis=1))}")
        print(f"  FAIL  window row counts: {list(m.sum(axis=1))}")

    # Decode with an offset: q_abs = q_off, so a window of w keeps the last w.
    c = dict(tq=1, tkv=100, window=8, q_off=99, kv_off=0)
    m = keep_mask(c)
    if m.sum() == 8 and m[0, 92] and m[0, 99] and not m[0, 91]:
        print("  ok    offset decode: window lands on the last w keys")
    else:
        FAILURES.append(f"offset decode mask wrong: sum={m.sum()}")
        print(f"  FAIL  offset decode mask: sum={m.sum()}")

    # A window wider than the history is inert -- plain causal.
    a = keep_mask(dict(tq=8, tkv=8, window=999, q_off=0, kv_off=0))
    b = keep_mask(dict(tq=8, tkv=8, window=-1, q_off=0, kv_off=0))
    if (a == b).all():
        print("  ok    a window wider than the history is inert")
    else:
        FAILURES.append("wide window differs from causal")
        print("  FAIL  wide window differs from causal")

    # Nothing in the shipped set masks a row completely: a fully masked row is
    # a legitimate decode state the kernels answer with zeros, but torch SDPA
    # returns NaN for it, so a config that hit it would compare noise.
    empty = [c["label"] for c in CFGS if (keep_mask(c).sum(axis=1) == 0).any()]
    if empty:
        FAILURES.append(f"configs with a fully masked row: {empty}")
        print(f"  FAIL  fully masked rows in {empty}")
    else:
        print(f"  ok    no config in the set has a fully masked row ({len(CFGS)} checked)")

    # Live-pair counts must match the Rust FLOP accounting exactly, or the two
    # sides report GFLOP/s against different denominators.
    want = {"swa128_prefill_512": 131328, "swa128_decode_1k": 1024,
            "global512_decode_4k": 4096, "swa128_prefill_2048": 1573376}
    biffed = {k: (live_pairs(BY_LABEL[k]), v) for k, v in want.items()
              if live_pairs(BY_LABEL[k]) != v}
    if biffed:
        FAILURES.append(f"live-pair counts disagree with the Rust lane: {biffed}")
        print(f"  FAIL  live-pair counts: {biffed}")
    else:
        print(f"  ok    live-pair counts match the Rust FLOP accounting ({len(want)} checked)")


def tune_knob_contract():
    """The knob table and the environment it actually hands the binary.

    `attn_tune.run` clears every knob it is not sweeping, because a knob left
    over from the caller's shell would silently be measured as part of the
    value under test. That clearing list used to be a second hardcoded copy of
    the knob names, so adding a knob to KNOBS and forgetting the copy leaked
    it. Both the table's shape and the env the subprocess receives are checked
    here.
    """
    import types
    import attn_tune
    from flash_attn_torch_mlx import BY_LABEL
    print("\n-- attn_tune knob table and subprocess environment --")

    declared = {env for env, *_ in attn_tune.KNOBS.values()}
    if declared == set(attn_tune.KNOB_ENV):
        print(f"  ok    KNOB_ENV covers all {len(declared)} knob variables")
    else:
        FAILURES.append(f"KNOB_ENV {attn_tune.KNOB_ENV} != declared {sorted(declared)}")
        print("  FAIL  KNOB_ENV does not cover every knob")

    lanes = {"tessl", "tessl-tiled", "tessl-decode", "tessl-rows"}
    for knob, (env, values, lane, cfgs) in sorted(attn_tune.KNOBS.items()):
        bad = [c for c in cfgs if c not in BY_LABEL]
        if bad:
            FAILURES.append(f"knob {knob}: unknown configs {bad}")
            print(f"  FAIL  {knob}: unknown configs {bad}")
        elif lane not in lanes:
            FAILURES.append(f"knob {knob}: lane {lane!r} is not one the binary emits")
            print(f"  FAIL  {knob}: unknown lane {lane!r}")
        elif len(set(values)) != len(values):
            FAILURES.append(f"knob {knob}: duplicate values {values}")
            print(f"  FAIL  {knob}: duplicate values {values}")
        else:
            print(f"  ok    {knob}: {env}={values} on {lane}, {len(cfgs)} configs")

    # The decomposition report divides batched by solo, so the values are part
    # of the contract, not a default someone may reorder.
    if attn_tune.KNOBS["batched"][1] == ["1", "32"]:
        print("  ok    batched knob is ('1', '32'), the order the share assumes")
    else:
        FAILURES.append(f"batched values reordered: {attn_tune.KNOBS['batched'][1]}")
        print("  FAIL  batched knob values are not ('1', '32')")

    # A knob left in the caller's environment must not survive into a sweep of
    # a different knob.
    seen = {}

    def fake_run(cmd, env=None, **kw):
        seen.clear()
        seen.update(env or {})
        rows = [dict(cfg=c, runtime="tessl-rows", median_ms=1.0)
                for c in attn_tune.KNOBS["rows"][3]]
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")

    real_run, real_env = attn_tune.subprocess.run, dict(os.environ)
    try:
        attn_tune.subprocess.run = fake_run
        os.environ["BENCH_ATTN_DECODE_CHUNK"] = "256"
        os.environ["BENCH_ATTN_BATCHED"] = "32"
        env_name, _, lane, cfgs = attn_tune.KNOBS["rows"]
        attn_tune.run(env_name, "8", lane, cfgs, 3, 1)
    finally:
        attn_tune.subprocess.run = real_run
        os.environ.clear()
        os.environ.update(real_env)

    # BENCH_ATTN_BATCHED is not cleared but *set*, so a stray value in the
    # caller's shell is overridden rather than inherited: a sweep silently run
    # at someone else's batching is a sweep of the wrong thing.
    leaked = [k for k in attn_tune.KNOB_ENV
              if k not in ("BENCH_ATTN_ROWS_R", "BENCH_ATTN_BATCHED") and k in seen]
    if leaked:
        FAILURES.append(f"knobs leaked into a rows sweep: {leaked}")
        print(f"  FAIL  knobs leaked into the child env: {leaked}")
    elif seen.get("BENCH_ATTN_ROWS_R") != "8":
        FAILURES.append(f"swept knob not set: BENCH_ATTN_ROWS_R={seen.get('BENCH_ATTN_ROWS_R')}")
        print("  FAIL  the swept knob did not reach the child env")
    elif seen.get("BENCH_ATTN_BATCHED") != "1":
        FAILURES.append(f"stray BENCH_ATTN_BATCHED inherited: {seen.get('BENCH_ATTN_BATCHED')}")
        print(f"  FAIL  inherited BENCH_ATTN_BATCHED={seen.get('BENCH_ATTN_BATCHED')}")
    else:
        print("  ok    sweeping one knob clears the others and pins the batching")

    # A sweep of launches-per-submit cannot also be handed a fixed batching.
    import subprocess as _sp
    r = _sp.run([sys.executable, os.path.join(HERE_BENCH, "attn_tune.py"),
                 "--knob", "batched", "--batched", "32"],
                capture_output=True, text=True)
    if r.returncode != 0 and "Drop one of them" in (r.stderr + r.stdout):
        print("  ok    --knob batched with --batched is refused")
    else:
        FAILURES.append("--knob batched --batched 32 was accepted")
        print("  FAIL  --knob batched --batched 32 accepted")


def attn_cli_contract():
    import subprocess
    print("\n-- bench_flash_attn argument / environment contract --")
    binary = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "target", "release", "bench_flash_attn")
    if not os.path.exists(binary):
        print(f"  SKIP  {binary} not built; CLI contract unverified")
        return
    probe = subprocess.run([binary, "--dump-parity"], capture_output=True, text=True)
    if "requires a directory" not in probe.stderr and "panicked" not in probe.stderr:
        print(f"  SKIP  no usable Metal runtime: {probe.stderr.strip()[:90]}")
        return
    for label, argv, env, needle in ATTN_CLI_CASES:
        e = dict(os.environ, **env)
        for var in ("BENCH_ITERS", "BENCH_WARMUP", "BENCH_ATTN_DIST", "BENCH_ATTN_CFGS",
                    "BENCH_ATTN_DECODE_CHUNK", "BENCH_ATTN_DECODE_R",
                    "BENCH_ATTN_ROWS_R", "TESSL_ATTN_TILED"):
            if var not in env:
                e.pop(var, None)
        r = subprocess.run([binary] + argv, env=e, capture_output=True, text=True)
        out = r.stderr + r.stdout
        if "panicked at" in out or r.returncode < 0:
            FAILURES.append(f"attn CLI {label}: panicked")
            print(f"  FAIL  {label}: panic")
        elif r.returncode == 0:
            FAILURES.append(f"attn CLI {label}: exited 0, accepting bad input")
            print(f"  FAIL  {label}: accepted")
        elif needle.lower() not in out.lower():
            FAILURES.append(f"attn CLI {label}: error lacks {needle!r}")
            print(f"  FAIL  {label}: message lacks {needle!r}")
        else:
            print(f"  ok    {label}: exit {r.returncode}, names the cause")


def coverage_inventory():
    """`kernel_coverage.inventory` -- the kernel census.

    This is where the worst error in the audit came from: a scan for
    `kernel void` missed every kernel declared through the `NN_COOP_KERNEL`
    family of macros, so the inventory read 67 when the true count was 83 and
    16 kernels could never have shown as uncovered. The guard that makes that
    impossible is the one worth testing.
    """
    from kernel_coverage import inventory
    print("\n-- kernel inventory (the census coverage is measured against) --")
    tmp = tempfile.mkdtemp(prefix="kcov-")
    try:
        def write(body):
            d = tempfile.mkdtemp(dir=tmp)
            with open(os.path.join(d, "k.metal"), "w") as f:
                f.write(body)
            return d

        d = write("kernel void alpha(device float* x) {}\n"
                  "NN_COOP_KERNEL(beta, bfloat, 64, 64, 4, false)\n"
                  "TN_NT_COOP_KERNEL( gamma , float, 128, 64)\n")
        names, _ = inventory(d, metallib="/nonexistent")
        if names == {"alpha", "beta", "gamma"}:
            print("  ok    plain and macro-declared kernels are both counted")
        else:
            FAILURES.append(f"inventory missed declarations: {sorted(names)}")
            print(f"  FAIL  inventory got {sorted(names)}")

        # A macro *body* containing `kernel void NAME(` must not be scanned:
        # NAME is the macro parameter, and counting it invents a kernel that is
        # in no metallib. The metallib cross-check caught this for real when
        # the decode kernels landed.
        # The macro name must end in KERNEL for the pattern to recognise it,
        # which is the convention every kernel-declaring macro in the tree uses.
        d = write("#define MK_KERNEL(NAME, D) \\\n"
                  "kernel void NAME(device float* x) {}\n"
                  "MK_KERNEL(real_one, 128)\n")
        names, _ = inventory(d, metallib="/nonexistent")
        if names == {"real_one"}:
            print("  ok    a macro body's parameter is not counted as a kernel")
        else:
            FAILURES.append(f"macro body leaked into the inventory: {sorted(names)}")
            print(f"  FAIL  macro body leaked: {sorted(names)}")

        # A macro whose first argument is not an identifier cannot be parsed
        # for a kernel name, and must fail rather than be skipped.
        d = write("kernel void alpha(device float* x) {}\n"
                  'WEIRD_NEW_KERNEL("delta", bfloat)\n')
        try:
            inventory(d, metallib="/nonexistent")
        except SystemExit as exc:
            if "not parsed" in str(exc):
                print("  ok    an unparsed kernel macro fails loudly")
            else:
                FAILURES.append(f"unparsed macro raised the wrong error: {exc}")
                print(f"  FAIL  wrong error: {exc}")
        else:
            FAILURES.append("an unparsed kernel macro was silently undercounted")
            print("  FAIL  unparsed macro silently undercounted")

        # An empty kernel directory is a broken run, not zero kernels.
        d = tempfile.mkdtemp(dir=tmp)
        try:
            inventory(d, metallib="/nonexistent")
        except SystemExit:
            print("  ok    an empty kernel directory is refused, not read as zero")
        else:
            FAILURES.append("an empty kernel directory was accepted")
            print("  FAIL  empty kernel directory accepted")

        # A source scan that disagrees with the compiled metallib is wrong,
        # whichever way it disagrees, and must not be measured against.
        from kernel_coverage import metallib_symbols
        real = metallib_symbols()
        if real is None:
            print("  SKIP  metal-nm unavailable; metallib cross-check unverified")
        else:
            d = write("kernel void not_a_real_kernel(device float* x) {}\n")
            try:
                inventory(d)
            except SystemExit as exc:
                if "disagrees with the compiled metallib" in str(exc):
                    print(f"  ok    source scan vs metallib disagreement is fatal "
                          f"({len(real)} exported)")
                else:
                    FAILURES.append(f"metallib mismatch raised wrong error: {exc}")
                    print(f"  FAIL  wrong error: {str(exc)[:70]}")
            else:
                FAILURES.append("a source scan disagreeing with the metallib was accepted")
                print("  FAIL  metallib disagreement accepted")

        # The real tree must agree with the union of both declaration forms.
        names, _ = inventory()
        if len(names) >= 80 and "matmul2d_tensorops_bf16_f32_64x64_sg4" in names:
            print(f"  ok    real tree: {len(names)} entry points, macro-declared ones included")
        else:
            FAILURES.append(f"real inventory looks wrong: {len(names)} names")
            print(f"  FAIL  real inventory: {len(names)} names")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tmp = tempfile.mkdtemp(prefix="parity-harness-")
    try:
        print("\n-- a valid dump is accepted, and reported exactly once per lane --")
        good = build_dump(os.path.join(tmp, "good"), seeds=3)
        rows = check("valid dump", lambda: parity(good), expect_reject=False)
        if rows is not None:
            names = [r["lane"] for r in rows]
            dupes = {x for x in names if names.count(x) > 1}
            if dupes:
                FAILURES.append(f"duplicate lane rows: {sorted(dupes)}")
                print(f"  FAIL  duplicate lane rows: {sorted(dupes)}")
            else:
                print(f"  ok    {len(names)} lanes, no duplicates: {names}")
            for lane in LANES:
                row = next((r for r in rows if r["lane"] == lane), None)
                if row is None or row.get("seeds") != 3:
                    FAILURES.append(f"{lane} did not report all 3 seeds: {row}")
                    print(f"  FAIL  {lane} seeds={row and row.get('seeds')}")
            if all(r.get("seeds") == 3 for r in rows if r["lane"] in LANES):
                print("  ok    every dumped lane covers all 3 seeds")

        print("\n-- the aggregate reports the WORST seed, not the last or the mean --")
        # Seed 1 is perturbed hardest; a scorer that keeps the last seed or
        # averages would under-report it.
        w = build_dump(os.path.join(tmp, "worst"), seeds=3, err_scale=[1e-8, 5e-7, 1e-8])
        rows = check("worst-of-seeds", lambda: parity(w), expect_reject=False)
        if rows:
            row = next(r for r in rows if r["lane"] == "tensorops-f32")
            per = row["per_seed_rel_err"]
            if abs(row["max_rel_err"] - max(per)) > 1e-12:
                FAILURES.append(f"max_rel_err {row['max_rel_err']} != max(per_seed) {max(per)}")
                print("  FAIL  max_rel_err is not the worst seed")
            elif not (per[1] > per[0] * 10):
                FAILURES.append(f"perturbation did not register: {per}")
                print(f"  FAIL  perturbation did not register: {per}")
            else:
                print(f"  ok    worst seed dominates: per_seed={['%.2e' % x for x in per]}")

        print("\n-- structural rejections --")
        d = build_dump(os.path.join(tmp, "nomanifest"))
        os.remove(os.path.join(d, "parity_manifest.json"))
        check("missing manifest", lambda: parity(d), True, "manifest")

        d = build_dump(os.path.join(tmp, "badjson"))
        with open(os.path.join(d, "parity_manifest.json"), "w") as f:
            f.write("{not json")
        check("malformed manifest", lambda: parity(d), True, "unreadable")

        d = build_dump(os.path.join(tmp, "lanestr"))
        edit_manifest(d, lanes="tensorops-f32")
        check("manifest lanes not a list", lambda: parity(d), True, "non-empty list")

        d = build_dump(os.path.join(tmp, "laneint"))
        edit_manifest(d, lanes=[1, 2])
        check("manifest lanes not strings", lambda: parity(d), True, "strings")

        d = build_dump(os.path.join(tmp, "emptylanes"))
        edit_manifest(d, lanes=[])
        check("manifest lanes empty", lambda: parity(d), True, "non-empty list")

        d = build_dump(os.path.join(tmp, "badk"))
        edit_manifest(d, k=0)
        check("manifest zero dim", lambda: parity(d), True, "positive integer")

        d = build_dump(os.path.join(tmp, "notobj"))
        with open(os.path.join(d, "parity_manifest.json"), "w") as f:
            json.dump([1, 2, 3], f)
        check("manifest not an object", lambda: parity(d), True, "object")

        print("\n-- coverage rejections (the original bug's shape) --")
        # Exactly the pre-fix state: the two reduced-precision lanes absent
        # while the manifest still claims four.
        d = build_dump(os.path.join(tmp, "shortlanes"))
        for s in os.listdir(d):
            if s.startswith("seed_"):
                for lane in ("tensorops-bf16", "tensorops-tf32"):
                    os.remove(os.path.join(d, s, f"parity_c_{lane}.npy"))
        check("lane in manifest, npy absent", lambda: parity(d), True, "missing")

        d = build_dump(os.path.join(tmp, "shortseed"), seeds=3)
        shutil.rmtree(os.path.join(d, "seed_02"))
        check("seed in manifest, directory absent", lambda: parity(d), True, "absent")

        print("\n-- numeric rejections --")
        for label, bad in (("NaN", np.nan), ("+Inf", np.inf), ("-Inf", -np.inf)):
            d = build_dump(os.path.join(tmp, f"nf{label}"))
            fn = os.path.join(d, "seed_00", "parity_c_tensorops-tf32.npy")
            c = np.load(fn)
            c[3, 4] = bad
            np.save(fn, c)
            check(f"{label} in a result", lambda d=d: parity(d), True, "non-finite")

        d = build_dump(os.path.join(tmp, "wrongshape"))
        np.save(os.path.join(d, "seed_00", "parity_c_tensorops-f32.npy"),
                np.zeros((M, N + 3), dtype=np.float32))
        check("result shape mismatch", lambda: parity(d), True, "shape")

        d = build_dump(os.path.join(tmp, "zeroops"))
        np.save(os.path.join(d, "seed_00", "parity_a.npy"), np.zeros((M, K), np.float32))
        check("all-zero operands (scale 0)", lambda: parity(d), True, "division by zero")

        d = build_dump(os.path.join(tmp, "nanops"))
        a = np.load(os.path.join(d, "seed_00", "parity_a.npy"))
        a[0, 0] = np.nan
        np.save(os.path.join(d, "seed_00", "parity_a.npy"), a)
        check("non-finite operands", lambda: parity(d), True, "non-finite")

        d = build_dump(os.path.join(tmp, "opshape"))
        np.save(os.path.join(d, "seed_00", "parity_b.npy"),
                np.ones((K + 2, N), dtype=np.float32))
        check("operand shape contradicts manifest", lambda: parity(d), True, "contradict")

        print("\n-- per-element error budget --")
        # A lane that writes nothing. The Rust side pre-zeroes C so this cannot
        # inherit the previous lane's result; the budget gate then rejects it
        # outright rather than merely scoring it 1.0.
        d = build_dump(os.path.join(tmp, "zeroed"))
        for s in os.listdir(d):
            if s.startswith("seed_"):
                np.save(os.path.join(d, s, "parity_c_tensorops-tf32.npy"),
                        np.zeros((M, N), dtype=np.float32))
        check("no-op lane (all-zero result)", lambda: parity(d), True, "budget exceeded")

        # A drift of 1e-5 relative: innocuous by the normwise metric, and ~2x
        # the f32 lane's per-element budget. This is the case the old harness
        # could not see at all.
        d = build_dump(os.path.join(tmp, "drifted"))
        for s in os.listdir(d):
            if s.startswith("seed_"):
                fn = os.path.join(d, s, "parity_c_tensorops-f32.npy")
                np.save(fn, (np.load(fn).astype(np.float64) * (1 + 1e-5)).astype(np.float32))
        check("f32 lane drifted 1e-5 (normwise would shrug)",
              lambda: parity(d), True, "kernel defect")

        # A lane with no declared precision must not be scored against a
        # budget nobody chose for it.
        d = build_dump(os.path.join(tmp, "unknownlane"), lanes=LANES + ["tensorops-fp8"])
        check("lane with no LANE_PRECISION entry", lambda: parity(d), True,
              "LANE_PRECISION")

        print("\n-- budget verdicts (which response the breach calls for) --")
        from gemm_sweep_mlx import adjudicate
        for name, over, needle in (
            ("tessl only", {"tensorops-bf16": 2.0}, "kernel defect"),
            ("comparison runtime only", {"torch-mps-bf16": 2.0}, "other runtime is the outlier"),
            ("every runtime", {"tensorops-bf16": 2.0, "torch-mps-bf16": 2.1, "mlx-bf16": 2.2},
             "points at the bound"),
        ):
            msg = adjudicate(over)
            if needle in msg:
                print(f"  ok    {name}: {needle!r}")
            else:
                FAILURES.append(f"adjudicate({name}) lacks {needle!r}: {msg}")
                print(f"  FAIL  {name}: lacks {needle!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    cli_contract()
    ladder_merge()
    speed_coverage()
    attention_semantics()
    attn_cli_contract()
    attn_paired_contract()
    tune_knob_contract()
    coverage_inventory()

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} case(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PASS: every adversarial dump was handled as specified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
