#!/usr/bin/env python3
"""Adversarial stress suite for scripts/bench_qwen38.py.

Every FAIL is a claim the harness would make that its data does not support.
No engine, no model weights, and no network are touched: engines are stubbed, and
the dflash-mlx version is pinned per-test so the correctness gates are exercised
in both directions regardless of what is installed.

    python3 scripts/test_bench_qwen38.py     # exit 0 = clean

Run from the repo root.
"""
import importlib.util, sys, json, tempfile, pathlib, subprocess, contextlib, io
import importlib.metadata as md

sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("b", "scripts/bench_qwen38.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

FAILS, PASSES = [], []
_orig_version = md.version


def check(name, cond, detail=""):
    (PASSES if cond else FAILS).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def arm(name, kind="lossless", tok=None, acc=None, ok=True, draft="d", runs=None):
    return {"arm": name, "kind": kind, "label": name, "target": "t", "draft": draft,
            "cmd": [], "all_ok": ok, "median_tok_s": tok, "median_accept_pct": acc,
            "median_peak_gb": None, "runs": runs or []}


def vt(results, blockers=None):
    return "\n".join(m.verdict(results, blockers))


def fake_dflash(version):
    """Pin the reported dflash-mlx version so gates can be tested independently
    of what happens to be installed."""
    def _v(name):
        if name == "dflash-mlx":
            if version is None:
                raise md.PackageNotFoundError(name)
            return version
        return _orig_version(name)
    md.version = _v


GOOD, BAD = "0.1.10", "0.1.8"

print("\n=== A. PARSER: real-world output shapes ===")
r = m.parse_metrics("""Prompt: 42 tokens, 512.30 tokens-per-sec
Generation: 256 tokens, 17.63 tokens-per-sec
Peak memory: 15.34 GB""")
check("A1 picks GENERATION tok/s, not prefill", r["tok_s_reported"] == 17.63, f"got {r['tok_s_reported']}")

r = m.parse_metrics("""Baseline MLX
Generation: 256 tokens, 17.63 tokens-per-sec
DFlash speculative
Generation: 256 tokens, 52.35 tokens-per-sec
acceptance rate: 80.5%""")
check("A2 two-leg output takes the measured leg, not the baseline", r["tok_s_reported"] == 52.35,
      f"got {r['tok_s_reported']}")
check("A2b ambiguity is flagged", any("distinct values" in f for f in r["parse_flags"]),
      f"flags={r['parse_flags']}")

# Acceptance with no percent sign is deliberately NOT guessed at. The forward-
# scanning fallback that used to handle it is what produced a false 100% from
# "copyspec 1 blocks" (see R1). None is the honest answer; the verdict renders it
# as "acceptance unreported, cannot confirm lossless" rather than shipping.
r = m.parse_metrics("acceptance rate: 0.805\nGeneration: 10 tokens, 50.0 tokens-per-sec")
check("A3 unmarked acceptance is not guessed at", r["accept_pct"] is None, f"got {r['accept_pct']}")
check("A3b an unparsed acceptance cannot ship",
      "SHIP AS DEFAULT" not in vt([arm("ar", tok=10.0), arm("dflash", tok=50.0, acc=None)]))

r = m.parse_metrics("Generation: 5 tokens, 999999.0 tokens-per-sec\nacceptance rate: 250%")
check("A5 absurd tok/s rejected", r["tok_s_reported"] is None, f"got {r['tok_s_reported']}")
check("A6 acceptance >100% rejected", r["accept_pct"] is None, f"got {r['accept_pct']}")
check("A7 rejections are flagged, not silent", len(r["parse_flags"]) >= 1, f"flags={r['parse_flags']}")

r = m.parse_metrics("")
check("A8 empty output yields no metrics", all(v is None for k, v in r.items() if k != "parse_flags"))
r = m.parse_metrics("<timeout after 1800s>")
check("A9 timeout sentinel yields no metrics",
      all(v is None for k, v in r.items() if k != "parse_flags"), f"got {r}")
r = m.parse_metrics("Generation: 256 tokens, 17.63 tokens-per-sec\n" * 500)
check("A10 huge repeated output does not crash", r["tok_s_reported"] == 17.63)
r = m.parse_metrics("Generation: 0 tokens, 0.0 tokens-per-sec")
check("A11 zero tok/s rejected as implausible", r["tok_s_reported"] is None, f"got {r['tok_s_reported']}")
r = m.parse_metrics("eval rate: 70.12 tokens/s\neval count: 256 token(s)")
check("A12 ollama --verbose format parsed", r["tok_s_reported"] == 70.12 and r["tokens_reported"] == 256.0,
      f"got {r['tok_s_reported']}, {r['tokens_reported']}")

print("\n=== A-REAL. Verbatim engine output (captured 2026-08-16) ===")
# These strings are copied from .artifacts/.../results.json stdout_tail. Every one
# of them defeated the synthetic fixtures above.

DFLASH_REAL = "256 tokens | 16.4 tok/s | 53.5% acceptance | copyspec 1 blocks / 3 tokens"
r = m.parse_metrics(DFLASH_REAL)
check("R1 dflash acceptance is 53.5%, not the 100% from 'copyspec 1 blocks'",
      r["accept_pct"] == 53.5, f"got {r['accept_pct']}")
check("R2 dflash tok/s parsed", r["tok_s_reported"] == 16.4, f"got {r['tok_s_reported']}")
check("R3 dflash token count parsed", r["tokens_reported"] == 256.0, f"got {r['tokens_reported']}")

MLXVLM_REAL = """==========
Prompt: 102 tokens, 512.300 tokens-per-sec
Generation: 256 tokens, 41.207 tokens-per-sec
Peak memory: 17.301 GB
Speculative decoding: 2.81 accepted tokens/round (1.81 accepted drafts/round, 91.2% of drafted, avg draft 1.99) over 91 rounds"""
r = m.parse_metrics(MLXVLM_REAL)
check("R4 mlx-vlm --verbose tok/s parsed", r["tok_s_reported"] == 41.207, f"got {r['tok_s_reported']}")
check("R5 mlx-vlm acceptance '% of drafted' parsed", r["accept_pct"] == 91.2, f"got {r['accept_pct']}")
check("R6 mlx-vlm accepted-tokens/round parsed", r["accept_len"] == 2.81, f"got {r['accept_len']}")
check("R7 mlx-vlm prefill rate not mistaken for decode", r["tok_s_reported"] != 512.300)

OLLAMA_REAL = """total duration:       4m1.5s
load duration:        4.441778958s
prompt eval count:    102 token(s)
prompt eval duration: 5.667103916s
prompt eval rate:     18.00 tokens/s
eval count:           6580 token(s)
eval duration:        3m47.081636666s
eval rate:            28.98 tokens/s"""
r = m.parse_metrics(OLLAMA_REAL)
check("R8 ollama decode rate, not prompt-eval rate", r["tok_s_reported"] == 28.98, f"got {r['tok_s_reported']}")
check("R9 ollama eval count, not prompt eval count", r["tokens_reported"] == 6580.0, f"got {r['tokens_reported']}")

# The mlx-vlm arm before --verbose was added: ran fine, reported nothing.
NO_VERBOSE = "Loading drafter (mtp): mlx-community/Qwen3.8-27B-MTP-4bit\nSpeculative decoding: 2.81 accepted tokens/round (1.81 accepted drafts/round, 91.2% of drafted, avg draft 1.99) over 91 rounds"
r = m.parse_metrics(NO_VERBOSE)
check("R10 missing throughput stays None rather than being invented", r["tok_s_reported"] is None)

src = pathlib.Path("scripts/bench_qwen38.py").read_text()
for a in ("mtp", "nvfp4-mtp"):
    seg = src.split(f'if arm == "{a}":')[1].split("if arm ==")[0]
    check(f"R11 {a} arm passes --verbose (mlx-vlm gates stats behind it)", '"--verbose"' in seg)

print("\n=== B. VERDICT: decision-rule integrity ===")
check("B1 no acceptance data -> not shipped",
      "SHIP AS DEFAULT" not in vt([arm("ar", tok=17.6), arm("dflash", tok=52.3, acc=None)]))
check("B2 failed runs -> not shipped",
      "SHIP AS DEFAULT" not in vt([arm("ar", tok=17.6), arm("dflash", tok=52.3, acc=80.5, ok=False)]))
v = vt([arm("ar", tok=0.001), arm("dflash", tok=52.3, acc=80.5)])
check("B3 degenerate baseline warns and blocks", "implausibly low" in v and "SHIP AS DEFAULT" not in v)
check("B4 failed AR baseline warns",
      "baseline had failing runs" in vt([arm("ar", tok=17.6, ok=False), arm("dflash", tok=52.3, acc=80.5)]))
check("B5 slower-than-AR rejected", "DO NOT SHIP" in vt([arm("ar", tok=17.6), arm("dflash", tok=17.0, acc=80.5)]))
check("B6 collapsed acceptance rejected", "DO NOT SHIP" in vt([arm("ar", tok=17.6), arm("dflash", tok=52.3, acc=12.0)]))
check("B7 genuine win still ships", "SHIP AS DEFAULT" in vt([arm("ar", tok=17.6), arm("dflash", tok=52.3, acc=80.5)]))
check("B8 marginal win kept but not defaulted",
      "KEEP AVAILABLE" in vt([arm("ar", tok=17.6), arm("dflash", tok=25.0, acc=70.0)]))
check("B9 missing baseline handled", "No usable AR baseline" in vt([arm("dflash", tok=52.3, acc=80.5)]))
check("B10 empty results handled", "No usable AR baseline" in vt([]))
check("B11 boundary 1.8x/60% not over-shipped",
      "SHIP AS DEFAULT" not in vt([arm("ar", tok=10.0), arm("dflash", tok=18.0, acc=60.0)]))

print("\n=== C. CORRECTNESS GATE ===")
fake_dflash(BAD)
b = m.correctness_blockers(["ar", "dflash"])
check("C1 old dflash produces a blocker", "dflash" in b)
v = vt([arm("ar", tok=17.6), arm("dflash", tok=52.3, acc=80.5)], b)
check("C2 blocked arm is INVALID, never SHIP", "INVALID" in v and "SHIP AS DEFAULT" not in v)
check("C3 blocker does not suppress the mtp arm",
      "SHIP AS DEFAULT" in vt([arm("ar", tok=17.6), arm("mtp", tok=40.0, acc=75.0),
                               arm("dflash", tok=52.3, acc=80.5)], b))

fake_dflash(None)
check("C4 undetermined version also blocks", "dflash" in m.correctness_blockers(["dflash"]))
fake_dflash("not-a-version")
check("C5 unparseable version blocks", "dflash" in m.correctness_blockers(["dflash"]))
fake_dflash(GOOD)
check("C6 fixed version does NOT block", m.correctness_blockers(["ar", "dflash"]) == {})
check("C7 blocker only applies to the dflash arm", m.correctness_blockers(["ar", "mtp"]) == {})
fake_dflash(BAD)
check("C8 blocker absent when dflash arm not selected", m.correctness_blockers(["ar", "mtp"]) == {})

print("\n=== D. ARTIFACTS ===")
for label, ver in (("blocked", BAD), ("clean", GOOD)):
    fake_dflash(ver)
    with tempfile.TemporaryDirectory() as td:
        notes, bl = m.preflight(["ar", "dflash"], 256)
        res = [arm("ar", tok=17.6, runs=[{"parse_flags": ["tok_s: 2 distinct values [17.6, 52.3]; used last"]}]),
               arm("dflash", tok=52.3, acc=80.5, runs=[{"parse_flags": []}])]
        rep = m.write_artifacts(pathlib.Path(td), res, notes,
                                {"prompt": "p", "max_tokens": 256, "repeat": 1, "cooldown": 0,
                                 "arms": ["ar", "dflash"]}, bl)
        body = rep.read_text()
        p = json.loads((pathlib.Path(td) / "results.json").read_text())
        blocked = ver == BAD
        check(f"D-{label} lossless_claim_valid correct",
              p["validity"]["lossless_claim_valid"] is (not blocked))
        check(f"D-{label} unconditional lossless sentence gated",
              ("Lossless arms all verify" in body) is (not blocked))
        check(f"D-{label} blocker section present iff blocked",
              ("Losslessness NOT established" in body) is blocked)
        check(f"D-{label} parse flags surfaced in report", "Parse warnings" in body)
        check(f"D-{label} json is round-trippable", isinstance(p["validity"]["blockers"], dict))

print("\n=== E. PREFLIGHT ===")
fake_dflash(BAD)
n, b = m.preflight(["ar", "dflash"], 256)
check("E1 version reported", any("dflash-mlx version: 0.1.8" in x for x in n))
check("E2 blocker surfaced in notes", any("BLOCKER [dflash]" in x for x in n))
check("E3 upgrade hint included", any("git+https://github.com" in x for x in n))
check("E4 thermal note at small max-tokens", any("THERMAL" in x for x in n))
n2, _ = m.preflight(["ar", "dflash"], 8192)
check("E5 no thermal note at large max-tokens", not any("THERMAL" in x for x in n2))
n3, b3 = m.preflight(["ar"], 4096)
check("E6 no dflash noise when arm unselected", not any("dflash" in x.lower() for x in n3) and b3 == {})

print("\n=== G. STABILITY / THERMAL ===")


def run_with_series(series, repeat):
    """Stub an engine whose tok/s changes per invocation, to exercise run_arm."""
    with tempfile.TemporaryDirectory() as td:
        counter = pathlib.Path(td) / "n"
        counter.write_text("0")
        prog = (
            "import pathlib,sys;"
            f"p=pathlib.Path({str(counter)!r});"
            "i=int(p.read_text());p.write_text(str(i+1));"
            f"s={series!r};"
            "print(f'Generation: 256 tokens, {s[min(i,len(s)-1)]} tokens-per-sec')"
        )
        saved = m.build_cmd
        m.build_cmd = lambda a, pr, mt: [sys.executable, "-c", prog]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return m.run_arm("ar", "p", 256, repeat=repeat, cooldown=0, timeout=60)
        finally:
            m.build_cmd = saved


r = run_with_series([17.6, 17.6, 17.6], 3)
check("G1 stable runs produce no stability warning", r["stability"] == [], f"got {r['stability']}")
check("G1b spread computed as ~0", r["spread_pct"] == 0.0, f"got {r['spread_pct']}")

r = run_with_series([50.0, 20.0, 45.0], 3)
check("G2 high spread flagged", any("spread" in s for s in r["stability"]), f"got {r['stability']}")

r = run_with_series([52.0, 40.0, 28.0], 3)
check("G3 monotonic decline flagged as thermal",
      any("thermal throttling" in s for s in r["stability"]), f"got {r['stability']}")
check("G3b thermal arm still reports a median", r["median_tok_s"] == 40.0, f"got {r['median_tok_s']}")

v = vt([arm("ar", tok=17.6), dict(arm("dflash", tok=52.3, acc=80.5),
                                  stability=["tok/s fell on every successive run"])])
check("G4 stability surfaced in verdict", "!" in v and "fell on every successive run" in v)

with tempfile.TemporaryDirectory() as td:
    res = [dict(arm("ar", tok=17.6, runs=[{"parse_flags": []}]), spread_pct=42.0,
                stability=["spread 42.0% across 3 runs"])]
    rep = m.write_artifacts(pathlib.Path(td), res, [], {"prompt": "p", "max_tokens": 256,
                            "repeat": 3, "cooldown": 0, "arms": ["ar"]}, {})
    body = rep.read_text()
    check("G5 spread column rendered with warning marker", "42.0% ⚠" in body)
    p = json.loads((pathlib.Path(td) / "results.json").read_text())
    check("G6 unstable arms recorded in json", "ar" in p["validity"]["unstable_arms"])

print("\n=== F. CLI / END-TO-END ===")
# --strict is exercised IN-PROCESS with the version pinned. Doing this by
# subprocess would depend on whatever dflash-mlx happens to be installed, and on
# a healthy install there is no blocker — so bench would sail past the gate and
# start a real 27B run. Never let the suite be able to launch an engine.
def strict_exit_code(pinned_version, argv):
    fake_dflash(pinned_version)
    saved_argv, saved_run = sys.argv, m.subprocess.run

    def _forbidden(*a, **k):
        raise AssertionError("subprocess spawned before the strict gate — a regression "
                             "here would launch a real model run")

    class _ReachedRunner(Exception):
        pass

    def _forbidden(*a, **k):
        raise _ReachedRunner

    m.subprocess.run = _forbidden
    sys.argv = ["bench_qwen38.py"] + argv
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            m.main()
    except SystemExit as e:
        return e.code, err.getvalue()
    except _ReachedRunner:
        # Reaching the runner means the gate let this through — the correct
        # outcome for a healthy install, and the thing --strict must prevent
        # when a blocker is present.
        return "RAN", err.getvalue()
    finally:
        sys.argv, m.subprocess.run = saved_argv, saved_run
    return 0, err.getvalue()


rc, err = strict_exit_code(BAD, ["--arms", "dflash", "--strict"])
check("F1 --strict refuses to run under a blocker (exit 3)", rc == 3, f"rc={rc}")
check("F2 --strict explains why on stderr", "BLOCKER" in err)
check("F2b --strict never reaches the runner when blocked", rc != "RAN")
rc2, _ = strict_exit_code(GOOD, ["--arms", "dflash", "--strict"])
check("F2c a healthy install passes the gate and proceeds", rc2 == "RAN", f"rc={rc2}")

md.version = _orig_version
r = subprocess.run([sys.executable, "scripts/bench_qwen38.py", "--arms", "bogus"],
                   capture_output=True, text=True, timeout=180)
check("F3 unknown arm rejected (exit 2)", r.returncode == 2, f"rc={r.returncode}")
r = subprocess.run([sys.executable, "scripts/bench_qwen38.py", "--help"],
                   capture_output=True, text=True, timeout=180)
check("F4 --help works and documents --strict", r.returncode == 0 and "--strict" in r.stdout)

print("\n=== H. SHARED GUARD (canonical owner) ===")
sys.path.insert(0, "scripts")
import dflash_guard as g

fake_dflash(BAD)
check("H1 old version blocks", g.dflash_lossless_blocker() is not None)
fake_dflash(GOOD)
check("H2 fixed version does not block", g.dflash_lossless_blocker() is None)
fake_dflash(None)
check("H3 missing package blocks", g.dflash_lossless_blocker() is not None)
fake_dflash("0.1.9")
check("H4 exact minimum version passes", g.dflash_lossless_blocker() is None)
fake_dflash("0.2.0")
check("H5 future version passes", g.dflash_lossless_blocker() is None)
fake_dflash("1.0.0")
check("H6 major bump passes", g.dflash_lossless_blocker() is None)

fake_dflash(BAD)
buf = io.StringIO()
ret = g.warn_unless_lossless("unit test", stream=buf)
check("H7 warn_unless_lossless returns the blocker", ret is not None)
check("H8 banner is loud and names the context",
      "LOSSLESSNESS NOT ESTABLISHED" in buf.getvalue() and "unit test" in buf.getvalue())
fake_dflash(GOOD)
buf = io.StringIO()
check("H9 silent when trustworthy",
      g.warn_unless_lossless("unit test", stream=buf) is None and buf.getvalue() == "")

src = pathlib.Path("scripts/bench_qwen38.py").read_text()
check("H10 bench has no duplicate version logic", 'md.version("dflash-mlx")' not in src)
check("H11 bench imports the shared guard", "from dflash_guard import" in src)

for f in ("scripts/run_qwen_inference.py", "scripts/serve_qwen.py"):
    s = pathlib.Path(f).read_text()
    check(f"H12 {pathlib.Path(f).name} imports the guard", "from dflash_guard import" in s)
    # Count real subprocess invocations only: `"dflash", "<subcommand>"` opening a
    # command list. A bare "dflash" in an argparse choices= list is not a call.
    import re as _re
    invocations = _re.findall(r'"dflash",\s*"(generate|serve|benchmark)"', s)
    guards = s.count("warn_unless_lossless(f")
    check(f"H13 {pathlib.Path(f).name} gates every dflash invocation",
          guards >= len(invocations),
          f"{guards} guards vs {len(invocations)} invocations {invocations}")

    # A guard placed after the subprocess call would warn about tokens already
    # emitted. Every invocation must be preceded by a guard.
    lines = s.splitlines()
    guard_at = [i for i, ln in enumerate(lines) if "warn_unless_lossless(f" in ln]
    call_at = [i for i, ln in enumerate(lines)
               if _re.search(r'"dflash",\s*"(generate|serve|benchmark)"', ln)]
    check(f"H15 {pathlib.Path(f).name} guards precede their invocations",
          all(any(gi < ci for gi in guard_at) for ci in call_at),
          f"guards at {guard_at}, calls at {call_at}")

md.version = _orig_version
r = subprocess.run([sys.executable, "-c",
                    "import sys; sys.path.insert(0,'scripts'); import run_qwen_inference, serve_qwen; print('ok')"],
                   capture_output=True, text=True, timeout=300)
check("H14 siblings still import cleanly", r.returncode == 0 and "ok" in r.stdout,
      (r.stderr or "")[-300:])

print(f"\n{'='*66}\nPASS {len(PASSES)}   FAIL {len(FAILS)}")
if FAILS:
    print("\nFAILING:")
    for f in FAILS:
        print("  -", f)
sys.exit(1 if FAILS else 0)
