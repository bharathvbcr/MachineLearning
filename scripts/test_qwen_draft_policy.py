#!/usr/bin/env python3
"""Adversarial suite for scripts/qwen_draft_policy.py and its call sites.

Every FAIL here is a way a Qwen3.8-27B target could reach the GPU with no drafter
and nothing saying so — a silent 2.18x regression that answers every request.

No engine, no model weights, and no network are touched: only argument resolution
and config auditing run, and the oMLX settings file is faked in a temp dir.

    python3 scripts/test_qwen_draft_policy.py     # exit 0 = clean

Run from the repo root.
"""
import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile

sys.argv = ["x"]
sys.path.insert(0, "scripts")
# Imported by name, the same way the call sites import it, so `is` comparisons
# below actually prove single ownership instead of comparing two loaded copies.
import qwen_draft_policy as p  # noqa: E402

FAILS, PASSES = [], []


def check(name, cond, detail=""):
    (PASSES if cond else FAILS).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def quiet(fn, *a, **kw):
    """Run fn capturing the policy's stderr banners; return (result, stderr)."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


def run_and_catch(target, draft, engine):
    """resolve_draft, swallowing the refusal so its stderr can be inspected."""
    try:
        return p.resolve_draft(target, draft, engine=engine, context="t")
    except p.BareLoadRefused:
        return None


print("\n=== A. TARGET RECOGNITION ===")
check("A1 canonical 4-bit repo id is guarded", p.is_guarded_target(p.TARGET_4BIT))
check("A2 8-bit and nvfp4 variants are guarded",
      p.is_guarded_target(p.TARGET_8BIT) and p.is_guarded_target(p.TARGET_NVFP4))
check("A3 oMLX '--' spelling is guarded",
      p.is_guarded_target("mlx-community--Qwen3.8-27B-4bit"))
check("A4 local snapshot path is guarded",
      p.is_guarded_target("/Users/x/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/abc"))
check("A5 an unrecognized 3.8-27B quant still trips the guard (no silent pass-through)",
      p.is_guarded_target("someone/Qwen3.8-27B-mxfp6"))
check("A6 a different model is NOT guarded",
      not p.is_guarded_target("mlx-community/Qwen3.6-27B-4bit")
      and not p.is_guarded_target("mlx-community/gemma-4-31B-it-4bit"))

print("\n=== B. OMISSION: no --draft given ===")
got, err = quiet(p.resolve_draft, p.TARGET_4BIT, None, engine="mlx-vlm", context="t")
check("B1 mlx-vlm attaches the MTP drafter", got == p.MTP_4BIT, f"got {got}")
check("B2 the attachment is announced, not silent", "attaching" in err, f"stderr={err!r}")
got, _ = quiet(p.resolve_draft, p.TARGET_NVFP4, None, engine="mlx-vlm", context="t")
check("B3 nvfp4 target attaches the nvfp4 MTP drafter", got == p.MTP_NVFP4, f"got {got}")
got, _ = quiet(p.resolve_draft, p.TARGET_4BIT, None, engine="dspark", context="t")
check("B4 dspark attaches the 3.8-native drafter", got == p.DSPARK_4BIT, f"got {got}")
got, _ = quiet(p.resolve_draft, p.TARGET_4BIT, None, engine="dflash", context="t")
check("B5 dflash attaches the 3.6 cross-apply", got == p.DFLASH_36, f"got {got}")

try:
    quiet(p.resolve_draft, p.TARGET_8BIT, None, engine="mlx-vlm", context="t")
    check("B6 unpaired (engine,target) refuses rather than guessing", False, "no raise")
except p.BareLoadRefused as e:
    check("B6 unpaired (engine,target) refuses rather than guessing",
          "no measured drafter" in e.message)

print("\n=== C. SILENT REINTERPRETATION: --draft none ===")
# Pre-change, serve_qwen.py mapped "none" -> None and then `draft or DEFAULT_MTP_DRAFT`
# put the drafter straight back: the documented way to ask for AR silently served MTP.
try:
    quiet(p.resolve_draft, p.TARGET_4BIT, "none", engine="mlx-vlm", context="t")
    check("C1 'none' is refused without --allow-bare", False, "no raise")
except p.BareLoadRefused as e:
    check("C1 'none' is refused without --allow-bare", "REFUSING" in e.message)

got, err = quiet(p.resolve_draft, p.TARGET_4BIT, "none", engine="mlx-vlm", context="t",
                 allow_bare=True, bare_reason="baseline")
check("C2 'none' + --allow-bare yields a bare load", got is None, f"got {got}")
check("C3 an approved bare load still shouts", "BARE LOAD" in err and "baseline" in err)

try:
    quiet(p.resolve_draft, p.TARGET_4BIT, "none", engine="mlx-vlm", context="t", allow_bare=True)
    check("C4 allow_bare with no reason is refused", False, "no raise")
except p.BareLoadRefused as e:
    check("C4 allow_bare with no reason is refused", "no reason" in e.message)

print("\n=== D. DRAFTERLESS ENGINE ===")
try:
    quiet(p.resolve_draft, p.TARGET_4BIT, None, engine="mlx-lm", context="t")
    check("D1 mlx-lm on a guarded target is refused", False, "no raise")
except p.BareLoadRefused as e:
    check("D1 mlx-lm on a guarded target is refused", "no drafter path" in e.message)
    check("D2 refusal names the fix", "--allow-bare" in e.message)

got, _ = quiet(p.resolve_draft, "mlx-community/Qwen3.6-27B-4bit", None, engine="mlx-lm", context="t")
check("D3 an unguarded model is untouched by the policy", got is None, f"got {got}")

print("\n=== E. EXPLICIT DRAFT ALWAYS WINS ===")
got, _ = quiet(p.resolve_draft, p.TARGET_4BIT, "someone/custom-drafter", engine="mlx-vlm", context="t")
check("E1 an explicit --draft is passed through", got == "someone/custom-drafter", f"got {got}")
buf = io.StringIO()
check("E2 exit code is 2, not a bare crash",
      p.BareLoadRefused("x", stream=buf).code == 2)
# SystemExit(2) prints nothing and SystemExit(msg) exits 1; a guard that stops the
# process without saying why is indistinguishable from a crash.
check("E3 the refusal reason actually reaches stderr", "x" in buf.getvalue())

_, err = quiet(lambda: run_and_catch(p.TARGET_4BIT, None, "mlx-lm"))
check("E4 a refused CLI call prints the reason, not just an exit code",
      "REFUSING" in err and "--allow-bare" in err, f"stderr={err!r}")

print("\n=== F. CALL SITES ===")
srv = importlib.util.spec_from_file_location("srv", "scripts/serve_qwen.py")
serve = importlib.util.module_from_spec(srv)
srv.loader.exec_module(serve)


def run_main(mod, argv):
    """Invoke a script's main() with argv, returning ('exit', code) or ('ran', calls)."""
    calls = []
    real = getattr(mod, "subprocess").run
    mod.subprocess.run = lambda cmd, *a, **kw: calls.append(cmd)
    old = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            mod.main()
        return "ran", calls
    except SystemExit as e:
        return "exit", e.code
    finally:
        sys.argv = old
        mod.subprocess.run = real


kind, val = run_main(serve, ["serve_qwen.py", "--backend", "mlx-lm"])
check("F1 serve --backend mlx-lm no longer starts a bare 27B server",
      kind == "exit" and val == 2, f"got {kind}:{val}")

kind, val = run_main(serve, ["serve_qwen.py", "--backend", "mlx-lm", "--allow-bare"])
check("F2 ...but does with --allow-bare", kind == "ran" and len(val) == 1, f"got {kind}:{val}")

kind, val = run_main(serve, ["serve_qwen.py", "--backend", "mlx-vlm", "--draft", "none"])
check("F3 serve --draft none no longer silently serves MTP",
      kind == "exit" and val == 2, f"got {kind}:{val}")

kind, val = run_main(serve, ["serve_qwen.py", "--backend", "mlx-vlm"])
ok = kind == "ran" and len(val) == 1 and p.MTP_4BIT in val[0]
check("F4 default serve still attaches the MTP drafter", ok, f"got {kind}:{val}")

kind, val = run_main(serve, ["serve_qwen.py", "--backend", "dspark", "--draft", "none",
                             "--allow-bare"])
check("F5 an approved bare load on a drafterless-incapable backend is refused, not crashed",
      kind == "exit" and val == 2, f"got {kind}:{val}")

print("\n=== F2. OMLX BACKEND: the drafter lives in a config file, not argv ===")
with tempfile.TemporaryDirectory() as d:
    cfg = pathlib.Path(d) / "model_settings.json"
    calls = []
    real_run, real_cli = serve.subprocess.run, serve.OMLX_CLI
    serve.subprocess.run = lambda cmd, *a, **kw: calls.append(cmd)
    serve.OMLX_CLI = "/bin/echo"  # exists, so the launch path is exercised
    try:
        # Config missing entirely: oMLX would load bare. Repair, then launch.
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            serve.serve_omlx(p.TARGET_4BIT, draft=p.DFLASH2_38, settings_path=cfg)
        written = json.loads(cfg.read_text())
        check("F6 a missing oMLX config is repaired before the server starts",
              written["models"][p.omlx_model_id(p.TARGET_4BIT)]["dflash_enabled"] is True)
        check("F7 the server is then actually launched", len(calls) == 1 and "serve" in calls[0])

        # Drifted config: entry present but dflash switched off.
        cfg.write_text(json.dumps({"version": 1, "models": {
            p.omlx_model_id(p.TARGET_4BIT): {"dflash_enabled": False}}}))
        calls.clear()
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            serve.serve_omlx(p.TARGET_4BIT, draft=p.DFLASH2_38, settings_path=cfg)
        written = json.loads(cfg.read_text())
        check("F8 a drifted config (dflash toggled off) is repaired, not trusted",
              written["models"][p.omlx_model_id(p.TARGET_4BIT)]["dflash_enabled"] is True
              and len(calls) == 1)

        # No oMLX installed: a clear message beats a confusing exec failure.
        serve.OMLX_CLI = "/nonexistent/omlx-cli"
        calls.clear()
        try:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                serve.serve_omlx(p.TARGET_4BIT, draft=p.DFLASH2_38, settings_path=cfg)
            check("F9 a missing oMLX install exits cleanly", False, "no raise")
        except SystemExit as e:
            check("F9 a missing oMLX install exits cleanly", e.code == 2 and not calls)
    finally:
        serve.subprocess.run, serve.OMLX_CLI = real_run, real_cli

print("\n=== G. BENCH: the one legitimate bare load ===")
bspec = importlib.util.spec_from_file_location("b", "scripts/bench_qwen38.py")
b = importlib.util.module_from_spec(bspec)
bspec.loader.exec_module(b)

_, err = quiet(b.build_cmd, "ar", "hi", 8)
check("G1 the ar arm is exempt and still builds", True)
check("G2 the ar exemption is announced with its reason",
      "BARE LOAD" in err and "anchor" in err, f"stderr={err!r}")
notes, _ = b.preflight(["ar", "mtp"])
check("G3 the exemption lands in preflight notes (so results.json carries it)",
      any(n.startswith("EXEMPT [ar]") for n in notes), f"notes={notes}")
notes_no_ar, _ = b.preflight(["mtp"])
check("G4 no exemption note when the ar arm is not run",
      not any(n.startswith("EXEMPT") for n in notes_no_ar))
check("G5 bench and policy agree on the target id (no second copy to drift)",
      b.TARGET_4BIT is p.TARGET_4BIT and b.MTP_4BIT is p.MTP_4BIT)

print("\n=== H. OMLX CONFIG AUDIT ===")
mid = p.omlx_model_id(p.TARGET_4BIT)
ok, f = p.audit_omlx({"version": 1, "models": {mid: p.OMLX_DFLASH2_SETTINGS}})
check("H1 a configured target passes", ok and any(x.startswith("OK:") for x in f), f"{f}")

ok, f = p.audit_omlx({"version": 1, "models": {}})
check("H2 a missing entry fails (oMLX would load it bare)",
      not ok and any("MISSING" in x for x in f), f"{f}")

ok, f = p.audit_omlx({"version": 1, "models": {mid: {"dflash_enabled": False}}})
check("H3 dflash_enabled=false fails", not ok and any("BARE" in x for x in f), f"{f}")

ok, f = p.audit_omlx({"version": 1, "models": {mid: {"dflash_enabled": True}}})
check("H4 enabled-but-no-draft-model fails", not ok and any("names no" in x for x in f), f"{f}")

ok, f = p.audit_omlx({"version": 1, "models": {
    mid: p.OMLX_DFLASH2_SETTINGS,
    "mlx-community--Qwen3.8-27B-nvfp4": {"dflash_enabled": False}}})
check("H5 a second guarded target that would load bare is caught too",
      not ok and any("nvfp4" in x and "BARE" in x for x in f), f"{f}")

with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / "model_settings.json"
    path.write_text(json.dumps({"version": 1, "models": {
        "other--model": {"ttl_seconds": 60}}}))
    p.apply_omlx_settings(path)
    written = json.loads(path.read_text())
    check("H6 apply writes the pairing", written["models"][mid]["dflash_draft_model"] == p.DFLASH2_38)
    check("H7 apply preserves unrelated models", "other--model" in written["models"])
    ok, _ = p.audit_omlx(written)
    check("H8 the written config passes its own audit", ok)

print(f"\n{'='*70}")
print(f"{len(PASSES)} passed, {len(FAILS)} failed")
if FAILS:
    print("FAILED: " + ", ".join(FAILS))
print(f"{'='*70}")
sys.exit(1 if FAILS else 0)
