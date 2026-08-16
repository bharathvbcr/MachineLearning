#!/usr/bin/env python3
"""Adversarial suite for scripts/bench_agents.py.

The verifier is the only thing standing between a lying agent and a green
result, so it gets attacked first. No agent, no server, and no network are
touched: every case is a synthetic file or a synthetic log line.

    python3 scripts/test_bench_agents.py     # exit 0 = clean

Run from the repo root.
"""
import importlib.util, sys, tempfile, pathlib, subprocess

sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ba", "scripts/bench_agents.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

FAILS, PASSES = [], []


def check(name, cond, detail=""):
    (PASSES if cond else FAILS).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


F = m.FIXTURE
CORRECT = F.replace("import sys\n", "").replace("import time\n", "")


print("\n=== A. VERIFIER: the cases that actually happened ===")

ok, why = m.verify(F, CORRECT)
check("A1 correct edit passes", ok, why)

# The real oh-my-pi failure: removed `re`, which is used, then claimed success.
omp_failure = CORRECT.replace("import re\n", "")
ok, why = m.verify(F, omp_failure)
check("A2 removing an in-use import FAILS", not ok, "this is the exact omp bug")
check("A2b and the reason names the import", any("re" in w for w in why), why)

# It still parses — which is why the agent's own ast.parse check passed.
check("A3 the broken file still parses (why naive checks miss it)",
      m.module_imports(omp_failure) is not None)

ok, _ = m.verify(F, F)
check("A4 no-op is a failure, not a pass", not ok)

ok, why = m.verify(F, F.replace("import sys\n", ""))
check("A5 partial removal fails", not ok, why)
check("A5b reason names the leftover", any("time" in w for w in why), why)

ok, why = m.verify(F, CORRECT + "\nimport json\n")
check("A6 invented import fails", not ok, why)

ok, why = m.verify(F, CORRECT.replace('print(f"{name}: {score}")', 'print(name, score)'))
check("A7 collateral edit fails", not ok, why)
check("A7b reason says so", any("outside the imports" in w for w in why), why)

ok, why = m.verify(F, "def broken(:\n")
check("A8 unparseable result fails", not ok, why)
check("A8b reason says SyntaxError", any("parse" in w for w in why), why)

ok, why = m.verify(F, "")
check("A9 emptied file fails", not ok, why)

# Aliases and from-imports must not fool the import extractor.
check("A10 aliased import counted by alias",
      m.module_imports("import numpy as np\n") == {"np"})
check("A11 from-import counted by module",
      m.module_imports("from os import path\n") == {"os"})
check("A12 dotted import counted by root",
      m.module_imports("import os.path\n") == {"os"})
check("A13 nested (non-top-level) import ignored",
      m.module_imports("def f():\n    import sys\n") == set())

# A comment mentioning the import must not count as usage or as an import.
ok, _ = m.verify(F, CORRECT.replace("import os\n", "# import os\n"))
check("A14 commented-out in-use import fails", not ok)


print("\n=== B. SERVER LOG PARSING ===")

LOG = """2026-08-16 15:43:55,423 - INFO - Request completed: endpoint=/chat/completions model=m prompt_tokens=3020 generated_tokens=141 elapsed=5.753s prefill=2880.2 tok/s decode=30.2 tok/s finish_reason=stop in_flight=0
2026-08-16 15:44:10,000 - INFO - Request completed: endpoint=/chat/completions model=m prompt_tokens=39782 generated_tokens=71 elapsed=8.247s prefill=12108.4 tok/s decode=17.3 tok/s finish_reason=stop in_flight=0
2026-08-16 15:44:20,000 - INFO - something else entirely
"""
with tempfile.TemporaryDirectory() as td:
    lp = pathlib.Path(td) / "s.log"
    lp.write_text(LOG)
    import datetime as _dt
    t = _dt.datetime.strptime("2026-08-16 15:43:00,000", m.LOG_TS).timestamp()
    reqs = m.server_requests(str(lp), t, t + 3600)
    check("B1 parses both completed requests", len(reqs) == 2, f"got {len(reqs)}")
    check("B2 prompt tokens correct", [r["prompt_tokens"] for r in reqs] == [3020, 39782])
    check("B3 prefill parsed", reqs[0]["prefill_tok_s"] == 2880.2)
    check("B4 decode parsed", reqs[0]["decode_tok_s"] == 30.2)

    # Time windowing: a request outside the window belongs to another agent's run.
    narrow = m.server_requests(str(lp), t, t + 56)
    check("B5 window excludes later requests", len(narrow) == 1, f"got {len(narrow)}")

    check("B6 missing log yields [] (unknown), not a crash",
          m.server_requests(str(lp.parent / "nope.log"), t, t + 10) == [])
    check("B7 None log path yields []", m.server_requests(None, t, t + 10) == [])

    lp.write_text("garbage\nnot a log\n")
    check("B8 unparseable log yields []", m.server_requests(str(lp), t, t + 10) == [])

    # An aborted request logs prefill progress but never "Request completed".
    # Qwen Code hits this: ~29.7k tokens at ~100 tok/s exceeds its own 240s
    # stream-idle timeout, so it kills the request it just made.
    ABORTED = """2026-08-16 16:15:42,131 - INFO - Prefill progress: request=abc tokens=2048/29766 (6.9%)
2026-08-16 16:15:59,000 - INFO - Prefill progress: request=abc tokens=14336/29766 (48.2%)
"""
    lp.write_text(ABORTED)
    t2 = _dt.datetime.strptime("2026-08-16 16:15:00,000", m.LOG_TS).timestamp()
    check("B9 aborted request still yields its prompt size",
          m.server_prefill_totals(str(lp), t2, t2 + 3600) == [29766])
    check("B10 aborted request has no completed record",
          m.server_requests(str(lp), t2, t2 + 3600) == [])
    check("B11 prefill totals honour the time window",
          m.server_prefill_totals(str(lp), t2 - 7200, t2 - 3600) == [])
    check("B12 missing log yields [] for prefill too",
          m.server_prefill_totals(None, t2, t2 + 10) == [])


print("\n=== C. UNKNOWN != ZERO ===")
base = {"agent": "x", "status": "PASS", "note": "", "cmd": [], "returncode": 0,
        "wall_s": 1.0, "correct": True, "problems": [], "requests": 0,
        "max_prompt_tokens": None, "median_prompt_tokens": None,
        "median_prefill_tok_s": None, "median_decode_tok_s": None,
        "server_requests": [], "result_file": "", "stdout_tail": ""}
txt = "\n".join(m.report([dict(base)], None))
check("C1 unmeasured prompt renders 'unknown', never 0", "unknown" in txt and " 0 " not in txt.split("max prompt")[-1][:40])
check("C2 a run with no token data warns loudly", "MISSING, not zero" in txt)

txt2 = "\n".join(m.report([dict(base, agent="a", max_prompt_tokens=3020),
                           dict(base, agent="b", max_prompt_tokens=39782)], "/tmp/x.log"))
check("C3 spread computed between agents", "13.2x" in txt2, txt2)
check("C4 no false warning when data exists", "MISSING, not zero" not in txt2)


print("\n=== D. FIXTURE INTEGRITY ===")
check("D1 fixture parses", m.module_imports(F) is not None)
check("D2 fixture imports are exactly expected+removable",
      m.module_imports(F) == m.EXPECTED_IMPORTS | m.REMOVABLE_IMPORTS,
      f"got {m.module_imports(F)}")
check("D3 removable imports really are unused",
      all(f"{n}." not in "".join(m.non_import_lines(F)) for n in m.REMOVABLE_IMPORTS))
check("D4 expected imports really are used",
      all(f"{n}." in "".join(m.non_import_lines(F)) for n in m.EXPECTED_IMPORTS))
check("D5 fixture compiles as a program", compile(F, "f", "exec") is not None)


print("\n=== E. AGENT REGISTRY / CLI ===")
for name, spec_ in m.AGENTS.items():
    cmd = spec_["cmd"]("TASK", pathlib.Path("/tmp/w"), "target.py")
    check(f"E1 {name} cmd is a non-empty list of str",
          isinstance(cmd, list) and cmd and all(isinstance(c, str) for c in cmd))
    check(f"E2 {name} cmd carries the task text", any("TASK" in c for c in cmd), cmd)
    check(f"E3 {name} binary name matches cmd[0]", cmd[0] == spec_["bin"], cmd[:1])

# Env-configured agents: Qwen Code has no CLI flag for the endpoint, so a
# dropped env dict would silently send the task to whatever OPENAI_BASE_URL the
# shell happens to have — possibly a paid cloud endpoint.
src_ba = pathlib.Path("scripts/bench_agents.py").read_text()
check("E3b per-agent env is merged over os.environ", "env.update(spec.get(\"env\"" in src_ba)
check("E3c env is actually passed to subprocess", "env=env" in src_ba)
qwen = m.AGENTS.get("qwen", {})
check("E3d qwen carries a full endpoint env",
      {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"} <= set(qwen.get("env", {})),
      qwen.get("env"))
check("E3d2 qwen disables its 240s stream-idle timeout",
      qwen.get("env", {}).get("QWEN_STREAM_IDLE_TIMEOUT_MS") == "0",
      "without this it aborts its own request mid-prefill")
check("E3e qwen env points at the local server",
      qwen.get("env", {}).get("OPENAI_BASE_URL") == m.BASE_URL)
check("E3f agents without env still work", "env" not in m.AGENTS["pi"])

# Regression: an inherited stdin makes stdin-reading agents (qwen -p, kon -p)
# block for an EOF that never arrives. The run then burns the full timeout
# having sent zero requests, which reads exactly like a very slow agent.
check("E3g stdin is closed for every agent", "stdin=subprocess.DEVNULL" in src_ba)

# Qwen Code refuses every edit/write/shell tool in headless mode without -y. It
# then reasons correctly and changes nothing, which the verifier would otherwise
# score as a wrong answer — an unfair comparison caused by the harness.
check("E3i qwen passes its auto-approve flag", "-y" in m.AGENTS["qwen"]["cmd"]("t", pathlib.Path("/tmp"), "f.py"))
check("E3j BLOCKED is distinguished from FAIL", '"BLOCKED"' in src_ba)
check("E3k approval markers cover the real message",
      any("requires user approval" in k for k in m.APPROVAL_MARKERS))
check("E3l blocked detection requires an unchanged file",
      "result == FIXTURE" in src_ba)

_probe = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.exit(0 if sys.stdin.read()=='' else 1)"],
    capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
check("E3h DEVNULL really yields immediate EOF", _probe.returncode == 0)

r = subprocess.run([sys.executable, "scripts/bench_agents.py", "--list"],
                   capture_output=True, text=True, timeout=120)
check("E4 --list works", r.returncode == 0 and "pi" in r.stdout)
r = subprocess.run([sys.executable, "scripts/bench_agents.py", "--agents", "nope"],
                   capture_output=True, text=True, timeout=120)
check("E5 unknown agent rejected (exit 2)", r.returncode == 2, f"rc={r.returncode}")

r = run = subprocess.run([sys.executable, "scripts/bench_agents.py", "--help"],
                         capture_output=True, text=True, timeout=120)
check("E6 --help documents --server-log", r.returncode == 0 and "--server-log" in r.stdout)


print("\n=== F. ISOLATION: an agent must never reach a real file ===")
src = pathlib.Path("scripts/bench_agents.py").read_text()
check("F1 fixture written to a TemporaryDirectory", "TemporaryDirectory" in src)
check("F2 agent runs with cwd set to the temp dir", "cwd=work" in src)
check("F3 no repo path is passed to any agent",
      not any("parameter_golf" in c for s in m.AGENTS.values()
              for c in s["cmd"]("t", pathlib.Path("/tmp/w"), "target.py")))


print(f"\n{'='*66}\nPASS {len(PASSES)}   FAIL {len(FAILS)}")
if FAILS:
    print("\nFAILING:")
    for f in FAILS:
        print("  -", f)
sys.exit(1 if FAILS else 0)
