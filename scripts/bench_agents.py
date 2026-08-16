#!/usr/bin/env python3
"""Compare CLI coding harnesses against the local Qwen server.

Answers one question: which harness is actually usable on THIS machine?

The deciding metric is prompt size, not model speed. Cold prefill runs at
~74-101 tok/s regardless of harness, so a harness that ships a 39k-token system
prompt pays ~390s per cold request while a 3k-token one pays ~20s. Decode
throughput barely matters — see docs/qwen_mlx_dflash_guide.md section 6a.

Three rules this harness enforces, each learned from a real failure:

  1. NEVER trust the agent's self-report. oh-my-pi deleted an in-use import and
     then stated it had "verified" the file still had it. Correctness is decided
     here by parsing the AST, never by reading what the agent claimed.
  2. NEVER let an agent touch a real file. Every run gets a fresh copy of the
     fixture in a temp dir, so a bad edit costs nothing.
  3. NEVER invent a number. prompt_tokens comes from the server's own
     "Request completed" log line. If it cannot be read, it is reported as
     unknown -- not estimated, not omitted.

Usage:
    # start the server first, keeping its log:
    #   APC_ENABLED=1 python3 scripts/serve_qwen.py --backend mlx-vlm --port 8000 > /tmp/qwen.log 2>&1 &
    python3 scripts/bench_agents.py --server-log /tmp/qwen.log
    python3 scripts/bench_agents.py --server-log /tmp/qwen.log --agents pi,kon
    python3 scripts/bench_agents.py --list

Artifacts land in .artifacts/agents/<timestamp>/.
"""

import argparse
import ast
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"
BASE_URL = "http://127.0.0.1:8000/v1"

# --------------------------------------------------------------------------- #
# Fixture: real code with a known-correct answer
# --------------------------------------------------------------------------- #

FIXTURE = '''#!/usr/bin/env python3
import subprocess
import re
import sys
import os
import time

def collect(paths):
    found = []
    for p in paths:
        full = os.path.join("build", p)
        if not os.path.exists(full):
            continue
        out = subprocess.run(["cat", full], capture_output=True, text=True).stdout
        m = re.search(r"score=([\\d.]+)", out)
        if m:
            found.append((os.path.dirname(full), float(m.group(1))))
    return found

def main():
    for name, score in collect(["a.txt", "b.txt"]):
        print(f"{name}: {score}")

if __name__ == "__main__":
    main()
'''

# `sys` and `time` are imported and never referenced. Everything else is used.
EXPECTED_IMPORTS = {"subprocess", "re", "os"}
REMOVABLE_IMPORTS = {"sys", "time"}

TASK = ("Remove the unused imports from {fname}. "
        "Keep every import that is actually referenced. Change nothing else.")

# An agent whose edit tool was refused did not fail the task — the harness failed
# to configure it. Scoring those identically would publish an unfair comparison,
# so they are reported as BLOCKED with the fix named.
APPROVAL_MARKERS = (
    "requires user approval",
    "cannot execute in non-interactive mode",
    "denied permission",
    "yolo mode",
)


# --------------------------------------------------------------------------- #
# Agent registry
# --------------------------------------------------------------------------- #

def _pi(task, cwd, fname):
    return ["pi", "-p", "--model", f"qwen-local/{MODEL_ID}", task]


def _omp(task, cwd, fname):
    return ["omp", "-p", "--cwd", str(cwd), "--model", f"qwen-local/{MODEL_ID}", task]


def _kon(task, cwd, fname):
    # kon takes the endpoint on the command line; no config file needed.
    return ["kon", "-p", task, "--provider", "openai", "--model", MODEL_ID,
            "--base-url", BASE_URL, "--openai-compat-auth", "none"]


def _zero(task, cwd, fname):
    return ["zero", "exec", task]


def _qwen(task, cwd, fname):
    # -y is mandatory for headless use: without it every edit/write/shell tool is
    # refused with "requires user approval but cannot execute in non-interactive
    # mode", and the agent reasons correctly then changes nothing.
    return ["qwen", "-p", task, "-m", MODEL_ID, "-y"]


# `env` is merged over os.environ for that agent only. Qwen Code takes its
# endpoint entirely through environment variables, with no CLI equivalent.
AGENTS = {
    "pi":   {"cmd": _pi,   "bin": "pi",
             "note": "earendil-works/pi — npm i -g --ignore-scripts @earendil-works/pi-coding-agent"},
    "kon":  {"cmd": _kon,  "bin": "kon",
             "note": "0xku/kon — uv tool install kon-coding-agent  [WINNER]"},
    "zero": {"cmd": _zero, "bin": "zero",
             "note": "gitlawb/zero — npm i -g @gitlawb/zero"},
    # QWEN_STREAM_IDLE_TIMEOUT_MS=0 is required, not a tuning preference. Qwen
    # Code aborts a stream after 240s without a chunk; its ~29.7k-token prompt
    # needs ~300s of prefill at ~100 tok/s, so at defaults it reliably kills the
    # request it just issued and never completes one. 0 disables the window.
    "qwen": {"cmd": _qwen, "bin": "qwen",
             "note": "QwenLM/qwen-code — npm i -g @qwen-code/qwen-code (Gemini CLI fork)",
             "env": {"OPENAI_API_KEY": "local",
                     "OPENAI_BASE_URL": BASE_URL,
                     "OPENAI_MODEL": MODEL_ID,
                     "QWEN_STREAM_IDLE_TIMEOUT_MS": "0"}},
    "omp":  {"cmd": _omp,  "bin": "omp",
             "note": "can1357/oh-my-pi — npm i -g @oh-my-pi/pi-coding-agent bun (990M, ~13 min/task)"},
}


# --------------------------------------------------------------------------- #
# Independent verification (never trust the agent)
# --------------------------------------------------------------------------- #

def module_imports(src):
    """Top-level imported names, or None if the source does not parse."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def non_import_lines(src):
    """Every line that is not a top-level import, for collateral-damage checks."""
    return [ln for ln in src.splitlines()
            if not re.match(r"\s*(import|from)\s+\w", ln)]


def verify(original, result):
    """Decide correctness from the file itself. Returns (ok, [reasons])."""
    problems = []

    if result == original:
        return False, ["file unchanged — agent did not perform the edit"]

    imports = module_imports(result)
    if imports is None:
        return False, ["result does not parse (SyntaxError)"]

    missing = EXPECTED_IMPORTS - imports
    if missing:
        # The failure mode that matters: removing an import that is still used.
        # This is a runtime NameError, not a syntax error, so `python -c compile`
        # and ast.parse both pass. Only an explicit set check catches it.
        problems.append(f"removed in-use import(s): {sorted(missing)}")

    left = REMOVABLE_IMPORTS & imports
    if left:
        problems.append(f"left unused import(s) behind: {sorted(left)}")

    extra = imports - EXPECTED_IMPORTS - REMOVABLE_IMPORTS
    if extra:
        problems.append(f"invented import(s): {sorted(extra)}")

    if non_import_lines(original) != non_import_lines(result):
        problems.append("changed lines outside the imports (task said change nothing else)")

    return (not problems), problems


# --------------------------------------------------------------------------- #
# Server-side token accounting
# --------------------------------------------------------------------------- #

LOG_TS = "%Y-%m-%d %H:%M:%S,%f"
REQ_RE = re.compile(
    r"^(?P<ts>[\d-]+ [\d:,]+).*Request completed:.*?"
    r"prompt_tokens=(?P<prompt>\d+)\s+generated_tokens=(?P<gen>\d+)\s+"
    r"elapsed=(?P<elapsed>[\d.]+)s\s+prefill=(?P<prefill>[\d.]+)\s*tok/s\s+"
    r"decode=(?P<decode>[\d.]+)"
)


PREFILL_RE = re.compile(
    r"^(?P<ts>[\d-]+ [\d:,]+).*Prefill progress:.*tokens=\d+/(?P<total>\d+)")


def server_prefill_totals(log_path, t0, t1):
    """Prompt sizes seen in prefill-progress lines, for requests that never finished.

    A client that aborts mid-prefill (Qwen Code gives up after a 240s stream-idle
    timeout) never produces a "Request completed" line, so its prompt size would
    read as unknown — precisely for the agents whose prompts are too large to
    prefill in time, which is the measurement that matters most.
    """
    if not log_path:
        return []
    try:
        text = Path(log_path).read_text(errors="ignore")
    except OSError:
        return []

    totals = []
    for line in text.splitlines():
        m = PREFILL_RE.search(line)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group("ts"), LOG_TS).timestamp()
        except ValueError:
            continue
        if t0 <= ts <= t1:
            totals.append(int(m.group("total")))
    return sorted(set(totals))


def server_requests(log_path, t0, t1):
    """Completed requests the server logged inside [t0, t1].

    Returns [] when the log is unreadable — callers must treat that as unknown,
    never as zero. A missing measurement and a measurement of zero are different
    facts and must not render identically.
    """
    if not log_path:
        return []
    try:
        text = Path(log_path).read_text(errors="ignore")
    except OSError:
        return []

    out = []
    for line in text.splitlines():
        m = REQ_RE.search(line)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group("ts"), LOG_TS).timestamp()
        except ValueError:
            continue
        if t0 <= ts <= t1:
            out.append({
                "prompt_tokens": int(m.group("prompt")),
                "generated_tokens": int(m.group("gen")),
                "elapsed_s": float(m.group("elapsed")),
                "prefill_tok_s": float(m.group("prefill")),
                "decode_tok_s": float(m.group("decode")),
            })
    return out


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_agent(name, server_log, timeout, keep_going):
    spec = AGENTS[name]
    if not shutil.which(spec["bin"]):
        return {"agent": name, "status": "MISSING",
                "detail": f"`{spec['bin']}` not on PATH", "runs": []}

    with tempfile.TemporaryDirectory(prefix=f"bench-{name}-") as td:
        work = Path(td)
        fname = "target.py"
        target = work / fname
        target.write_text(FIXTURE)

        cmd = spec["cmd"](TASK.format(fname=fname), work, fname)
        print(f"\n{'='*74}\nAGENT: {name}  — {spec['note']}\n  {' '.join(cmd[:6])} ...\n{'='*74}")

        env = os.environ.copy()
        env.update(spec.get("env", {}))

        t0 = time.time()
        try:
            # stdin MUST be closed. Several agents (Qwen Code, kon) append stdin
            # to the -p prompt and block for EOF that an inherited stdin never
            # delivers — the run hangs until the timeout with zero requests sent,
            # which is indistinguishable from a very slow agent.
            proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                                  timeout=timeout, env=env,
                                  stdin=subprocess.DEVNULL)
            rc, stdout = proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")
            timed_out = False
        except subprocess.TimeoutExpired as e:
            rc, timed_out = -1, True
            stdout = f"<timeout after {timeout}s>\n" + (e.stdout or b"").decode(errors="ignore")[-2000:]
        except FileNotFoundError as e:
            return {"agent": name, "status": "MISSING", "detail": str(e), "runs": []}
        t1 = time.time()

        result = target.read_text() if target.exists() else ""
        ok, problems = verify(FIXTURE, result)
        reqs = server_requests(server_log, t0 - 2, t1 + 2)

        prompts = [r["prompt_tokens"] for r in reqs]
        prompt_source = "completed-requests"
        if not prompts:
            # Nothing completed. Fall back to prefill-progress totals so an agent
            # that aborts mid-prefill still reports its prompt size, which is the
            # very reason it aborted.
            prompts = server_prefill_totals(server_log, t0 - 2, t1 + 2)
            if prompts:
                prompt_source = "prefill-progress (request aborted before completing)"

        # Distinguish "the agent got it wrong" from "the harness never let it act".
        blocked = (not ok and result == FIXTURE
                   and any(k in stdout.lower() for k in APPROVAL_MARKERS))
        if blocked:
            problems = ["tool approval was refused — harness misconfiguration, "
                        "not an agent failure (pass the agent's auto-approve flag)"]

        status = ("TIMEOUT" if timed_out
                  else "BLOCKED" if blocked
                  else ("PASS" if ok else "FAIL"))
        print(f"  {status}  wall={t1-t0:.1f}s  requests={len(reqs)}  "
              f"max_prompt={max(prompts) if prompts else '—'}")
        if prompt_source != "completed-requests":
            print(f"    ! prompt size from {prompt_source}")
        for p in problems:
            print(f"    ! {p}")

        return {
            "agent": name,
            "status": status,
            "note": spec["note"],
            "cmd": cmd,
            "returncode": rc,
            "wall_s": round(t1 - t0, 2),
            "correct": ok,
            "problems": problems,
            "requests": len(reqs),
            "prompt_source": prompt_source if prompts else "unmeasured",
            "max_prompt_tokens": max(prompts) if prompts else None,
            "median_prompt_tokens": int(statistics.median(prompts)) if prompts else None,
            "median_prefill_tok_s": round(
                statistics.median([r["prefill_tok_s"] for r in reqs]), 1) if reqs else None,
            "median_decode_tok_s": round(
                statistics.median([r["decode_tok_s"] for r in reqs]), 1) if reqs else None,
            "server_requests": reqs,
            "result_file": result,
            "stdout_tail": stdout[-4000:],
        }


def report(results, server_log):
    lines = ["", "=" * 74, "RESULTS", "=" * 74]
    lines.append(f"{'agent':<8} {'status':<8} {'wall':>8} {'reqs':>5} "
                 f"{'max prompt':>11} {'prefill':>10} {'decode':>8}")
    for r in sorted(results, key=lambda x: (x["status"] != "PASS",
                                            x.get("max_prompt_tokens") or 1 << 30)):
        if r["status"] == "MISSING":
            lines.append(f"{r['agent']:<8} {'MISSING':<8}   {r.get('detail','')}")
            continue
        mp = r["max_prompt_tokens"]
        mp_s = f"{mp:,}" if mp is not None else "unknown"
        pf_s = str(r["median_prefill_tok_s"]) if r["median_prefill_tok_s"] else "—"
        dc_s = str(r["median_decode_tok_s"]) if r["median_decode_tok_s"] else "—"
        lines.append(
            f"{r['agent']:<8} {r['status']:<8} {r['wall_s']:>7.1f}s {r['requests']:>5} "
            f"{mp_s:>11} {pf_s:>10} {dc_s:>8}")
        for p in r["problems"]:
            lines.append(f"{'':<17} ! {p}")

    measured = [r for r in results if r.get("max_prompt_tokens")]
    if not measured and any(r["status"] != "MISSING" for r in results):
        lines += ["", "WARNING: no prompt_tokens could be read from the server log.",
                  f"  --server-log was {server_log!r}. Numbers above are wall-clock only;",
                  "  the prompt-size comparison — the entire point — is MISSING, not zero."]
    elif measured:
        best = min(measured, key=lambda r: r["max_prompt_tokens"])
        worst = max(measured, key=lambda r: r["max_prompt_tokens"])
        if best is not worst and best["max_prompt_tokens"]:
            ratio = worst["max_prompt_tokens"] / best["max_prompt_tokens"]
            lines += ["", f"Prompt spread: {best['agent']} {best['max_prompt_tokens']:,} vs "
                          f"{worst['agent']} {worst['max_prompt_tokens']:,}  ({ratio:.1f}x)"]
    return lines


def main():
    p = argparse.ArgumentParser(description="Benchmark CLI coding harnesses on the local Qwen server.")
    p.add_argument("--agents", default="all",
                   help=f"Comma-separated, or 'all'. Available: {', '.join(AGENTS)}")
    p.add_argument("--server-log", default=None,
                   help="Path to the serve_qwen.py log. Without it, prompt_tokens is UNKNOWN "
                        "and the comparison loses its point.")
    p.add_argument("--timeout", type=int, default=1800, help="Per-agent timeout (s).")
    p.add_argument("--outdir", default=None)
    p.add_argument("--list", action="store_true", help="List agents and whether they are installed.")
    args = p.parse_args()

    if args.list:
        for n, s in AGENTS.items():
            print(f"  {n:<6} {'installed' if shutil.which(s['bin']) else 'MISSING  '}  {s['note']}")
        return

    names = list(AGENTS) if args.agents == "all" else [a.strip() for a in args.agents.split(",") if a.strip()]
    unknown = [a for a in names if a not in AGENTS]
    if unknown:
        print(f"Unknown agent(s): {unknown}. Available: {list(AGENTS)}", file=sys.stderr)
        sys.exit(2)

    if not args.server_log:
        print("WARNING: --server-log not given; prompt sizes will read as unknown.\n",
              file=sys.stderr)

    results = [run_agent(n, args.server_log, args.timeout, True) for n in names]

    lines = report(results, args.server_log)
    print("\n".join(lines))

    outdir = Path(args.outdir) if args.outdir else Path(".artifacts/agents") / datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "machine": platform.platform(),
        "model": MODEL_ID,
        "server_log": args.server_log,
        "results": results,
    }, indent=2))
    (outdir / "REPORT.md").write_text(
        "# CLI coding harness comparison\n\n```\n" + "\n".join(lines) + "\n```\n")
    print(f"\nArtifacts: {outdir}")

    if any(r["status"] == "FAIL" for r in results):
        sys.exit(4)


if __name__ == "__main__":
    main()
