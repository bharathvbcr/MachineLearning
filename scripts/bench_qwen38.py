#!/usr/bin/env python3
"""
Multi-arm throughput benchmark for Qwen3.8-27B on Apple Silicon.

Answers one question: which engine actually decodes fastest on THIS machine?

Two classes of arm:

  LOSSLESS  — every emitted token is verified against `mlx-community/Qwen3.8-27B-4bit`,
              so output is the same model's output, just produced faster. These arms
              are directly comparable to each other and to the AR baseline.

    ar          mlx-lm autoregressive baseline (no drafter)
    mtp         + official Qwen3.8 MTP drafter, via mlx-vlm      [same checkpoint]
    dflash      + z-lab Qwen3.6-27B-DFlash, via dflash-mlx       [cross-applied]
    dspark      + DimInfer Qwen3.8 DSpark drafter, via mlx-dspark [3.8-native]

  MAX-THROUGHPUT — different quantization. Faster numbers here are NOT free: the
              model itself differs, so quality is not held constant and tok/s is
              not comparable to the lossless arms on equal terms.

    nvfp4-mtp   Qwen3.8-27B-nvfp4 + nvfp4 MTP drafter, via mlx-vlm
    ollama      `ollama run qwen3.8:27b-mlx` (modelopt mixed-precision)
    dspark-8bit Qwen3.8-27B-8bit + RadixArk DSpark drafter, via mlx-dspark

              dspark-8bit sits here because it verifies against the 8-bit target,
              not the 4-bit one — so it is not comparable to the lossless arms on
              equal terms. Note the class name reads backwards for this arm: 8-bit
              is quality-SUPERIOR to the 4-bit lossless class, not a shortcut. What
              the class encodes is "different model", not "worse model".

Usage:
    python3 scripts/bench_qwen38.py                       # lossless arms only
    python3 scripts/bench_qwen38.py --arms all            # everything
    python3 scripts/bench_qwen38.py --arms ar,mtp --repeat 3

Artifacts (JSON + markdown) land in .artifacts/dflash/qwen38_bench/<timestamp>/.

NOTE: this harness only measures. It does not change any default. Read the
verdict, then decide.
"""

import argparse
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# scripts/ is not a package; make sibling modules importable whether this file is
# run directly or loaded by path (as the test suite does).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflash_guard import (  # noqa: E402
    DFLASH_UPGRADE_HINT,
    dflash_lossless_blocker,
    dflash_version,
)
from serve_qwen import resolve_drafter  # noqa: E402

TARGET_4BIT = "mlx-community/Qwen3.8-27B-4bit"
TARGET_NVFP4 = "mlx-community/Qwen3.8-27B-nvfp4"
MTP_4BIT = "mlx-community/Qwen3.8-27B-MTP-4bit"
MTP_NVFP4 = "mlx-community/Qwen3.8-27B-MTP-nvfp4"
DFLASH_36 = "z-lab/Qwen3.6-27B-DFlash"
TARGET_8BIT = "mlx-community/Qwen3.8-27B-8bit"
# First drafters trained against 3.8 itself, so neither is a cross-application.
# Each is matched to the precision it was trained for: DimInfer to the 4-bit class,
# RadixArk to the FP8/8-bit verifier. Pairing them the other way costs acceptance.
DSPARK_4BIT = "DimInfer/Qwen3.8-27B-Dspark-v1"
DSPARK_8BIT = "RadixArk/Qwen3.8-27B-DSpark"
OLLAMA_TAG = "qwen3.8:27b-mlx"

# Same functional-equation prompt dflash-mlx uses for its published numbers, so
# our figures sit on the same footing as the upstream Qwen3.6 table.
DEFAULT_PROMPT = (
    r"The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] "
    r"for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that "
    r"$f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, "
    r"and put your final answer within \boxed{}."
)

LOSSLESS_ARMS = ["ar", "mtp", "dflash", "dspark"]
MAXTP_ARMS = ["nvfp4-mtp", "ollama", "dspark-8bit"]


# --------------------------------------------------------------------------- #
# CLI resolution
# --------------------------------------------------------------------------- #

def resolve_cli(dotted: str, unified: str, subcommand: str):
    """mlx-lm / mlx-vlm ship console scripts like `mlx_vlm.generate`; some installs
    expose a unified `mlx_vlm generate`. Probe both, fall back to the module form."""
    if shutil.which(dotted):
        return [dotted]
    if shutil.which(unified):
        return [unified, subcommand]
    return [sys.executable, "-m", dotted]


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #

# Plausibility bounds. A value outside these is a parse artifact, not a
# measurement — recording it as real is how a loose regex becomes a shipped number.
BOUNDS = {
    "tok_s": (0.01, 100_000.0),
    "tokens": (1.0, 10_000_000.0),
    "peak_gb": (0.01, 4096.0),
    "accept_pct": (0.0, 100.0),
    "accept_len": (0.0, 1024.0),
}


def _search_all(patterns, text):
    """Every value matched by the first pattern that hits, in document order.

    Later patterns are consulted only when earlier ones find nothing, so specific
    formats win over generic fallbacks.
    """
    for pat in patterns:
        vals = []
        for mo in re.finditer(pat, text, re.IGNORECASE):
            try:
                vals.append(float(mo.group(1)))
            except (ValueError, IndexError):
                continue
        if vals:
            return vals
    return []


def _pick(patterns, text, key, flags, prefer_last=True):
    """Bounded, ambiguity-aware extraction.

    prefer_last matters for engines that print a baseline leg before the measured
    leg (`dflash` does): taking the first match would silently report the baseline
    as the accelerated number.
    """
    vals = _search_all(patterns, text)
    if not vals:
        return None

    lo, hi = BOUNDS[key]
    kept = [v for v in vals if lo <= v <= hi]
    if len(kept) < len(vals):
        dropped = sorted(set(vals) - set(kept))
        flags.append(f"{key}: discarded out-of-range match(es) {dropped}")
    if not kept:
        return None

    distinct = sorted(set(kept))
    if len(distinct) > 1:
        flags.append(f"{key}: {len(distinct)} distinct values {distinct}; "
                     f"used {'last' if prefer_last else 'first'}")
    return kept[-1] if prefer_last else kept[0]


# One pattern per supported engine, and nothing speculative. A loose fallback
# that never fires is not free insurance: an unanchored pattern's failure mode is
# a confident wrong number (see the bounded gaps below), while a missing pattern's
# failure mode is None — which the verdict already reports as "no result" with the
# raw stdout preserved in results.json. Loud beats plausible.
TOK_S_PATTERNS = [
    r"Generation:\s*\d+\s*tokens?,\s*([\d.]+)\s*tokens-per-sec",  # mlx-lm, mlx-vlm --verbose
    # ollama --verbose prints BOTH "prompt eval rate" (prefill) and "eval rate"
    # (decode). Anchor to line start so the prefill rate can never be read as
    # decode throughput — it is roughly half, and silently wrong.
    r"(?m)^\s*eval rate:\s*([\d.]+)\s*tokens?/s",
    r"([\d.]+)\s*tok/s",                                          # dflash + mlx-dspark summary line
]

TOKENS_PATTERNS = [
    r"Generation:\s*(\d+)\s*tokens?",
    r"(?m)^\s*eval count:\s*(\d+)\s*token",   # not "prompt eval count"
    r"(\d+)\s*tokens?\s*\|",                  # dflash: "256 tokens | 16.4 tok/s | ..."
    r"(\d+)\s*tokens?\s*\u00b7",              # mlx-dspark: "33 tokens \u00b7 3.35s \u00b7 9.8 tok/s \u00b7 ..."
]

PEAK_MEM_PATTERNS = [
    r"Peak memory:\s*([\d.]+)\s*GB",
]

# Both engines report "share of drafted tokens accepted", on opposite sides of
# the word:
#   dflash   : "256 tokens | 16.4 tok/s | 53.5% acceptance | copyspec 1 blocks"
#   mlx-vlm  : "... (1.81 accepted drafts/round, 91.2% of drafted, avg draft 1.99)"
# Earlier forward-scanning variants (acceptance[^\d\n]*([\d.]+)) are deliberately
# gone: on the dflash line they skipped past the real figure and captured the 1
# from "copyspec 1 blocks", reporting 100% acceptance for a 53.5% run.
ACCEPT_PATTERNS = [
    r"([\d.]+)\s*%\s*acceptance",
    r"([\d.]+)\s*%\s*of\s*drafted",
]

# mlx-dspark reports accepted tokens PER ROUND and no percentage at all, so it
# lands in accept_len and leaves accept_pct None. Do not synthesise a percentage
# from it: accept_len includes the target's bonus token, so it is not the same
# quantity as "share of drafted tokens accepted" and the two must not be compared
# as if they were.
ACCEPT_LEN_PATTERNS = [
    r"([\d.]+)\s*accepted\s*tokens?/round",   # mlx-vlm
    r"accept\s+([\d.]+)\s*/\s*round",         # mlx-dspark
]


def parse_metrics(stdout: str):
    flags = []
    metrics = {
        "tok_s_reported": _pick(TOK_S_PATTERNS, stdout, "tok_s", flags),
        "tokens_reported": _pick(TOKENS_PATTERNS, stdout, "tokens", flags),
        "peak_gb": _pick(PEAK_MEM_PATTERNS, stdout, "peak_gb", flags),
        "accept_pct": _pick(ACCEPT_PATTERNS, stdout, "accept_pct", flags),
        "accept_len": _pick(ACCEPT_LEN_PATTERNS, stdout, "accept_len", flags),
    }
    metrics["parse_flags"] = flags
    return metrics


# --------------------------------------------------------------------------- #
# Arm definitions
# --------------------------------------------------------------------------- #

def build_cmd(arm: str, prompt: str, max_tokens: int):
    if arm == "ar":
        return resolve_cli("mlx_lm.generate", "mlx_lm", "generate") + [
            "--model", TARGET_4BIT,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temp", "0",
        ]

    # --verbose is REQUIRED, not cosmetic: mlx-vlm gates its
    # "Generation: N tokens, X tokens-per-sec" summary behind it
    # (generate/dispatch.py). Without it the arm runs fine and reports no
    # throughput at all, which reads as a failed arm.
    if arm == "mtp":
        return resolve_cli("mlx_vlm.generate", "mlx_vlm", "generate") + [
            "--model", TARGET_4BIT,
            "--draft-model", MTP_4BIT,
            "--draft-kind", "mtp",
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temperature", "0",
            "--verbose",
        ]

    if arm == "nvfp4-mtp":
        return resolve_cli("mlx_vlm.generate", "mlx_vlm", "generate") + [
            "--model", TARGET_NVFP4,
            "--draft-model", MTP_NVFP4,
            "--draft-kind", "mtp",
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temperature", "0",
            "--verbose",
        ]

    if arm == "dflash":
        # --draft is mandatory: the dflash registry has no Qwen3.8 entry and will
        # reject the target rather than auto-resolving a drafter.
        return [
            "dflash", "generate",
            "--model", TARGET_4BIT,
            "--draft", DFLASH_36,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--verify-mode", "adaptive",
        ]

    # --max-draft is deliberately NOT pinned. mlx-dspark measures this machine's
    # verify/drafter cost curves once and derives the cap from them (it picked 4
    # here, where upstream's M4 Pro derived 7 for the same drafter). Pinning a cap
    # would report a number this machine does not choose on its own.
    if arm == "dspark":
        return [
            "mlx-dspark", "generate",
            "--model", TARGET_4BIT,
            "--drafter", resolve_drafter(DSPARK_4BIT),
            "--mode", "dspark",
            "--prompt", prompt,
            "--max-new-tokens", str(max_tokens),
            "--temperature", "0",
            "--no-stream",
        ]

    if arm == "dspark-8bit":
        return [
            "mlx-dspark", "generate",
            "--model", TARGET_8BIT,
            "--drafter", resolve_drafter(DSPARK_8BIT),
            "--mode", "dspark",
            "--prompt", prompt,
            "--max-new-tokens", str(max_tokens),
            "--temperature", "0",
            "--no-stream",
        ]

    if arm == "ollama":
        return ["ollama", "run", OLLAMA_TAG, "--verbose", prompt]

    raise ValueError(f"unknown arm: {arm}")


ARM_META = {
    "ar":        ("lossless", "mlx-lm AR baseline", TARGET_4BIT, None),
    "mtp":       ("lossless", "official MTP drafter (mlx-vlm)", TARGET_4BIT, MTP_4BIT),
    "dflash":    ("lossless", "3.6 DFlash cross-applied (dflash-mlx)", TARGET_4BIT, DFLASH_36),
    "dspark":    ("lossless", "3.8-native DSpark drafter (mlx-dspark)", TARGET_4BIT, DSPARK_4BIT),
    "nvfp4-mtp": ("max-throughput", "NVFP4 target + NVFP4 MTP (mlx-vlm)", TARGET_NVFP4, MTP_NVFP4),
    "dspark-8bit": ("max-throughput", "8-bit target + RadixArk DSpark (mlx-dspark)", TARGET_8BIT, DSPARK_8BIT),
    "ollama":    ("max-throughput", "Ollama MLX build (modelopt mixed-precision)", OLLAMA_TAG, None),
}


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #

def correctness_blockers(arms):
    """Conditions under which an arm's tok/s is real but its LOSSLESS claim is not.

    Distinct from a preflight note. A blocked arm still runs and still produces a
    number; what it must not do is win the comparison, because the engine is not
    verifying against the target correctly. Speed bought by a broken verifier is
    not a speedup — it is a different model.
    """
    blockers = {}

    if "dflash" in arms:
        blocker = dflash_lossless_blocker()
        if blocker:
            blockers["dflash"] = blocker

    return blockers


def preflight(arms, max_tokens=None):
    """Return (notes, blockers). Notes are advisory; blockers void a lossless claim."""
    notes = []
    blockers = correctness_blockers(arms)

    if "dflash" in arms:
        if not shutil.which("dflash"):
            notes.append("MISSING: `dflash` not on PATH — the dflash arm will fail.")
        ver, _ = dflash_version()
        notes.append(f"dflash-mlx version: {ver or 'UNDETERMINED'}")

    for arm_name, why in blockers.items():
        notes.append(f"BLOCKER [{arm_name}]: {why} {DFLASH_UPGRADE_HINT}")

    if any(a in arms for a in ("mtp", "nvfp4-mtp")):
        try:
            import importlib.metadata as md
            ver = md.version("mlx-vlm")
            notes.append(f"mlx-vlm version: {ver}")
            m = re.match(r"(\d+)\.(\d+)\.(\d+)", ver)
            if m and tuple(int(x) for x in m.groups()) < (0, 6, 13):
                notes.append("WARNING: mlx-vlm < 0.6.13 — --draft-model/--draft-kind may be absent.")
        except Exception:
            notes.append("MISSING: mlx-vlm not importable — MTP arms will fail. `pip install -U mlx-vlm`")

    if any(a in arms for a in ("dspark", "dspark-8bit")):
        if not shutil.which("mlx-dspark"):
            notes.append("MISSING: `mlx-dspark` not on PATH — the dspark arms will fail. "
                         "`pip install mlx-dspark`")
        try:
            import importlib.metadata as md
            notes.append(f"mlx-dspark version: {md.version('mlx-dspark')}")
        except Exception:
            notes.append("mlx-dspark version: UNDETERMINED")

    if "ollama" in arms and not shutil.which("ollama"):
        notes.append("MISSING: `ollama` not on PATH — the ollama arm will fail.")

    if max_tokens is not None and max_tokens < 2000:
        notes.append(
            f"THERMAL: --max-tokens {max_tokens} decodes in seconds, so every arm here "
            "reports a fresh-GPU number. Apple Silicon throttles sustained 27B decode "
            "after ~2-3 min; steady-state tok/s will be lower, and speculative arms lose "
            "more than AR because the drafter competes for the same GPU. For a sustained "
            "figure use `dflash benchmark --sustained-minutes N` (requires v0.1.10)."
        )

    return notes, blockers


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

# Run-to-run spread above this makes the median untrustworthy as a point estimate.
SPREAD_WARN_PCT = 15.0

def run_arm(arm, prompt, max_tokens, repeat, cooldown, timeout):
    kind, label, target, draft = ARM_META[arm]
    cmd = build_cmd(arm, prompt, max_tokens)

    print(f"\n{'='*74}")
    print(f"ARM: {arm}  [{kind}]  — {label}")
    print(f"  target: {target}")
    print(f"  draft : {draft or '—'}")
    print(f"  cmd   : {' '.join(cmd[:6])} ...")
    print(f"{'='*74}")

    runs = []
    for i in range(repeat):
        print(f"\n  run {i+1}/{repeat} ...", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            wall = time.time() - t0
            stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
            metrics = parse_metrics(stdout)
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            wall = time.time() - t0
            stdout = f"<timeout after {timeout}s>"
            metrics = parse_metrics("")
            ok = False
        except FileNotFoundError as e:
            wall = 0.0
            stdout = f"<command not found: {e}>"
            metrics = parse_metrics("")
            ok = False

        tokens = metrics["tokens_reported"]
        wall_tok_s = (tokens / wall) if (tokens and wall > 0) else None

        runs.append({
            "run": i + 1,
            "ok": ok,
            "wall_s": round(wall, 3),
            "wall_tok_s": round(wall_tok_s, 2) if wall_tok_s else None,
            **{k: v for k, v in metrics.items()},
            "stdout_tail": stdout[-4000:],
        })

        status = "ok" if ok else "FAILED"
        rep = metrics["tok_s_reported"]
        print(f"    {status}  wall={wall:.1f}s  reported={rep if rep else '—'} tok/s"
              f"  accept={metrics['accept_pct'] if metrics['accept_pct'] else '—'}")

        if i < repeat - 1 and cooldown:
            print(f"    cooling down {cooldown}s ...", flush=True)
            time.sleep(cooldown)

    good = [r for r in runs if r["ok"] and r["tok_s_reported"]]
    median_tok_s = statistics.median([r["tok_s_reported"] for r in good]) if good else None
    accepts = [r["accept_pct"] for r in runs if r["accept_pct"] is not None]
    accept_lens = [r["accept_len"] for r in runs if r["accept_len"] is not None]
    peaks = [r["peak_gb"] for r in runs if r["peak_gb"] is not None]

    # A median over throttling runs is a number with no referent. Record the
    # spread, and flag a monotonic decline separately — that shape is thermal,
    # not noise, and it means the median overstates steady state.
    series = [r["tok_s_reported"] for r in good]
    spread_pct = None
    stability = []
    if len(series) >= 2 and median_tok_s:
        spread_pct = round((max(series) - min(series)) / median_tok_s * 100, 1)
        if spread_pct > SPREAD_WARN_PCT:
            stability.append(f"spread {spread_pct}% across {len(series)} runs "
                             f"(min {min(series)}, max {max(series)}) — median is not a stable estimate")
        if len(series) >= 3 and all(a > b for a, b in zip(series, series[1:])):
            stability.append(f"tok/s fell on every successive run ({series}) — thermal throttling "
                             "signature; steady state is below the median reported here")

    # An arm that ignored --max-tokens is not measuring the same workload as the
    # rest. `ollama run` has no token-budget flag, so it generates to EOS: a
    # multi-minute throttled figure compared against 256-token fresh-GPU bursts.
    counts = [r["tokens_reported"] for r in good if r["tokens_reported"]]
    if counts:
        med_tokens = statistics.median(counts)
        if not (0.5 * max_tokens <= med_tokens <= 1.5 * max_tokens):
            stability.append(
                f"generated ~{int(med_tokens)} tokens against --max-tokens {max_tokens} — this arm "
                "did not honour the token budget, so its tok/s covers a different (and here, much "
                "longer and therefore throttled) workload than the other arms")

    return {
        "arm": arm,
        "spread_pct": spread_pct,
        "stability": stability,
        "kind": kind,
        "label": label,
        "target": target,
        "draft": draft,
        "cmd": cmd,
        "all_ok": all(r["ok"] for r in runs),
        "median_tok_s": round(median_tok_s, 2) if median_tok_s else None,
        "median_accept_pct": round(statistics.median(accepts), 1) if accepts else None,
        "median_accept_len": round(statistics.median(accept_lens), 2) if accept_lens else None,
        "median_peak_gb": round(statistics.median(peaks), 2) if peaks else None,
        "runs": runs,
    }


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #

# Acceptance arrives in one of two units and they are NOT interconvertible:
#   share    — MTP / dflash: percent of drafted tokens accepted
#   perround — mlx-dspark: accepted tokens per round, INCLUDING the target's bonus
#              token, so 1.0 means nothing drafted survived verification
# Converting one into the other would manufacture a number the engine never
# reported, so each unit carries its own floor instead.
ACCEPT_SHARE_COLLAPSED_PCT = 30.0
ACCEPT_SHARE_HEALTHY_PCT = 60.0
ACCEPT_PERROUND_COLLAPSED = 1.05
ACCEPT_PERROUND_HEALTHY = 2.0    # >=1 drafted token accepted per round on average


def acceptance(r):
    """(display, unit, value) for an arm's acceptance, or ('unreported', None, None)."""
    pct = r.get("median_accept_pct")
    if pct is not None:
        return f"{pct:.0f}%", "share", pct
    per_round = r.get("median_accept_len")
    if per_round is not None:
        return f"{per_round:.2f}/rnd", "perround", per_round
    return "unreported", None, None


# An AR baseline below this is a broken run, not a slow machine. Every speedup is
# a ratio against it, so a near-zero baseline manufactures spectacular nonsense.
MIN_PLAUSIBLE_BASELINE_TOK_S = 0.5


def verdict(results, blockers=None):
    """Decision rule, applied only within the lossless class."""
    blockers = blockers or {}
    by_arm = {r["arm"]: r for r in results}
    baseline = by_arm.get("ar")
    lines = []

    if not baseline or not baseline["median_tok_s"]:
        return ["No usable AR baseline — cannot compute speedups. "
                "Check the `ar` arm's stdout_tail in the JSON artifact."]

    base = baseline["median_tok_s"]
    lines.append(f"AR baseline: {base:.2f} tok/s ({TARGET_4BIT})")
    for s in baseline.get("stability", []):
        lines.append(f"  ! baseline {s}")
    if not baseline.get("all_ok", True):
        lines.append("  WARNING: the AR baseline had failing runs — every ratio below is suspect.")
    if base < MIN_PLAUSIBLE_BASELINE_TOK_S:
        lines.append(f"  WARNING: baseline {base:.3f} tok/s is implausibly low. Speedups below are "
                     "meaningless until the `ar` arm is fixed — do not act on them.")
    lines.append("")

    # Derived from what ran, never a hardcoded list: a hardcoded tuple silently
    # drops any lossless arm added later (it dropped `dspark`) and invents a
    # "no result" line for arms the user never asked for.
    lossless = [r["arm"] for r in results if r["kind"] == "lossless" and r["arm"] != "ar"]
    if not lossless:
        lines.append("No lossless arm ran besides the baseline — nothing to compare.")

    for arm in lossless:
        r = by_arm.get(arm)
        if not r or not r["median_tok_s"]:
            lines.append(f"{arm:<10} no result — see artifact")
            continue
        speedup = r["median_tok_s"] / base
        acc_s, acc_unit, acc_val = acceptance(r)

        # Order matters: correctness gates precede any performance judgement.
        if arm in blockers:
            call = "INVALID — losslessness unverified (see BLOCKER above)"
        elif not r.get("all_ok", True):
            call = "DISCARD — not every run succeeded"
        elif base < MIN_PLAUSIBLE_BASELINE_TOK_S:
            call = "INCONCLUSIVE — baseline implausible"
        elif speedup <= 1.0:
            call = "DO NOT SHIP — slower than plain AR"
        elif acc_unit == "share" and acc_val < ACCEPT_SHARE_COLLAPSED_PCT:
            call = "DO NOT SHIP — acceptance collapsed"
        elif acc_unit == "perround" and acc_val <= ACCEPT_PERROUND_COLLAPSED:
            call = "DO NOT SHIP — acceptance collapsed"
        elif acc_unit is None:
            call = "INCONCLUSIVE — acceptance unreported, cannot confirm lossless"
        elif speedup > 1.8 and (
                (acc_unit == "share" and acc_val >= ACCEPT_SHARE_HEALTHY_PCT)
                or (acc_unit == "perround" and acc_val >= ACCEPT_PERROUND_HEALTHY)):
            call = "SHIP AS DEFAULT"
        else:
            call = "KEEP AVAILABLE, default to AR"

        lines.append(f"{arm:<10} {r['median_tok_s']:>7.2f} tok/s  "
                     f"{speedup:>5.2f}x  accept={acc_s:<11} {call}")
        for s in r.get("stability", []):
            lines.append(f"{'':<10}   ! {s}")

    lines.append("")
    maxtp = [r for r in results if r["kind"] == "max-throughput" and r["median_tok_s"]]
    if maxtp:
        lines.append("Max-throughput arms (different quantization — quality NOT held constant,")
        lines.append("so these speedups are not free and are not comparable to the above):")
        for r in maxtp:
            lines.append(f"  {r['arm']:<12} {r['median_tok_s']:>7.2f} tok/s  "
                         f"({r['median_tok_s']/base:.2f}x vs the 4-bit AR baseline)")
    return lines


def write_artifacts(outdir: Path, results, notes, config, blockers=None):
    outdir.mkdir(parents=True, exist_ok=True)
    blockers = blockers or {}

    parse_flags = sorted({f for r in results for run in r["runs"]
                          for f in run.get("parse_flags", [])})

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
        },
        "config": config,
        "validity": {
            "lossless_claim_valid": not blockers,
            "blocked_arms": sorted(blockers),
            "blockers": blockers,
            "parse_flags": parse_flags,
            "arms_with_failed_runs": sorted(r["arm"] for r in results if not r["all_ok"]),
            "unstable_arms": {r["arm"]: r["stability"] for r in results if r.get("stability")},
        },
        "preflight_notes": notes,
        "results": results,
        "verdict": verdict(results, blockers),
    }
    (outdir / "results.json").write_text(json.dumps(payload, indent=2))

    md = ["# Qwen3.8-27B multi-arm benchmark", ""]
    md.append(f"Generated: {payload['generated_at']}  ")
    md.append(f"Machine: {payload['machine']['platform']}  ")
    md.append(f"Prompt tokens requested: {config['max_tokens']}, "
              f"repeats: {config['repeat']}, cooldown: {config['cooldown']}s")
    md.append("")
    if notes:
        md.append("## Preflight")
        md.append("")
        for n in notes:
            md.append(f"- {n}")
        md.append("")
    md.append("## Results")
    md.append("")
    md.append("| Arm | Class | Target | Draft | tok/s (median) | Spread | Accept | Peak GB |")
    md.append("| :-- | :-- | :-- | :-- | --: | --: | --: | --: |")
    for r in results:
        spread = r.get("spread_pct")
        spread_s = f"{spread}%" + (" ⚠" if spread and spread > SPREAD_WARN_PCT else "") if spread is not None else "—"
        md.append(
            f"| `{r['arm']}` | {r['kind']} | `{r['target']}` | "
            f"{'`'+r['draft']+'`' if r['draft'] else '—'} | "
            f"{r['median_tok_s'] if r['median_tok_s'] else 'n/a'} | "
            f"{spread_s} | "
            f"{acceptance(r)[0].replace('unreported', '—')} | "
            f"{r['median_peak_gb'] if r['median_peak_gb'] else '—'} |"
        )
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append("```")
    md.extend(payload["verdict"])
    md.append("```")
    md.append("")

    if blockers:
        md.append("## ⚠ Losslessness NOT established")
        md.append("")
        md.append("The lossless framing below does **not** hold for this run. These arms produced "
                  "a throughput number, but nothing here shows their output matches the target:")
        md.append("")
        for arm_name, why in sorted(blockers.items()):
            md.append(f"- **`{arm_name}`** — {why}")
        md.append("")
        md.append("Fix the blocker and re-run before using any of it to pick a default.")
        md.append("")
    else:
        md.append("Lossless arms all verify against the same 4-bit target, so their outputs are "
                  "interchangeable and only speed differs.")
        md.append("")

    md.append("Max-throughput arms use a different quantization: treat their numbers as a "
              "separate question, not a free win.")

    if parse_flags:
        md.append("")
        md.append("## Parse warnings")
        md.append("")
        md.append("Ambiguous or out-of-range values were seen while reading engine output. "
                  "Check `results.json` -> `stdout_tail` before trusting the affected metric:")
        md.append("")
        for f in parse_flags:
            md.append(f"- `{f}`")

    (outdir / "REPORT.md").write_text("\n".join(md) + "\n")

    return outdir / "REPORT.md"


def main():
    p = argparse.ArgumentParser(
        description="Benchmark Qwen3.8-27B decoding engines on Apple Silicon.")
    p.add_argument("--arms", type=str, default="lossless",
                   help="Comma-separated arm names, or 'lossless' (default), 'max-throughput', 'all'. "
                        f"Available: {', '.join(ARM_META)}")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--cooldown", type=int, default=60,
                   help="Seconds between runs. Apple Silicon throttles sustained 27B decode "
                        "after ~2-3 min; too short a cooldown flatters whichever arm runs first.")
    p.add_argument("--timeout", type=int, default=1800, help="Per-run timeout in seconds.")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero instead of running when a correctness blocker is "
                        "present (e.g. a dflash-mlx build whose verifier is known wrong). "
                        "Use in CI so a broken comparison cannot be mistaken for a result.")
    args = p.parse_args()

    if args.arms == "lossless":
        arms = list(LOSSLESS_ARMS)
    elif args.arms == "max-throughput":
        arms = list(MAXTP_ARMS)
    elif args.arms == "all":
        arms = LOSSLESS_ARMS + MAXTP_ARMS
    else:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    unknown = [a for a in arms if a not in ARM_META]
    if unknown:
        print(f"Unknown arm(s): {unknown}. Available: {list(ARM_META)}", file=sys.stderr)
        sys.exit(2)

    # The AR baseline anchors every speedup; pull it in whenever a lossless arm runs.
    if any(ARM_META[a][0] == "lossless" for a in arms) and "ar" not in arms:
        arms.insert(0, "ar")

    print(f"\nArms: {arms}")
    notes, blockers = preflight(arms, args.max_tokens)
    if notes:
        print("\nPreflight:")
        for n in notes:
            print(f"  - {n}")

    if blockers and args.strict:
        print("\nRefusing to run under --strict: the lossless comparison is already "
              "invalid before any tokens are generated.", file=sys.stderr)
        for arm_name, why in sorted(blockers.items()):
            print(f"  BLOCKER [{arm_name}]: {why}", file=sys.stderr)
        sys.exit(3)

    results = [run_arm(a, args.prompt, args.max_tokens, args.repeat, args.cooldown, args.timeout)
               for a in arms]

    outdir = Path(args.outdir) if args.outdir else Path(
        ".artifacts/dflash/qwen38_bench") / datetime.now().strftime("%Y%m%d-%H%M%S")
    report = write_artifacts(outdir, results, notes, {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "repeat": args.repeat,
        "cooldown": args.cooldown,
        "arms": arms,
    }, blockers)

    print(f"\n{'='*74}")
    print("VERDICT")
    print(f"{'='*74}")
    for line in verdict(results, blockers):
        print(line)
    print(f"\nArtifacts: {outdir}")
    print(f"Report:    {report}")

    if blockers:
        print(f"\nLosslessness NOT established for: {', '.join(sorted(blockers))}. "
              "Throughput above is real; the lossless claim is not.", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
