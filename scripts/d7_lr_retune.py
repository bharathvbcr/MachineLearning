#!/usr/bin/env python3
"""D7: re-tune the funnel finalists' learning rates at the scale that selected them.

Background.  The optimizer funnel picked ``muon_polar_adamw`` at matrix LR 0.05 over
``normuon_adamw`` at 0.1 by 0.031226 BPB.  Both LRs were tuned at ``arch02-16m`` and
neither was re-tuned at 128M.  This script runs both finalists across a matched LR grid
at the *exact* protocol that made the selection (``exact_128m_1000``: 128M, 1000 steps),
so the ordering can be read at matched tuning quality.

Result of the first 24 jobs (2026-08-23, seeds 42 + 2026): ``normuon_adamw`` leads at
all five matched LRs on both seeds, and at each candidate's best tested LR by 0.016317
BPB.  The recorded selection is retired.  See ``research/d7-lr-retune.json``.  A 500-step
spot-check had suggested Polar's optimum was near 0.035 with a 1.95x penalty; both
figures were artifacts of the short horizon and a truncated grid, and are withdrawn.

Round 3 adds seed 1337 and lr 0.008 -- see the GRID comment for why neither a champion
nor an optimum can be declared without them.

Protocol is taken from ``nanolab.native_funnel.job_argv`` for the ``exact_128m_*``
stages so these jobs are directly comparable to the recorded funnel results.  Both
candidates are in ``MUON_FAMILY``, so only ``--matrix-lr`` varies; embedding and
scalar LRs are left at their preset defaults exactly as in the original stage.

The ledger is rewritten after **every** job.  The predecessor of this experiment lost
three completed runs because its ledger was written once, at launch, and never
updated again (tracked as D8); that is not repeated here.

Usage:
    python3 scripts/d7_lr_retune.py --dry-run     # print every command, run nothing
    python3 scripts/d7_lr_retune.py               # run sequentially, resumable
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METAL_DIR = ROOT / "Rust_MLKit/arch_02_value_resid/metal-native"
TRAIN_BIN = METAL_DIR / "target/release/train"
STUDY = ROOT / "research/optimizer-study.json"
DATA = ROOT / "parameter-golf/data/datasets/fineweb10B_sp1024"
TOKEN_BYTES = ROOT / "Rust_MLKit/arch_02_value_resid/burn-port/token_bytes.json"

OUT_ROOT = ROOT / "out/funnel/d7_lr_retune_1000"
LEDGER = OUT_ROOT / "ledger.json"

STEPS = 1000
SEEDS = (42, 2026)          # the two seeds of the original exact_128m_1000 stage
SEEDS_N3 = (42, 2026, 1337)  # third seed added 2026-08-23, see below
# Each grid brackets that candidate's own expected optimum.  Polar's 500-step
# spot-check minimum sat at 0.035; NorMuon's selected value was 0.1 and has never
# been swept at 128M, so its grid reaches down toward Polar's optimum.
#
# Grid extended twice on 2026-08-22, both times because a candidate's minimum landed
# on the left edge of its grid.  A boundary minimum measures "lower is better within
# this range", not an optimum, and comparing one bracketed candidate against one
# boundary-limited candidate would repeat the funnel's own unequal-tuning error.
#
#   Extension 1 (after job 3).  The 500-step spot-check's minimum at 0.035 does not
#   reproduce at 1000 steps.  Polar, seed 42: 0.025 -> 2.144780, 0.035 -> 2.153202,
#   0.05 -> 2.165343 -- monotone, minimum at the left edge.  Added 0.0125 and 0.018.
#
#   Extension 2 (after job 5).  NorMuon is monotone the same way: 0.05 -> 2.148214,
#   0.07 -> 2.165703, 0.1 -> 2.202106 (recorded).  Its minimum is also on its left
#   edge, so its grid was matched to Polar's over the range where both optima lie.
#
# Both candidates' 16M-tuned LRs are therefore at least one grid step too high at
# 128M -- Polar's selected 0.05 by >= 0.0206 BPB, NorMuon's selected 0.1 by >= 0.0539.
# The funnel's LR ladder was mis-scaled for the whole 128M stage, not just its winner.
# The optimum falling as scale and horizon grow is the direction Bjorck et al.
# (arXiv:2405.18392) predict, and is itself a recipe-dependence worth recording.
#
# Polar is not swept at 0.07 / 0.1: it is already rising steeply by 0.05, so those
# points would cost four hours to confirm something both curves already show.  The
# grids are therefore asymmetric only on the uninformative side; they are identical
# across {0.0125 .. 0.05}, which is where both minima sit.
# GRID maps candidate -> {lr: seeds}.  It was a flat tuple of LRs until 2026-08-23;
# per-LR seed lists were needed once a third seed was added only where it changes a
# conclusion, rather than everywhere (lr 0.07/0.1 are far from either optimum and a
# third seed there buys nothing).
#
#   Round 3 (2026-08-23).  The 24-job matched grid established the SIGN -- normuon_adamw
#   leads at all five matched LRs, 2-of-2 -- but two things block locking a champion:
#     (a) n=2 supports a sign, not a magnitude; the paper's own D1 fix sets n>=3 as the
#         threshold for an informative interval, so a third seed is required.
#     (b) neither optimum is bracketed.  normuon_adamw is monotone to its left edge
#         0.0125; muon_polar_adamw ties between 0.0125 and 0.018 on seed 2026 but is
#         still falling on seed 42.  lr 0.008 is added to both to test for a bracket.
#   If 0.008 wins again for either candidate, the grid needs another step down and the
#   optimum is still not located -- that result would be reported, not papered over.
#   Round 4 (2026-08-23, after job 26).  normuon_adamw is now BRACKETED: seed 42 gives
#   0.008 -> 2.115936, 0.0125 -> 2.113670, 0.018 -> 2.124600, an interior minimum.
#   muon_polar_adamw is not: it is still falling at 0.008 (2.118333, better than its
#   0.0125 by 0.0138).  That collapsed the best-tested gap from 0.018414 to 0.004663,
#   so which optimizer is actually better now turns on where Polar's optimum sits.
#   0.005 and 0.0035 are added to find it.  Comparing a bracketed candidate against an
#   unbracketed one is the same unequal-tuning error the funnel made.
GRID = {
    "muon_polar_adamw": {
        0.0035: SEEDS_N3, 0.005: SEEDS_N3,
        0.008: SEEDS_N3, 0.0125: SEEDS_N3, 0.018: SEEDS_N3,
        0.025: SEEDS_N3, 0.035: SEEDS_N3, 0.05: SEEDS_N3,
    },
    #   Round 5 (2026-08-23, after job 42).  muon_polar_adamw's optimum turned out to
    #   be 0.005 -- an interior minimum, but BELOW normuon_adamw's grid floor of 0.008.
    #   Round 4 had given the low points to Polar only, on the strength of a seed-42
    #   reading that job 28 later overturned. Comparing a candidate swept over
    #   {0.0035..0.05} against one swept over {0.008..0.1} is unequal tuning depth --
    #   the exact error this experiment exists to document. The grids are made
    #   symmetric here rather than arguing that the difference is probably small.
    "normuon_adamw": {
        0.0035: SEEDS_N3, 0.005: SEEDS_N3,
        0.008: SEEDS_N3, 0.0125: SEEDS_N3, 0.018: SEEDS_N3,
        0.025: SEEDS_N3, 0.035: SEEDS_N3, 0.05: SEEDS_N3,
        0.07: SEEDS, 0.1: SEEDS,
    },
}


def job_id(candidate: str, lr: float, seed: int) -> str:
    return f"{candidate}_lr{lr:g}".replace(".", "p") + f"_seed{seed}_128m"


def jobs() -> list[dict]:
    out = []
    # Seed-major ordering: a full grid at seed 42 lands before any seed-2026 job, so
    # an interrupted run still yields one complete, readable sweep rather than a
    # ragged half of each.
    for seed in SEEDS_N3:
        for candidate, lrs in GRID.items():
            for lr, seeds in lrs.items():
                if seed not in seeds:
                    continue
                out.append({
                    "candidate": candidate,
                    "lr": lr,
                    "seed": seed,
                    "steps": STEPS,
                    "id": job_id(candidate, lr, seed),
                })
    return out


def argv_for(job: dict) -> list[str]:
    out = OUT_ROOT / job["id"]
    return [
        str(TRAIN_BIN),
        "--preset", "arch02-128m",
        "--optimizer", job["candidate"],
        "--total-steps", str(job["steps"]),
        "--seed", str(job["seed"]),
        "--eval-every", str(job["steps"]),
        "--log-every", "50",
        "--research-manifest", str(STUDY),
        "--data-dir", str(DATA),
        "--token-bytes", str(TOKEN_BYTES),
        "--no-final-weight-save",
        "--out", str(out),
        "--matrix-lr", f"{job['lr']:.9g}",
    ]


def read_bpb(out_dir: Path) -> float | None:
    """Final EMA sliding BPB from the last metrics row that carries one.

    Returns None rather than a sentinel when the run produced no such row, so an
    unfinished job can never be mistaken for a measured one.
    """
    metrics = out_dir / "metrics.jsonl"
    if not metrics.exists():
        return None
    value = None
    for line in metrics.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("final_ema_sliding_bpb") is not None:
            value = float(row["final_ema_sliding_bpb"])
    return value


def write_ledger(records: list[dict], started: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    done = [r for r in records if r["status"] == "done"]
    payload = {
        "schema_version": 1,
        "id": "d7-lr-retune-1000",
        "purpose": (
            "Re-tune muon_polar_adamw and normuon_adamw at the exact_128m_1000 protocol "
            "that selected the champion, so the ordering can be read at matched tuning "
            "quality. Closes D7."
        ),
        "protocol": {
            "preset": "arch02-128m",
            "steps": STEPS,
            "seeds": list(SEEDS),
            "eval_every": STEPS,
            "metric": "final_ema_sliding_bpb",
            "varies": "matrix_lr only (both candidates are MUON_FAMILY)",
            "matched_to": "native-optimizer-funnel.json stage exact_128m_1000",
        },
        "grid": {k: {str(lr): list(sd) for lr, sd in v.items()} for k, v in GRID.items()},
        "prior": {
            "selected": {"candidate": "muon_polar_adamw", "lr": 0.05,
                         "mean_validation_bpb": 2.1699185},
            "runner_up": {"candidate": "normuon_adamw", "lr": 0.1,
                          "mean_validation_bpb": 2.201145},
            "selection_margin_bpb": 0.031226,
            "spot_check_500step_optimum_lr": 0.035,
            "spot_check_champion_penalty_bpb": 0.060996,
        },
        "started_at": started,
        "jobs_total": len(records),
        "jobs_done": len(done),
        "jobs": records,
        "note": (
            "This ledger is rewritten after every job. Its predecessor "
            "(out/funnel/polar_exact_lr_spot/) was written once at launch and lost three "
            "completed runs; see D8 in docs/ISSUES_AND_GAPS_2026-08-22.md."
        ),
    }
    tmp = LEDGER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, LEDGER)


def summarize(records: list[dict]) -> None:
    print("\n=== D7 LR re-tune: means over completed seeds ===")
    by = {}
    for r in records:
        if r["status"] == "done" and r.get("validation_bpb") is not None:
            by.setdefault((r["candidate"], r["lr"]), []).append(r["validation_bpb"])
    if not by:
        print("  (no completed jobs yet)")
        return
    for (cand, lr), vals in sorted(by.items()):
        mean = sum(vals) / len(vals)
        seeds = ", ".join(f"{v:.6f}" for v in vals)
        flag = "  <- previously selected" if (cand, lr) in (
            ("muon_polar_adamw", 0.05), ("normuon_adamw", 0.1)) else ""
        print(f"  {cand:<20} lr {lr:<6g} n={len(vals)}  mean {mean:.6f}  [{seeds}]{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = jobs()

    if args.dry_run:
        for j in todo:
            print(" ".join(argv_for(j)))
        print(f"\n{len(todo)} jobs", file=sys.stderr)
        return 0

    for path, label in ((TRAIN_BIN, "train binary"), (STUDY, "study manifest"),
                        (TOKEN_BYTES, "token bytes")):
        if not path.exists():
            print(f"missing {label}: {path}", file=sys.stderr)
            return 1
    if not DATA.is_dir():
        print(f"missing data dir: {DATA}", file=sys.stderr)
        return 1

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Carry forward whatever a previous pass recorded.  Rebuilding purely from the run
    # directories recovers status and BPB but silently drops elapsed_s / finished_at,
    # so a resumed pass would erase timings the first pass had measured.
    prior = {}
    if LEDGER.exists():
        try:
            for rec in json.loads(LEDGER.read_text(encoding="utf-8")).get("jobs", []):
                if isinstance(rec, dict) and rec.get("id"):
                    prior[rec["id"]] = rec
        except (json.JSONDecodeError, OSError):
            prior = {}

    records = []
    for j in todo:
        out_dir = OUT_ROOT / j["id"]
        existing = read_bpb(out_dir)
        rec = dict(prior.get(j["id"], {}))
        rec.update({
            "id": j["id"], "candidate": j["candidate"], "lr": j["lr"], "seed": j["seed"],
            "steps": j["steps"], "output": str(out_dir.relative_to(ROOT)),
            "status": "done" if existing is not None else "pending",
            "validation_bpb": existing,
        })
        # A job that never ran in this or any prior pass must not inherit a stale timing.
        if existing is None:
            rec.pop("elapsed_s", None)
            rec.pop("finished_at", None)
        records.append(rec)
    write_ledger(records, started)

    for i, (j, rec) in enumerate(zip(todo, records), 1):
        if rec["status"] == "done":
            print(f"[{i}/{len(todo)}] skip {j['id']} (already {rec['validation_bpb']:.6f})")
            continue
        out_dir = OUT_ROOT / j["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        rec["status"] = "running"
        rec["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_ledger(records, started)

        print(f"[{i}/{len(todo)}] {j['id']} ...", flush=True)
        t0 = time.time()
        log = out_dir / "run.log"
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(argv_for(j), stdout=fh, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0

        bpb = read_bpb(out_dir)
        rec["elapsed_s"] = round(elapsed, 1)
        rec["returncode"] = proc.returncode
        rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["validation_bpb"] = bpb
        # A non-zero exit or a missing metric is a failure even if the other looks fine:
        # a job that could not be measured must never read as a measured job.
        rec["status"] = "done" if (proc.returncode == 0 and bpb is not None) else "failed"
        if rec["status"] == "failed":
            rec["failure_reason"] = (
                f"returncode={proc.returncode}, final_ema_sliding_bpb="
                f"{'absent' if bpb is None else bpb}; see {log.relative_to(ROOT)}"
            )
        write_ledger(records, started)
        print(f"    -> {rec['status']} {bpb if bpb is not None else ''} "
              f"({elapsed/60:.1f} min)", flush=True)

    summarize(records)
    failed = [r for r in records if r["status"] != "done"]
    print(f"\nledger: {LEDGER.relative_to(ROOT)}")
    if failed:
        print(f"{len(failed)} job(s) did not complete: "
              + ", ".join(r["id"] for r in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
