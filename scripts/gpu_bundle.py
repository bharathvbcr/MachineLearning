#!/usr/bin/env python3
"""GPU bundle: the three outstanding rented-GPU suites, as one resumable runner.

Covers the experiments the paper names but has not run:

  E1  The parametrization arm specified in PAPER section 8.4. Four sub-arms:
        e1_mup          muP cells of the 2x2 (the SP cells are suite 24, already run)
        e1_proxy        proxy-width LR sweep at mup_base_width, to tune before transfer
        e1_perlayer_sp  per-layer standard-parametrization prescription
        e1_embed_lr     embedding-LR-only ablation
  E2  Suite 26 never reran attention/minGRU at 50M matched batch 32; its board's top
      and bottom rows are suite 22's sample, capping the combined ranking at
      Medium-High. Ten jobs close it.
  D10 Suite 20's horizon claim is withdrawn: run128m_20k is eight resumed segments
      with a broken token counter, and its LR moves with its horizon. A matched PAIR
      (10k and 20k at the SAME learning rate, uninterrupted) is what the claim needs.
      Note this is NOT a reproduction of suite 20 -- that would need the original
      RTX 3070 Ti. It answers the science on new hardware; report it that way.

Cadence comes from ``crossover_replicate.scale_to_token_budget`` so eval markers line
up with suites 22-26 and loss-vs-tokens curves remain comparable.

The ledger is rewritten after EVERY job. A predecessor experiment lost three completed
runs to a ledger written once at launch (gap D8); that is not repeated.

Usage:
  python3 scripts/gpu_bundle.py --plan                 # print the matrix and exit
  python3 scripts/gpu_bundle.py --dry-run              # print every job config
  python3 scripts/gpu_bundle.py --smoke                # 40-step check of one job per suite
  python3 scripts/gpu_bundle.py --only e1_mup          # one sub-arm
  python3 scripts/gpu_bundle.py --workers 8            # full run, 8 concurrent
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
sys.path.insert(0, str(ROOT))

from nanolab.crossover_replicate import (  # noqa: E402
    SEEDS, TOKEN_BUDGET, LOCKED20_TOKEN_BUDGET, scale_to_token_budget,
)

OUT_ROOT = ROOT / "nanolab/out/gpu_bundle"
LEDGER = OUT_ROOT / "ledger.json"

# suite-24 recipe: batch 32, 20M stop, 50M cosine horizon, eval_iters 20
S24 = dict(batch=32, budget=LOCKED20_TOKEN_BUDGET, horizon=TOKEN_BUDGET,
           eval_iters=20)
ARMS = ("attention", "mingru")
BASE_WIDTH = 256          # cfg.mup_base_width; the proxy sweep runs here
TARGET_WIDTH = 768        # the 12L/768d target every suite uses
PROXY_LRS = (1.5e-4, 3e-4, 6e-4, 1.2e-3, 2.4e-3)


def _cadence(batch: int, budget: int, horizon: int | None) -> dict:
    return scale_to_token_budget(batch_size=batch, block_size=512, grad_accum=1,
                                 token_budget=budget, lr_horizon_tokens=horizon)


def _job(suite: str, arm: str, seed: int, extra: dict, *,
         batch: int, budget: int, horizon: int | None, eval_iters: int,
         d_model: int = TARGET_WIDTH, tag: str = "") -> dict:
    c = _cadence(batch, budget, horizon)
    jid = f"{suite}_{arm}{('_' + tag) if tag else ''}_s{seed}"
    return {
        "id": jid, "suite": suite, "arm": arm, "seed": seed, "tag": tag,
        "overrides": dict(
            run_name=jid, mixer=arm, layer_mixers="", seed=seed,
            out_dir=str(OUT_ROOT), d_model=d_model,
            batch_size=c["batch_size"], grad_accum=c["grad_accum"],
            block_size=c["block_size"], max_steps=c["max_steps"],
            lr_max_steps=c["lr_max_steps"], warmup_steps=c["warmup_steps"],
            eval_interval=c["eval_interval"], ckpt_interval=c["ckpt_interval"],
            log_interval=c["log_interval"], eval_train=False,
            eval_iters=eval_iters, compile=False, mem_fraction=0.0, **extra),
    }


def build_matrix() -> list[dict]:
    jobs: list[dict] = []

    # --- E1a: muP cells of the 2x2. SP cells are suite 24 and are NOT rerun. ---
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_job("e1_mup", arm, seed,
                             dict(mup=True, mup_base_width=BASE_WIDTH), **S24))

    # --- E1b: proxy-width LR sweep. Tune at BASE_WIDTH, transfer to TARGET_WIDTH.
    # One seed: this locates a peak, it does not rank arms. Cheap by construction --
    # a 256-wide model is ~1/9 the FLOPs of the 768-wide target.
    for arm in ARMS:
        for lr in PROXY_LRS:
            jobs.append(_job("e1_proxy", arm, SEEDS[0],
                             dict(mup=True, mup_base_width=BASE_WIDTH, lr=lr),
                             d_model=BASE_WIDTH, tag=f"lr{lr:g}", **S24))

    # --- E1c: per-layer SP (Everett et al.). See the caveats in optim.py: the
    # prescription is stated for pure Adam and our hybrid sends hidden matrices to
    # Muon, and tied embeddings prevent a separate readout rate. This arm is an
    # APPROXIMATION of their prescription and must be reported as one.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_job("e1_perlayer_sp", arm, seed,
                             dict(per_layer_sp=True, mup_base_width=BASE_WIDTH), **S24))

    # --- E1d: embedding-LR-only. Raise ONLY the embedding LR by the width ratio.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_job("e1_embed_lr", arm, seed,
                             dict(embed_lr_mult=TARGET_WIDTH / BASE_WIDTH,
                                  mup_base_width=BASE_WIDTH), **S24))

    # --- E2: suite 26's missing attention/minGRU cells at 50M, matched batch 32. ---
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_job("e2_matched32_50m", arm, seed, {},
                             batch=32, budget=TOKEN_BUDGET,
                             horizon=None, eval_iters=20))

    # --- D10: matched horizon pair, attention only, ONE seed each, uninterrupted.
    # Both at the same LR so horizon is the only variable -- the confound that made
    # suite 20 unusable was LR moving with horizon. ctx1024/bs32 matches suite 20.
    for steps, budget in (("10k", 327_680_000), ("20k", 655_360_000)):
        c = scale_to_token_budget(batch_size=32, block_size=1024, grad_accum=1,
                                  token_budget=budget, lr_horizon_tokens=budget)
        jid = f"d10_horizon_{steps}"
        jobs.append({
            "id": jid, "suite": "d10_horizon", "arm": "attention",
            "seed": SEEDS[0], "tag": steps,
            "overrides": dict(
                run_name=jid, mixer="attention", layer_mixers="", seed=SEEDS[0],
                out_dir=str(OUT_ROOT), batch_size=32, grad_accum=1, block_size=1024,
                max_steps=c["max_steps"], lr_max_steps=c["lr_max_steps"],
                warmup_steps=c["warmup_steps"], eval_interval=c["eval_interval"],
                ckpt_interval=c["ckpt_interval"], log_interval=c["log_interval"],
                eval_train=False, eval_iters=20, compile=False, mem_fraction=0.0),
        })
    return jobs


def read_result(out_dir: Path) -> float | None:
    """best_val from the run's `done` record, or None if it never finished.

    None rather than a sentinel: a job that could not be measured must never read
    as a measured job.
    """
    m = out_dir / "metrics.jsonl"
    if not m.exists():
        return None
    val = None
    for line in m.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "done":
            val = row.get("best_val")
    return val


def write_ledger(records: list[dict], started: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "id": "gpu-bundle",
        "started_at": started,
        "suites": {
            "e1_mup": "PAPER 8.4 muP cells of the 2x2 (SP cells = suite 24, not rerun)",
            "e1_proxy": "proxy-width LR sweep at mup_base_width, 1 seed, locates a peak only",
            "e1_perlayer_sp": "per-layer SP (Everett et al.) -- APPROXIMATION, see optim.py caveats",
            "e1_embed_lr": "embedding-LR-only ablation (Kalra & Barkeshli)",
            "e2_matched32_50m": "suite 26's missing attention/minGRU cells at 50M / batch 32",
            "d10_horizon": "matched 10k vs 20k at ONE learning rate; NOT a suite-20 reproduction",
        },
        "jobs_total": len(records),
        "jobs_done": sum(1 for r in records if r["status"] == "done"),
        "jobs": records,
        "note": "Rewritten after every job (see gap D8).",
    }
    tmp = LEDGER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, LEDGER)


def run_one(job: dict, smoke: bool) -> tuple[int, float]:
    ov = dict(job["overrides"])
    if smoke:
        ov.update(max_steps=40, lr_max_steps=40, eval_interval=20, eval_iters=4,
                  log_interval=5, warmup_steps=5, ckpt_interval=40, batch_size=8)
    code = (
        "import json,sys;sys.path.insert(0,%r);"
        "from nanolab.config import build_config;from nanolab.train import train;"
        "train(build_config('crossover50m', json.loads(%r)))"
        % (str(ROOT), json.dumps(ov))
    )
    out_dir = OUT_ROOT / ov["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (out_dir / "run.log").open("w", encoding="utf-8") as fh:
        p = subprocess.run([sys.executable, "-c", code], stdout=fh,
                           stderr=subprocess.STDOUT, cwd=str(ROOT))
    return p.returncode, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None, help="run one suite id")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    jobs = build_matrix()
    if args.only:
        jobs = [j for j in jobs if j["suite"] == args.only]
        if not jobs:
            sys.exit(f"no jobs for suite {args.only!r}")
    if args.smoke:
        seen, picked = set(), []
        for j in jobs:
            if j["suite"] not in seen:
                seen.add(j["suite"]); picked.append(j)
        jobs = picked

    if args.plan:
        from collections import Counter
        c = Counter(j["suite"] for j in jobs)
        print(f"{'suite':<20}{'jobs':>6}")
        for k, v in c.items():
            print(f"  {k:<18}{v:>6}")
        print(f"  {'TOTAL':<18}{len(jobs):>6}")
        return 0
    if args.dry_run:
        for j in jobs:
            print(f"{j['id']}: {json.dumps(j['overrides'], sort_keys=True)}")
        return 0

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    records = []
    for j in jobs:
        existing = read_result(OUT_ROOT / j["id"])
        records.append({"id": j["id"], "suite": j["suite"], "arm": j["arm"],
                        "seed": j["seed"], "tag": j["tag"],
                        "status": "done" if existing is not None else "pending",
                        "best_val": existing})
    write_ledger(records, started)

    for i, (j, rec) in enumerate(zip(jobs, records), 1):
        if rec["status"] == "done":
            print(f"[{i}/{len(jobs)}] skip {j['id']} ({rec['best_val']:.4f})", flush=True)
            continue
        rec["status"] = "running"; write_ledger(records, started)
        print(f"[{i}/{len(jobs)}] {j['id']} ...", flush=True)
        code, elapsed = run_one(j, args.smoke)
        val = read_result(OUT_ROOT / j["id"])
        rec.update(returncode=code, elapsed_s=round(elapsed, 1), best_val=val,
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        # Both conditions required: a non-zero exit OR a missing metric is a failure.
        rec["status"] = "done" if (code == 0 and val is not None) else "failed"
        if rec["status"] == "failed":
            rec["failure_reason"] = f"returncode={code}, best_val={'absent' if val is None else val}"
        write_ledger(records, started)
        print(f"    -> {rec['status']} {val if val is not None else ''} "
              f"({elapsed/60:.1f} min)", flush=True)

    failed = [r for r in records if r["status"] != "done"]
    print(f"\nledger: {LEDGER.relative_to(ROOT)}")
    if failed:
        print(f"{len(failed)} did not complete: " + ", ".join(r["id"] for r in failed),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
