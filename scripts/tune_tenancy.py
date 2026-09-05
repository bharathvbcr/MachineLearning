"""Aggregate GPU throughput as a function of tenancy (jobs per GPU).

`workers` is a recipe field (crossover_replicate.current_recipe), and every board
in this repo has picked it by hand: 2 for the ladder, 3 for E18/E19/E20, 4 for
the recall grid. Nothing ever measured the knee. At the board shape (batch 32,
ctx 512) a 124M model leaves a GH200 ~86% idle (nanolab.sweep_gpu arm), so the
question is not academic: if aggregate throughput is still climbing at 4, every
50M board in the backlog is being run at a fraction of the box's rate.

The measurement is N concurrent processes running the SAME arm through the same
`job_config` the suite runner uses, all timed over ONE shared wall-clock window
so start-up skew cannot inflate a high-tenancy row:

    parent: t_start = now + warmup_grace ; t_stop = t_start + seconds
    worker: build, warm up, spin until t_start, count steps until t_stop

Each worker reports steps completed and its own peak allocation; the parent sums
the token rate and reports the per-GPU total plus the VRAM headroom that sets the
ceiling. Synthetic in-GPU batches (bench_gpu.synth_batch), so the row is GPU
contention, not the dataloader.

    python scripts/tune_tenancy.py --arm attention --tenancies 1,2,3,4,6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_worker(args) -> None:
    sys.path.insert(0, str(REPO))
    import torch
    from nanolab.bench_gpu import synth_batch
    from nanolab.crossover_replicate import ARMS, job_config
    from nanolab.model import build_model
    from nanolab.optim import build_optimizers

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    arm = next((a for a in ARMS if a.name == args.arm), None)
    if arm is None:
        raise SystemExit(f"unknown arm {args.arm!r}")
    os.environ["CROSSOVER_BATCH"] = str(args.batch_size)
    os.environ["CROSSOVER_BLOCK"] = str(args.block_size)
    os.environ["CROSSOVER_TOKEN_BUDGET"] = str(args.token_budget)
    job = {"id": f"ten_{args.arm}", "arm": arm.name, "mixer": arm.mixer,
           "layer_mixers": arm.layer_mixers, "seed": 1337}
    cfg = job_config(job, REPO / "nanolab/out/_tune")

    model = build_model(cfg).to("cuda")
    optimizers = build_optimizers(model, cfg)
    x, y = synth_batch(cfg, "cuda")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    def step():
        with autocast:
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    while time.time() < args.start_at:      # shared window: all workers align here
        time.sleep(0.002)
    t0 = time.time()
    steps = 0
    while time.time() < args.stop_at:
        step()
        steps += 1
    torch.cuda.synchronize()
    wall = time.time() - t0

    print("RESULT " + json.dumps({
        "steps": steps, "wall": wall,
        "tok_s": cfg.batch_size * cfg.block_size * steps / wall,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }), flush=True)


def run_parent(args) -> None:
    out: dict = {"arm": args.arm, "batch": args.batch_size,
                 "block": args.block_size, "seconds": args.seconds, "rows": {}}
    for n in [int(t) for t in args.tenancies.split(",")]:
        # Build+warm-up is serialised by the driver, not the GPU, so give it room
        # to finish on every worker before the shared window opens.
        start_at = time.time() + args.grace + args.build_grace * n
        stop_at = start_at + args.seconds
        procs = []
        for _ in range(n):
            cmd = [sys.executable, "-u", __file__, "--worker",
                   "--arm", args.arm, "--batch_size", str(args.batch_size),
                   "--block_size", str(args.block_size),
                   "--token_budget", str(args.token_budget),
                   "--warmup", str(args.warmup),
                   "--start-at", repr(start_at), "--stop-at", repr(stop_at)]
            procs.append(subprocess.Popen(cmd, cwd=str(REPO), text=True,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE))
        rows, failed = [], []
        for p in procs:
            so, se = p.communicate()
            line = next((l for l in so.splitlines() if l.startswith("RESULT ")), None)
            if p.returncode != 0 or line is None:
                failed.append((se or so).strip().splitlines()[-3:])
            else:
                rows.append(json.loads(line[len("RESULT "):]))
        if failed:
            print(f"  tenancy {n}: {len(failed)}/{n} workers FAILED -> {failed[0]}")
            out["rows"][n] = {"ok": False, "failed": len(failed), "n": n}
            continue
        agg = sum(r["tok_s"] for r in rows)
        peak = sum(r["peak_gb"] for r in rows)
        res = sum(r["reserved_gb"] for r in rows)
        out["rows"][n] = {"ok": True, "n": n, "agg_tok_s": agg,
                          "per_job_tok_s": agg / n, "sum_peak_gb": peak,
                          "sum_reserved_gb": res,
                          "spread": max(r["tok_s"] for r in rows) / min(r["tok_s"] for r in rows)}
        print(f"  tenancy {n}: agg {agg/1e3:8.1f}K tok/s | per-job {agg/n/1e3:7.1f}K "
              f"| {peak:5.1f}GB alloc / {res:5.1f}GB reserved "
              f"| worker spread {out['rows'][n]['spread']:.2f}x")
    ok = [r for r in out["rows"].values() if r.get("ok")]
    if ok:
        best = max(ok, key=lambda r: r["agg_tok_s"])
        base = min(ok, key=lambda r: r["n"])
        out["best_tenancy"] = best["n"]
        out["speedup_vs_t1"] = best["agg_tok_s"] / base["agg_tok_s"]
        print(f"\n  {args.arm}: best tenancy {best['n']} "
              f"({best['agg_tok_s']/1e3:.1f}K tok/s, "
              f"{out['speedup_vs_t1']:.2f}x vs tenancy {base['n']})")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--arm", default="attention")
    p.add_argument("--tenancies", default="1,2,3,4,6")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--token_budget", type=int, default=50_000_000)
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--grace", type=float, default=25.0)
    p.add_argument("--build-grace", dest="build_grace", type=float, default=6.0)
    p.add_argument("--start-at", dest="start_at", type=float, default=0.0)
    p.add_argument("--stop-at", dest="stop_at", type=float, default=0.0)
    p.add_argument("--out", default="")
    args = p.parse_args()
    if args.worker:
        run_worker(args)
    else:
        print(f"=== tenancy sweep: {args.arm} bs{args.batch_size} ctx{args.block_size} "
              f"({args.seconds:.0f}s shared window per row) ===")
        run_parent(args)


if __name__ == "__main__":
    main()
