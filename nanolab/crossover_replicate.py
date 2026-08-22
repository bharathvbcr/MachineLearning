"""
Token-budget mixer comparison on GH200 (suite 22) plus a locked 20M follow-up.

The 50M n=5 grid did **not** replicate suite 14's 6.6–7.4M overtake location.
Measured on the matched attention/minGRU pair (bs32, eval_iters=20): minGRU
overtakes ~1.05M tokens; attention overtakes for good ~12.4M. Short rankings
lie, and the token of the flip is recipe-dependent.

    python -m nanolab.crossover_replicate list
    python -m nanolab.crossover_replicate smoke          # 40-step GPU/CPU check
    python -m nanolab.crossover_replicate launch --workers 4
    python -m nanolab.crossover_replicate isolates      # matched20 → bs8 → matched32
    python -m nanolab.crossover_replicate matched20
    python -m nanolab.crossover_replicate bs8
    python -m nanolab.crossover_replicate matched32
    python -m nanolab.crossover_replicate status
    python -m nanolab.crossover_replicate plot
    python -m nanolab.crossover_replicate table
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import build_config, parse_layer_mixers
from .train import train


SEEDS = (1337, 42, 100, 2026, 777)
TOKEN_BUDGET = 50_000_000
LOCKED20_TOKEN_BUDGET = 20_000_000
LOCKED20_OUT = Path("nanolab/out/crossover20m_locked")
LOCKED20_ARMS = ("attention", "mingru")
SUITE14_TOKEN_BUDGET = 8_192_000  # 2000 steps × bs8 × ctx512
MATCHED20_OUT = Path("nanolab/out/crossover20m_matched_lr")
BS8_OUT = Path("nanolab/out/crossover8m_bs8")
MATCHED32_OUT = Path("nanolab/out/crossover50m_matched32")
DRIFTED_ARMS = (
    "mamba2", "gdn", "mla",
    "hybrid_gdn10_attn2", "hybrid_gdn_periodic", "hybrid_gdn_bookend",
    "hybrid_mingru10_attn2", "hybrid_mamba10_attn2",
)
# Suite 14 locked these in *tokens* (bs8 × ctx512). Cluster runs may widen
# the microbatch; warmup/eval/ckpt stay on this token cadence.
SUITE14_TOKENS_PER_STEP = 8 * 1 * 512  # 4096
SUITE14_WARMUP_TOKENS = 256 * SUITE14_TOKENS_PER_STEP
SUITE14_EVAL_TOKENS = 200 * SUITE14_TOKENS_PER_STEP
SUITE14_CKPT_TOKENS = 2000 * SUITE14_TOKENS_PER_STEP
SUITE14_LOG_TOKENS = 20 * SUITE14_TOKENS_PER_STEP
TOKENS_PER_STEP = SUITE14_TOKENS_PER_STEP
MAX_STEPS = TOKEN_BUDGET // TOKENS_PER_STEP  # 12207 at bs8
EVAL_EVERY_TOKENS = SUITE14_EVAL_TOKENS
# Original suite-14 checkpoints, in millions of tokens.
SUITE14_MARKERS_M = (0.8, 4.1, 6.6, 7.4, 8.2)

# Student-t two-sided 95% critical values (df = n-1). n>=7 falls back to 1.96.
_T_CRIT_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}

DEFAULT_OUT = Path("nanolab/out/crossover50m")
QUEUE_NAME = "queue.json"
# GH200 Hopper dense BF16 tensor peak (no sparsity). Override with PEAK_FLOPS.
GH200_PEAK_FLOPS = 494.7e12


@dataclass(frozen=True)
class Arm:
    name: str
    mixer: str
    layer_mixers: str = ""
    note: str = ""


# Core replication (suite 14) plus expansions: MLA (suite 13 board) and hybrids
# that test "RNN/SSM bias early, attention capacity late" in one stack.
ARMS: tuple[Arm, ...] = (
    Arm("attention", "attention", note="capacity baseline"),
    Arm("mingru", "mingru", note="suite-14 early leader"),
    Arm("mamba2", "mamba2", note="chunk-parallel SSD"),
    Arm("gdn", "gdn", note="Gated DeltaNet"),
    Arm("mla", "mla", note="DeepSeek latent attention (suite 13)"),
    Arm("hybrid_gdn10_attn2", "gdn", "gdn*10,attention*2",
        "DeltaNet 6+2 scaled to 12L: last-2 attention"),
    Arm("hybrid_gdn_periodic", "gdn",
        "gdn*3,attention,gdn*3,attention,gdn*3,attention",
        "Qwen-style every-4th-layer attention (9 GDN + 3 Attn)"),
    Arm("hybrid_gdn_bookend", "gdn", "attention,gdn*10,attention",
        "attention at first and last layer"),
    Arm("hybrid_mingru10_attn2", "mingru", "mingru*10,attention*2",
        "minGRU inductive bias + last-2 attention"),
    Arm("hybrid_mamba10_attn2", "mamba2", "mamba2*10,attention*2",
        "Mamba-2 SSD + last-2 attention"),
)


def scale_to_token_budget(batch_size: int, block_size: int = 512,
                          grad_accum: int = 1,
                          token_budget: int = TOKEN_BUDGET,
                          lr_horizon_tokens: int | None = None) -> dict:
    """Widen the microbatch for cluster GPUs while keeping the suite-14 token cadence.

    Warmup / eval / ckpt intervals are converted from the original 4096-tok step
    so loss-vs-tokens plots stay aligned across batch sizes.

    ``lr_horizon_tokens`` is the cosine *length* in tokens. Truncating the run
    (e.g. 20M stop on a 50M schedule) must pass the long horizon or the late
    crossover moves with the schedule, not the architecture.
    """
    tps = int(batch_size) * int(grad_accum) * int(block_size)
    if tps <= 0:
        raise ValueError("tokens_per_step must be >0")
    horizon = int(lr_horizon_tokens) if lr_horizon_tokens is not None else int(token_budget)
    if horizon <= 0:
        raise ValueError("lr_horizon_tokens must be >0")
    return {
        "batch_size": int(batch_size),
        "grad_accum": int(grad_accum),
        "block_size": int(block_size),
        "tokens_per_step": tps,
        "max_steps": token_budget // tps,
        "lr_max_steps": max(1, horizon // tps),
        "warmup_steps": max(1, SUITE14_WARMUP_TOKENS // tps),
        "eval_interval": max(1, SUITE14_EVAL_TOKENS // tps),
        "ckpt_interval": max(1, SUITE14_CKPT_TOKENS // tps),
        "log_interval": max(1, SUITE14_LOG_TOKENS // tps),
    }


def cluster_batch() -> int:
    return int(os.environ.get("CROSSOVER_BATCH", "96"))


def cluster_eval_iters() -> int:
    return int(os.environ.get("CROSSOVER_EVAL_ITERS", "4"))


def cluster_token_budget() -> int:
    return int(os.environ.get("CROSSOVER_TOKEN_BUDGET", str(TOKEN_BUDGET)))


def cluster_lr_horizon() -> int | None:
    raw = os.environ.get("CROSSOVER_LR_HORIZON", "").strip()
    if not raw:
        return None
    return int(raw)


def job_prefix() -> str:
    return os.environ.get("CROSSOVER_JOB_PREFIX", "cx50")


def selected_arms() -> tuple[Arm, ...]:
    raw = os.environ.get("CROSSOVER_ARMS", "").strip()
    if not raw:
        return ARMS
    want = [x.strip() for x in raw.split(",") if x.strip()]
    by_name = {a.name: a for a in ARMS}
    missing = [n for n in want if n not in by_name]
    if missing:
        raise SystemExit(f"unknown CROSSOVER_ARMS: {missing}")
    return tuple(by_name[n] for n in want)


def current_recipe() -> dict:
    return {
        "batch_size": cluster_batch(),
        "eval_iters": cluster_eval_iters(),
        "token_budget": cluster_token_budget(),
        "lr_horizon": cluster_lr_horizon(),
        "arms": [a.name for a in selected_arms()],
        "prefix": job_prefix(),
        "compile": False,
    }


def lock_recipe(out_root: Path) -> dict:
    """Refuse to mix two training recipes in one out dir."""
    rec = current_recipe()
    path = Path(out_root) / "recipe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != rec:
            raise SystemExit(
                f"refusing to mix recipes in {path}:\n  have {old}\n  want {rec}")
        return old
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


ISOLATE_STAGES: tuple[dict, ...] = (
    {
        "name": "matched20",
        "out": MATCHED20_OUT,
        "batch": 32,
        "eval_iters": 20,
        "token_budget": LOCKED20_TOKEN_BUDGET,
        "lr_horizon": TOKEN_BUDGET,
        "arms": ",".join(LOCKED20_ARMS),
        "prefix": "cx20h",
        "workers": 2,
    },
    {
        "name": "bs8",
        "out": BS8_OUT,
        "batch": 8,
        "eval_iters": 20,
        "token_budget": SUITE14_TOKEN_BUDGET,
        "lr_horizon": None,
        "arms": ",".join(LOCKED20_ARMS),
        "prefix": "cx8",
        "workers": 2,
    },
    {
        "name": "matched32",
        "out": MATCHED32_OUT,
        "batch": 32,
        "eval_iters": 20,
        "token_budget": TOKEN_BUDGET,
        "lr_horizon": None,
        "arms": ",".join(DRIFTED_ARMS),
        "prefix": "cx32",
        "workers": 2,
    },
)


def apply_isolate(stage: dict) -> None:
    os.environ["CROSSOVER_BATCH"] = str(stage["batch"])
    os.environ["CROSSOVER_EVAL_ITERS"] = str(stage["eval_iters"])
    os.environ["CROSSOVER_TOKEN_BUDGET"] = str(stage["token_budget"])
    if stage.get("lr_horizon"):
        os.environ["CROSSOVER_LR_HORIZON"] = str(stage["lr_horizon"])
    else:
        os.environ.pop("CROSSOVER_LR_HORIZON", None)
    os.environ["CROSSOVER_ARMS"] = stage["arms"]
    os.environ["CROSSOVER_JOB_PREFIX"] = stage["prefix"]


def job_batch(job: dict) -> int:
    """GDN NaNs at bs96 once Muon LR peaks (~step 39). Cap those arms at 32."""
    text = f"{job.get('arm', '')} {job.get('mixer', '')} {job.get('layer_mixers', '')}".lower()
    if "gdn" in text:
        return min(32, cluster_batch())
    return cluster_batch()


def expand_grid(seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    jobs = []
    prefix = job_prefix()
    for arm in selected_arms():
        for seed in seeds:
            jobs.append({
                "id": f"{prefix}_{arm.name}_s{seed}",
                "arm": arm.name,
                "mixer": arm.mixer,
                "layer_mixers": arm.layer_mixers,
                "seed": seed,
                "note": arm.note,
                "status": "pending",
            })
    return jobs


def job_config(job: dict, out_root: Path, smoke: bool = False):
    scaled = scale_to_token_budget(
        job_batch(job),
        token_budget=cluster_token_budget(),
        lr_horizon_tokens=cluster_lr_horizon(),
    )
    overrides = dict(
        run_name=job["id"],
        mixer=job["mixer"],
        layer_mixers=job.get("layer_mixers") or "",
        seed=int(job["seed"]),
        out_dir=str(out_root),
        batch_size=scaled["batch_size"],
        grad_accum=scaled["grad_accum"],
        block_size=scaled["block_size"],
        max_steps=scaled["max_steps"],
        lr_max_steps=scaled["lr_max_steps"],
        warmup_steps=scaled["warmup_steps"],
        eval_interval=scaled["eval_interval"],
        ckpt_interval=scaled["ckpt_interval"],
        log_interval=scaled["log_interval"],
        eval_train=False,
        eval_iters=cluster_eval_iters(),
        compile=False,
        mem_fraction=0.0,
    )
    if smoke:
        overrides.update(
            max_steps=40, lr_max_steps=40, eval_interval=20, eval_iters=4,
            log_interval=5, warmup_steps=5, ckpt_interval=40, compile=False,
            batch_size=8,
        )
    return build_config("crossover50m", overrides)


def mean_ci(xs: list[float]) -> tuple[float, float, float]:
    """Return (mean, lower, upper) using a Student-t 95% interval."""
    n = len(xs)
    if n == 0:
        raise ValueError("mean_ci on empty sample")
    mu = statistics.fmean(xs)
    if n == 1:
        return mu, mu, mu
    sd = statistics.stdev(xs)
    sem = sd / math.sqrt(n)
    t = _T_CRIT_95.get(n, 1.96)
    half = t * sem
    return mu, mu - half, mu + half


def first_crossover_tokens(
    tokens: list[float], series_a: list[float], series_b: list[float],
) -> float | None:
    """Token count where (a - b) first changes sign, linearly interpolated."""
    flips = all_crossover_tokens(tokens, series_a, series_b)
    return flips[0] if flips else None


def all_crossover_tokens(
    tokens: list[float], series_a: list[float], series_b: list[float],
) -> list[float]:
    """Every token where (a - b) changes sign, linearly interpolated."""
    if len(tokens) != len(series_a) or len(tokens) != len(series_b):
        raise ValueError("crossover series length mismatch")
    if len(tokens) < 2:
        return []
    out: list[float] = []
    prev_gap = series_a[0] - series_b[0]
    for i in range(1, len(tokens)):
        gap = series_a[i] - series_b[i]
        if prev_gap == 0.0:
            out.append(float(tokens[i - 1]))
        elif gap == 0.0 or (prev_gap > 0) != (gap > 0):
            t0, t1 = tokens[i - 1], tokens[i]
            g0, g1 = prev_gap, gap
            if g1 == g0:
                out.append(float(t1))
            else:
                frac = -g0 / (g1 - g0)
                out.append(float(t0 + frac * (t1 - t0)))
        prev_gap = gap
    return out


def _run_dir(out_root: Path, job_id: str) -> Path:
    return Path(out_root) / job_id


def job_done(out_root: Path, job_id: str) -> bool:
    metrics = _run_dir(out_root, job_id) / "metrics.jsonl"
    if not metrics.exists():
        return False
    done = False
    with metrics.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "done":
                done = True
    return done


def load_eval_curve(run_dir: Path) -> list[dict]:
    metrics = run_dir / "metrics.jsonl"
    cfg_path = run_dir / "config.json"
    tps = TOKENS_PER_STEP
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        tps = int(cfg.get("batch_size", 8)) * int(cfg.get("grad_accum", 1)) * int(
            cfg.get("block_size", 512)
        )
    points = []
    if not metrics.exists():
        return points
    with metrics.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "eval":
                continue
            if "val_loss" not in rec:
                continue
            step = int(rec["step"])
            tokens = rec.get("tokens")
            if tokens is None:
                tokens = step * tps
            points.append({
                "step": step,
                "tokens": float(tokens),
                "val_loss": float(rec["val_loss"]),
            })
    points.sort(key=lambda p: p["tokens"])
    return points


def load_run_timing(run_dir: Path) -> dict:
    """Wall clock and throughput for one finished run.

    ``done.elapsed_s`` is authoritative and is written by ``Logger.done``.  Runs
    finished before 2026-08-22 do not carry it -- the trainer computed the
    elapsed time, printed it, and threw it away -- so for those we fall back to
    the median per-log ``tok_s`` and mark the row ``estimated``.  A row that had
    to be estimated must never be reported as a measurement.
    """
    metrics = run_dir / "metrics.jsonl"
    out = {"run": run_dir.name, "elapsed_s": None, "mean_tok_s": None,
           "median_tok_s": None, "median_mfu": None, "tokens": None,
           "source": "missing"}
    if not metrics.exists():
        return out
    tok_s, mfu = [], []
    with metrics.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = rec.get("event")
            if event == "train":
                if rec.get("tok_s"):
                    tok_s.append(float(rec["tok_s"]))
                if rec.get("mfu"):
                    mfu.append(float(rec["mfu"]))
            elif event == "done":
                out["tokens"] = rec.get("tokens")
                if rec.get("elapsed_s") is not None:
                    out["elapsed_s"] = float(rec["elapsed_s"])
                    out["mean_tok_s"] = rec.get("mean_tok_s")
                    out["source"] = "measured"
    if tok_s:
        out["median_tok_s"] = statistics.median(tok_s)
    if mfu:
        out["median_mfu"] = statistics.median(mfu)
    if out["elapsed_s"] is None and out["median_tok_s"] and out["tokens"]:
        out["elapsed_s"] = float(out["tokens"]) / out["median_tok_s"]
        out["source"] = "estimated_from_median_tok_s"
    return out


def timing_summary(out_root: Path) -> dict:
    """Per-suite GPU time. Measured and estimated rows are counted separately."""
    rows = [load_run_timing(d) for d in sorted(Path(out_root).glob("*"))
            if d.is_dir() and (d / "metrics.jsonl").exists()]
    measured = [r for r in rows if r["source"] == "measured"]
    estimated = [r for r in rows if r["source"].startswith("estimated")]
    timed = measured + estimated
    total_s = sum(r["elapsed_s"] for r in timed if r["elapsed_s"])
    return {
        "out": str(out_root),
        "runs_seen": len(rows),
        "runs_measured": len(measured),
        "runs_estimated": len(estimated),
        "runs_untimed": len(rows) - len(timed),
        "gpu_seconds_total": total_s,
        "gpu_hours_total": total_s / 3600.0,
        "gpu_hours_measured": sum(
            r["elapsed_s"] for r in measured if r["elapsed_s"]) / 3600.0,
        "gpu_hours_estimated": sum(
            r["elapsed_s"] for r in estimated if r["elapsed_s"]) / 3600.0,
        "median_run_seconds": (
            statistics.median([r["elapsed_s"] for r in timed if r["elapsed_s"]])
            if any(r["elapsed_s"] for r in timed) else None),
        "median_tok_s": (
            statistics.median([r["median_tok_s"] for r in rows if r["median_tok_s"]])
            if any(r["median_tok_s"] for r in rows) else None),
        "median_mfu": (
            statistics.median([r["median_mfu"] for r in rows if r["median_mfu"]])
            if any(r["median_mfu"] for r in rows) else None),
        "runs": rows,
    }


def cmd_timing(args) -> None:
    summary = timing_summary(Path(args.out))
    print(f"out={summary['out']}")
    print(f"  runs seen      : {summary['runs_seen']} "
          f"(measured {summary['runs_measured']}, "
          f"estimated {summary['runs_estimated']}, "
          f"untimed {summary['runs_untimed']})")
    print(f"  GPU hours      : {summary['gpu_hours_total']:.2f} total "
          f"({summary['gpu_hours_measured']:.2f} measured, "
          f"{summary['gpu_hours_estimated']:.2f} estimated)")
    if summary["median_run_seconds"]:
        print(f"  median run     : {summary['median_run_seconds']/60:.1f} min")
    if summary["median_tok_s"]:
        print(f"  median tok/s   : {summary['median_tok_s']:,.0f}")
    if summary["median_mfu"]:
        print(f"  median MFU     : {summary['median_mfu']*100:.1f}%")
    if summary["runs_untimed"]:
        print(f"  NOTE: {summary['runs_untimed']} run(s) carry no timing at all "
              f"and are excluded from the totals above.")
    if args.json:
        path = Path(args.out) / "timing.json"
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# queue (file-locked JSON; one worker claims one pending job)
# ---------------------------------------------------------------------------
def _lock_load(path: Path) -> tuple[object, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("r+" if path.exists() else "w+")
    fcntl.flock(fh, fcntl.LOCK_EX)
    raw = fh.read()
    state = json.loads(raw) if raw.strip() else {"jobs": []}
    return fh, state


def _lock_save(fh, state: dict) -> None:
    fh.seek(0)
    fh.truncate()
    json.dump(state, fh, indent=2)
    fh.flush()
    os.fsync(fh.fileno())
    fcntl.flock(fh, fcntl.LOCK_UN)
    fh.close()


def init_queue(path: Path, jobs: list[dict], out_root: Path) -> None:
    if path.exists():
        fh, state = _lock_load(path)
        have = {j["id"] for j in state["jobs"]}
        for job in jobs:
            if job["id"] not in have:
                state["jobs"].append(job)
            elif job_done(out_root, job["id"]):
                for existing in state["jobs"]:
                    if existing["id"] == job["id"]:
                        existing["status"] = "done"
        _lock_save(fh, state)
        return
    path.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")


def claim_job(path: Path, worker_id: int, out_root: Path) -> dict | None:
    fh, state = _lock_load(path)
    claimed = None
    now = time.time()
    for job in state["jobs"]:
        if job_done(out_root, job["id"]):
            job["status"] = "done"
            continue
        if job["status"] == "pending":
            job["status"] = "running"
            job["worker"] = worker_id
            job["started"] = now
            claimed = dict(job)
            break
    _lock_save(fh, state)
    return claimed


def finish_job(path: Path, job_id: str, status: str, detail: str = "") -> None:
    fh, state = _lock_load(path)
    for job in state["jobs"]:
        if job["id"] == job_id:
            job["status"] = status
            job["finished"] = time.time()
            if detail:
                job["detail"] = detail[:2000]
            break
    _lock_save(fh, state)


def cmd_list(_args) -> None:
    print(f"{'arm':<28} {'mixer':<12} {'layer_mixers':<48} note")
    for arm in ARMS:
        print(f"{arm.name:<28} {arm.mixer:<12} {arm.layer_mixers:<48} {arm.note}")
    n = len(ARMS) * len(SEEDS)
    scaled = scale_to_token_budget(cluster_batch())
    print(f"\n{len(ARMS)} arms × {len(SEEDS)} seeds = {n} jobs")
    print(f"token budget {TOKEN_BUDGET:,}  CROSSOVER_BATCH={cluster_batch()}  "
          f"tok/step {scaled['tokens_per_step']}  steps {scaled['max_steps']}")


def cmd_smoke(args) -> None:
    out_root = Path(args.out)
    seed = SEEDS[0]
    arms = [a for a in ARMS if a.name in ("attention", "mingru")]
    for arm in arms:
        job = {
            "id": f"smoke_{arm.name}_s{seed}",
            "arm": arm.name,
            "mixer": arm.mixer,
            "layer_mixers": arm.layer_mixers,
            "seed": seed,
        }
        cfg = job_config(job, out_root, smoke=True)
        print(f"\n--- smoke {job['id']} kinds={parse_layer_mixers(cfg)} ---")
        val = train(cfg)
        print(f"smoke {job['id']} best_val={val:.4f}")


def cmd_run(args) -> None:
    out_root = Path(args.out)
    arm = next((a for a in ARMS if a.name == args.arm), None)
    if arm is None:
        raise SystemExit(f"unknown arm {args.arm!r}. choices: {[a.name for a in ARMS]}")
    job = {
        "id": f"cx50_{arm.name}_s{args.seed}",
        "arm": arm.name,
        "mixer": arm.mixer,
        "layer_mixers": arm.layer_mixers,
        "seed": args.seed,
    }
    if job_done(out_root, job["id"]) and not args.force:
        print(f"{job['id']} already done; pass --force to rerun")
        return
    cfg = job_config(job, out_root, smoke=args.smoke)
    os.environ.setdefault("PEAK_FLOPS", str(GH200_PEAK_FLOPS))
    if job_done(out_root, job["id"]) is False:
        ckpt = _run_dir(out_root, job["id"]) / "ckpt.pt"
        if ckpt.exists():
            os.environ["RESUME"] = "1"
    val = train(cfg)
    print(f"{job['id']} best_val={val:.4f}")


def cmd_launch(args) -> None:
    if cluster_batch() >= 64 and args.workers > 1:
        print(f"CROSSOVER_BATCH={cluster_batch()} needs ~80GB/job; "
              f"forcing --workers 1 (asked {args.workers})")
        args.workers = 1
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    lock_recipe(out_root)
    queue = out_root / QUEUE_NAME
    jobs = expand_grid()
    if args.arm:
        jobs = [j for j in jobs if j["arm"] == args.arm]
    if args.seed is not None:
        jobs = [j for j in jobs if j["seed"] == args.seed]
    init_queue(queue, jobs, out_root)
    if getattr(args, "unhold", False):
        fh, state = _lock_load(queue)
        n = 0
        for job in state["jobs"]:
            if job.get("status") == "held":
                job["status"] = "pending"
                n += 1
        _lock_save(fh, state)
        print(f"unheld {n} jobs")
    n_pending = sum(
        1 for j in json.loads(queue.read_text(encoding="utf-8"))["jobs"]
        if j["status"] == "pending"
    )
    print(f"queue {queue}  pending≈{n_pending}  workers={args.workers}")
    workers = []
    env = os.environ.copy()
    env.setdefault("PEAK_FLOPS", str(GH200_PEAK_FLOPS))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CROSSOVER_BATCH", str(cluster_batch()))
    env.setdefault("CROSSOVER_EVAL_ITERS", str(cluster_eval_iters()))
    env.setdefault("CROSSOVER_TOKEN_BUDGET", str(cluster_token_budget()))
    env.setdefault("CROSSOVER_JOB_PREFIX", job_prefix())
    if os.environ.get("CROSSOVER_ARMS"):
        env["CROSSOVER_ARMS"] = os.environ["CROSSOVER_ARMS"]
    if os.environ.get("CROSSOVER_LR_HORIZON"):
        env["CROSSOVER_LR_HORIZON"] = os.environ["CROSSOVER_LR_HORIZON"]
    for stale in out_root.glob("worker_*.log"):
        try:
            n = int(stale.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if n >= args.workers:
            stale.unlink(missing_ok=True)
    for wid in range(args.workers):
        cmd = [
            sys.executable, "-m", "nanolab.crossover_replicate", "worker",
            "--out", str(out_root), "--worker-id", str(wid),
        ]
        log = (out_root / f"worker_{wid}.log").open("w", encoding="utf-8")
        workers.append(subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        ))
        print(f"  started worker {wid} pid={workers[-1].pid}")
    if args.detach:
        print("detached; monitor with: python -m nanolab.crossover_replicate status")
        return
    rc = 0
    for p in workers:
        rc = max(rc, p.wait())
    raise SystemExit(rc)


def cmd_worker(args) -> None:
    out_root = Path(args.out)
    queue = out_root / QUEUE_NAME
    os.environ.setdefault("PEAK_FLOPS", str(GH200_PEAK_FLOPS))
    while True:
        job = claim_job(queue, args.worker_id, out_root)
        if job is None:
            print(f"worker {args.worker_id}: idle, queue empty")
            return
        job_id = job["id"]
        print(f"worker {args.worker_id}: start {job_id}", flush=True)
        try:
            cfg = job_config(job, out_root, smoke=False)
            ckpt = _run_dir(out_root, job_id) / "ckpt.pt"
            if ckpt.exists() and not job_done(out_root, job_id):
                os.environ["RESUME"] = "1"
            else:
                os.environ.pop("RESUME", None)
            val = train(cfg)
            finish_job(queue, job_id, "done", f"best_val={val:.6f}")
            print(f"worker {args.worker_id}: done {job_id} val={val:.4f}", flush=True)
        except Exception as e:
            finish_job(queue, job_id, "failed", f"{type(e).__name__}: {e}")
            print(f"worker {args.worker_id}: FAIL {job_id}: {type(e).__name__}: {e}",
                  flush=True)


def _worker_pids() -> list[int]:
    try:
        raw = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    except Exception:
        return []
    pids = []
    for line in raw.splitlines():
        if "crossover_replicate worker" not in line:
            continue
        try:
            pids.append(int(line.strip().split()[0]))
        except ValueError:
            continue
    return pids


def cmd_repack(args) -> None:
    """Hold pending jobs so live workers drain, then relaunch 1-wide bs96.

    Two packed Mamba/GDN jobs at bs32 already fill ~70GB. The exclusive probe
    at bs96 is faster wall-clock for the remaining slow mixers, but two such
    jobs OOM, so this waits for the current pair to finish first.
    """
    out_root = Path(args.out)
    queue = out_root / QUEUE_NAME
    fh, state = _lock_load(queue)
    n_hold = 0
    for job in state["jobs"]:
        if job.get("status") == "pending":
            job["status"] = "held"
            n_hold += 1
    _lock_save(fh, state)
    print(f"held {n_hold} pending jobs; waiting for workers to drain", flush=True)
    while True:
        pids = _worker_pids()
        running = [
            j["id"] for j in json.loads(queue.read_text(encoding="utf-8"))["jobs"]
            if j.get("status") == "running"
        ]
        if not pids:
            break
        print(f"  still live pids={pids} running={running}", flush=True)
        time.sleep(30)
    args.workers = 1
    args.unhold = True
    args.arm = None
    args.seed = None
    args.detach = True
    os.environ["CROSSOVER_BATCH"] = str(cluster_batch())
    cmd_launch(args)


def cmd_status(args) -> None:
    out_root = Path(args.out)
    queue = out_root / QUEUE_NAME
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0, "held": 0, "unknown": 0}
    rows = []
    if queue.exists():
        state = json.loads(queue.read_text(encoding="utf-8"))
        jobs = state["jobs"]
    else:
        jobs = expand_grid()
    for job in jobs:
        st = job.get("status", "pending")
        if job_done(out_root, job["id"]):
            st = "done"
        counts[st] = counts.get(st, 0) + 1
        curve = load_eval_curve(_run_dir(out_root, job["id"]))
        last = curve[-1]["val_loss"] if curve else None
        tok = curve[-1]["tokens"] if curve else 0
        rows.append((job["id"], st, last, tok))
    print(f"out={out_root}  {counts}")
    for job_id, st, last, tok in rows:
        val = f"{last:.4f}" if last is not None else "—"
        print(f"  {job_id:<32} {st:<10} val={val:<8} tokens={tok/1e6:6.2f}M")


def _arm_from_run_dir(run_dir: Path) -> str | None:
    """Parse arm name from ``cx50_<arm>_s<seed>`` / config, not from process env."""
    name = run_dir.name
    body = name
    for pfx in ("cx20h_", "cx50_", "cx32_", "cx20_", "cx8_", "smoke_"):
        if body.startswith(pfx):
            body = body[len(pfx):]
            break
    if "_s" in body:
        body = body.rsplit("_s", 1)[0]
    known = {a.name for a in ARMS}
    if body in known:
        return body
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    mixer = cfg.get("mixer")
    layers = cfg.get("layer_mixers") or ""
    for arm in ARMS:
        if arm.mixer == mixer and (arm.layer_mixers or "") == layers:
            return arm.name
    return mixer if isinstance(mixer, str) else None


def _val_at_or_after(curve: list[dict], target: float) -> float | None:
    for point in curve:
        if point["tokens"] + 1.0 >= target:
            return float(point["val_loss"])
    return None


def _align_at_or_after(
    curves: list[list[dict]], ref_tokens: list[float],
) -> tuple[list[float], list[list[float]]]:
    """Map every seed onto ``ref_tokens`` by first eval at-or-after each target.

    Exact token-grid intersection drops mixed-batch arms to a handful of
    shared points and can emit a flat early curve. At-or-after keeps the
    dense reference grid and degrades to a step function on sparser seeds.
    """
    if not curves or not ref_tokens:
        return [], []
    kept: list[float] = []
    columns: list[list[float]] = []
    for target in ref_tokens:
        col: list[float] = []
        for curve in curves:
            val = _val_at_or_after(curve, target)
            if val is None:
                col = []
                break
            col.append(val)
        if col:
            kept.append(target)
            columns.append(col)
    if not columns:
        return [], []
    series = [list(row) for row in zip(*columns)]
    return kept, series


def discover_eval_curves(out_root: Path) -> dict[str, list[list[dict]]]:
    by_arm: dict[str, list[list[dict]]] = {}
    for metrics in sorted(Path(out_root).glob("*/metrics.jsonl")):
        run_dir = metrics.parent
        arm = _arm_from_run_dir(run_dir)
        if arm is None:
            continue
        curve = load_eval_curve(run_dir)
        if curve:
            by_arm.setdefault(arm, []).append(curve)
    return by_arm


def _collect(out_root: Path) -> dict:
    """Per-arm mean curves + all gap flips vs attention. Discovers runs on disk."""
    by_arm = discover_eval_curves(out_root)
    ref = []
    if by_arm.get("attention"):
        ref = [p["tokens"] for p in max(by_arm["attention"], key=len)]
    arms_out = {}
    for name, curves in by_arm.items():
        tokens_ref = ref if ref else [p["tokens"] for p in max(curves, key=len)]
        tokens, series = _align_at_or_after(curves, tokens_ref)
        if not tokens:
            arms_out[name] = {"n": 0, "tokens": [], "mean": [], "lo": [], "hi": []}
            continue
        mean, lo, hi = [], [], []
        for col in zip(*series):
            m, a, b = mean_ci(list(col))
            mean.append(m)
            lo.append(a)
            hi.append(b)
        arms_out[name] = {
            "n": len(curves),
            "tokens": tokens,
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "seeds": series,
        }

    attn = arms_out.get("attention", {})
    crossovers: dict[str, list[float]] = {}
    firsts: dict[str, float | None] = {}
    if attn.get("n"):
        tok_a, m_a = attn["tokens"], attn["mean"]
        for name, payload in arms_out.items():
            if name == "attention" or payload["n"] == 0:
                continue
            mapped_b = []
            mapped_a = []
            shared = []
            for t, va in zip(tok_a, m_a):
                vb = None
                for tb, mb in zip(payload["tokens"], payload["mean"]):
                    if tb + 1.0 >= t:
                        vb = mb
                        break
                if vb is None:
                    continue
                shared.append(t)
                mapped_a.append(va)
                mapped_b.append(vb)
            flips = all_crossover_tokens(shared, mapped_b, mapped_a)
            crossovers[name] = flips
            firsts[name] = flips[0] if flips else None
    return {
        "arms": arms_out,
        "crossovers_vs_attention": firsts,
        "all_crossovers_vs_attention": crossovers,
    }


def cmd_plot(args) -> None:
    out_root = Path(args.out)
    summary = _collect(out_root)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; wrote summary.json only")
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = {
        "attention": "#0072B2",
        "mingru": "#D55E00",
        "mamba2": "#009E73",
        "gdn": "#CC79A7",
        "mla": "#56B4E9",
        "hybrid_gdn10_attn2": "#E69F00",
        "hybrid_gdn_periodic": "#F0E442",
        "hybrid_gdn_bookend": "#882255",
        "hybrid_mingru10_attn2": "#AA4499",
        "hybrid_mamba10_attn2": "#44AA99",
    }
    for name, payload in summary["arms"].items():
        if payload["n"] == 0:
            continue
        tok_m = [t / 1e6 for t in payload["tokens"]]
        ax.plot(tok_m, payload["mean"], color=colors.get(name, "black"),
                label=f"{name} (n={payload['n']})", linewidth=1.8)
        ax.fill_between(tok_m, payload["lo"], payload["hi"],
                        color=colors.get(name, "black"), alpha=0.15, linewidth=0)
    for m in SUITE14_MARKERS_M:
        ax.axvline(m, color="#888888", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("Validation loss (mean, 95% t-interval)")
    ax.set_title("Mixer crossover @ 124M, FineWeb-edu, 5 seeds")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig_dir = out_root / "figures"
    fig_dir.mkdir(exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(fig_dir / f"crossover_val_loss.{ext}")
    plt.close(fig)
    print(f"wrote {fig_dir}/crossover_val_loss.{{svg,pdf,png}} and summary.json")


def cmd_locked20(args) -> None:
    """Finished short-cosine 20M run. New jobs go to matched20 (50M horizon)."""
    print(f"locked20 artifacts stay in {LOCKED20_OUT} (short cosine). "
          "launching matched20 (20M stop, 50M cosine) instead")
    cmd_matched20(args)


def _stage_launch(args, stage: dict) -> None:
    apply_isolate(stage)
    args.out = str(stage["out"])
    args.arm = None
    args.seed = None
    args.unhold = False
    if not getattr(args, "workers", None):
        args.workers = stage["workers"]
    print(f"{stage['name']} recipe: arms={stage['arms']} bs{stage['batch']} "
          f"eval_iters={stage['eval_iters']} tokens={stage['token_budget']/1e6:g}M "
          f"lr_horizon={stage['lr_horizon'] or stage['token_budget']} "
          f"n={len(SEEDS)} workers={args.workers} out={args.out}")
    cmd_launch(args)


def cmd_matched20(args) -> None:
    if not getattr(args, "detach", False):
        args.detach = True
    args.workers = getattr(args, "workers", None) or 2
    _stage_launch(args, ISOLATE_STAGES[0])


def cmd_bs8(args) -> None:
    if not getattr(args, "detach", False):
        args.detach = True
    args.workers = getattr(args, "workers", None) or 2
    _stage_launch(args, ISOLATE_STAGES[1])


def cmd_matched32(args) -> None:
    if not getattr(args, "detach", False):
        args.detach = True
    args.workers = getattr(args, "workers", None) or 2
    _stage_launch(args, ISOLATE_STAGES[2])


def _launch_blocking(args, stage: dict) -> None:
    apply_isolate(stage)
    args.out = str(stage["out"])
    args.workers = stage["workers"]
    args.arm = None
    args.seed = None
    args.unhold = False
    args.detach = False
    print(f"=== isolate {stage['name']} ===", flush=True)
    try:
        cmd_launch(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
        if code:
            raise
    queue = Path(stage["out"]) / QUEUE_NAME
    state = json.loads(queue.read_text(encoding="utf-8"))
    unfinished = [j["id"] for j in state["jobs"]
                  if j.get("status") != "done" and not job_done(Path(stage["out"]), j["id"])]
    if unfinished:
        raise SystemExit(f"{stage['name']} unfinished: {unfinished}")


def cmd_isolates(args) -> None:
    """matched20, then bs8, then matched32. One GPU, no mixed tables."""
    if not getattr(args, "wait", False):
        log_path = Path("nanolab/out/isolates.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        cmd = [sys.executable, "-m", "nanolab.crossover_replicate", "isolates", "--wait"]
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"isolates pid={proc.pid} log={log_path}")
        print("monitor: python -m nanolab.crossover_replicate status "
              f"--out {MATCHED20_OUT}")
        return
    ns = argparse.Namespace(arm=None, seed=None, unhold=False, detach=False, workers=2,
                            out=str(MATCHED20_OUT))
    for stage in ISOLATE_STAGES:
        _launch_blocking(ns, stage)
    print("isolates complete")


def cmd_table(args) -> None:
    out_root = Path(args.out)
    summary = _collect(out_root)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    markers = list(SUITE14_MARKERS_M) + [50.0]
    lines = [
        r"% Auto-generated by nanolab.crossover_replicate. Do not hand-edit.",
        r"\begin{tabular}{l" + "r" * (len(markers) + 1) + r"}",
        r"\toprule",
        "Architecture & " + " & ".join(f"{m:g}M" for m in markers)
        + r" & gap flips vs Attn (M tok) \\",
        r"\midrule",
    ]
    for arm in ARMS:
        payload = summary["arms"].get(arm.name, {"n": 0})
        cells = []
        for m in markers:
            target = m * 1e6
            if payload.get("n", 0) == 0 or not payload["tokens"]:
                cells.append("--")
                continue
            # nearest eval at or after the marker
            idx = min(range(len(payload["tokens"])),
                      key=lambda i: abs(payload["tokens"][i] - target))
            mu = payload["mean"][idx]
            lo = payload["lo"][idx]
            hi = payload["hi"][idx]
            cells.append(f"{mu:.3f} [{lo:.3f},{hi:.3f}]")
        flips = summary.get("all_crossovers_vs_attention", {}).get(arm.name)
        if arm.name == "attention":
            xo_s = "--"
        elif not flips:
            xo_s = "--"
        else:
            xo_s = ", ".join(f"{x / 1e6:.2f}" for x in flips)
        lines.append(arm.name.replace("_", r"\_") + " & " + " & ".join(cells)
                     + f" & {xo_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines) + "\n"
    table_dir = out_root / "tables"
    table_dir.mkdir(exist_ok=True)
    (table_dir / "crossover.tex").write_text(tex, encoding="utf-8")
    print(tex)


def _nv_query() -> dict:
    """Live compute snapshot. GPU% is occupancy of *some* work, not tensor-core math."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,power.draw,power.limit,"
        "clocks.sm,clocks.max.sm,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, text=True).strip().split(",")
        keys = ("gpu_pct", "mem_ctl_pct", "power_w", "power_cap_w", "sm_mhz",
                "sm_max_mhz", "mem_used_mb", "mem_total_mb", "temp_c")
        vals = [float(x.strip()) for x in raw]
        return dict(zip(keys, vals))
    except Exception as e:
        return {"error": str(e)}


def cmd_compute(args) -> None:
    nv = _nv_query()
    out_root = Path(args.out)
    recent = []
    now = time.time()
    for log in sorted(out_root.glob("worker_*.log")):
        try:
            if now - log.stat().st_mtime > 900:
                continue
        except OSError:
            continue
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            if "tok/s" in line and "mfu" in line:
                recent.append(f"{log.name}: {line.strip()}")
                break
    print("=== compute (not GPU%) ===")
    try:
        ga = subprocess.check_output(
            ["systemctl", "is-active", "lambda-guest-agent"], text=True
        ).strip()
    except Exception:
        ga = "missing"
    print(f"lambda-guest-agent  {ga}  (console: GPU%/VRAM; local truth: power + MFU + tok/s)")
    if "error" in nv:
        print(f"nvidia-smi: {nv['error']}")
    else:
        frac_tdp = nv["power_w"] / max(nv["power_cap_w"], 1.0)
        print(f"power     {nv['power_w']:.0f} / {nv['power_cap_w']:.0f} W  "
              f"({frac_tdp*100:.0f}% TDP)")
        print(f"SM clock  {nv['sm_mhz']:.0f} / {nv['sm_max_mhz']:.0f} MHz")
        print(f"mem ctl   {nv['mem_ctl_pct']:.0f}%   VRAM {nv['mem_used_mb']:.0f}/"
              f"{nv['mem_total_mb']:.0f} MiB   GPU% {nv['gpu_pct']:.0f}%")
        print(f"temp      {nv['temp_c']:.0f} C")
        if nv["gpu_pct"] >= 90 and frac_tdp < 0.70:
            print("verdict   BUSY but NOT compute-saturated "
                  "(high GPU%, low TDP — small GEMMs / host stalls / contention)")
        elif frac_tdp >= 0.80:
            print("verdict   compute-heavy (power near cap)")
        else:
            print("verdict   under-fed")
    if recent:
        print("latest worker tok/s lines:")
        for line in recent:
            print(f"  {line}")
    cmd_status(args)


def cmd_probe(args) -> None:
    """Exclusive-GPU batch sweep: tok/s, MFU, peak VRAM. Picks CROSSOVER_BATCH."""
    import torch
    from .train import train as _train

    batches = [int(x) for x in args.batches.split(",")]
    mixers = [m.strip() for m in args.mixers.split(",")]
    out_root = Path(args.out) / "_probe"
    results = []
    print(f"{'mixer':<12} {'bs':>4} {'tok/s':>10} {'mfu':>8} {'peakGB':>8} {'W':>6} status")
    for mixer in mixers:
        for bs in batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            scaled = scale_to_token_budget(bs)
            job = {"id": f"probe_{mixer}_bs{bs}", "arm": mixer, "mixer": mixer,
                   "layer_mixers": "", "seed": 1337}
            os.environ["CROSSOVER_BATCH"] = str(bs)
            cfg = job_config(job, out_root, smoke=False)
            cfg.max_steps = args.steps
            cfg.eval_interval = 10 ** 9
            cfg.ckpt_interval = 10 ** 9
            cfg.log_interval = max(1, args.steps // 4)
            cfg.warmup_steps = 2
            cfg.compile = False
            t0 = time.time()
            status = "ok"
            val = float("nan")
            try:
                val = _train(cfg)
            except Exception as e:
                status = f"{type(e).__name__}: {e}"[:80]
            dt = time.time() - t0
            tokens = cfg.batch_size * cfg.grad_accum * cfg.block_size * cfg.max_steps
            tok_s = tokens / max(dt, 1e-9)
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            nv = _nv_query()
            watts = nv.get("power_w", float("nan"))
            mfu = _mfu_from_toks(mixer, tok_s, cfg)
            print(f"{mixer:<12} {bs:4d} {tok_s:10.0f} {mfu*100:7.1f}% {peak:8.2f} "
                  f"{watts:6.0f} {status}")
            results.append({
                "mixer": mixer, "batch_size": bs, "tok_s": tok_s, "mfu": mfu,
                "peak_gb": peak, "watts": watts, "status": status, "val": val,
                **scaled,
            })
    rec = Path(args.out) / "probe.json"
    rec.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # pick largest batch that ran ok on the slowest mixer at > half of that mixer's peak tok/s
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        by_bs = {}
        for r in ok:
            by_bs.setdefault(r["batch_size"], []).append(r["tok_s"])
        # min tok/s across mixers at each batch (bottleneck mixer)
        ranked = sorted(by_bs.items(), key=lambda kv: min(kv[1]), reverse=True)
        best_bs, rates = ranked[0]
        print(f"\nrecommend CROSSOVER_BATCH={best_bs}  "
              f"(min mixer {min(rates):.0f} tok/s, wrote {rec})")


def _mfu_from_toks(mixer: str, tok_s: float, cfg) -> float:
    n = cfg.estimate_params()
    flops = 6 * n
    if mixer in ("attention", "mla"):
        flops += 12 * cfg.n_layer * cfg.n_head * cfg.head_dim * cfg.block_size
    peak = float(os.environ.get("PEAK_FLOPS", str(GH200_PEAK_FLOPS)))
    return (flops * tok_s) / peak


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=str(DEFAULT_OUT))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", parents=[common])
    sub.add_parser("smoke", parents=[common])
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--arm", required=True, choices=[a.name for a in ARMS])
    run.add_argument("--seed", type=int, default=SEEDS[0])
    run.add_argument("--force", action="store_true")
    run.add_argument("--smoke", action="store_true")
    launch = sub.add_parser("launch", parents=[common])
    launch.add_argument("--workers", type=int, default=1)
    launch.add_argument("--arm", default=None, choices=[a.name for a in ARMS])
    launch.add_argument("--seed", type=int, default=None)
    launch.add_argument("--detach", action="store_true")
    launch.add_argument("--unhold", action="store_true",
                        help="release held jobs back to pending")
    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--worker-id", type=int, required=True)
    sub.add_parser("status", parents=[common])
    timing = sub.add_parser("timing", parents=[common],
                            help="GPU hours and throughput for a suite")
    timing.add_argument("--json", action="store_true",
                        help="also write <out>/timing.json")
    sub.add_parser("compute", parents=[common])
    sub.add_parser("repack", parents=[common])
    probe = sub.add_parser("probe", parents=[common])
    probe.add_argument("--batches", default="8,32,64,96,128")
    probe.add_argument("--mixers", default="attention,mingru,gdn,mamba2")
    probe.add_argument("--steps", type=int, default=30)
    sub.add_parser("plot", parents=[common])
    sub.add_parser("table", parents=[common])
    locked = sub.add_parser("locked20", parents=[common],
                            help="redirects to matched20; short-cosine artifacts stay put")
    locked.add_argument("--workers", type=int, default=2)
    locked.add_argument("--detach", action="store_true")
    for name, help_txt in (
        ("matched20", "attn vs minGRU, 20M stop, 50M cosine, bs32, n=5"),
        ("bs8", "attn vs minGRU, suite-14 8.192M tokens, bs8, n=5"),
        ("matched32", "8 drifted arms, 50M, bs32, eval_iters=20, n=5"),
    ):
        sp = sub.add_parser(name, parents=[common], help=help_txt)
        sp.add_argument("--workers", type=int, default=2)
        sp.add_argument("--detach", action="store_true")
    iso = sub.add_parser("isolates", parents=[common],
                         help="run matched20, then bs8, then matched32")
    iso.add_argument("--wait", action="store_true",
                     help="run in the foreground (used by the detached supervisor)")

    args = p.parse_args()
    {
        "list": cmd_list,
        "smoke": cmd_smoke,
        "run": cmd_run,
        "launch": cmd_launch,
        "worker": cmd_worker,
        "status": cmd_status,
        "timing": cmd_timing,
        "compute": cmd_compute,
        "repack": cmd_repack,
        "probe": cmd_probe,
        "plot": cmd_plot,
        "table": cmd_table,
        "locked20": cmd_locked20,
        "matched20": cmd_matched20,
        "bs8": cmd_bs8,
        "matched32": cmd_matched32,
        "isolates": cmd_isolates,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
