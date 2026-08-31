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
    python -m nanolab.crossover_replicate swa32
    python -m nanolab.crossover_replicate swa2k
    python -m nanolab.crossover_replicate status
    python -m nanolab.crossover_replicate plot
    python -m nanolab.crossover_replicate table
"""

from __future__ import annotations

import argparse
import fcntl
import torch
import torch.nn.functional as F
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
# E10 gets its own out dir rather than joining matched32's: `current_recipe()`
# records the arm list and prefix, and `lock_recipe` refuses to mix two recipes in
# one directory. Sharing the directory would either trip that guard or, worse,
# silently blend two arm sets under one recipe.json.
RATIO32_OUT = Path("nanolab/out/crossover50m_ratio32")
# E12: sliding-window attention, the arm the section 4.5 board never had.
# arXiv:2608.28444 reports SWA(w=64, s=4) matching or beating post-trained
# linear attention; this board's linear-attention family is GDN, already at n=5
# on the matched32 recipe, so the comparison drops straight in. Own out dir for
# the same reason E10 has one: `lock_recipe` compares the arm list.
#
# NOT the paper's experiment, and the note is the point. The paper is
# TRAINING-FREE -- it masks a pretrained model at inference. These arms are
# pretrained from scratch at 50M tokens. That answers "is a local-attention
# stack competitive with a linear-attention stack when both are trained the same
# way", which is what this board can ask; it does not answer "can you retrofit a
# pretrained model", which needs a pretrained model this repo does not train.
SWA32_OUT = Path("nanolab/out/crossover50m_swa32")
# E15: the same SWA arms at 4x the context, on E9's recipe (batch 8 x ctx 2048 =
# 16,384 tokens/step, identical to suite 26's cadence). This is where the paper's
# claim actually lives: at ctx 512 a 64-wide window already spans 1/8 of the
# sequence and there is almost nothing for locality to save, which is why E12
# alone would understate SWA. Windows here span 1/32, 1/8 and 1/4 of context.
SWA2K_OUT = Path("nanolab/out/crossover50m_swa2k")
# E9: the same board at 4x the context. batch 8 x ctx 2048 = 16,384 tokens/step,
# which is exactly suite 26's bs32 x ctx512 cadence, so eval markers land on the
# same token counts and the loss-vs-token curves are directly comparable. The only
# variable that moves is sequence length.
CTX2048_OUT = Path("nanolab/out/crossover50m_ctx2048")
# The five distinct families of the section 4.5 board, one per mixer story.
CTX_ARMS = ("attention", "mingru", "gdn",
            "hybrid_mingru10_attn2", "hybrid_gdn_periodic")
# E11 phase 2: the board's top four by token-matched loss, run for the same
# WALL CLOCK rather than the same tokens. Each arm therefore gets its OWN token
# budget, and -- this is the part phase 1 could not do -- its own cosine over that
# budget. Phase 1 re-read curves that were annealed over 50M and stopped early,
# which penalises slow arms exactly as PAPER 4.3 measures; these runs anneal over
# the budget they actually get.
WALLCLOCK_OUT = Path("nanolab/out/crossover_wallclock32")
WALLCLOCK_ARMS = ("attention", "hybrid_mingru10_attn2",
                  "hybrid_gdn_periodic", "hybrid_gdn_bookend")
# Chosen as phase 1's budget: what the fastest arm needs for the 50M board.
WALLCLOCK_SECONDS = 691.0
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
# Floor for the marker window when an arm has too few evals to measure its own
# spacing. The real bound is one eval interval -- see `_marker_window`.
MARKER_TOLERANCE = 0.02

# Student-t multipliers come from native_funnel, which is the one table in this
# package: tabulated to df=30 at six decimals and covered by its own tests.
# This module used to keep a second copy keyed by n instead of df, rounded to
# three decimals, and falling back to the normal quantile from n>=7 -- so an
# eight-seed arm would silently have been given 1.96 where t_7 = 2.365 applies,
# a 21% understatement, and every interval it did compute sat ~1e-5 off the
# canonical value.
from .native_funnel import _t_critical_95

DEFAULT_OUT = Path("nanolab/out/crossover50m")
QUEUE_NAME = "queue.json"
# GH200 Hopper dense BF16 tensor peak (no sparsity). Override with PEAK_FLOPS.
#
# Was 494.7e12 until 2026-08-27, which disagreed 2x with
# scripts/gpu_bundle.py's DEVICE_PEAK_FLOPS["gh200"] and was settled by
# measurement rather than by picking: a dense BF16 8192^3 matmul on the box
# sustains 786.6 TFLOP/s, which is 159% of 494.7e12 -- impossible -- and 79.5%
# of this value, which is an ordinary large-matmul efficiency. Every GH200 MFU
# this module printed before that date was 2x too high. No published table used
# one (paper 6.3's MFU column is the 3070 Ti laptop: back-solving its logged
# tok_s and mfu gives a 40.0e12 peak).
GH200_PEAK_FLOPS = 989.5e12
MEASURED_GH200_DENSE_BF16 = 786.6e12   # 8192^3 matmul, 2026-08-27, torch 2.7.0

# bf16 DENSE peaks, never the sparsity-doubled marketing figure. Canonical here
# because `scripts/gpu_bundle.py` already imports from this module and used to
# keep a second copy of the same table.
#
# H100 SXM and GH200 share a die and a number; H100 PCIe is a lower-clocked part
# and is NOT the same figure, which is why it has its own key. Treat every row
# as a starting point, not a fact: this repo has already been burned once by a
# peak constant that was 2x wrong for months (see above). `--measure-peak`
# settles it on the box the way the GH200 row was settled.
DEVICE_PEAK_FLOPS = {
    "gh200": 989.5e12,
    "h100 pcie": 756e12,        # checked before the h100 key: substring order matters
    "h100": 989.5e12,
    "a100": 312e12,
    "a10": 125e12,
}


def device_peak_flops(name: str) -> tuple[float, str]:
    """(peak, matched-key) for a device name, or (0.0, "") when unrecognised.

    Returns zero rather than a default on purpose. An unknown accelerator that
    silently inherits GH200's 989.5e12 produces an MFU column that looks fine and
    is wrong by whatever the ratio happens to be -- the exact defect this repo
    already found once.
    """
    low = (name or "").lower()
    for key, peak in DEVICE_PEAK_FLOPS.items():
        if key in low:
            return peak, key
    return 0.0, ""


def live_device_name() -> str:
    try:
        import torch as _t
        if _t.cuda.is_available():
            return _t.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def resolve_peak_flops(strict: bool = True) -> float:
    """Peak bf16 dense FLOP/s for the accelerator actually present.

    Was `os.environ.setdefault("PEAK_FLOPS", GH200_PEAK_FLOPS)` at four call
    sites, which is correct on exactly one machine. Every suite in this repo was
    measured on a GH200; the moment one is not, that default silently rescales
    every MFU the run reports.
    """
    env = os.environ.get("PEAK_FLOPS")
    if env:
        return float(env)
    name = live_device_name()
    peak, key = device_peak_flops(name)
    if peak:
        os.environ["PEAK_FLOPS"] = str(peak)
        return peak
    if not strict:
        return 0.0
    raise SystemExit(
        f"unknown accelerator {name or '<none detected>'!r}: refusing to assume "
        f"a peak-FLOP figure, because every MFU this run prints would inherit it. "
        f"Known: {sorted(DEVICE_PEAK_FLOPS)}. Measure it with "
        f"`python -m nanolab.crossover_replicate measure-peak` and export "
        f"PEAK_FLOPS, or export PEAK_FLOPS yourself.")


def measure_dense_bf16(n: int = 8192, iters: int = 8) -> float:
    """Achieved dense bf16 FLOP/s from an n^3 matmul -- how the GH200 row above
    was settled after the tabulated value disagreed with reality by 2x."""
    import torch as _t
    if not _t.cuda.is_available():
        raise SystemExit("measure-peak needs a CUDA device")
    a = _t.randn(n, n, device="cuda", dtype=_t.bfloat16)
    b = _t.randn(n, n, device="cuda", dtype=_t.bfloat16)
    for _ in range(3):
        a @ b
    _t.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        a @ b
    _t.cuda.synchronize()
    return (2.0 * n ** 3 * iters) / (time.time() - t0)


@dataclass(frozen=True)
class Arm:
    name: str
    mixer: str
    layer_mixers: str = ""
    note: str = ""
    # Extra Config fields this arm needs, as (key, value) pairs -- a tuple, not a
    # dict, so the dataclass stays hashable. `job_config` applies them by ARM
    # NAME rather than from the job dict, so a job built by any of the four
    # construction sites gets them; and `current_recipe` records them, so
    # changing an arm's window cannot silently reuse the directory that holds
    # runs measured at the old one.
    overrides: tuple[tuple[str, object], ...] = ()


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
    # E10 (backlog 2026-08-26): the board's best hybrid family exists at exactly
    # ONE ratio and placement (10+2, last-2) while GDN got three variants, so
    # "the best hybrid ties attention" is a claim about one point on the
    # ratio/placement axis. These four vary that axis and nothing else. The
    # field's converged 3:1 periodic ratio has never been run on the family that
    # actually ties attention.
    Arm("hybrid_mingru11_attn1", "mingru", "mingru*11,attention",
        "how little attention is enough: 11+1"),
    Arm("hybrid_mingru_periodic", "mingru",
        "mingru*3,attention,mingru*3,attention,mingru*3,attention",
        "Qwen-style every-4th-layer attention (9 minGRU + 3 Attn)"),
    Arm("hybrid_mingru_bookend", "mingru", "attention,mingru*10,attention",
        "attention at first and last layer"),
    Arm("hybrid_mingru8_attn4", "mingru", "mingru*8,attention*4",
        "1:2 ratio upper arm"),
    # E12: SWA(w, s) at three windows plus the sink ablation. The window sweep
    # already has its top endpoint on the board for free -- at block_size 512 a
    # window of 512 IS full causal attention, i.e. the `attention` arm at n=5 --
    # so these four arms buy the 1/8, 1/4 and 1/2 points of that curve.
    Arm("swa_w64", "swa", note="SWA(64,4) -- the paper's primary config",
        overrides=(("swa_window", 64), ("swa_sinks", 4))),
    Arm("swa_w128", "swa", note="SWA(128,4) -- 1/4 of the 512 context",
        overrides=(("swa_window", 128), ("swa_sinks", 4))),
    Arm("swa_w256", "swa", note="SWA(256,4) -- 1/2 of the 512 context",
        overrides=(("swa_window", 256), ("swa_sinks", 4))),
    Arm("swa_w64_nosink", "swa", note="SWA(64,0) -- isolates the sinks",
        overrides=(("swa_window", 64), ("swa_sinks", 0))),
    # E15 only: at ctx 512 a 512-wide window IS full attention, so this arm is
    # meaningless there and is deliberately absent from SWA_ARMS.
    Arm("swa_w512", "swa", note="SWA(512,4) -- 1/4 of the 2048 context",
        overrides=(("swa_window", 512), ("swa_sinks", 4))),
    # The backlog's E12 question proper: must the hybrid's attention be GLOBAL?
    # `hybrid_mingru10_attn2` is the board's co-leader; this is its twin with the
    # two attention layers windowed. If it holds, those layers were doing local
    # work all along; if it drops, the global retrieval story earns its cost.
    # Window 128 at ctx 512 is the backlog's own choice -- wide enough to train,
    # narrow enough that the window binds.
    Arm("hybrid_mingru10_swa2", "mingru", "mingru*10,swa*2",
        "E12: the board's co-leader with its attention windowed",
        overrides=(("swa_window", 128), ("swa_sinks", 4))),
)
# `attention` and `gdn` are carried IN these suites rather than read across from
# suite 26 / E9. Those rows were measured on a GH200; these will not be. Joining
# them would compare architectures across hardware, which is the confound this
# paper is about -- and `current_recipe` records the device precisely so that
# such a join cannot happen quietly. Ten extra runs buys a self-contained board.
SWA_ARMS = ("attention", "gdn",
            "swa_w64", "swa_w128", "swa_w256", "swa_w64_nosink",
            "hybrid_mingru10_swa2")
# Same questions at ctx 2048: three locality ratios, the sink ablation, and the
# same two references for the same reason.
SWA2K_ARMS = ("attention", "gdn",
              "swa_w64", "swa_w256", "swa_w512", "swa_w64_nosink")
# The four E10 arms, as one name so a launcher cannot list three of them.
RATIO_ARMS = ("hybrid_mingru11_attn1", "hybrid_mingru_periodic",
              "hybrid_mingru_bookend", "hybrid_mingru8_attn4")


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


def cluster_swa_chunk() -> int:
    """SWA attention chunking: -1 auto, 0 dense, >0 that chunk size.

    A recipe field, not a tuning knob. It is numerically identical either way
    (asserted in tests), so it never moves a quality row -- but it changes tok/s
    by up to 2x, so a suite that mixed two settings would produce a throughput
    column nothing could interpret. `probe` measures both and prints which to
    pin here.
    """
    return int(os.environ.get("SWA_CHUNK", "-1"))


def cluster_gpus() -> int:
    """How many GPUs this launch spreads over. 1 unless set.

    Kept SEPARATE from `workers`, which stays what it has always been: jobs per
    GPU, i.e. tenancy. Total processes are `gpus * workers`. Conflating the two
    would silently redefine `workers` in every recipe.json already on disk, and
    every throughput number those recipes make interpretable.
    """
    return max(1, int(os.environ.get("CROSSOVER_GPUS", "1")))


def inherited_gpu_ids() -> list[str]:
    """The device ids this process may use, in CUDA's own order.

    Honours an inherited CUDA_VISIBLE_DEVICES. Without this, a launcher that
    hardcodes "0".."N-1" for its children silently ESCAPES an operator's
    restriction: `CUDA_VISIBLE_DEVICES=4,5,6,7 ... --gpus 4` would have run on
    physical GPUs 0-3, i.e. on somebody else's cards.
    """
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def gpu_id_for_worker(wid: int, gpus: int) -> str:
    """Physical device id for worker `wid`, round-robin over the allowed set."""
    ids = inherited_gpu_ids()
    return ids[wid % len(ids)] if ids else str(wid % gpus)


def visible_gpu_count() -> int:
    """GPUs this process can actually see, honouring CUDA_VISIBLE_DEVICES."""
    try:
        import torch as _t
        return _t.cuda.device_count() if _t.cuda.is_available() else 0
    except Exception:
        return 0


def cluster_workers() -> int:
    """How many jobs share the GPU. Set by ``cmd_launch``; 1 when unset.

    This is a RECIPE field, not a scheduling detail, because throughput is only
    interpretable together with it -- see ``_suite_tenancy``.
    """
    return max(1, int(os.environ.get("CROSSOVER_WORKERS", "1")))


def _suite_tenancy(suite_dir: Path) -> int | None:
    """Jobs per GPU while this suite ran, or None when that is unknowable.

    A tok/s number means nothing on its own. The same arm runs ~1.8x faster
    single-tenant than three-to-a-GPU, and the speedup is *not* uniform across
    arms: compute-bound mixers recover most of the contention loss, latency-bound
    ones barely move. Sizing a budget from a rate measured at another tenancy is
    exactly what silently broke the first wall-clock suite (1.70x spread on a
    board whose whole claim was equal wall clock).

    Recorded in recipe.json from now on. For suites predating that, the number of
    worker_N.log files is a real artifact of the last launch -- ``cmd_launch``
    unlinks logs with id >= workers -- so it is derived, not assumed. Returns
    None when neither signal exists; callers must fail closed on that rather
    than defaulting to 1.
    """
    rec = Path(suite_dir) / "recipe.json"
    if rec.exists():
        try:
            recorded = json.loads(rec.read_text(encoding="utf-8")).get("workers")
        except json.JSONDecodeError:
            recorded = None
        if isinstance(recorded, int) and recorded > 0:
            return recorded
    n = len(list(Path(suite_dir).glob("worker_*.log")))
    if not n:
        return None
    # Log files count TOTAL processes. On a multi-GPU launch that is gpus x
    # tenancy, so dividing is the difference between "three to a GPU" and
    # "twenty-four processes", which are wildly different throughput regimes.
    gpus = 1
    if rec.exists():
        try:
            gpus = int(json.loads(rec.read_text(encoding="utf-8")).get("gpus") or 1)
        except (json.JSONDecodeError, TypeError, ValueError):
            gpus = 1
    return max(1, -(-n // max(1, gpus)))


RATE_SUITES = (
    "nanolab/out/crossover50m",
    "nanolab/out/crossover50m_matched32",
    "nanolab/out/crossover50m_ratio32",
    # The first wall-clock attempt missed its target badly and is archived rather
    # than deleted: its runs are the only SINGLE-TENANT throughput measurement in
    # the repo, which is exactly what sizing the retry requires.
    "nanolab/out/crossover_wallclock32_unmatched",
    "nanolab/out/crossover_wallclock32",
)


def measured_rate_by_arm(root: Path | None = None, tenancy: int | None = None,
                         suites: tuple[str, ...] = RATE_SUITES) -> dict[str, float]:
    """Median tok/s per arm at ``tenancy`` jobs per GPU, from committed records.

    Derived from this repository, never typed. Same source the GPU-bundle cost
    model uses, so the two price the same hardware the same way.

    ``tenancy=None`` keeps the tenancy-blind behaviour and is for *reporting only*
    -- it mixes rates measured under different contention and must never size a
    budget. Anything that sizes a budget passes the tenancy it will really run at;
    suites whose tenancy cannot be established are skipped rather than assumed.
    """
    base = Path(root) if root else Path(__file__).resolve().parents[1]
    per: dict[str, list[float]] = {}
    for suite in suites:
        if tenancy is not None and _suite_tenancy(base / suite) != tenancy:
            continue
        for cfgp in sorted((base / suite).glob("*/config.json")):
            mp = cfgp.with_name("metrics.jsonl")
            if not mp.exists():
                continue
            try:
                cfg = json.loads(cfgp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if cfg.get("batch_size") != 32 or cfg.get("block_size") != 512:
                continue
            arm = cfgp.parent.name.split("_s")[0].split("_", 1)[-1]
            ts = []
            for line in mp.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "train" and r.get("tok_s"):
                    ts.append(float(r["tok_s"]))
            if ts:
                per.setdefault(arm, []).append(statistics.median(ts))
    return {a: statistics.median(v) for a, v in per.items()}


def effective_rate_by_arm(root: Path | None = None, tenancy: int | None = None,
                          suites: tuple[str, ...] = RATE_SUITES) -> dict[str, float]:
    """Median tokens per WALL-CLOCK second per arm, from terminal run records.

    Not the same thing as ``measured_rate_by_arm``, and the difference is what
    sizes a wall-clock budget correctly. That one reports instantaneous step
    throughput; this one is tokens/elapsed_s, so it also carries eval, checkpoint
    and startup time. The overhead is not a constant factor across arms --
    measured single-tenant, attention realises 81% of its step rate over the whole
    run while gdn_bookend realises 89% -- so sizing from step rate alone reinstates
    a several-percent mismatch in the one quantity the suite holds constant.
    """
    base = Path(root) if root else Path(__file__).resolve().parents[1]
    per: dict[str, list[float]] = {}
    for suite in suites:
        if tenancy is not None and _suite_tenancy(base / suite) != tenancy:
            continue
        for mp in sorted((base / suite).glob("*/metrics.jsonl")):
            name = mp.parent.name
            if "_s" not in name:
                continue
            arm = name.split("_s")[0].split("_", 1)[-1]
            for line in mp.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (rec.get("event") == "done" and rec.get("elapsed_s")
                        and rec.get("tokens")):
                    per.setdefault(arm, []).append(
                        float(rec["tokens"]) / float(rec["elapsed_s"]))
    return {a: statistics.median(v) for a, v in per.items() if v}


def wallclock_budgets(seconds: float, arms: tuple[str, ...],
                      batch: int = 32, block: int = 512,
                      rates: dict[str, float] | None = None,
                      tenancy: int | None = None) -> dict[str, int]:
    """{arm: token budget} such that every arm trains for the same wall clock.

    ``tenancy`` is the number of jobs that will share the GPU during the run, and
    it is REQUIRED unless explicit ``rates`` are supplied. Rates are only valid at
    the tenancy they were measured under: the first attempt at this suite sized
    budgets from three-to-a-GPU rates and then ran single-tenant, so attention got
    387.8s and gdn_bookend 661.1s against a 691s target -- a 1.70x spread in the
    one quantity the suite exists to hold constant.

    Floored to a whole number of steps, because a partial step is not trained and
    a budget that is not a step multiple silently rounds differently per arm --
    which would put the wall clock back out of match.
    """
    if rates is None:
        if tenancy is None:
            raise SystemExit(
                "wallclock_budgets needs the tenancy it will run at; a tok/s "
                "measured at another tenancy does not transfer across arms.")
        rates = effective_rate_by_arm(tenancy=tenancy)
    missing = [a for a in arms if a not in rates]
    if missing:
        raise SystemExit(
            f"no throughput for {missing} at tenancy={tenancy}; cannot match "
            f"wall clock without it. Run those arms at bs{batch}/ctx{block} "
            f"with --workers {tenancy} first.")
    tps = batch * block
    return {a: max(tps, int(rates[a] * seconds) // tps * tps) for a in arms}


# Equal wall clock is this suite's entire claim, so it is verified against the
# artifacts rather than trusted from the budgets that were requested.
WALLCLOCK_TOLERANCE = 0.05


def verify_wallclock(out_root: Path, target_s: float,
                     tolerance: float = WALLCLOCK_TOLERANCE) -> dict:
    """Did the runs actually train for the same wall clock? Measured, not assumed.

    Reads ``elapsed_s`` out of each run's terminal metrics record. Returns the
    per-arm means, the observed spread, and an ``ok`` flag. The first wall-clock
    suite missed by 1.70x and still emitted a board that looked publishable;
    nothing in the code objected, which is why this exists.
    """
    per: dict[str, list[float]] = {}
    for mp in sorted(Path(out_root).glob("*/metrics.jsonl")):
        arm = None
        cfgp = mp.with_name("config.json")
        if cfgp.exists():
            try:
                arm = json.loads(cfgp.read_text(encoding="utf-8")).get("mixer")
            except json.JSONDecodeError:
                arm = None
        name = mp.parent.name
        arm = name.split("_s")[0].split("_", 1)[-1] or arm
        for line in mp.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "done" and rec.get("elapsed_s"):
                per.setdefault(arm, []).append(float(rec["elapsed_s"]))
    rows = {a: statistics.mean(v) for a, v in per.items() if v}
    if not rows:
        return {"ok": False, "reason": "no elapsed_s in any run record",
                "arms": {}, "target_s": target_s, "spread": None}
    worst = max(abs(v - target_s) / target_s for v in rows.values())
    return {
        "ok": worst <= tolerance,
        "reason": None if worst <= tolerance else
                  f"worst arm is {worst:.1%} off the {target_s:.0f}s target "
                  f"(tolerance {tolerance:.0%})",
        "arms": rows,
        "target_s": target_s,
        "spread": max(rows.values()) / min(rows.values()),
        "n": {a: len(v) for a, v in per.items() if v},
    }


def budget_by_arm() -> dict[str, int]:
    raw = os.environ.get("CROSSOVER_BUDGET_BY_ARM", "").strip()
    if not raw:
        return {}
    return {k: int(v) for k, v in json.loads(raw).items()}


def cluster_block() -> int:
    """Context length. 512 is every committed suite; E9 varies it.

    This is a RECIPE field, so `current_recipe()` records it and `lock_recipe`
    refuses to mix two context lengths in one out dir. Without that, a 2048-context
    run dropped into a 512-context directory would be averaged into the same board
    -- the exact class of error this paper is about.
    """
    return int(os.environ.get("CROSSOVER_BLOCK", "512"))


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
        "block_size": cluster_block(),
        # Tenancy is a recipe field, not a scheduling detail: throughput measured
        # at one jobs-per-GPU does not transfer to another, and a suite that
        # silently mixed the two would produce rates nothing could interpret.
        "workers": cluster_workers(),
        # Total processes are gpus * workers; `workers` remains jobs-per-GPU.
        # Recorded because a 2-per-GPU suite on 8 GPUs and one on 1 GPU are the
        # same tenancy but not the same run, and the second cannot be resumed
        # into the first without changing what "elapsed" means.
        "gpus": cluster_gpus(),
        # Recorded so `lock_recipe` refuses to mix two wall-clock budgets, or a
        # wall-clock-matched run and a token-matched one, in one directory.
        "budget_by_arm": budget_by_arm() or None,
        # An arm's window is part of the architecture it names. Recorded so that
        # editing SWA(64,4) to SWA(64,8) and relaunching refuses the directory
        # instead of averaging two architectures under one arm name.
        "arm_overrides": selected_arm_overrides(),
        "swa_chunk": cluster_swa_chunk(),
        "eval_iters": cluster_eval_iters(),
        "token_budget": cluster_token_budget(),
        "lr_horizon": cluster_lr_horizon(),
        "arms": [a.name for a in selected_arms()],
        "prefix": job_prefix(),
        # The largest recipe axis of all, and the one this module did not record.
        # Two boxes produced identical recipe.json, so `lock_recipe` would have
        # mixed a GH200 suite and an H100 suite in one directory without a word --
        # in a repo whose paper is about rankings moving with the recipe.
        "device": live_device_name() or None,
        "compile": False,
    }


def lock_recipe(out_root: Path) -> dict:
    """Refuse to mix two training recipes in one out dir."""
    rec = current_recipe()
    path = Path(out_root) / "recipe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        # A field added after a suite ran is unrecorded, not conflicting. Backfill
        # it when everything actually recorded agrees; a real disagreement on any
        # shared field still refuses, which is the point of the lock.
        conflicts = {k: (old[k], rec[k]) for k in old.keys() & rec.keys()
                     if old[k] != rec[k]}
        if conflicts:
            raise SystemExit(
                f"refusing to mix recipes in {path}:\n  have {old}\n  want {rec}"
                f"\n  conflicting fields: {sorted(conflicts)}")
        if old.keys() != rec.keys():
            merged = {**rec, **old}
            if "workers" not in old:
                # Never backfill tenancy from the CURRENT launch -- that would
                # write a guess about a run that already happened into its own
                # record. The worker-log count is the suite's own artifact; when
                # it disagrees with this launch, that is a real conflict.
                prior = _suite_tenancy(path.parent)
                if prior is None:
                    merged.pop("workers", None)
                elif prior != rec["workers"]:
                    raise SystemExit(
                        f"refusing to mix tenancies in {path}: suite ran at "
                        f"workers={prior}, this launch is workers={rec['workers']}")
                else:
                    merged["workers"] = prior
            path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
            return merged
        return old
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


# What `isolates` runs, in order. Named explicitly, because it used to iterate the
# whole ISOLATE_STAGES tuple: adding a stage for a different experiment silently
# enlarged an existing command and would have run 20 unrelated jobs under it.
ISOLATE_SEQUENCE: tuple[str, ...] = ("matched20", "bs8", "matched32")

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
    # E10: identical to `matched32` in every recipe field -- batch 32,
    # eval_iters 20, 50M budget, 50M cosine -- so its rows drop straight into the
    # section 4.5 board. Only the arm list differs, which is the point.
    # E11 phase 2: every arm trains the SAME WALL CLOCK, each with its own token
    # budget and its own cosine over that budget. Single-tenant by design -- the
    # artifact is `elapsed_s`, and a co-located job measures the scheduler.
    {
        "name": "wallclock32",
        "out": WALLCLOCK_OUT,
        "batch": 32,
        "eval_iters": 20,
        "token_budget": TOKEN_BUDGET,      # overridden per arm; kept for the record
        "wall_clock_s": WALLCLOCK_SECONDS,
        "lr_horizon": None,
        "arms": ",".join(WALLCLOCK_ARMS),
        "prefix": "cxwc",
        "workers": 1,
    },
    {
        "name": "ctx2048",
        "out": CTX2048_OUT,
        "batch": 8,
        "block": 2048,
        "eval_iters": 20,
        "token_budget": TOKEN_BUDGET,
        "lr_horizon": None,
        "arms": ",".join(CTX_ARMS),
        "prefix": "cx2k",
        "workers": 2,
    },
    # E12: identical to `matched32` in every recipe field, so its rows drop
    # straight into the section 4.5 board. Only the arm list differs.
    {
        "name": "swa32",
        "out": SWA32_OUT,
        "batch": 32,
        "eval_iters": 20,
        "token_budget": TOKEN_BUDGET,
        "lr_horizon": None,
        "arms": ",".join(SWA_ARMS),
        "prefix": "cx32swa",
        "workers": 2,
    },
    # E15: E9's recipe exactly -- batch 8, block 2048, 50M budget -- so its rows
    # sit beside the five families already measured at that context.
    {
        "name": "swa2k",
        "out": SWA2K_OUT,
        "batch": 8,
        "block": 2048,
        "eval_iters": 20,
        "token_budget": TOKEN_BUDGET,
        "lr_horizon": None,
        "arms": ",".join(SWA2K_ARMS),
        "prefix": "cx2kswa",
        "workers": 2,
    },
    {
        "name": "ratio32",
        "out": RATIO32_OUT,
        "batch": 32,
        "eval_iters": 20,
        "token_budget": TOKEN_BUDGET,
        "lr_horizon": None,
        "arms": ",".join(RATIO_ARMS),
        "prefix": "cx32r",
        "workers": 2,
    },
)


def apply_isolate(stage: dict) -> None:
    # Exported before budgets are sized: the tenancy the stage will run at is the
    # same tenancy its throughput must have been measured at.
    os.environ["CROSSOVER_WORKERS"] = str(stage["workers"])
    if stage.get("wall_clock_s"):
        arms = tuple(stage["arms"].split(","))
        budgets = wallclock_budgets(stage["wall_clock_s"], arms,
                                    batch=stage["batch"],
                                    block=stage.get("block", 512),
                                    tenancy=stage["workers"])
        os.environ["CROSSOVER_BUDGET_BY_ARM"] = json.dumps(budgets, sort_keys=True)
    else:
        os.environ.pop("CROSSOVER_BUDGET_BY_ARM", None)
    os.environ["CROSSOVER_BATCH"] = str(stage["batch"])
    os.environ["CROSSOVER_BLOCK"] = str(stage.get("block", 512))
    os.environ["CROSSOVER_EVAL_ITERS"] = str(stage["eval_iters"])
    os.environ["CROSSOVER_TOKEN_BUDGET"] = str(stage["token_budget"])
    if stage.get("lr_horizon"):
        os.environ["CROSSOVER_LR_HORIZON"] = str(stage["lr_horizon"])
    else:
        os.environ.pop("CROSSOVER_LR_HORIZON", None)
    os.environ["CROSSOVER_ARMS"] = stage["arms"]
    os.environ["CROSSOVER_JOB_PREFIX"] = stage["prefix"]


def arm_overrides(name: str) -> dict:
    """Config overrides for an arm, looked up by name.

    Read from the registry rather than carried in the job dict on purpose: jobs
    are built at four sites (`expand_grid`, `cmd_smoke`, `cmd_run`, `cmd_probe`)
    and a fifth would silently drop the window, producing an arm called
    `swa_w64` that ran at the default window.
    """
    for arm in ARMS:
        if arm.name == name:
            return dict(arm.overrides)
    return {}


def selected_arm_overrides() -> dict:
    """`{arm: overrides}` for the arms this launch selects, or None if none set."""
    got = {a.name: dict(a.overrides) for a in selected_arms() if a.overrides}
    return got or None


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
    # A wall-clock-matched stage gives each arm its OWN token budget, and with
    # lr_horizon unset the cosine spans that same budget -- so every arm anneals
    # over the tokens it actually gets. That is the whole difference from E11
    # phase 1, which re-read curves annealed over 50M and stopped early.
    per_arm = budget_by_arm().get(job.get("arm", ""))
    scaled = scale_to_token_budget(
        job_batch(job),
        block_size=cluster_block(),
        token_budget=per_arm if per_arm else cluster_token_budget(),
        lr_horizon_tokens=None if per_arm else cluster_lr_horizon(),
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
        swa_chunk=cluster_swa_chunk(),
        eval_iters=cluster_eval_iters(),
        compile=False,
        mem_fraction=0.0,
    )
    # Applied last: an arm's own knobs (e.g. the SWA window) are what makes it
    # that arm, so they are not overridable by the generic scaling above.
    overrides.update(arm_overrides(job.get("arm", "")))
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
    half = _t_critical_95(n - 1) * sem
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
    scaled = scale_to_token_budget(cluster_batch(), block_size=cluster_block())
    print(f"\n{len(ARMS)} arms × {len(SEEDS)} seeds = {n} jobs")
    print(f"token budget {TOKEN_BUDGET:,}  CROSSOVER_BATCH={cluster_batch()}  "
          f"tok/step {scaled['tokens_per_step']}  steps {scaled['max_steps']}")


def cmd_smoke(args) -> None:
    # Smoke runs land in their own subtree, NOT in the suite directory.
    #
    # They are 40-step runs that write a real metrics.jsonl with a real `done`
    # record and a real best_val, and `_collect` reads every directory it finds.
    # Writing them beside the suite put `smoke_attention_s1337` next to 50 real
    # cx50 runs, which broke the aligner outright: the committed summary said
    # n=5 for attention and a fresh _collect found 0, because the smoke runs'
    # token grid shares no marker with the real ones. The paper's ten-arm board is
    # built from that summary.
    #
    # `docs/GPU_BUNDLE.md` records this same defect in `scripts/gpu_bundle.py`,
    # where "following the documented procedure corrupted the matrix". It was
    # fixed there and left here.
    out_root = Path(args.out) / "_smoke"
    seed = SEEDS[0]
    # Was hardcoded to attention/mingru, so no smoke run ever built a windowed
    # layer and `swa32`'s first real failure would have been on the rented box.
    # Defaults still cover the original pair.
    want = [x.strip() for x in (args.arms or "attention,mingru").split(",") if x.strip()]
    by = {a.name: a for a in ARMS}
    missing = [n for n in want if n not in by]
    if missing:
        raise SystemExit(f"unknown smoke arms: {missing}")
    arms = [by[n] for n in want]
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
    resolve_peak_flops()
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
    # Record the tenancy this launch will ACTUALLY run at. `cluster_workers()`
    # reads CROSSOVER_WORKERS, which `launch --workers N` does not set, so a
    # suite launched with --workers 3 recorded `workers: 1` while running three
    # to a GPU. Throughput is only interpretable together with tenancy, and
    # `_suite_tenancy` and every rate model downstream trust this field -- a
    # recipe field set by assumption rather than by what happened is the defect
    # this suite's own analysis exists to catch.
    os.environ["CROSSOVER_WORKERS"] = str(args.workers)
    gpus = max(1, int(getattr(args, "gpus", 0) or 0) or cluster_gpus())
    seen = visible_gpu_count()
    if seen and gpus > seen:
        raise SystemExit(
            f"--gpus {gpus} but only {seen} visible. Refusing rather than "
            f"oversubscribing: workers past the last GPU would pile onto one "
            f"device and the tenancy this suite records would be a fiction.")
    os.environ["CROSSOVER_GPUS"] = str(gpus)
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
    total = gpus * args.workers
    print(f"queue {queue}  pending≈{n_pending}  gpus={gpus} "
          f"workers/gpu={args.workers}  processes={total}")
    workers = []
    env = os.environ.copy()
    env.setdefault("PEAK_FLOPS", str(resolve_peak_flops()))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CROSSOVER_BATCH", str(cluster_batch()))
    env.setdefault("CROSSOVER_EVAL_ITERS", str(cluster_eval_iters()))
    env.setdefault("CROSSOVER_TOKEN_BUDGET", str(cluster_token_budget()))
    env.setdefault("CROSSOVER_JOB_PREFIX", job_prefix())
    # Workers is exported, not defaulted: the recipe every job records must say
    # how many jobs actually shared the GPU with it.
    env["CROSSOVER_WORKERS"] = str(args.workers)
    env["CROSSOVER_GPUS"] = str(gpus)
    if os.environ.get("CROSSOVER_ARMS"):
        env["CROSSOVER_ARMS"] = os.environ["CROSSOVER_ARMS"]
    if os.environ.get("CROSSOVER_LR_HORIZON"):
        env["CROSSOVER_LR_HORIZON"] = os.environ["CROSSOVER_LR_HORIZON"]
    for stale in out_root.glob("worker_*.log"):
        try:
            n = int(stale.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if n >= total:
            stale.unlink(missing_ok=True)
    for wid in range(total):
        cmd = [
            sys.executable, "-m", "nanolab.crossover_replicate", "worker",
            "--out", str(out_root), "--worker-id", str(wid),
        ]
        wenv = dict(env)
        if gpus > 1:
            # Round-robin, so gpus*workers processes land exactly `workers` to a
            # device.
            #
            # Pinned with CUDA_VISIBLE_DEVICES rather than torch.cuda.set_device
            # on purpose. The pinned card is renumbered to index 0 inside the
            # child, so `pick_device`'s bare "cuda" and the nineteen no-argument
            # torch.cuda.* calls in this package -- max_memory_allocated and
            # reset_peak_memory_stats among them -- are all correct with no code
            # change, because the process can see exactly one device. set_device
            # would need injecting before anything touches CUDA, and any path
            # that got there first would bind to device 0 and then report a peer
            # GPU's memory as this job's.
            #
            # Ids come from the inherited allow-list, not from range(gpus): the
            # naive form escapes an operator's CUDA_VISIBLE_DEVICES onto cards
            # they did not offer.
            wenv["CUDA_VISIBLE_DEVICES"] = gpu_id_for_worker(wid, gpus)
        log = (out_root / f"worker_{wid}.log").open("w", encoding="utf-8")
        workers.append(subprocess.Popen(
            cmd, env=wenv, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        ))
        print(f"  started worker {wid} pid={workers[-1].pid}"
              + (f" gpu={gpu_id_for_worker(wid, gpus)}" if gpus > 1 else ""))
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
    resolve_peak_flops()
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
    elif not out_root.exists():
        # Falling back to the default grid here printed 70 pending jobs of a
        # different suite for a directory that did not exist -- a suite nobody
        # had launched read as a suite fully queued.
        raise SystemExit(f"{out_root} does not exist; nothing has been launched")
    else:
        jobs = expand_grid()
        print(f"note: no {QUEUE_NAME} in {out_root}; showing the default grid, "
              "which may not be this suite's")
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


def stage_by_name(name: str) -> dict:
    """Look a stage up by name.

    These were indexed positionally -- ISOLATE_STAGES[0], [1], [2] -- so inserting
    a stage anywhere but the end silently re-pointed a subcommand at someone
    else's recipe. A name is what the caller actually means.
    """
    for stage in ISOLATE_STAGES:
        if stage["name"] == name:
            return stage
    raise SystemExit(f"unknown stage {name!r}; have "
                     + ", ".join(s["name"] for s in ISOLATE_STAGES))


def _stage_cmd(name: str):
    def run(args) -> None:
        stage = stage_by_name(name)
        if not getattr(args, "detach", False):
            args.detach = True
        # The stage's own tenancy is the default, not a hardcoded 2. A wall-clock
        # stage sizes its budgets for a specific jobs-per-GPU, so launching it at
        # any other tenancy silently invalidates the run -- refuse rather than
        # quietly honour the flag.
        asked = getattr(args, "workers", None)
        args.workers = asked or stage["workers"]
        if stage.get("wall_clock_s") and args.workers != stage["workers"]:
            raise SystemExit(
                f"{name} sizes its token budgets for workers={stage['workers']}; "
                f"running it at workers={args.workers} would put the arms back "
                "out of wall-clock match. Re-size for that tenancy or drop the flag.")
        _stage_launch(args, stage)
    return run


cmd_matched20 = _stage_cmd("matched20")
cmd_bs8 = _stage_cmd("bs8")
cmd_matched32 = _stage_cmd("matched32")
cmd_ratio32 = _stage_cmd("ratio32")
cmd_swa32 = _stage_cmd("swa32")
cmd_swa2k = _stage_cmd("swa2k")
cmd_ctx2048 = _stage_cmd("ctx2048")
cmd_wallclock32 = _stage_cmd("wallclock32")


def _launch_blocking(args, stage: dict) -> None:
    apply_isolate(stage)
    args.out = str(stage["out"])
    args.workers = stage["workers"]
    args.gpus = max(1, int(getattr(args, "gpus", 1) or 1))
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


# ---------------------------------------------------------------------------
# E12+E13+E14: the whole sliding-window board, one GPU, in order.
# ---------------------------------------------------------------------------
# Deliberately NOT added to ISOLATE_SEQUENCE: that command has its own three
# stages and a documented history of being silently enlarged.
SWA_BOARD_PHASES: tuple[str, ...] = (
    "probe", "swa32", "swa2k", "mqar-calibrate", "mqar-grid")
# Batches the calibration pass tries, smallest first. E8 calibrated 256 at
# sequence 15; a 511-token cell holds 34x more tokens per row, so the batch that
# fits and the batch that trains are both open questions there -- hence a sweep
# rather than a constant.
MQAR_CALIB_BATCHES = (64, 128, 256, 512)
MQAR_OUT = Path("nanolab/out/mqar_e16")


def _swa_path_report(cfg, device, batch) -> str:
    """Time the dense and chunked SWA paths at this stage's real shape.

    The auto rule in `swa_auto_chunk` was fitted on Apple silicon; this is the
    same measurement on the box that will actually be billed, so the operator
    can pin SWA_CHUNK from data instead of inheriting someone else's hardware.
    """
    import time
    from .mixers import sliding_window_mask, swa_chunked, swa_auto_chunk
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    H, D, T = cfg.n_head, cfg.head_dim, cfg.block_size
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    q, k, v = (torch.randn(batch, H, T, D, device=dev, dtype=dt, requires_grad=True)
               for _ in range(3))
    mask = sliding_window_mask(T, cfg.swa_window, cfg.swa_sinks, dev, {})
    auto = swa_auto_chunk(T, cfg.swa_window)

    def timed(fn, n=5):
        for _ in range(2):
            fn().sum().backward()
        _sync(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            fn().sum().backward()
        _sync(dev)
        return (time.perf_counter() - t0) / n * 1000

    dense = timed(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
    best, bc = dense, 0
    for c in sorted({256, max(256, cfg.swa_window), auto} - {0}):
        if c >= T:
            continue
        t = timed(lambda c=c: swa_chunked(q, k, v, cfg.swa_window, cfg.swa_sinks,
                                          1.0 / math.sqrt(D), c))
        if t < best:
            best, bc = t, c
    note = "" if bc == auto else f"  <- auto picked {auto}; pin SWA_CHUNK={bc}"
    return (f"dense {dense:.1f}ms, best chunk {bc or 'dense'} {best:.1f}ms "
            f"({dense / best:.2f}x){note}")


def _sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def swa_backend_report_only(cfg, device, batch):
    """`assert_swa_backend_is_viable` without the refusal, for SWA_ALLOW_MATH=1."""
    from .mixers import swa_sdpa_backends
    return swa_sdpa_backends(cfg, device, batch)


def _mqar(argv: list[str]) -> None:
    """Run the MQAR suite as a subprocess.

    Subprocess, not import: `mqar_suite` imports ARMS from this module, so an
    import here would be circular -- and a multi-hour phase is better off with
    its own address space anyway.
    """
    cmd = [sys.executable, "-u", "-m", "nanolab.mqar_suite", *argv]
    print(f"$ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc:
        raise SystemExit(f"mqar phase failed (exit {rc}); board not complete")


def cmd_swaboard(args) -> None:
    """E12 -> E13 -> E14 on one GPU, in dependency order.

    Phases are resumable (`--from`) and individually runnable (`--only`) because
    the grid is hours long and a failure in phase 4 should not re-bill phases
    1-3. Every phase is idempotent on its own artifacts: the CE stages skip runs
    already marked done, and the MQAR ledger skips runs already recorded.
    """
    want = list(SWA_BOARD_PHASES)
    if args.only:
        if args.only not in SWA_BOARD_PHASES:
            raise SystemExit(f"unknown phase {args.only!r}; have {SWA_BOARD_PHASES}")
        want = [args.only]
    elif args.start_from:
        if args.start_from not in SWA_BOARD_PHASES:
            raise SystemExit(f"unknown phase {args.start_from!r}; have {SWA_BOARD_PHASES}")
        want = list(SWA_BOARD_PHASES[SWA_BOARD_PHASES.index(args.start_from):])
    print(f"=== swaboard phases: {' -> '.join(want)} ===\n", flush=True)

    for phase in want:
        print(f"\n########## phase {phase} ##########", flush=True)
        if phase == "probe":
            # Preflight, and it is not a formality: on CUDA an explicit attn_mask
            # cannot use the flash kernel, and if SDPA falls back to the math path
            # at context 2048 the score matrix is materialised and the stage OOMs.
            # Better to learn that in 15 minutes than four hours in.
            from .mixers import assert_swa_backend_is_viable
            for block, batches in ((512, "32"), (2048, "8")):
                os.environ["CROSSOVER_BLOCK"] = str(block)
                bs = int(batches)
                for arm in ("swa_w64", "swa_w512"):
                    cfg = job_config({"id": "pf", "arm": arm, "mixer": "swa",
                                      "layer_mixers": "", "seed": 1337},
                                     Path("nanolab/out"), smoke=True)
                    cfg.block_size = block
                    rep = (swa_backend_report_only(cfg, args.device, bs)
                           if os.environ.get("SWA_ALLOW_MATH")
                           else assert_swa_backend_is_viable(cfg, args.device, bs))
                    served = [n for n, r in rep.items() if r["ok"]]
                    print(f"  sdpa ctx{block} {arm} bs{bs}: {', '.join(served) or 'NONE'}",
                          flush=True)
                    print(f"  swa  ctx{block} {arm}: {_swa_path_report(cfg, args.device, bs)}",
                          flush=True)
                cmd_probe(argparse.Namespace(
                    out=str(Path("nanolab/out") / f"swa_probe_ctx{block}"),
                    batches=batches, steps=args.probe_steps,
                    mixers="attention,swa_w64,swa_w512"))
            os.environ.pop("CROSSOVER_BLOCK", None)
        elif phase in ("swa32", "swa2k"):
            ns = argparse.Namespace(**vars(args))
            ns.gpus = args.gpus or 1
            _launch_blocking(ns, stage_by_name(phase))
        elif phase == "mqar-calibrate":
            _mqar(["--out", str(MQAR_OUT), "--device", args.device,
                   "--cells", ",".join(str(p) for p in _mqar_cells()),
                   "--calibrate", ",".join(str(b) for b in MQAR_CALIB_BATCHES),
                   "--steps", str(args.mqar_steps)])
        elif phase == "mqar-grid":
            calib = MQAR_OUT / "calibration.json"
            if not calib.exists():
                raise SystemExit(
                    f"{calib} missing: run the mqar-calibrate phase first. One "
                    "batch does not serve every sequence length, and an "
                    "uncalibrated cell reports 'not solved' for a model that was "
                    "merely untrainable at that batch.")
            by_cell = json.loads(calib.read_text(encoding="utf-8"))
            _mqar(["--out", str(MQAR_OUT), "--device", args.device,
                   "--cells", ",".join(str(p) for p in _mqar_cells()),
                   "--batch-by-cell", json.dumps(by_cell),
                   "--arms", ",".join(_mqar_arms()),
                   "--seeds", str(args.mqar_seeds),
                   "--workers", str(args.mqar_workers),
                   "--gpus", str(args.gpus or 1),
                   "--steps", str(args.mqar_steps)])
    print("\n=== swaboard complete ===")
    print("  CE boards : nanolab/out/crossover50m_swa32, .../crossover50m_swa2k")
    print(f"  recall    : {MQAR_OUT}/runs.jsonl")


def _mqar_cells() -> tuple[int, ...]:
    from .mqar_suite import E16_PAIRS
    return E16_PAIRS


def _mqar_arms() -> tuple[str, ...]:
    from .mqar_suite import E16_ARMS
    return E16_ARMS


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
    for name in ISOLATE_SEQUENCE:
        _launch_blocking(ns, stage_by_name(name))
    print("isolates complete")


def _marker_window(tokens: list[float], target: float) -> float:
    """How far from a marker an eval may sit and still be a reading of it.

    One eval interval. The marker set is inherited from suite 14 and is
    approximate by construction; an eval cadence that steps in whole batches
    will rarely land on a round number, and blanking those cells would throw
    away real measurements. What it must still reject is an arm whose curve
    ended well short of the marker.
    """
    if len(tokens) < 2:
        return MARKER_TOLERANCE * target
    gaps = [b - a for a, b in zip(tokens, tokens[1:]) if b > a]
    if not gaps:
        return MARKER_TOLERANCE * target
    return max(statistics.median(gaps), MARKER_TOLERANCE * target)


def cmd_table(args) -> None:
    out_root = Path(args.out)
    rec_path = out_root / "recipe.json"
    if rec_path.exists():
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rec = {}
        if rec.get("budget_by_arm"):
            raise SystemExit(
                f"{out_root} matches arms on WALL CLOCK, so its arms stop at "
                "different token counts and a token-grid table cannot compare "
                "them -- the shared columns would be reading different amounts "
                "of training.\n  use: python -m nanolab.crossover_replicate "
                f"wcboard --out {out_root}")
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
            # An arm that never reached the marker has no value there. Taking
            # its nearest eval printed a short arm's FINAL loss under a column
            # header it never trained to -- a fabricated cell that looked like
            # a measurement.
            idx = min(range(len(payload["tokens"])),
                      key=lambda i: abs(payload["tokens"][i] - target))
            # An arm that never reached the marker has no value there. Taking its
            # nearest eval printed a short arm's FINAL loss under a column header
            # it never trained to -- a fabricated cell that read as a measurement.
            # The window is one eval interval: a grid that steps 0.819M at a time
            # answers a 0.8M marker with its 0.836M eval and cannot do better,
            # whereas an arm that stopped at 18.9M is not answering 50M at all.
            if abs(payload["tokens"][idx] - target) > _marker_window(
                    payload["tokens"], target):
                cells.append("--")
                continue
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


def _final_checkpoint_val(run_dir: Path) -> float | None:
    """Loss at the end of the schedule, recovered from ``final.pt``.

    Runs from before ``final_val`` was logged still carry the number: train.py
    has always written the end-of-schedule eval into the final checkpoint. Read
    it rather than reruning 4 GPU-hours, and rather than silently substituting
    ``best_val``.
    """
    fp = Path(run_dir) / "final.pt"
    if not fp.exists():
        return None
    try:
        import torch
    except ImportError:
        return None
    try:
        blob = torch.load(fp, map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        return None
    val = blob.get("val_loss")
    return None if val is None else float(val)


def _final_by_seed(out_root: Path) -> dict[str, dict[str, float]]:
    """{arm: {seed: final_val}} -- the loss at the end of each arm's schedule.

    Every arm anneals its own cosine over its own budget, so the end of the
    curve is where the arms are comparable. Deliberately NOT ``best_val``: that
    is a minimum over however many evals fired, and a minimum over more draws
    sits lower. In a wall-clock suite the fast arm takes more steps and so gets
    more draws, which biases the board toward exactly the arm the design is
    trying to measure. Measured on the 2026-08-27 suite the gap between
    best_val and final_val ran -0.0133 for attention (89 evals) against -0.0032
    for hybrid_gdn_bookend (24) -- monotone in eval count, and pointing the
    same way as the throughput advantage.

    Prefers the logged ``final_val``; falls back to ``final.pt`` for runs that
    predate that field. Raises rather than falling back to ``best_val``.
    """
    per: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for mp in sorted(Path(out_root).glob("*/metrics.jsonl")):
        name = mp.parent.name
        if "_s" not in name:
            continue
        arm = name.split("_s")[0].split("_", 1)[-1]
        seed = name.rsplit("_s", 1)[1]
        val = None
        for line in mp.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "done" and rec.get("final_val") is not None:
                val = float(rec["final_val"])
        if val is None:
            val = _final_checkpoint_val(mp.parent)
        if val is None:
            missing.append(name)
            continue
        per.setdefault(arm, {})[seed] = val
    if missing:
        raise SystemExit(
            f"{len(missing)} run(s) record no end-of-schedule loss and have no "
            f"final.pt to recover it from: {', '.join(sorted(missing)[:5])}"
            + (" ..." if len(missing) > 5 else "")
            + "\nRefusing to substitute best_val: it is a minimum over an "
              "arm-dependent number of evals and would bias the board toward "
              "whichever arm ran the most steps.")
    return per


def cmd_wcboard(args) -> None:
    """Final loss at equal wall clock, with the clock itself verified first."""
    out_root = Path(args.out)
    rec = {}
    if (out_root / "recipe.json").exists():
        rec = json.loads((out_root / "recipe.json").read_text(encoding="utf-8"))
    target = args.seconds or rec.get("wall_clock_s") or WALLCLOCK_SECONDS

    v = verify_wallclock(out_root, target)
    print(f"wall-clock check against {target:.0f}s target")
    for arm, el in sorted(v["arms"].items(), key=lambda kv: kv[1]):
        off = (el - target) / target
        print(f"  {arm:<24}{el:>9.1f}s  {off:+6.1%}  n={v['n'][arm]}")
    if v["spread"]:
        print(f"  spread {v['spread']:.2f}x")
    if not v["ok"]:
        msg = (f"REFUSING to emit a wall-clock board: {v['reason']}.\n"
               "  Equal wall clock is the claim; these runs did not train for "
               "equal wall clock, so the board would not mean what it says.\n"
               "  Re-size budgets from rates measured at this tenancy, or pass "
               "--allow-unmatched to print it as a diagnostic only.")
        if not args.allow_unmatched:
            raise SystemExit(msg)
        print(f"\n!! {msg}\n")

    per = _final_by_seed(out_root)
    if not per:
        raise SystemExit(f"no finished runs under {out_root}")
    print(f"\nfinal loss at equal wall clock ({target:.0f}s)")
    ranked = []
    for arm, by_seed in per.items():
        mu, lo, hi = mean_ci(list(by_seed.values()))
        ranked.append((mu, lo, hi, arm, by_seed))
    ranked.sort()
    for i, (mu, lo, hi, arm, by_seed) in enumerate(ranked, 1):
        ci = f"[{lo:.4f},{hi:.4f}]" if lo is not None else "[n=1, no interval]"
        print(f"  {i}. {arm:<24}{mu:.4f}  {ci}  n={len(by_seed)}")

    print("\npaired per-seed sign tests")
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            a, da = ranked[i][3], ranked[i][4]
            b, db = ranked[j][3], ranked[j][4]
            shared = sorted(set(da) & set(db))
            if not shared:
                print(f"  {a} vs {b}: no shared seeds")
                continue
            wins = sum(1 for sd in shared if da[sd] < db[sd])
            delta = statistics.mean(db[sd] - da[sd] for sd in shared)
            print(f"  {a:<24} beats {b:<24} "
                  f"delta={delta:+.4f}  {wins}/{len(shared)} seeds")


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


def cmd_measure_peak(args) -> None:
    name = live_device_name() or "<none detected>"
    tabled, key = device_peak_flops(name)
    got = measure_dense_bf16(args.n, args.iters)
    print(f"device      : {name}")
    print(f"measured    : {got/1e12:.1f} TFLOP/s dense bf16 ({args.n}^3 matmul)")
    if tabled:
        print(f"tabled ({key}): {tabled/1e12:.1f} TFLOP/s  -> achieved "
              f"{got/tabled*100:.1f}% of peak")
        if got > tabled:
            print("  MEASURED EXCEEDS THE TABLE, which is impossible: the table "
                  "row is wrong, exactly as the GH200 row once was.")
    else:
        print("tabled      : unknown device, no row")
    print(f"\nexport PEAK_FLOPS={tabled or got:.0f}")


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
            spec = next((a for a in ARMS if a.name == mixer), None)
            job = {"id": f"probe_{mixer}_bs{bs}", "arm": mixer,
                   "mixer": spec.mixer if spec else mixer,
                   "layer_mixers": spec.layer_mixers if spec else "",
                   "seed": 1337}
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
            mfu = _mfu_from_toks(job["mixer"], tok_s, cfg)
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
    from .model import mixer_flops_per_token
    # Delegated, not reimplemented: this used to hardcode `attention`/`mla` and
    # charge every other mixer ZERO attention FLOPs, so adding a mixer produced
    # a quietly wrong MFU here and a correct one in model.py.
    flops = 6 * cfg.estimate_params() + mixer_flops_per_token(cfg)
    # NaN, not 0.0 and not a guessed peak, when the device is unknown: a run
    # command resolves PEAK_FLOPS strictly before any job starts, so reaching
    # here without one means nobody established a device. `nan%` in the table is
    # unmistakable; `0.0%` reads as a slow arm and a guess reads as a fast one.
    peak = resolve_peak_flops(strict=False)
    return (flops * tok_s) / peak if peak else float("nan")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=str(DEFAULT_OUT))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", parents=[common])
    sm = sub.add_parser("smoke", parents=[common])
    sm.add_argument("--arms", default="attention,mingru",
                    help="comma-separated arm names to smoke (40 steps each)")
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--arm", required=True, choices=[a.name for a in ARMS])
    run.add_argument("--seed", type=int, default=SEEDS[0])
    run.add_argument("--force", action="store_true")
    run.add_argument("--smoke", action="store_true")
    launch = sub.add_parser("launch", parents=[common])
    launch.add_argument("--workers", type=int, default=1,
                        help="jobs per GPU (tenancy), NOT total processes")
    launch.add_argument("--gpus", type=int, default=0,
                        help="spread over this many GPUs; total processes are "
                             "gpus*workers. 0 = CROSSOVER_GPUS or 1.")
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
    mp = sub.add_parser("measure-peak", parents=[common],
                        help="achieved dense bf16 FLOP/s on this box")
    mp.add_argument("--n", type=int, default=8192)
    mp.add_argument("--iters", type=int, default=8)
    probe = sub.add_parser("probe", parents=[common])
    probe.add_argument("--batches", default="8,32,64,96,128")
    probe.add_argument("--mixers", default="attention,mingru,gdn,mamba2")
    probe.add_argument("--steps", type=int, default=30)
    sub.add_parser("plot", parents=[common])
    sub.add_parser("table", parents=[common])
    wc = sub.add_parser("wcboard", parents=[common],
                        help="E11 p2 board: final loss at equal wall clock")
    wc.add_argument("--seconds", type=float, default=None,
                    help="target wall clock; defaults to the suite recipe")
    wc.add_argument("--allow-unmatched", action="store_true",
                    help="print the board even when the clock did not match "
                         "(diagnostic only, never for publication)")
    locked = sub.add_parser("locked20", parents=[common],
                            help="redirects to matched20; short-cosine artifacts stay put")
    locked.add_argument("--workers", type=int, default=2)
    locked.add_argument("--detach", action="store_true")
    for name, help_txt in (
        ("matched20", "attn vs minGRU, 20M stop, 50M cosine, bs32, n=5"),
        ("bs8", "attn vs minGRU, suite-14 8.192M tokens, bs8, n=5"),
        ("matched32", "8 drifted arms, 50M, bs32, eval_iters=20, n=5"),
        ("ratio32", "E10: 4 minGRU hybrid ratios/placements, 50M, bs32, n=5"),
        ("swa32", "E12: SWA(64/128/256, 4) + sink ablation, 50M, bs32, n=5"),
        ("swa2k", "E15: the SWA arms at context 2048, 50M, bs8, n=5"),
        ("ctx2048", "E9: 5 families at context 2048, 50M, bs8, n=5"),
        ("wallclock32", "E11 p2: top-4 arms matched on WALL CLOCK, own cosine, n=5"),
    ):
        sp = sub.add_parser(name, parents=[common], help=help_txt)
        # No default: a stage carries its own tenancy, and "unspecified" must be
        # distinguishable from an explicit flag so the wall-clock guard can tell
        # an operator override from its own default.
        sp.add_argument("--workers", type=int, default=None)
        sp.add_argument("--detach", action="store_true")
    board = sub.add_parser(
        "swaboard", parents=[common],
        help="E12+E13+E14: the whole sliding-window board on one GPU, in order")
    board.add_argument("--only", default="",
                       help=f"run one phase only; one of {', '.join(SWA_BOARD_PHASES)}")
    board.add_argument("--from", dest="start_from", default="",
                       help="resume from this phase onward (phases are idempotent)")
    board.add_argument("--workers", type=int, default=None)
    board.add_argument("--detach", action="store_true")
    board.add_argument("--device", default="cuda")
    board.add_argument("--gpus", type=int, default=0,
                       help="spread every phase over this many GPUs (0 = 1). "
                            "The board is 315 independent runs, so this scales "
                            "very nearly linearly where tenancy does not.")
    board.add_argument("--probe-steps", type=int, default=30)
    board.add_argument("--mqar-seeds", type=int, default=15)
    board.add_argument("--mqar-workers", type=int, default=4,
                       help="MQAR runs are ~9.5M-param models that leave the GPU "
                            "mostly idle one at a time; this is the grid's main "
                            "cost lever. Lower it if the probe reports tight memory.")
    board.add_argument("--mqar-steps", type=int, default=3000)
    iso = sub.add_parser("isolates", parents=[common],
                         help="run matched20, then bs8, then matched32")
    iso.add_argument("--wait", action="store_true",
                     help="run in the foreground (used by the detached supervisor)")
    return p


def main():
    args = build_parser().parse_args()
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
        "measure-peak": cmd_measure_peak,
        "plot": cmd_plot,
        "table": cmd_table,
        "locked20": cmd_locked20,
        "matched20": cmd_matched20,
        "bs8": cmd_bs8,
        "matched32": cmd_matched32,
        "ratio32": cmd_ratio32,
        "swa32": cmd_swa32,
        "swa2k": cmd_swa2k,
        "ctx2048": cmd_ctx2048,
        "wallclock32": cmd_wallclock32,
        "isolates": cmd_isolates,
        "swaboard": cmd_swaboard,
        "wcboard": cmd_wcboard,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
