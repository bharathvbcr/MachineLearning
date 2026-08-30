#!/usr/bin/env python3
"""GPU bundle: the outstanding rented-GPU suites, as one resumable runner.

Covers the experiments the paper names but has not run:

  E1  The parametrization arm specified in PAPER section 8.4, at all THREE
      recipes its pre-registered readouts are statements about:
        e1_proxy        proxy-width matrix-LR sweep, PER ARM, 5 seeds, to tune
        e1_sp_rerun     SP cells of the 2x2 at the s24 recipe, re-run on THIS box
        e1_mup          muP cells of the 2x2 at s24, at the transferred LR
        e1_mup_basin    s24 muP at 0.25/0.5/2/4x that LR -- verifies the transfer
        e1_sp_sched20   SP at the s23 recipe (20M cosine)   -- readouts 2 and 3
        e1_mup_sched20  muP at the s23 recipe               -- readouts 2 and 3
        e1_sp_bs8       SP at the s25 recipe (batch 8)      -- readout 4
        e1_mup_bs8      muP at the s25 recipe               -- readout 4
        e1_perlayer_sp  per-layer standard-parametrization prescription
        e1_embed_lr     embedding-LR-only ablation
  E2  Suite 26 never reran attention/minGRU at 50M matched batch 32; its first and
      eighth board rows are suite 22's sample, capping the combined ranking at
      Medium-High. Ten jobs close it. GH200-gated -- see below.
  D10 OPT-IN (--with-d10). Suite 20's horizon claim is withdrawn: run128m_20k is
      eight resumed segments with a broken token counter, and its LR moves with
      its horizon. A matched TRIO of pairs (10k and 20k at the SAME learning rate,
      uninterrupted, 3 seeds) is what the claim needs. This is NOT a reproduction
      of suite 20 -- that would need the original RTX 3070 Ti.

WHAT CHANGED, AND WHY IT HAD TO. The earlier matrix ran every E1 cell at the s24
recipe only, at one proxy seed, at one transferred learning rate, and would have
produced a result that could not be claimed:

  1. Three of the four pre-registered readouts are comparisons against the s23
     (20M cosine) and s25 (batch 8) recipes, which had no cells. The run answered
     one readout of four and the other three would have read "not measured."
  2. The proxy sweep decided the transferred LR from ONE seed. It is the upstream
     dependency of every muP cell, so a wrong value mis-tunes both arms on the
     exact axis section 8.4 exists to control -- the D7 failure one axis over,
     whose own record is "three times we drew a conclusion from two seeds that a
     third seed overturned."
  3. Each muP arm was measured at a SINGLE learning rate. Section 8.3's mechanism
     is offset flat basins: two arms each measured at one point can sit on opposite
     sides of their own optima, and the ordering that falls out is an artifact of
     where they were measured. That is the defect this arm exists to remove, and
     the design reproduced it with a different number.
  4. muP transfer from width 256 to 768 was asserted and never measured, though
     the rule it rests on (optim.py: matrix_lr / width_mult) is the *Adam* muP
     rule applied to a Muon group whose update is already orthogonalized and
     RMS-matched -- a rule the Muon literature does not agree with.

WHY e1_sp_rerun EXISTS. Section 8.4's 2x2 puts muP cells against suite 24's SP
cells. Suite 24 ran on a GH200. On any other box the 2x2 is confounded by hardware
and proves nothing -- PAPER section 7.1 refuses exactly this comparison, because the
same architecture pair differs by ~0.18-0.3 nats at matched token markers across two
GPUs. So unless this box IS a GH200, all four cells must be measured here.
``--sp-cells suite24`` drops the re-run and now REQUIRES a GH200 by device name.

WHY E2 IS GH200-GATED. Its cells are merged into suite 26's published board, whose
other eight rows were measured on a Lambda GH200. Filling that board's two
`source: suite22` rows from an A100 replaces a same-box caveat with a
cross-hardware one, leaving the board worse than it found it. Preflight fails
closed; --allow-cross-hardware-board records the caveat and proceeds.

Cadence comes from ``crossover_replicate.scale_to_token_budget`` so eval markers line
up with suites 22-26 and loss-vs-tokens curves remain comparable.

Usage:
  python3 scripts/gpu_bundle.py --plan          # the matrix, with blockers, no work
  python3 scripts/gpu_bundle.py --cost          # GPU-hours and $ , derived not typed
  python3 scripts/gpu_bundle.py --preflight     # gate the box before it bills
  python3 scripts/gpu_bundle.py --smoke         # 40-step check, ISOLATED from the matrix
  python3 scripts/gpu_bundle.py --only e1_proxy # tune first; everything muP is blocked
  python3 scripts/gpu_bundle.py --workers 1     # everything else
  python3 scripts/gpu_bundle.py --report        # read the ledger, no work
  python3 scripts/gpu_bundle.py --analyse       # crossing tokens + the 8.4 readouts
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanolab.config import build_config  # noqa: E402
from nanolab.crossover_replicate import (  # noqa: E402
    SEEDS, TOKEN_BUDGET, LOCKED20_TOKEN_BUDGET, SUITE14_TOKEN_BUDGET,
    all_crossover_tokens, mean_ci, scale_to_token_budget,
)

PRESET = "crossover50m"
OUT_ROOT = ROOT / "nanolab/out/gpu_bundle"
# Smoke runs land in their own subtree. They are 40-step runs that write a real
# `done` record with a real best_val; sharing a directory with the matrix meant a
# documented "--smoke first" would mark one job per suite complete and the full
# run would skip it, publishing 40-step numbers as measurements.
SMOKE_ROOT = OUT_ROOT / "_smoke"
ARCHIVE_ROOT = OUT_ROOT / "_archived"
TRANSFER = OUT_ROOT / "transfer.json"
# The basin publishes an ANCHOR the way the proxy publishes a TRANSFER: the
# multiplier, per arm, at which the target-width curve actually bottoms out. The
# transfer was the hypothesis; the anchor is the measurement. Both are written only
# when their optimum is bracketed AND sign-consistent, so a downstream suite can
# never inherit a direction mistaken for an optimum.
ANCHOR = OUT_ROOT / "anchor.json"

# The three recipes section 8.4's readouts are read against. Each is the exact
# recipe of a published suite, so a muP cell here and an SP cell there differ in
# the parametrization and nothing else.
#
#   S24  suite 24  batch 32, 20M stop, 50M cosine horizon   -> late crossing 12.34M
#   S23  suite 23  batch 32, 20M stop, 20M cosine horizon   -> late crossing 14.58M
#   S25  suite 25  batch  8, 8.19M stop, own cosine         -> NO crossing at all
#
# S23 and S25 were absent from this matrix, and three of the four pre-registered
# readouts are statements about them: rows 2 and 3 compare the 20M cosine against
# the 50M cosine *under muP*, and row 4 asks whether the batch-8 arm develops a
# crossing *under muP*. With only S24 cells the run answers one readout of four.
S24 = dict(batch=32, budget=LOCKED20_TOKEN_BUDGET, horizon=TOKEN_BUDGET, eval_iters=20)
S23 = dict(batch=32, budget=LOCKED20_TOKEN_BUDGET, horizon=None, eval_iters=20)
S25 = dict(batch=8, budget=SUITE14_TOKEN_BUDGET, horizon=None, eval_iters=20)
RECIPES = {"s24": S24, "s23": S23, "s25": S25}
ARMS = ("attention", "mingru")
HEAD_DIM = 64
BASE_WIDTH = 256          # cfg.mup_base_width; the proxy sweep runs here
TARGET_WIDTH = 768        # the 12L/768d target every suite uses

# The proxy sweep tunes MATRIX_LR, not lr.
#
# Under muon_ns5_adamw -- the optimizer every suite in section 4 uses --
# build_optimizers sends 2-D hidden matrices to Muon at cfg.matrix_lr and sends
# embeddings, head and scalars to AdamW at cfg.lr. An earlier version of this
# sweep varied cfg.lr, which leaves the Muon group pinned at 0.025 across every
# point: it tuned the embedding LR and left the hidden LR -- the quantity muP
# exists to transfer -- inherited and untested. (optim.py's `hidden_lr` is only
# consumed on the non-Muon paths, so under Muon muP's hidden-layer rule reaches
# the model solely through matrix_lr.)
#
# The grid spans 0.0002..0.05, a 250x range at factor-2 spacing, deliberately
# reaching far below the inherited 0.025 rather than bracketing it symmetrically.
# D7 found both optimizer finalists 6-12x above their true optima and needed four
# grid extensions to find the bottom; a symmetric +/-2x grid around an inherited
# value is how that started.
#
# EXTENDED ONCE, 2026-08-26, and the extension is part of the record rather than
# presented as the plan. The first round ran 0.0016..0.05 and found minGRU
# bracketed at 0.00625 (beating both neighbours 3/3 seeds) but attention still
# falling at the LOW edge -- 5.265662 at 0.0016 against 5.600789 at the inherited
# 0.025, monotone across the whole grid. `transfer.json` published minGRU and
# excluded attention, and the muP suites stayed blocked, which is the gate working.
#
# Both arms get the new points, not just attention. D7's own list of what made its
# rounds unequal includes "one candidate given lower learning rates than the
# other," and these two curves are compared against each other.
PROXY_MATRIX_LRS = (0.0002, 0.0004, 0.0008, 0.0016, 0.003125, 0.00625, 0.0125,
                    0.025, 0.05)

# The sweep that decides the transferred value runs at THREE seeds, not one.
#
# It is the upstream dependency of every muP cell in the bundle: a wrong value
# here mis-tunes both arms on the exact axis section 8.4 exists to control, which
# is the D7 failure one axis over. D7's own record is the argument -- "three times
# we drew a conclusion from two seeds that a third seed overturned" (PAPER 8.3) --
# and its per-cell seed spread (~0.006) was the same size as the basin it was
# resolving. Deciding an optimum from one seed at that noise level is not a
# measurement. Proxy jobs are the cheapest in the bundle, so this costs ~1 GPU-h.
#
# Raised 3 -> 5 on 2026-08-27. At three seeds the sweep located attention's
# optimum cleanly (3/3 against both neighbours) but could not locate minGRU's:
# 0.003125 won on the mean and lost a seed to 0.00625, so `--analyse` refused
# to price minGRU's inherited LR at all. That refusal is correct, and the
# instruction it printed -- "Add seeds and re-run" -- is this line.
PROXY_SEEDS = SEEDS[:5]

# muP transfer is an ASSUMPTION until it is measured at the target width.
#
# The bundle tunes matrix_lr at width 256 and transfers it to 768, and optim.py
# implements the transfer as `matrix_lr / (d_model / mup_base_width)`. That is the
# *Adam* muP hidden-layer rule, and the group it is applied to is Muon, whose
# update is orthogonalized and already carries a `max(1, m/n)**0.5` RMS-match
# factor -- the Muon literature reports width-transfer WITHOUT a 1/width factor.
# We do not change the rule here; changing a parametrization the paper documents
# on the strength of an argument is what section 8.3 is a cautionary tale about.
# We measure it instead: e1_mup_basin runs the target width at four more points
# spanning a 16x range around the transferred value, so the published 2x2 cell is
# either interior to a bracketed target-width minimum -- transfer verified, and the
# crossing token comes with an LR-sensitivity band -- or it is not, which retires
# the arm honestly instead of silently.
#
# EXTENDED 2026-08-26, and like the proxy grid the extension is part of the record.
# The first basin ran 0.25/0.5/2/4x, a 16x span chosen to bracket a 3x rule error.
# It did not bracket. Both arms improved monotonically with the multiplier and the
# minimum sat at the HIGH edge:
#
#   mult      0.25x    0.5x      1x      2x      4x      (attention, n=3, 20M tok)
#   val      5.6986  5.5332  5.2900  5.1288  5.1034   <- still falling at the edge
#
# and every muP cell was worse than the SP baseline measured beside it (4.7454).
# So the transferred value is not the target-width optimum, the arm as configured is
# mis-tuned, and the direction is the one the competing rule predicts: dividing an
# already-orthogonalized, RMS-matched Muon update by width_mult overshoots downward.
# This is the failed-transfer outcome the basin was specified to detect, and it is
# reported as a failed transfer rather than as a muP measurement.
#
# Extended to 32x (a 128x span). SP reaches its 0.025 at mult = 46.9 for attention,
# so this covers most of the way there without assuming the two parametrizations
# share an optimum -- muP also rescales the init and the logits, so they need not.
BASIN_MULTS = (0.25, 0.5, 2.0, 4.0, 8.0, 16.0, 32.0)

# The SP target-width sweep. PAPER 8.1 names the inherited global 6e-4/0.025, never
# re-tuned at these recipes, as the largest uncontrolled factor in the paper, and
# 8.2 says seed agreement cannot bound it because a wrong global LR is an error all
# five seeds share. This measures it directly, per arm, at the width and recipe
# section 4's headline result was measured at.
#
# It is also what makes the parametrization comparison fair. Setting muP at its own
# optimum against SP at an inherited value would measure tuning quality and report
# it as parametrization -- the error this arm exists to remove, committed on the
# other side of the table.
#
# Centred on the inherited 0.025 but spanning 64x rather than a symmetric +/-2x:
# D7 found both funnel finalists 6-12x off, and a symmetric grid around an
# inherited value is how that started.
# EXTENDED 2026-08-26, the third grid extension in this bundle and recorded like
# the other two. The first SP sweep ran 0.003125..0.2 and did not bracket: BOTH arms
# improved monotonically all the way down, minimum at the low edge.
#
#   matrix_lr   0.003125  0.00625   0.0125    0.025*   0.05     0.1      0.2
#   attention     4.5402   4.5753   4.6505   4.7454   4.8608   4.9834   5.1243
#   mingru        4.8162   4.8177   4.8599   4.9359   5.0526   5.1714   5.2908
#                                            *inherited
#
# So the inherited 0.025 is at least 8x too high for both arms at the target width,
# costing attention >= 0.205 nats and minGRU >= 0.120 -- unequally, and of the same
# order as the -0.227 nat gap section 4.2 reports as its headline. PAPER 8.1 named
# this as the paper's largest uncontrolled factor; this is the measurement.
#
# Extended down to 0.00039 (a 512x grid). Note where that heads: the proxy found
# attention's optimum at width 256 to be 0.0016, and if SP's optimum at width 768
# lands near there too, that is a third independent line of support for a Muon
# learning rate that does not scale with width.
SP_TARGET_LRS = (0.000390625, 0.00078125, 0.0015625, 0.003125, 0.00625, 0.0125,
                 0.025, 0.05, 0.1, 0.2)
BASIN_SEEDS = SEEDS[:3]
# The SP curve at the target width gets five, not three. It is the curve that
# PRICES the inherited 0.025 -- a section 8.4 claim -- and at three seeds it
# could not: attention's argmin was sign-consistent but minGRU's won on the mean
# and lost a seed to its neighbour, so `--analyse` refused to price minGRU at all
# and printed "Add seeds and re-run". This line is that instruction. It is kept
# separate from BASIN_SEEDS because e1_mup_basin answers a different question
# (did the transfer land) into a parametrization E13 shows is handicapped, and
# does not warrant the same spend until that is settled.
SP_BASIN_SEEDS = SEEDS[:5]

# Fields that define the recipe. After every job the written config.json is
# compared against the config the overrides actually build; a mismatch fails the
# job. Suites 22-26 verified this fingerprint per job (PAPER 3.3) and this bundle
# did not.
RECIPE_FIELDS = (
    "mixer", "layer_mixers", "seed", "n_layer", "d_model", "n_head", "head_dim",
    "batch_size", "grad_accum", "block_size", "max_steps", "lr_max_steps",
    "warmup_steps", "eval_interval", "eval_iters", "eval_train", "compile",
    "optimizer", "schedule", "lr", "matrix_lr", "mup", "mup_base_width",
    "per_layer_sp", "embed_lr_mult", "dataset", "hf_dataset", "hf_config",
)

SUITE_DOC = {
    "e1_proxy": "proxy-width matrix-LR sweep at mup_base_width, 5 seeds, locates a peak only",
    "e1_sp_rerun": "SP cells of PAPER 8.4's 2x2 at the s24 recipe, on THIS box",
    "e1_mup_spattn": "muP at the transferred LR with SP's 1/sqrt(d) attention temperature; ablates muP's 1/d rule",
    "e1_mup_sched20_spattn": "e1_mup_sched20 with SP's attention temperature; the s23 half rows 2-3 need",
    "e1_mup_bs8_spattn": "e1_mup_bs8 with SP's attention temperature; row 4 without the temperature confound",
    "e1_mup_basin_spattn": "e1_mup_basin with SP's attention temperature; re-measures whether the muP transfer lands",
    "e1_mup": "muP cells of PAPER 8.4's 2x2 at s24; matrix_lr transferred from e1_proxy",
    "e1_mup_basin": "s24 muP either side of the transferred LR -- verifies the transfer",
    "e1_mup_tuned": "s24 muP at its OWN target-width optimum, n=5 -- muP's actual answer",
    "e1_sp_basin": "s24 SP matrix-LR sweep at the TARGET width, 5 seeds -- prices the inherited LR",
    "e1_sp_sched20": "SP at the s23 recipe (20M cosine) -- readout rows 2 and 3",
    "e1_mup_sched20": "muP at the s23 recipe (20M cosine) -- readout rows 2 and 3",
    "e1_sp_bs8": "SP at the s25 recipe (batch 8) -- readout row 4",
    "e1_mup_bs8": "muP at the s25 recipe (batch 8) -- readout row 4",
    "e1_perlayer_sp": "per-layer SP (Everett et al.) -- APPROXIMATION, see optim.py caveats",
    "e1_embed_lr": "embedding-LR-only ablation (Kalra & Barkeshli)",
    "e2_matched32_50m": "suite 26's missing attention/minGRU cells at 50M / batch 32",
    "d10_horizon": "OPT-IN (--with-d10). Matched 10k vs 20k, 3 seeds; NOT a suite-20 replication",
}
SUITE_ORDER = tuple(SUITE_DOC)

# Suites whose muP cells cannot be built before e1_proxy publishes a bracketed
# optimum. Listed once so a new muP suite cannot forget to be blocked.
MUP_TRANSFER_SUITES = ("e1_mup", "e1_mup_basin", "e1_mup_tuned",
                       "e1_mup_sched20", "e1_mup_bs8", "e1_mup_spattn",
                       "e1_mup_sched20_spattn", "e1_mup_bs8_spattn",
                       "e1_mup_basin_spattn")
# Suites that run muP at its MEASURED target-width optimum rather than at the
# transferred value, and so wait on the basin as well as on the proxy.
MUP_ANCHOR_SUITES = ("e1_mup_tuned", "e1_mup_sched20", "e1_mup_bs8",
                     "e1_mup_sched20_spattn", "e1_mup_bs8_spattn")

# Every suite whose cells are merged into, or compared against, a board measured
# on the GH200 that ran suites 22-26. Running these anywhere else replaces a
# same-box caveat with a cross-hardware one, which PAPER 7.1 refuses.
GH200_REQUIRED = ("e2_matched32_50m",)

_ledger_lock = threading.Lock()


def _rel(path: Path) -> str:
    """Repo-relative for display, absolute when it is not under the repo.

    ``Path.relative_to`` raises rather than degrading, so a display path outside
    ROOT took down the caller. A formatting helper must never be able to fail a
    run that has already produced its data.
    """
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------
def heads_for(width: int) -> int:
    """Head count at a given width, holding head_dim fixed.

    The proxy previously inherited n_head=12/head_dim=64 from the preset while
    only d_model shrank, so a "256-wide" proxy carried an attention block of
    inner width 768 -- identical to the target's and 3x its own residual stream.
    That is not a narrow version of the target, and a learning rate transferred
    from it is not a muP transfer. Scaling n_head with width and holding head_dim
    is the standard convention muP's width rules assume.
    """
    if width % HEAD_DIM:
        raise ValueError(f"width {width} is not a multiple of head_dim {HEAD_DIM}")
    return width // HEAD_DIM


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
        # max_steps floors the budget, so the tokens actually trained are up to
        # one step short of the nominal figure -- 49,987,584 rather than 50M, the
        # number suites 22-26 report. Record what runs, not what was asked for.
        "token_budget": c["tokens_per_step"] * c["max_steps"],
        "token_budget_requested": budget,
        "overrides": dict(
            run_name=jid, mixer=arm, layer_mixers="", seed=seed,
            d_model=d_model, n_head=heads_for(d_model), head_dim=HEAD_DIM,
            batch_size=c["batch_size"], grad_accum=c["grad_accum"],
            block_size=c["block_size"], max_steps=c["max_steps"],
            lr_max_steps=c["lr_max_steps"], warmup_steps=c["warmup_steps"],
            eval_interval=c["eval_interval"], ckpt_interval=c["ckpt_interval"],
            log_interval=c["log_interval"], eval_train=False,
            eval_iters=eval_iters, compile=False, mem_fraction=0.0, **extra),
    }


def _tuned_lr(transfer: dict | None, arm: str) -> dict | None:
    return (transfer or {}).get("arms", {}).get(arm)


def _mup_job(suite: str, arm: str, seed: int, recipe: dict, transfer: dict | None,
             *, mult: float = 1.0, tag: str = "", anchor: dict | None = None,
             extra_cfg: dict | None = None) -> dict:
    """One muP cell, blocked unless e1_proxy published a bracketed optimum.

    Without a transfer these jobs inherit the preset's 0.025 -- an arm mis-tuned on
    the exact axis under test, which is the D7 failure one axis over -- so they are
    marked blocked and refused at launch rather than run wrong.
    """
    tuned = _tuned_lr(transfer, arm)
    anchored = (anchor or {}).get("arms", {}).get(arm) if anchor is not None else None
    if anchor is not None:
        # An anchored suite runs muP at its MEASURED target-width optimum, not at
        # the transferred value the basin showed to be wrong. It therefore waits on
        # the basin as well as on the proxy.
        if not anchored:
            j = _job(suite, arm, seed,
                     dict(mup=True, mup_base_width=BASE_WIDTH, **(extra_cfg or {})),
                     tag=tag, **recipe)
            j["blocked_on"] = ("e1_mup_basin: no located target-width optimum for "
                               f"{arm!r} in {ANCHOR.name}")
            return j
        mult = anchored["mult"]
    extra = dict(mup=True, mup_base_width=BASE_WIDTH, **(extra_cfg or {}))
    if tuned:
        extra["matrix_lr"] = tuned["matrix_lr"] * mult
    j = _job(suite, arm, seed, extra, tag=tag, **recipe)
    if tuned:
        j["transfer"] = {"matrix_lr": tuned["matrix_lr"], "mult": mult,
                         "applied": extra["matrix_lr"],
                         "source": tuned.get("source", "e1_proxy"),
                         "bracketed": tuned.get("bracketed"),
                         "anchored": bool(anchored)}
    else:
        j["blocked_on"] = ("e1_proxy: no tuned matrix_lr for "
                           f"{arm!r} in {TRANSFER.name}")
    return j


def build_matrix(sp_cells: str = "rerun", transfer: dict | None = None,
                 with_d10: bool = False, anchor: dict | None = None) -> list[dict]:
    jobs: list[dict] = []

    # --- E1a: proxy-width matrix-LR sweep, PER ARM, THREE seeds. Tune at
    # BASE_WIDTH, transfer to TARGET_WIDTH. Per arm because attention and minGRU
    # may have different optima: D7 found two optimizers with offset basins whose
    # ordering crossed over, and transferring one arm's optimum to both would
    # rebuild the unequal-tuning error that retired the funnel's champion.
    #
    # This locates a peak, it does not rank arms. Cheap by construction -- a
    # 256-wide model is ~1/9 the FLOPs of the 768-wide target -- which is why it
    # can afford the seeds that make its minimum a measurement rather than a draw.
    #
    # cfg.lr stays at the suite value. Under muP the embedding/scalar LR is
    # width-constant, so transferring it unchanged is correct by the rule; that it
    # is itself inherited rather than tuned is a stated limitation of this arm.
    for arm in ARMS:
        for mlr in PROXY_MATRIX_LRS:
            for seed in PROXY_SEEDS:
                jobs.append(_job("e1_proxy", arm, seed,
                                 dict(mup=True, mup_base_width=BASE_WIDTH, matrix_lr=mlr),
                                 d_model=BASE_WIDTH, tag=f"mlr{mlr:g}", **S24))

    # --- E1b: the SP cells of the 2x2 at the s24 recipe, on THIS box. See the
    # module docstring. Also the drift control: on the GH200 these are directly
    # comparable to suite 24's published values, so the re-run measures how far
    # the environment has moved rather than assuming it has not.
    if sp_cells == "rerun":
        for arm in ARMS:
            for seed in SEEDS:
                jobs.append(_job("e1_sp_rerun", arm, seed, {}, **S24))

    # --- E1c: the muP cells of the 2x2, at the transferred optimum. n = 5, the
    # sample every suite in section 4 uses, so the crossing tokens are comparable
    # to 22/24 without a sample-size caveat.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_mup_job("e1_mup", arm, seed, S24, transfer))

    # --- E1c-spattn: muP with SP's attention temperature. An ablation of ONE
    # muP term, run at the same transferred LR as e1_mup so the only difference
    # is the logit scale.
    #
    # muP prescribes 1/d attention logits, which is correct only when q.k grows
    # as Theta(d). This tree initialises every Linear at a fixed std=0.02, so q.k
    # grows as Theta(sqrt(d)) and 1/d leaves attention at 99.8% of uniform entropy
    # at init against SP's 89% (measured, d_model=768/head_dim=64; the assertion
    # is nanolab.tests.mup_attention_temperature_is_the_arm_asymmetric_term).
    # minGRU has no attention logits, so the term is arm-asymmetric -- the shape of
    # every NOT A VALID COMPARATOR verdict, where attention is hurt 4-7x more than
    # minGRU. e1_mup_tuned rules out an LR explanation: it re-tunes at the target
    # width and attention is still +0.35 nats. If this suite closes that gap, the
    # muP cells were measuring a broken attention temperature rather than muP.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_mup_job("e1_mup_spattn", arm, seed, S24, transfer,
                                 extra_cfg=dict(mup_sqrt_attn_scale=True)))

    # --- E1c': the target-width basin around the transferred value. Four more
    # points per arm at 0.25/0.5/2/4x, three seeds each. Together with e1_mup's
    # 1x cell this is a five-point LR curve at the TARGET width, which is the only
    # thing that can show the transfer landed: if the minimum of that curve is the
    # transferred value, muP transferred; if it is an end point, it did not, and
    # the arm is reported as a failed transfer rather than as a muP measurement.
    #
    # It answers a second question the single-point design cannot. PAPER 8.3's
    # mechanism is OFFSET FLAT BASINS: two arms each measured at one LR can sit on
    # opposite sides of their own optima and produce an ordering that is an artifact
    # of where they were measured. A crossing token that is stable across a 16x LR
    # range is a far stronger claim than one measured at a single point, and it is
    # the claim section 8.4 actually needs.
    for arm in ARMS:
        for mult in BASIN_MULTS:
            for seed in BASIN_SEEDS:
                jobs.append(_mup_job("e1_mup_basin", arm, seed, S24, transfer,
                                     mult=mult, tag=f"x{mult:g}"))

    # The same basin with SP's attention temperature. e1_mup_basin's verdict --
    # "TRANSFER MISSED: the interior minimum is at 4x (attention) / 2x (minGRU),
    # not 1.0x" -- was measured through the temperature E13 identifies, so it is
    # not evidence about muP transfer. It is the last piece of the muP story: if
    # the interior minimum moves to 1.0x once attention can actually attend, the
    # transfer lands and the miss was the temperature; if it stays out at 4x/2x,
    # muP transfer genuinely misses at this width and that stands on its own.
    for arm in ARMS:
        for mult in BASIN_MULTS:
            for seed in BASIN_SEEDS:
                jobs.append(_mup_job("e1_mup_basin_spattn", arm, seed, S24,
                                     transfer, mult=mult, tag=f"x{mult:g}",
                                     extra_cfg=dict(mup_sqrt_attn_scale=True)))

    # --- E1c-tuned: muP at its OWN target-width optimum, n = 5. This is muP's
    # actual answer. e1_mup above stays at 1x: it is the pre-registered transfer
    # cell, and the fact that the transfer MISSED is a result, kept rather than
    # overwritten by re-pointing the same job ids at a better learning rate.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_mup_job("e1_mup_tuned", arm, seed, S24, transfer,
                                 anchor=anchor if anchor is not None else {}))

    # --- E1c'': SP's own learning-rate curve at the target width. The inherited
    # value is a point on this curve and is contributed by e1_sp_rerun, so only the
    # other six are new jobs. Two things come out of it: how far PAPER 4's global
    # 0.025 sits from SP's own optimum at this width and recipe (section 8.1's
    # concern, measured rather than argued), and an SP anchor that makes the
    # parametrization comparison a comparison of parametrizations.
    inherited_lr = build_config(PRESET, {"run_name": "probe"}).matrix_lr
    for arm in ARMS:
        for lr in SP_TARGET_LRS:
            if lr == inherited_lr:
                continue            # e1_sp_rerun already measures this cell, at n=5
            for seed in SP_BASIN_SEEDS:
                jobs.append(_job("e1_sp_basin", arm, seed, dict(matrix_lr=lr),
                                 tag=f"mlr{lr:g}", **S24))

    # --- E1d: the s23 recipe (20M stop under a 20M cosine), SP and muP.
    # Readout rows 2 and 3 are statements about this recipe versus s24:
    #   row 2  "the ordering of recipes is preserved -- 20M cosine still crosses
    #           later than 50M cosine"          -> needs muP at BOTH schedules
    #   row 3  "muP crossings collapse onto a single token across schedules"
    #                                            -> needs muP at BOTH schedules
    # The SP half is the same-box control that makes the muP delta a magnitude
    # rather than a direction; suite 23's own 14.58M was measured in a different
    # session and only the paired difference survives that gap.
    if sp_cells == "rerun":
        for arm in ARMS:
            for seed in SEEDS:
                jobs.append(_job("e1_sp_sched20", arm, seed, {}, **S23))
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_mup_job("e1_mup_sched20", arm, seed, S23, transfer,
                                 anchor=anchor if anchor is not None else {}))
    # The same cell with SP's attention temperature. Rows 2 and 3 need a muP
    # crossing at BOTH schedules and e1_mup_sched20 never crossed on any seed --
    # for the reason E13 identifies. At s24 the same one-term ablation turned 0
    # crossings into 5/5 at 10.12M, so the s23 half is the other measurement those
    # rows need. If it also crosses, rows 2 and 3 become answerable.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_mup_job("e1_mup_sched20_spattn", arm, seed, S23, transfer,
                                 anchor=anchor if anchor is not None else {},
                                 extra_cfg=dict(mup_sqrt_attn_scale=True)))

    # --- E1e: the s25 recipe (batch 8), SP and muP. Readout row 4 asks whether
    # the batch-8 arm "develops a crossing under muP within 7.4M tokens" -- a
    # within-recipe comparison that needs a batch-8 muP cell, which this matrix
    # did not contain. The SP half reproduces suite 25's no-crossing result on the
    # same box, so "develops a crossing" is measured against a control rather than
    # against a number from another session.
    if sp_cells == "rerun":
        for arm in ARMS:
            for seed in SEEDS:
                jobs.append(_job("e1_sp_bs8", arm, seed, {}, **S25))
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_mup_job("e1_mup_bs8", arm, seed, S25, transfer,
                                 anchor=anchor if anchor is not None else {}))
    # Row 4 asks whether batch 8 develops a crossing under muP; e1_mup_bs8 answered
    # "no" through the same broken temperature, so the answer is not about batch 8.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_mup_job("e1_mup_bs8_spattn", arm, seed, S25, transfer,
                                 anchor=anchor if anchor is not None else {},
                                 extra_cfg=dict(mup_sqrt_attn_scale=True)))

    # --- E1f: per-layer SP (Everett et al.). See the caveats in optim.py: the
    # prescription is stated for pure Adam and our hybrid sends hidden matrices to
    # Muon, and tied embeddings prevent a separate readout rate. This arm is an
    # APPROXIMATION of their prescription and must be reported as one.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_job("e1_perlayer_sp", arm, seed,
                             dict(per_layer_sp=True, mup_base_width=BASE_WIDTH), **S24))

    # --- E1g: embedding-LR-only. Raise ONLY the embedding LR by the width ratio.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_job("e1_embed_lr", arm, seed,
                             dict(embed_lr_mult=TARGET_WIDTH / BASE_WIDTH,
                                  mup_base_width=BASE_WIDTH), **S24))

    # --- E2: suite 26's missing attention/minGRU cells at 50M, matched batch 32.
    # These are MERGED INTO a board whose other eight rows were measured on the
    # GH200 (26-matched32_lock.json), so this suite is gated on GH200 hardware --
    # measured elsewhere it would replace a same-box caveat with the cross-hardware
    # one PAPER 7.1 refuses, leaving the board worse than it found it.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_job("e2_matched32_50m", arm, seed, {},
                             batch=32, budget=TOKEN_BUDGET,
                             horizon=None, eval_iters=20))

    # --- D10: matched horizon pair, attention only, uninterrupted, THREE seeds.
    # Both at the same LR so horizon is the only variable -- the confound that made
    # suite 20 unusable was LR moving with horizon. ctx1024/bs32 matches suite 20.
    #
    # OPT-IN (--with-d10), for two reasons that are both about this bundle rather
    # than about the experiment. It is not on section 8.4's critical path, and at
    # one seed it was not claimable -- a single-seed horizon pair is the shape of
    # the suite-14 observation section 4 spent five seeds overturning. At three
    # seeds it is claimable and it is also ~3x the compute of everything else here
    # combined, in jobs no amount of parallelism shortens. Decide it separately.
    if with_d10:
        for steps, budget in (("10k", 327_680_000), ("20k", 655_360_000)):
            c = scale_to_token_budget(batch_size=32, block_size=1024, grad_accum=1,
                                      token_budget=budget, lr_horizon_tokens=budget)
            for seed in BASIN_SEEDS:
                jid = f"d10_horizon_{steps}_s{seed}"
                jobs.append({
                    "id": jid, "suite": "d10_horizon", "arm": "attention",
                    "seed": seed, "tag": steps,
                    "token_budget": 32 * 1024 * c["max_steps"],
                    "token_budget_requested": budget,
                    "overrides": dict(
                        run_name=jid, mixer="attention", layer_mixers="", seed=seed,
                        d_model=TARGET_WIDTH, n_head=heads_for(TARGET_WIDTH),
                        head_dim=HEAD_DIM, batch_size=32, grad_accum=1, block_size=1024,
                        max_steps=c["max_steps"], lr_max_steps=c["lr_max_steps"],
                        warmup_steps=c["warmup_steps"], eval_interval=c["eval_interval"],
                        ckpt_interval=c["ckpt_interval"], log_interval=c["log_interval"],
                        eval_train=False, eval_iters=20, compile=False, mem_fraction=0.0),
                })
    return jobs


# ---------------------------------------------------------------------------
# run inspection: a partial run and a finished run must never look alike
# ---------------------------------------------------------------------------
def inspect_run(d: Path) -> dict:
    """Classify a run directory from its metrics.jsonl.

    ``Logger`` opens metrics.jsonl in APPEND mode, so re-running a job that died
    mid-flight leaves one file holding two `start` records and possibly two
    `done`s -- which is precisely what made run128m_20k (gap D10) unusable, and
    what this bundle exists to avoid producing. Statuses:

      missing  no metrics.jsonl
      partial  started, never finished -- must be archived before a re-run
      suspect  more than one segment in one file -- not a single measurement
      done     exactly one start, one done, and a best_val
    """
    m = d / "metrics.jsonl"
    out = {"status": "missing", "starts": 0, "dones": 0, "best_val": None,
           "final_val": None, "tokens": None, "elapsed_s": None,
           "mean_tok_s": None, "curve": []}
    if not m.exists():
        return out
    for line in m.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = row.get("event")
        if ev == "start":
            out["starts"] += 1
        elif ev == "eval":
            if row.get("tokens") is not None and row.get("val_loss") is not None:
                out["curve"].append([int(row["tokens"]), float(row["val_loss"])])
        elif ev == "done":
            out["dones"] += 1
            out["best_val"] = row.get("best_val")
            out["tokens"] = row.get("tokens")
            out["elapsed_s"] = row.get("elapsed_s")
            out["mean_tok_s"] = row.get("mean_tok_s")
    if out["curve"]:
        out["final_val"] = out["curve"][-1][1]
    if out["starts"] > 1 or out["dones"] > 1:
        out["status"] = "suspect"
    elif out["dones"] == 1 and out["best_val"] is not None:
        out["status"] = "done"
    elif out["starts"] >= 1:
        out["status"] = "partial"
    return out


def verify_fingerprint(job: dict, d: Path) -> str | None:
    """None if the written config matches the requested recipe, else the reason.

    PAPER 3.3: the isolates "verify the recipe fingerprint on every individual job
    config before the results are read". This bundle did not, so a job whose
    config silently differed from the request would have been read as if it
    matched.
    """
    cfgp = d / "config.json"
    if not cfgp.exists():
        return "no config.json written"
    try:
        got = json.loads(cfgp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return f"unreadable config.json: {e}"
    want = build_config(PRESET, job["overrides"]).to_dict()
    bad = []
    for f in RECIPE_FIELDS:
        if f not in want:
            continue
        a, b = want[f], got.get(f, "<absent>")
        if isinstance(a, float) or isinstance(b, float):
            same = isinstance(b, (int, float)) and abs(float(a) - float(b)) <= 1e-12
        else:
            same = a == b
        if not same:
            bad.append(f"{f}: want {a!r} got {b!r}")
    return "; ".join(bad) if bad else None


def classify_result(code: int, st: dict, fp: str | None) -> tuple[str, str]:
    """('done'|'failed', reason). Every condition must hold.

    A job that exited 0 but wrote no metric, or wrote a metric under a config
    that is not the one requested, is not a measurement -- and must never read as
    one. Kept pure and separate from the worker so the decision is testable
    without running a trainer.
    """
    reasons = []
    if code != 0:
        reasons.append(f"returncode={code}")
    if st.get("status") != "done":
        reasons.append(f"run status={st.get('status')} "
                       f"({st.get('starts')} start/{st.get('dones')} done)")
    if st.get("best_val") is None:
        reasons.append("best_val absent")
    if fp:
        reasons.append(f"recipe fingerprint mismatch -- {fp}")
    return ("failed" if reasons else "done"), "; ".join(reasons)


# ---------------------------------------------------------------------------
# ledger: merge, never clobber
# ---------------------------------------------------------------------------
def ledger_path(root: Path) -> Path:
    return root / "ledger.json"


def write_ledger(root: Path, records: list[dict], started: str, meta: dict) -> None:
    """Merge `records` into any existing ledger by id.

    The previous version serialized only the jobs of the current invocation, so
    `--only e1_proxy` after `--only e1_mup` left a ledger with no trace of the
    first -- the same class of loss as gap D8, by a different mechanism.
    """
    with _ledger_lock:
        root.mkdir(parents=True, exist_ok=True)
        path = ledger_path(root)
        merged: dict[str, dict] = {}
        if path.exists():
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
                for r in prior.get("jobs", []):
                    merged[r["id"]] = r
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # a corrupt prior ledger must not block the current write
        for r in records:
            merged[r["id"]] = r
        jobs = list(merged.values())
        by_suite: dict[str, dict[str, int]] = {}
        for r in jobs:
            s = by_suite.setdefault(r.get("suite", "?"), {})
            s[r["status"]] = s.get(r["status"], 0) + 1
        payload = {
            "schema_version": 2,
            "id": "gpu-bundle",
            "started_at": started,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "suites": SUITE_DOC,
            "meta": meta,
            "jobs_total": len(jobs),
            "jobs_done": sum(1 for r in jobs if r["status"] == "done"),
            "by_suite": by_suite,
            "gpu_hours_measured": round(
                sum(r.get("elapsed_s") or 0.0 for r in jobs) / 3600.0, 3),
            "jobs": jobs,
            "note": ("Rewritten after every job and MERGED by id (gap D8). "
                     "`gpu_hours_measured` counts only jobs this ledger timed; "
                     "it is a measurement, never an estimate."),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def run_one(job: dict, root: Path, smoke: bool, device: str | None) -> tuple[int, float]:
    ov = dict(job["overrides"])
    ov["out_dir"] = str(root)
    if smoke:
        ov.update(max_steps=40, lr_max_steps=40, eval_interval=20, eval_iters=4,
                  log_interval=5, warmup_steps=5, ckpt_interval=40, batch_size=8)
    code = (
        "import json,sys;sys.path.insert(0,%r);"
        "from nanolab.config import build_config;from nanolab.train import train;"
        "train(build_config(%r, json.loads(%r)))"
        % (str(ROOT), PRESET, json.dumps(ov))
    )
    out_dir = root / ov["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    t0 = time.time()
    with (out_dir / "run.log").open("a", encoding="utf-8") as fh:
        fh.write(f"\n--- launch {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                 f"CUDA_VISIBLE_DEVICES={device} ---\n")
        fh.flush()
        p = subprocess.run([sys.executable, "-c", code], stdout=fh,
                           stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    return p.returncode, time.time() - t0


def detect_gpus() -> int:
    """Device count, asked of a child so the parent never initializes CUDA."""
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch;print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"],
            capture_output=True, text=True, timeout=180, cwd=str(ROOT))
        return int(r.stdout.strip() or 0)
    except (subprocess.SubprocessError, ValueError):
        return 0


def device_total_vram_gib() -> float:
    """Total VRAM on device 0, or 0.0 when there is no CUDA device."""
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch;print(torch.cuda.get_device_properties(0).total_memory "
             "if torch.cuda.is_available() else 0)"],
            capture_output=True, text=True, timeout=180, cwd=str(ROOT))
        return int(r.stdout.strip() or 0) / 1024 ** 3
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def job_vram_gib(job: dict) -> float:
    o = job["overrides"]
    return JOB_VRAM_GIB.get((o["mixer"], o["d_model"], o["batch_size"]),
                            JOB_VRAM_DEFAULT_GIB)


def vram_safe_workers(jobs: list[dict], total_gib: float) -> tuple[int, float]:
    """(workers that fit under the headroom, the per-job figure used).

    Sized on the HEAVIEST job in the set, not the average: the scheduler is free to
    put the expensive ones together, and it does.
    """
    if not jobs or total_gib <= 0:
        return 0, 0.0
    per = max(job_vram_gib(j) for j in jobs)
    return max(1, int(total_gib * VRAM_HEADROOM / per)), per


def device_name() -> str:
    """Device 0's model name, or "" if there is no CUDA device.

    The docstring used to say the runner "does not check that the box is that
    GH200, because it cannot." It cannot identify the individual machine, but it
    can identify the MODEL, which is what the hardware control is about -- suite
    26's board is a GH200 board, and a cell measured on an A100 does not belong in
    it whichever GH200 the original was.
    """
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch;print(torch.cuda.get_device_name(0) "
             "if torch.cuda.is_available() else '')"],
            capture_output=True, text=True, timeout=180, cwd=str(ROOT))
        return r.stdout.strip()
    except subprocess.SubprocessError:
        return ""


def is_gh200(name: str) -> bool:
    return "gh200" in name.lower()


# ---------------------------------------------------------------------------
# proxy sweep -> transfer
# ---------------------------------------------------------------------------
def proxy_points(records: list[dict]) -> dict[str, dict[float, dict[int, float]]]:
    """{arm: {matrix_lr: {seed: final_val}}} from the finished proxy cells.

    final_val at a fixed token count, not best_val. PAPER 3.2: "A best_val field
    is not a paired snapshot. Minimum-over-all-evaluations is not a ranking and is
    never reported as one." Every proxy cell stops at the same token budget, so the
    last eval IS the paired snapshot.
    """
    out: dict[str, dict[float, dict[int, float]]] = {}
    for r in records:
        if r.get("suite") != "e1_proxy" or r.get("status") != "done":
            continue
        v = r.get("final_val")
        if v is None:
            continue
        lr = float(str(r["tag"]).replace("mlr", ""))
        out.setdefault(r["arm"], {}).setdefault(lr, {})[int(r["seed"])] = float(v)
    return out


def _paired_wins(a: dict[int, float], b: dict[int, float]) -> tuple[int, int]:
    """(seeds where a < b, seeds present in both). Sign test, not a mean."""
    shared = sorted(set(a) & set(b))
    return sum(1 for sd in shared if a[sd] < b[sd]), len(shared)


def analyse_lr_curve(by_arm: dict[str, dict[float, dict[int, float]]],
                     x_name: str) -> dict:
    """Per-arm minimum of an LR curve, plus whether it is real.

    The proxy sweep, the muP target-width basin and the SP target-width basin ask
    the identical question of identically shaped data -- "where is this arm's
    minimum, and is it an optimum or an artifact?" -- so they ask it in one place.
    Three near-copies would drift, and the gates below are the whole point:

      bracketed        a boundary minimum measures "lower is better within this
                       range", not an optimum. D7 needed four grid extensions to
                       find its bottom and twice reported a boundary minimum as an
                       optimum before the next point contradicted it. Both grids in
                       this bundle have already needed extending once.

      sign_consistent  the winner must beat BOTH neighbours on every seed the two
                       cells share. A mean that wins by less than the seed spread
                       is the n=2 result PAPER 8.3 had overturned three times.
    """
    out: dict[str, dict] = {}
    for arm, by_x in sorted(by_arm.items()):
        curve = []
        for x in sorted(by_x):
            vals = [by_x[x][sd] for sd in sorted(by_x[x])]
            mean, lo, hi = mean_ci(vals)
            informative = len(vals) >= 2
            curve.append({x_name: x, "mean": round(mean, 6), "n": len(vals),
                          "ci_informative": informative,
                          "lo": round(lo, 6) if informative else None,
                          "hi": round(hi, 6) if informative else None,
                          "per_seed": {str(sd): round(v, 6)
                                       for sd, v in sorted(by_x[x].items())}})
        best = min(curve, key=lambda c: c["mean"])
        i = curve.index(best)
        interior = len(curve) > 2 and 0 < i < len(curve) - 1
        neighbours, consistent = [], interior
        for k in (i - 1, i + 1):
            if 0 <= k < len(curve):
                wins, n = _paired_wins(by_x[best[x_name]], by_x[curve[k][x_name]])
                neighbours.append({x_name: curve[k][x_name], "seeds_won": wins,
                                   "seeds_paired": n})
                if n == 0 or wins < n:
                    consistent = False
        out[arm] = {x_name: best[x_name], "final_val": best["mean"],
                    "points": len(curve), "n_seeds": best["n"],
                    "bracketed": bool(interior), "sign_consistent": bool(consistent),
                    "beats_neighbours": neighbours, "curve": curve}
    return out


def report_lr_curve(arms: dict, x_name: str, unit: str, n_grid: int) -> None:
    """Print an LR curve and say plainly when its minimum is not an optimum."""
    unbracketed, unstable = [], []
    for arm, a in sorted(arms.items()):
        print(f"  {arm}:  ({a['points']}/{n_grid} points done)")
        for c in a["curve"]:
            mark = "  <- min" if c[x_name] == a[x_name] else ""
            ci = (f"[{c['lo']:.6f}, {c['hi']:.6f}]" if c["ci_informative"]
                  else "(n=1, no interval)")
            print(f"    {x_name} {c[x_name]:<10g}{unit} {c['mean']:.6f}  {ci}"
                  f"  n={c['n']}{mark}")
        for nb in a["beats_neighbours"]:
            print(f"      vs {nb[x_name]:<10g}{unit} wins on "
                  f"{nb['seeds_won']}/{nb['seeds_paired']} seeds")
        if not a["bracketed"]:
            unbracketed.append((arm, a[x_name]))
        elif not a["sign_consistent"]:
            unstable.append((arm, a[x_name]))
    if unbracketed:
        print("\n  WARNING: boundary minimum -- the grid does not bracket the optimum:")
        for arm, x in unbracketed:
            print(f"    {arm}: best is {x_name} {x:g}{unit}, an end of the swept range")
        print("  That measures 'lower/higher is better within this range', not an")
        print("  optimum. Extend the grid past that end and re-run before anything")
        print("  downstream reads this value.")
    if unstable:
        print("\n  WARNING: the minimum is not sign-consistent against its neighbours:")
        for arm, x in unstable:
            print(f"    {arm}: {x_name} {x:g}{unit} wins on the mean but loses on a seed")
        print("  The argmin is then inside the seed spread, which is how section 8.3's")
        print("  two-seed rounds were overturned three times. Add seeds and re-run.")
    if not unbracketed and not unstable:
        print("\n  Both arms bracketed and sign-consistent against both neighbours.")


def analyse_proxy(records: list[dict]) -> dict:
    """Per-arm proxy-width optimum, via the shared LR-curve analyser."""
    out = {"base_width": BASE_WIDTH, "target_width": TARGET_WIDTH,
           "swept": list(PROXY_MATRIX_LRS), "seeds": list(PROXY_SEEDS),
           "arms": analyse_lr_curve(proxy_points(records), "matrix_lr")}
    for v in out["arms"].values():
        v["source"] = "e1_proxy"
    return out


def report_proxy(analysis: dict) -> None:
    if not analysis.get("arms"):
        return
    print("\n=== e1_proxy: matrix-LR sweep at the PROXY width, per arm "
          f"(n={len(analysis.get('seeds', []))}) ===")
    report_lr_curve(analysis["arms"], "matrix_lr", "", len(PROXY_MATRIX_LRS))


def sp_basin_points(records: list[dict]) -> dict[str, dict[float, dict[int, float]]]:
    """{arm: {matrix_lr: {seed: final_val}}} for the SP target-width sweep.

    e1_sp_rerun's cells are the inherited-LR point of this same curve -- same
    recipe, same width, same parametrization, differing only in matrix_lr -- so
    they are folded in rather than measured twice.
    """
    out: dict[str, dict[float, dict[int, float]]] = {}
    for r in records:
        if r.get("status") != "done" or r.get("final_val") is None:
            continue
        if r.get("suite") not in ("e1_sp_basin", "e1_sp_rerun"):
            continue
        lr = r.get("matrix_lr")
        if lr is None:
            lr = build_config(PRESET, {"run_name": "probe"}).matrix_lr
        out.setdefault(r["arm"], {}).setdefault(float(lr), {})[
            int(r["seed"])] = float(r["final_val"])
    return out


def analyse_sp_basin(records: list[dict]) -> dict:
    """SP's own target-width optimum, and how far the inherited value sits from it.

    PAPER 8.1 names a single global learning rate, inherited from the 3070 Ti
    suites and never re-tuned at these recipes, as the largest uncontrolled factor
    in the paper. Section 8.2 states plainly that seed agreement cannot bound it,
    because a wrong global LR is a systematic error shared by all five seeds. This
    sweep is the direct measurement: it prices the inherited 0.025 against SP's own
    optimum at the target width, per arm, the way section 8.3 priced the funnel's
    inherited learning rates against theirs.

    It also makes the parametrization comparison a fair one. Comparing muP at its
    optimum against SP at an inherited value would measure tuning quality and call
    it parametrization -- the error this whole arm exists to remove.
    """
    inherited = build_config(PRESET, {"run_name": "probe"}).matrix_lr
    arms = analyse_lr_curve(sp_basin_points(records), "matrix_lr")
    for arm, a in arms.items():
        pts = sp_basin_points(records)[arm]
        # A price against an optimum needs an optimum. With a partial sweep the
        # argmin can be the inherited point itself simply because it is the only
        # point measured, and reporting "1x its own optimum, costs +0.000000" reads
        # exactly like a completed check that found no penalty. A check that could
        # not run must not report what a check that ran and passed reports.
        if not (a["bracketed"] and a["sign_consistent"]):
            a["inherited_matrix_lr"] = inherited
            priced = _penalty_across_tied_optima(pts, a["matrix_lr"], inherited) \
                if a["bracketed"] else None
            a["inherited_penalty"] = priced or {
                "mean": None, "lo": None, "hi": None, "n": 0,
                "unavailable": ("optimum not located: "
                                + ("grid does not bracket it" if not a["bracketed"]
                                   else "minimum is inside the seed spread"))}
            continue
        if inherited in pts and a["matrix_lr"] in pts:
            best_v = [pts[a["matrix_lr"]][sd] for sd in sorted(pts[a["matrix_lr"]])]
            inh_v = [pts[inherited][sd] for sd in sorted(pts[inherited])]
            shared = sorted(set(pts[a["matrix_lr"]]) & set(pts[inherited]))
            paired = [pts[inherited][sd] - pts[a["matrix_lr"]][sd] for sd in shared]
            pen, plo, phi = mean_ci(paired) if paired else (None, None, None)
            a["inherited_matrix_lr"] = inherited
            a["inherited_penalty"] = {
                "mean": round(pen, 6) if pen is not None else None,
                "lo": round(plo, 6) if len(paired) >= 2 else None,
                "hi": round(phi, 6) if len(paired) >= 2 else None,
                "n": len(paired),
                "factor_off": round(inherited / a["matrix_lr"], 2)
                if a["matrix_lr"] else None,
                "best_mean": round(sum(best_v) / len(best_v), 6),
                "inherited_mean": round(sum(inh_v) / len(inh_v), 6)}
    return {"width": TARGET_WIDTH, "swept": list(SP_TARGET_LRS),
            "inherited_matrix_lr": inherited, "arms": arms}



# How far the price of the inherited LR may vary across statistically tied optima
# before the ambiguity stops being a rounding detail and starts being the answer.
PENALTY_ROBUST_FRAC = 0.10


def _tied_with_argmin(pts: dict, best_lr: float) -> list:
    """LRs the argmin fails to beat on every shared seed -- i.e. tied with it."""
    tied = []
    for lr, d in pts.items():
        if lr == best_lr:
            continue
        shared = sorted(set(d) & set(pts.get(best_lr, {})))
        if shared and not all(d[sd] - pts[best_lr][sd] > 0 for sd in shared):
            tied.append(lr)
    return tied


def _penalty_across_tied_optima(pts: dict, best_lr, inherited) -> dict | None:
    """Price the inherited LR when the argmin is real but unresolved.

    "Which LR is best" and "what does the inherited LR cost" are different
    questions, and a flat basin only defeats the first. On the 2026-08-27 SP
    basin minGRU's argmin sat inside the seed spread -- 0.003125 and 0.00625
    are 0.0024 apart on a 4/5 sign test and no seed count fixes a basin that
    flat -- yet the inherited 0.025 costs +0.126 against one and +0.124 against
    the other. Refusing to price it at all discards a number that is robust to
    the whole ambiguity.

    Fails closed, and the guards are the point: every tied optimum must beat the
    inherited value on EVERY seed, and their prices must agree to within
    PENALTY_ROBUST_FRAC. If the candidates disagree about the cost, then the
    unresolved argmin IS the answer and there is nothing safe to report.
    Returns None to fall back to refusal.
    """
    if best_lr is None or best_lr not in pts or inherited not in pts:
        return None
    cands = sorted({best_lr, *_tied_with_argmin(pts, best_lr)} - {inherited})
    if not cands:
        return None
    priced = []
    for c in cands:
        shared = sorted(set(pts[c]) & set(pts[inherited]))
        if len(shared) < 2:
            return None
        paired = [pts[inherited][sd] - pts[c][sd] for sd in shared]
        if not all(x > 0 for x in paired):     # not unanimously worse: fail closed
            return None
        mean, lo, hi = mean_ci(paired)
        priced.append({"matrix_lr": c, "mean": mean, "lo": lo, "hi": hi,
                       "n": len(paired)})
    means = [q["mean"] for q in priced]
    spread = max(means) - min(means)
    if min(means) <= 0 or spread > PENALTY_ROBUST_FRAC * min(means):
        return None
    # Report the smallest price of the set: the ambiguity may only cost us
    # confidence, never buy us a bigger number than the data supports.
    worst = min(priced, key=lambda q: q["mean"])
    return {
        "mean": round(worst["mean"], 6),
        "lo": round(worst["lo"], 6), "hi": round(worst["hi"], 6),
        "n": worst["n"],
        "factor_off": round(inherited / worst["matrix_lr"], 2),
        "argmin_unresolved": True,
        "candidates": [q["matrix_lr"] for q in priced],
        "factor_range": [round(inherited / max(cands), 2),
                         round(inherited / min(cands), 2)],
        "candidate_spread": round(spread, 6),
        "candidate_spread_frac": round(spread / min(means), 4),
    }


def free_gib() -> float:
    """Free space under ROOT, in GiB.

    A seam, not indirection for its own sake: preflight is tested, and reading
    the real filesystem inside a test makes the developer's spare disk part of
    the pass condition. Three preflight tests went red on a laptop with 208 GiB
    against a 213 GiB matrix while the box they gate had 3.2 TiB.
    """
    return shutil.disk_usage(ROOT).free / 1024 ** 3


def report_sp_basin(analysis: dict) -> None:
    if not analysis.get("arms"):
        return
    print(f"\n=== e1_sp_basin: matrix-LR sweep at the TARGET width, standard "
          f"parametrization ===")
    inh = analysis['inherited_matrix_lr']
    print(f"    the inherited suite value is {inh:g}")
    report_lr_curve(analysis["arms"], "matrix_lr", "", len(SP_TARGET_LRS))
    for arm, a in sorted(analysis["arms"].items()):
        pen = a.get("inherited_penalty")
        if not pen:
            continue
        if pen["mean"] is None:
            print(f"  {arm}: inherited LR not yet priced -- {pen.get('unavailable')}")
            continue
        if pen.get("argmin_unresolved"):
            f = pen["factor_range"]
            cands = " / ".join(f"{c:g}" for c in pen["candidates"])
            print(f"  {arm}: the inherited {inh:g} is {f[0]:g}-{f[1]:g}x its own "
                  f"optimum and costs AT LEAST {pen['mean']:+f} "
                  f"[{pen['lo']:+f}, {pen['hi']:+f}] nats (paired, n={pen['n']})")
            print(f"      argmin unresolved -- {cands} are tied inside the seed "
                  f"spread; each beats the inherited value on every seed and they "
                  f"price it to within {pen['candidate_spread']:.6f} nats "
                  f"({pen['candidate_spread_frac']:.1%})")
            continue
        ci = (f" [{pen['lo']:+.6f}, {pen['hi']:+.6f}]" if pen["lo"] is not None
              else " (n=1, no interval)")
        print(f"  {arm}: the inherited {a['inherited_matrix_lr']:g} is "
              f"{pen['factor_off']:g}x its own optimum {a['matrix_lr']:g} "
              f"and costs {pen['mean']:+.6f}{ci} nats (paired, n={pen['n']})")


def load_anchor() -> dict | None:
    if not ANCHOR.exists():
        return None
    try:
        return json.loads(ANCHOR.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_anchor(basin: dict) -> None:
    """Publish only arms whose target-width optimum is located.

    Same gate as the transfer, for the same reason: an unbracketed or
    seed-noise-width minimum is a direction, not an optimum, and an arm with no
    optimum has nothing to anchor to.
    """
    keep = {a: {"mult": v["best_mult"], "source": "e1_mup_basin",
                "bracketed": v["bracketed"], "sign_consistent": v["sign_consistent"]}
            for a, v in basin.get("arms", {}).items() if v.get("optimum_located")}
    if not keep:
        return
    payload = {"arms": keep, "width_mult": basin.get("width_mult"),
               "rule_verdict": basin.get("rule_verdict"),
               "generated_by": "scripts/gpu_bundle.py --only e1_mup_basin",
               "excluded": {a: "target-width optimum not located"
                            for a in basin.get("arms", {}) if a not in keep},
               "meaning": ("multiplier on the transferred base-width matrix_lr at "
                           "which the TARGET-width curve bottoms out; 1.0 would mean "
                           "the muP transfer landed")}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = ANCHOR.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, ANCHOR)
    print(f"  wrote {_rel(ANCHOR)}: "
          + ", ".join(f"{a} {v['mult']:g}x" for a, v in sorted(keep.items())))


def load_transfer() -> dict | None:
    if not TRANSFER.exists():
        return None
    try:
        return json.loads(TRANSFER.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_transfer(analysis: dict) -> None:
    """Publish only arms whose minimum is bracketed AND sign-consistent.

    Either gate failing means the sweep located a direction, not an optimum, and
    an arm with no optimum has nothing to transfer.
    """
    def _why(v: dict) -> str | None:
        if v["points"] != len(PROXY_MATRIX_LRS):
            return f"only {v['points']}/{len(PROXY_MATRIX_LRS)} grid points finished"
        if not v["bracketed"]:
            return "minimum not bracketed by the grid"
        if not v["sign_consistent"]:
            return "minimum not sign-consistent against its neighbours across seeds"
        return None

    keep = {a: v for a, v in analysis["arms"].items() if _why(v) is None}
    if not keep:
        return
    payload = dict(analysis)
    payload["arms"] = keep
    payload["generated_by"] = "scripts/gpu_bundle.py --only e1_proxy"
    payload["rule"] = ("muP transfers the tuned hidden-layer LR from the base width; "
                       "optim.py divides matrix_lr by d_model/mup_base_width, so the "
                       "value published here is the BASE-width value and is passed "
                       "through unscaled. That divisor is the Adam muP rule applied "
                       "to a Muon group and is NOT validated by this sweep -- "
                       "e1_mup_basin measures it at the target width.")
    payload["excluded"] = {a: _why(v) for a, v in analysis["arms"].items()
                           if a not in keep}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = TRANSFER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, TRANSFER)
    print(f"  wrote {_rel(TRANSFER)}: "
          + ", ".join(f"{a} matrix_lr={v['matrix_lr']:g}" for a, v in sorted(keep.items())))


# ---------------------------------------------------------------------------
# crossings: the quantity the pre-registered readouts are actually read against
# ---------------------------------------------------------------------------
# PAPER 4.2's per-seed bands for the suite 22/24 recipe under standard
# parametrization. The readouts compare against these, so they live in code and
# are cited, never retyped into a conclusion.
SUITE24_EARLY_BAND = (1.03e6, 1.09e6)
SUITE24_LATE_BAND = (11.93e6, 12.58e6)
BS8_HORIZON_TOKENS = 7.4e6      # section 8.4 readout row 4: "within 7.4M tokens"
CROSSINGS = OUT_ROOT / "crossings.json"


def crossing_tokens(records: list[dict]) -> dict:
    """Per-seed attention-vs-minGRU crossings for every (suite, tag) cell.

    Paired within a seed, then aggregated -- never the crossing of the mean curve.
    A crossing computed on averaged curves hides the per-seed spread that PAPER 4.2
    reports as its interval, and averaging first can manufacture a crossing that no
    individual seed has.
    """
    cells: dict[tuple[str, str], dict[str, dict[int, list]]] = {}
    for r in records:
        if r.get("status") != "done" or not r.get("curve"):
            continue
        if r.get("arm") not in ARMS:
            continue
        cells.setdefault((r["suite"], r.get("tag") or ""), {}) \
             .setdefault(r["arm"], {})[int(r["seed"])] = r["curve"]

    out: dict = {"generated_by": "scripts/gpu_bundle.py --analyse", "cells": {}}
    for (suite, tag), by_arm in sorted(cells.items()):
        a_runs, b_runs = by_arm.get("attention", {}), by_arm.get("mingru", {})
        seeds = sorted(set(a_runs) & set(b_runs))
        if not seeds:
            continue
        per_seed = {}
        for sd in seeds:
            a = {int(t): v for t, v in a_runs[sd]}
            b = {int(t): v for t, v in b_runs[sd]}
            shared = sorted(set(a) & set(b))
            if len(shared) < 2:
                continue
            per_seed[sd] = all_crossover_tokens(
                [float(t) for t in shared], [a[t] for t in shared], [b[t] for t in shared])
        if not per_seed:
            continue
        firsts = [v[0] for v in per_seed.values() if v]
        lasts = [v[-1] for v in per_seed.values() if v]
        cell = {"suite": suite, "tag": tag, "seeds": seeds,
                "n_seeds": len(per_seed),
                "seeds_with_a_crossing": sum(1 for v in per_seed.values() if v),
                "per_seed": {str(k): [round(x, 1) for x in v]
                             for k, v in sorted(per_seed.items())}}
        for name, xs in (("first", firsts), ("last", lasts)):
            if not xs:
                continue
            mu, lo, hi = mean_ci(xs)
            # Gap D1, one module over: `mean_ci` returns (mu, mu, mu) at n = 1, a
            # ZERO-WIDTH interval -- infinite precision for a quantity measured
            # once. `native_funnel` was corrected to yield an infinite half-width
            # for that case and to gate its tie-breakers on it; this does the same
            # rather than re-import the defect under a new name. A single-seed cell
            # reports its value and NO interval, and every readout that reads an
            # interval refuses to return a verdict from it.
            informative = len(xs) >= 2
            cell[name] = {"mean": round(mu, 1), "n": len(xs),
                          "ci_informative": informative,
                          "lo": round(lo, 1) if informative else None,
                          "hi": round(hi, 1) if informative else None,
                          "per_seed_min": round(min(xs), 1),
                          "per_seed_max": round(max(xs), 1)}
        out["cells"][f"{suite}|{tag}" if tag else suite] = cell
    return out


def analyse_basin(records: list[dict],
                  suites: tuple[str, ...] = ("e1_mup", "e1_mup_basin"),
                  label: str = "e1_mup_basin") -> dict:
    """Per-arm target-width LR curve, and whether the transfer actually landed.

    This is the readout for the fourth defect: muP transfer from width 256 to 768
    rests on `matrix_lr / width_mult` in optim.py, which is the *Adam* muP rule
    applied to a Muon group whose update is already orthogonalized and RMS-matched.
    We do not adjudicate that in code. We measure it: e1_mup's 1x cell plus
    e1_mup_basin's four points make a five-point curve at the TARGET width, and the
    transfer is verified only if the transferred value is that curve's interior
    minimum.

    If it is not, the honest report is a failed transfer, not a muP measurement --
    and the direction of the miss is itself informative, because the competing rule
    (no 1/width divisor) predicts the optimum sits at multiplier ~= width_mult.
    """
    by_arm: dict[str, dict[float, dict[int, float]]] = {}
    for r in records:
        if r.get("status") != "done" or r.get("final_val") is None:
            continue
        if r.get("suite") not in suites:
            continue
        mult = (r.get("transfer") or {}).get("mult", 1.0)
        by_arm.setdefault(r["arm"], {}).setdefault(float(mult), {})[
            int(r["seed"])] = float(r["final_val"])

    width_mult = TARGET_WIDTH / BASE_WIDTH
    out = {"width_mult": width_mult, "rule": "optim.py: matrix_lr / width_mult",
           "competing_rule_predicts_mult": width_mult, "arms": {}}
    for arm, by_mult in sorted(by_arm.items()):
        mults = sorted(by_mult)
        curve = []
        for mu_ in mults:
            vals = [by_mult[mu_][sd] for sd in sorted(by_mult[mu_])]
            mean, lo, hi = mean_ci(vals)
            curve.append({"mult": mu_, "mean": round(mean, 6), "n": len(vals),
                          "ci_informative": len(vals) >= 2,
                          "lo": round(lo, 6) if len(vals) >= 2 else None,
                          "hi": round(hi, 6) if len(vals) >= 2 else None,
                          "per_seed": {str(sd): round(v, 6)
                                       for sd, v in sorted(by_mult[mu_].items())}})
        best = min(curve, key=lambda c: c["mean"])
        i = curve.index(best)
        interior = len(curve) > 2 and 0 < i < len(curve) - 1
        # The 1x cell carries five seeds and the basin points three, so the means
        # are not measured at equal n. Locating a minimum by mean alone across
        # unequal samples, at effects that may sit inside the seed spread, is the
        # error PAPER 8.3 spends a section on. The winner must also beat both
        # neighbours on EVERY seed the two cells share.
        neighbours, consistent = [], interior
        for k in (i - 1, i + 1):
            if 0 <= k < len(curve):
                a, b = by_mult[best["mult"]], by_mult[curve[k]["mult"]]
                wins, n = _paired_wins(a, b)
                neighbours.append({"mult": curve[k]["mult"], "seeds_won": wins,
                                   "seeds_paired": n})
                if n == 0 or wins < n:
                    consistent = False
        # With the optimum located, the two candidate width rules become a
        # measurement rather than an argument. optim.py divides the tuned base-width
        # LR by width_mult, so it predicts the optimum at 1x; the Muon rule applies
        # no divisor, so it predicts width_mult x. Report how far each is off, in
        # grid steps, per arm -- a rule that is off in OPPOSITE directions on the two
        # arms is within resolution, while one off in the SAME direction on both is
        # biased, and that distinction is the whole finding.
        located = bool(interior and consistent)
        rules = None
        if located:
            rules = {
                "optim_py_divisor": {"predicts_mult": 1.0,
                                     "measured_over_predicted": round(best["mult"], 3)},
                "muon_no_divisor": {"predicts_mult": width_mult,
                                    "measured_over_predicted": round(
                                        best["mult"] / width_mult, 3)},
            }
        out["arms"][arm] = {
            "best_mult": best["mult"], "bracketed": bool(interior),
            "sign_consistent": bool(consistent), "optimum_located": located,
            "transfer_verified": bool(located and best["mult"] == 1.0),
            "rules": rules,
            "points": len(curve), "curve": curve, "beats_neighbours": neighbours,
        }

    # Which rule predicted better, across arms. A rule whose errors straddle 1.0 is
    # within grid resolution; one whose errors sit on the same side of 1.0 on every
    # arm is systematically wrong, and by how much.
    located = {a: v for a, v in out["arms"].items() if v.get("rules")}
    if len(located) >= 2:
        verdict = {}
        for rule in ("optim_py_divisor", "muon_no_divisor"):
            errs = [v["rules"][rule]["measured_over_predicted"] for v in located.values()]
            verdict[rule] = {
                "errors": errs,
                "worst_factor_off": round(max(max(errs), 1 / min(errs)), 3),
                "same_direction_on_every_arm": all(e > 1 for e in errs)
                or all(e < 1 for e in errs),
            }
        out["rule_verdict"] = verdict
    out["label"] = label
    return out


def report_basin(analysis: dict) -> None:
    if not analysis.get("arms"):
        return
    wm = analysis["width_mult"]
    print(f"\n=== {analysis.get('label', 'e1_mup_basin')}: target-width LR curve "
          f"(transferred value = 1.0x) ===")
    print(f"    optim.py applies matrix_lr / {wm:g}; the competing Muon rule applies")
    print(f"    no divisor, which would put the optimum at mult = {wm:g}x")
    for arm, a in sorted(analysis["arms"].items()):
        print(f"  {arm}:  ({a['points']} points)")
        for c in a["curve"]:
            mark = "  <- min" if c["mult"] == a["best_mult"] else ""
            ci = (f"  [{c['lo']:.6f}, {c['hi']:.6f}]" if c["ci_informative"]
                  else "  (n=1, no interval)")
            print(f"    {c['mult']:>5g}x  {c['mean']:.6f}{ci}  n={c['n']}{mark}")
        for nb in a["beats_neighbours"]:
            print(f"      vs {nb['mult']:g}x wins on {nb['seeds_won']}/"
                  f"{nb['seeds_paired']} shared seeds")
        if a.get("rules"):
            for name, r in a["rules"].items():
                print(f"      {name:<20} predicts {r['predicts_mult']:g}x, "
                      f"measured is {r['measured_over_predicted']:g}x that")
        if a["transfer_verified"]:
            print("    -> TRANSFER VERIFIED: the transferred value is the interior")
            print("       minimum, and beats both neighbours on every shared seed")
        elif a["bracketed"] and not a["sign_consistent"]:
            print(f"    -> NOT RESOLVED: the minimum at {a['best_mult']:g}x wins on the")
            print("       mean but loses to a neighbour on a seed, so it sits inside the")
            print("       seed spread. This is a flat basin, not a located optimum "
                  "(PAPER 8.3).")
        elif not a["bracketed"]:
            print(f"    -> NOT BRACKETED: minimum at {a['best_mult']:g}x, an end of the "
                  "range. Extend BASIN_MULTS past that end.")
        else:
            print(f"    -> TRANSFER MISSED: the interior minimum is at {a['best_mult']:g}x, "
                  "not 1.0x.")
            print("       Report this arm as a failed muP transfer, NOT as a muP")
            print("       measurement. The crossing tokens below are still measurements")
            print("       of THIS learning rate; they are not muP's answer.")


def report_rule_verdict(analysis: dict) -> None:
    v = analysis.get("rule_verdict")
    if not v:
        return
    print("\n  Which width rule predicted the target-width optimum:")
    for name, r in v.items():
        errs = ", ".join(f"{e:g}x" for e in r["errors"])
        bias = ("SAME direction on every arm -- systematically wrong"
                if r["same_direction_on_every_arm"]
                else "straddles 1.0 -- within grid resolution, unbiased")
        print(f"    {name:<20} off by [{errs}]  worst {r['worst_factor_off']:g}x"
              f"  ({bias})")
    print("  The grid is factor-2, so an error inside 2x is not resolvable; an error")
    print("  outside it, in the same direction on both arms, is.")


def _in_band(cell: dict | None, key: str, band: tuple[float, float]) -> bool | None:
    if not cell or key not in cell:
        return None
    return band[0] <= cell[key]["mean"] <= band[1]


def readouts(cx: dict) -> list[dict]:
    """Evaluate PAPER 8.4's four pre-registered rows against the measured cells.

    Generated from the crossings, not transcribed. Section 8.3's record is the
    reason: a hand-written conclusion survived the round that superseded it and
    asserted the opposite of the data, which is why `d7_analyze.py --check` exists.
    Any row whose cells did not run reads `unanswered` -- never `pass`. A check
    that could not run must not report what a check that ran and passed reports.
    """
    # muP's answer is muP measured at its OWN optimum. e1_mup is the transfer cell,
    # and the basin showed its learning rate to be 2-4x low; reading the readouts off
    # it would report a mis-tuned arm as muP's result, which is the error this whole
    # arm exists to remove. e1_mup_tuned is the cell, and until it runs these rows
    # read `unanswered` rather than falling back to the one that is present.
    c = cx["cells"]
    s24 = c.get("e1_mup_tuned")
    s23, s25 = c.get("e1_mup_sched20"), c.get("e1_mup_bs8")
    rows = []

    early = _in_band(s24, "first", SUITE24_EARLY_BAND)
    late = _in_band(s24, "last", SUITE24_LATE_BAND)
    # The bands are PER-SEED bands. A mean inside them computed over the subset of
    # seeds that happened to cross is not the same statement, so unanimity is part
    # of the verdict rather than a footnote under it.
    unanimous = s24 and s24["seeds_with_a_crossing"] == s24["n_seeds"]
    rows.append({
        "row": 1,
        "question": "Do both muP crossings land in the suite 22/24 per-seed bands?",
        "needs": ["e1_mup_tuned"],
        "verdict": (None if early is None or late is None
                    else bool(early and late and unanimous)),
        "reading": ("Recipe dependence is not a tuning artifact; 4.3's schedule "
                    "effect stands as stated."),
    })

    ordering = None
    if s23 and s24 and "last" in s23 and "last" in s24:
        ordering = s23["last"]["mean"] > s24["last"]["mean"]
    rows.append({
        "row": 2,
        "question": "Under muP, does the 20M cosine still cross LATER than the 50M cosine?",
        "needs": ["e1_mup_tuned", "e1_mup_sched20"],
        "verdict": ordering,
        "reading": ("The schedule effect is real; 4.3's magnitude is inflated by "
                    "mis-tuning and should be restated as an upper bound."),
    })

    # Overlap is a statement about intervals, so it is unanswerable when either
    # side has no informative one. Reporting a n=1 point comparison as a collapse
    # would be gap D1 with a new label.
    collapse = None
    if (s23 and s24 and "last" in s23 and "last" in s24
            and s23["last"]["ci_informative"] and s24["last"]["ci_informative"]):
        collapse = (s23["last"]["lo"] <= s24["last"]["hi"]
                    and s24["last"]["lo"] <= s23["last"]["hi"])
    rows.append({
        "row": 3,
        "question": "Do the muP crossings collapse onto one token across schedules?",
        "needs": ["e1_mup_tuned", "e1_mup_sched20"],
        "verdict": collapse,
        "reading": ("The schedule effect is a tuning artifact: 4.3 must be "
                    "withdrawn and 4.4's batch result re-examined."),
    })

    bs8 = None
    if s25:
        bs8 = (s25["seeds_with_a_crossing"] > 0 and "first" in s25
               and s25["first"]["mean"] <= BS8_HORIZON_TOKENS)
    rows.append({
        "row": 4,
        "question": f"Does the batch-8 arm develop a crossing within "
                    f"{BS8_HORIZON_TOKENS/1e6:g}M tokens under muP?",
        "needs": ["e1_mup_bs8"],
        "verdict": bs8,
        "reading": ("The batch effect is a tuning artifact and suite 14's original "
                    "6.6-7.4M window is partially rehabilitated."),
    })
    return rows


# muP is expected to be at worst neutral against standard parametrization at the
# base width, and better under transfer. An arm that is uniformly WORSE than its own
# SP control is not a tuned comparator -- it is a mis-specified one, and any ordering
# read off it is a property of the mis-specification. Worse still if the damage is
# ARM-DEPENDENT, because that is precisely the confound PAPER 8.1 names and this
# whole arm exists to remove: "one arm was closer to its own optimum than the other."
PARAMETRIZATION_CONTROL = {
    "e1_mup_tuned": "e1_sp_rerun", "e1_mup": "e1_sp_rerun",
    "e1_mup_spattn": "e1_sp_rerun",
    "e1_mup_sched20_spattn": "e1_sp_sched20",
    "e1_mup_bs8_spattn": "e1_sp_bs8",
    "e1_mup_sched20": "e1_sp_sched20", "e1_mup_bs8": "e1_sp_bs8",
    "e1_perlayer_sp": "e1_sp_rerun", "e1_embed_lr": "e1_sp_rerun",
}


def parametrization_health(records: list[dict]) -> dict:
    """Per-suite: is this parametrization competitive with its own SP control?"""
    import statistics
    finals: dict[str, dict[str, list[float]]] = {}
    for r in records:
        if r.get("status") != "done" or r.get("final_val") is None:
            continue
        if r.get("arm") not in ARMS or r.get("tag"):
            continue
        finals.setdefault(r["suite"], {}).setdefault(r["arm"], []).append(
            float(r["final_val"]))
    out = {}
    for suite, control in PARAMETRIZATION_CONTROL.items():
        a, b = finals.get(suite), finals.get(control)
        if not a or not b or set(a) != set(ARMS) or set(b) != set(ARMS):
            continue
        deltas = {arm: statistics.fmean(a[arm]) - statistics.fmean(b[arm])
                  for arm in ARMS}
        worse = [arm for arm, d in deltas.items() if d > 0]
        spread = max(deltas.values()) - min(deltas.values())
        out[suite] = {
            "control": control,
            "delta_vs_control": {k: round(v, 4) for k, v in deltas.items()},
            "worse_on_every_arm": len(worse) == len(ARMS),
            "damage_spread_between_arms": round(spread, 4),
            # Fairness is the spread, and it does not care about direction. The
            # rule used to require "worse on EVERY arm" as well, which let the
            # worst case through: e1_mup_bs8_spattn helps attention by 0.098 and
            # hurts minGRU by 0.099 -- spread 0.1968, wider than the 0.1730 that
            # failed -- and scored ok because one arm improved. A parametrization
            # that moves two arms in OPPOSITE directions is precisely the one
            # that manufactures an ordering, which is what this check exists to
            # catch. Uniform damage is still fine: both arms pay it equally, and
            # that is what the spread measures.
            "usable_as_comparator": spread <= 0.05,
        }
    return out


def report_parametrization_health(health: dict) -> None:
    if not health:
        return
    print("\n=== is each parametrization competitive with its own SP control? ===")
    print(f"{'suite':<20}{'control':<18}{'d attention':>13}{'d minGRU':>11}"
          f"{'spread':>9}  verdict")
    for suite, h in health.items():
        d = h["delta_vs_control"]
        v = ("ok" if h["usable_as_comparator"]
             else "NOT A VALID COMPARATOR")
        print(f"  {suite:<18}{h['control']:<18}{d['attention']:>+13.4f}"
              f"{d['mingru']:>+11.4f}{h['damage_spread_between_arms']:>9.4f}  {v}")
    bad = [s for s, h in health.items() if not h["usable_as_comparator"]]
    if bad:
        print("\n  Positive delta = WORSE than the same recipe under standard")
        print("  parametrization. A parametrization that loses on EVERY arm is")
        print("  mis-specified, not tuned; when it also loses by DIFFERENT amounts on")
        print("  the two arms, any ordering read off it is a property of that")
        print("  difference. That is the confound PAPER 8.1 names, reintroduced by the")
        print("  arm built to remove it.")
        print(f"  Affected: {', '.join(bad)}")
        print("  Do NOT read the pre-registered readouts off these cells.")


def report_crossings(records: list[dict], write: bool = True) -> int:
    cx = crossing_tokens(records)
    if not cx["cells"]:
        print("no finished attention/minGRU pairs to compute a crossing from")
        return 1
    print("=== crossing tokens (attention - minGRU sign changes, paired per seed) ===")
    print(f"{'cell':<26}{'n':>3}{'xings':>7}{'first (M)':>26}{'last (M)':>26}")
    for key, cell in cx["cells"].items():
        def fmt(k):
            if k not in cell:
                return f"{'-':>26}"
            v = cell[k]
            if not v["ci_informative"]:
                return f"{v['mean']/1e6:.2f} (n=1, no interval)".rjust(26)
            return (f"{v['mean']/1e6:>10.2f} [{v['lo']/1e6:.2f}, "
                    f"{v['hi']/1e6:.2f}]").rjust(26)
        print(f"  {key:<24}{cell['n_seeds']:>3}"
              f"{cell['seeds_with_a_crossing']:>7}{fmt('first')}{fmt('last')}")

    print("\n=== PAPER 8.4 pre-registered readouts ===")
    print("    (stated before the run; do not revise after seeing the numbers)")
    for r in readouts(cx):
        mark = {True: "YES", False: "no", None: "unanswered"}[r["verdict"]]
        print(f"  row {r['row']}  [{mark:^10}]  {r['question']}")
        if r["verdict"] is None:
            missing = [n for n in r["needs"] if n not in cx["cells"]]
            ran_but_flat = [n for n in r["needs"]
                            if n in cx["cells"]
                            and not cx["cells"][n]["seeds_with_a_crossing"]]
            print(f"              needs {', '.join(r['needs'])}")
            if missing:
                print(f"              not measured: {', '.join(missing)}")
            if ran_but_flat:
                # "not measured" and "measured, and the arms never crossed" are
                # different findings and must not print the same way. The second is
                # a RESULT -- the pre-registered table simply has no row for it.
                print(f"              measured, but NEVER CROSSED on any seed: "
                      f"{', '.join(ran_but_flat)}")
                print("              -- that is an outcome, not a missing cell, and "
                      "it is not in the")
                print("                 pre-registered table. Report it as such "
                      "rather than mapping it")
                print("                 onto the nearest row.")
        elif r["verdict"]:
            print(f"              -> {r['reading']}")
    _health = parametrization_health(records)
    report_parametrization_health(_health)
    cx["parametrization_health"] = _health
    _basin = analyse_basin(records)
    report_basin(_basin)
    # The same curve with SP's attention temperature. Reported SEPARATELY: the
    # uncorrected verdict is a real measurement of that parametrization and is
    # not overwritten, and `*_spattn` is muP-with-SP-attention rather than muP,
    # so its interior minimum answers a different question than rows 1-4 asked.
    _basin_sp = analyse_basin(records, ("e1_mup_spattn", "e1_mup_basin_spattn"),
                              "e1_mup_basin_spattn")
    report_basin(_basin_sp)
    report_rule_verdict(_basin)
    report_sp_basin(analyse_sp_basin(records))
    cx["readouts"] = readouts(cx)
    cx["basin"] = analyse_basin(records)
    cx["sp_basin"] = analyse_sp_basin(records)
    if write:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = CROSSINGS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cx, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, CROSSINGS)
        print(f"\nwrote {_rel(CROSSINGS)}")
    return 0


# ---------------------------------------------------------------------------
# cost: derived from the committed suite-22..26 runs, never typed into a doc
# ---------------------------------------------------------------------------
# Instances on the current price list, as ($/hr, GPUs). The unit that matters for
# 64 independent jobs is $/GPU-hr; the unit that matters for the CALENDAR is
# $/GPU-hr divided by per-GPU throughput, which is why the break-even ratios
# below are printed rather than a winner asserted.
# (name, $/hr, GPUs, VRAM GiB, (low, high) per-GPU throughput vs the GH200 the
# suites ran on). The ratio bracket is an ASSUMPTION, labelled as one everywhere
# it is used: GH200 carries an H100 die, so an H100 SXM5 is ~1.0x it, and A100 ->
# H100 on bf16 transformer training is commonly 1.8-2.2x, hence ~0.45-0.55.
INSTANCES = (
    # The GH200 is the reference box: suites 22-26 ran on a Lambda ParameterGolf
    # GH200, 97871 MiB HBM, aarch64 (experiment-notes/nanolab/22, 26 headers). Its
    # ratio is 1.0-1.0 by DEFINITION, not by assumption -- every measured rate in
    # this cost model was recorded on that hardware. Every other row's bracket is
    # an assumption, and the break-even table below is the assumption-free view.
    ("1x GH200 96GB", 2.29, 1, 96, (1.00, 1.00)),
    ("8x A100 40GB SXM4", 15.92, 8, 40, (0.45, 0.60)),
    ("1x A100 40GB SXM4", 1.99, 1, 40, (0.45, 0.60)),
    ("1x H100 80GB PCIe", 3.29, 1, 80, (0.70, 0.85)),
    ("4x H100 80GB SXM5", 16.36, 4, 80, (0.95, 1.05)),
    ("2x H100 80GB SXM5", 8.38, 2, 80, (0.95, 1.05)),
    ("1x H100 80GB SXM5", 4.29, 1, 80, (0.95, 1.05)),
    ("1x A10 24GB PCIe", 1.29, 1, 24, (0.20, 0.30)),
)
# The corpus suites 22-26 trained on: nanolab/data/HuggingFaceFW_fineweb-edu,
# train.bin = 995,000,000 bytes of uint16. It is part of the recipe, not merely a
# capacity requirement. The Batcher samples windows uniformly WITH replacement, so
# a 50M-token corpus and a 497.5M-token corpus give a 20M-token job two different
# training distributions -- 0.4 epochs against 0.04. Every cell here is compared
# against a published suite (the SP drift check against suite 24, E2's cells
# merged into suite 26's board), and a re-tokenized corpus of a different size is
# a second variable moving alongside the parametrization under test.
REFERENCE_CORPUS_TOKENS = 497_500_000
# Peak device memory per concurrent trainer, MEASURED on the GH200 during this
# bundle (nvidia-smi --query-compute-apps=used_memory, steady state after the first
# evals). It is NOT the 1.07/1.33 GiB weights+gradients+optimizer figure -- that is
# the part that can be computed from the parameter split, and activations plus the
# caching allocator are several times larger than it.
#
# The two arms differ enough to matter, and sizing a worker count on the cheaper one
# is how this went wrong: 4 workers chosen against attention's 17 GiB put four
# minGRU jobs on the card at 95.5 of 97.9 GiB, 97.6% full.
JOB_VRAM_GIB = {("attention", 768, 32): 17.0, ("mingru", 768, 32): 23.3}
JOB_VRAM_DEFAULT_GIB = 23.3     # the worst measured cell, so an unknown shape is
                                # sized pessimistically rather than optimistically
VRAM_HEADROOM = 0.85            # refuse a plan that would fill more than this
SETUP_HOURS = 1.0        # billed while the box boots, clones and tokenizes
# Extrapolation factors, stated so they can be argued with. Both are applied to a
# MEASURED rate and both are labelled `extrapolated` in the output.
PROXY_SPEEDUP = 3.0     # 256-wide vs 768-wide: 8.9x fewer matrix FLOPs, but a
                        # model this small is launch-bound, so cap the credit at 3x
CTX1024_FACTOR = 0.7    # ctx 512 -> 1024: attention is the only quadratic term


def measured_rates() -> dict[tuple[str, int, int], float]:
    """Median tok/s per (mixer, batch, ctx) from the committed suite-22..26 runs.

    PAPER section 10 notes that no suite reports GPU hours because the trainer
    discarded elapsed time. The per-step `tok_s` was never discarded, and those
    run records ARE committed -- so the estimate below is derived from this
    repository rather than recalled.
    """
    import statistics
    per_run: dict[tuple[str, int, int], list[float]] = {}
    for cfgp in sorted((ROOT / "nanolab/out").glob("crossover*/**/config.json")):
        mp = cfgp.with_name("metrics.jsonl")
        if not mp.exists():
            continue
        try:
            cfg = json.loads(cfgp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        key = (cfg.get("mixer"), cfg.get("batch_size"), cfg.get("block_size"))
        ts = []
        for line in mp.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "train" and r.get("tok_s"):
                ts.append(float(r["tok_s"]))
        if ts:
            per_run.setdefault(key, []).append(statistics.median(ts))
    return {k: statistics.median(v) for k, v in per_run.items()}


def job_rate(job: dict, rates: dict) -> tuple[float | None, str]:
    """(tok/s on the GH200 the suites ran on, provenance).

    Width is checked FIRST. Every measured rate comes from a 768-wide run, so a
    256-wide proxy that happens to match on (mixer, batch, ctx) is NOT measured --
    reading it as measured priced the cheapest jobs in the bundle at the most
    expensive jobs' rate.
    """
    o = job["overrides"]
    base = rates.get((o["mixer"], o["batch_size"], 512))
    exact = rates.get((o["mixer"], o["batch_size"], o["block_size"]))
    if o["d_model"] == TARGET_WIDTH and exact is not None:
        return exact, "measured"
    if base is None:
        return None, "no reference"
    factor, why = 1.0, []
    if o["block_size"] != 512:
        factor *= CTX1024_FACTOR
        why.append(f"ctx{o['block_size']} x{CTX1024_FACTOR}")
    if o["d_model"] != TARGET_WIDTH:
        factor *= PROXY_SPEEDUP
        why.append(f"d{o['d_model']} x{PROXY_SPEEDUP}")
    return base * factor, "extrapolated (" + ", ".join(why) + ")"


def makespan(hours: list[float], gpus: int) -> float:
    """Longest-processing-time packing of independent jobs onto `gpus` workers.

    The bundle's wall clock is not total/GPUs: one job is 4+ hours and no number
    of GPUs shortens it. LPT gives the schedule the runner actually achieves when
    the long jobs are queued first.
    """
    ends = [0.0] * max(1, gpus)
    for h in sorted(hours, reverse=True):
        i = min(range(len(ends)), key=lambda k: ends[k])
        ends[i] += h
    return max(ends)


def cost_report(jobs: list[dict]) -> int:
    rates = measured_rates()
    if not rates:
        print("no committed run records to derive throughput from", file=sys.stderr)
        return 1
    print("=== measured GH200 throughput (median tok/s, from committed run records) ===")
    for k in sorted(rates, key=lambda x: (str(x[0]), x[1], x[2])):
        print(f"  {str(k[0]):<12} bs{k[1]:<4} ctx{k[2]:<6} {rates[k]:>10,.0f} tok/s")

    print("\n=== per-suite GH200-hours ===")
    tot = 0.0
    extrap = 0.0
    per_suite: dict[str, list[float]] = {}
    notes: dict[str, str] = {}
    for j in jobs:
        r, why = job_rate(j, rates)
        if not r:
            continue
        h = j["token_budget"] / r / 3600.0
        per_suite.setdefault(j["suite"], []).append(h)
        notes[j["suite"]] = why
        tot += h
        if why != "measured":
            extrap += h
    print(f"{'suite':<20}{'jobs':>5}{'GPU-h':>9}{'longest job':>13}   basis")
    longest = (0.0, "")
    for s in SUITE_ORDER:
        hs = per_suite.get(s)
        if not hs:
            continue
        if max(hs) > longest[0]:
            longest = (max(hs), s)
        print(f"  {s:<18}{len(hs):>5}{sum(hs):>9.2f}{max(hs)*60:>10.0f} min   {notes[s]}")
    print(f"  {'TOTAL':<18}{sum(len(v) for v in per_suite.values()):>5}{tot:>9.2f}"
          f"      ({extrap:.2f} h of it extrapolated, not measured)")

    print(f"\n  CRITICAL PATH: the longest single job is in {longest[1]} at "
          f"{longest[0]*60:.0f} min.")
    print("  No number of GPUs makes the bundle finish faster than that, so schedule")
    print("  the longest jobs FIRST and let the short ones fill in behind them.")

    all_h = [h for hs in per_suite.values() for h in hs]
    no_d10 = [h for s, hs in per_suite.items() if s != "d10_horizon" for h in hs]
    has_d10 = bool(per_suite.get("d10_horizon"))

    print(f"\n=== projected wall clock and total, incl. {SETUP_HOURS:g} h setup ===")
    print("    (throughput vs GH200 is an ASSUMPTION for every row but the GH200,")
    print("     which IS the box every measured rate above was recorded on)")
    second = "without d10_horizon" if has_d10 else ""
    print(f"{'instance':<22}{'$/GPU-h':>9}{'full bundle':>22}{second:>22}")
    rows = []
    for name, price, gpus, vram, (lo, hi) in INSTANCES:
        def bracket(hours):
            a = makespan([h / hi for h in hours], gpus) + SETUP_HOURS
            b = makespan([h / lo for h in hours], gpus) + SETUP_HOURS
            return a, b
        fa, fb = bracket(all_h)
        na, nb = bracket(no_d10) if has_d10 else (fa, fb)
        rows.append((name, price / gpus, fa * price, fb * price, fa, fb, na, nb, price))
        col2 = (f'${na*price:,.0f}-{nb*price:,.0f} / {na:.1f}-{nb:.1f} h'
                if has_d10 else '')
        print(f"  {name:<20}{price/gpus:>9.2f}"
              f"{f'${fa*price:,.0f}-{fb*price:,.0f} / {fa:.1f}-{fb:.1f} h':>22}"
              f"{col2:>22}")

    # rows: (name, $/GPU-h, full_lo$, full_hi$, full_lo_h, full_hi_h, no_lo_h, no_hi_h, $/hr)
    cheap = min(rows, key=lambda r: r[3])
    fast = min(rows, key=lambda r: r[5])
    knee = min((r for r in rows if r[5] <= 8.0), key=lambda r: r[3], default=None)
    d10_h = sum(per_suite.get("d10_horizon", []))
    print(f"\n  cheapest (full bundle):  {cheap[0]:<20} "
          f"${cheap[2]:,.0f}-{cheap[3]:,.0f} over {cheap[4]:.0f}-{cheap[5]:.0f} h")
    print(f"  fastest (full bundle):   {fast[0]:<20} "
          f"${fast[2]:,.0f}-{fast[3]:,.0f} over {fast[4]:.1f}-{fast[5]:.1f} h")
    if knee:
        print(f"  cheapest under 8 h:      {knee[0]:<20} "
              f"${knee[2]:,.0f}-{knee[3]:,.0f} over {knee[4]:.1f}-{knee[5]:.1f} h")
    if has_d10:
        print(f"\n  d10_horizon is {d10_h:.1f} of the {tot:.1f} GPU-hours "
              f"({d10_h/tot*100:.0f}%), in {len(per_suite['d10_horizon'])} jobs, "
              "none of which")
        print("  any amount of parallelism shortens. It is also the lowest-value item")
        print("  here: suite 20's horizon claim is already withdrawn in either direction,")
        print("  so this adds a new measurement rather than settling a live question, and")
        print("  it is the only part needing an enlarged corpus. It is OPT-IN for exactly")
        print("  that reason -- the right-hand column above is the bundle without it.")
    else:
        print("\n  d10_horizon is NOT in this matrix (opt in with --with-d10). At three")
        print("  seeds it is roughly 3x everything above, in jobs no parallelism")
        print("  shortens, and it is not on section 8.4's critical path. Price it")
        print("  separately: `--cost --with-d10`.")

    ref = "1x A100 40GB SXM4"
    ref_price = {n: p / g for n, p, g, _, _ in INSTANCES}[ref]
    print(f"\n=== assumption-free view: break-even vs {ref} at ${ref_price:.2f}/GPU-hr ===")
    print(f"{'instance':<22}{'$/GPU-hr':>10}{'GPUs':>6}{'VRAM':>6}{'break-even':>12}")
    for name, price, gpus, vram, _ in INSTANCES:
        unit = price / gpus
        print(f"  {name:<20}{unit:>10.2f}{gpus:>6}{vram:>5}G{unit/ref_price:>11.2f}x")
    print("  Read as: this box must be faster than that multiple of an A100-40GB,")
    print("  per GPU, to cost less for the same work. No throughput claim needed.")
    print("\n  Nothing above is measured on this machine -- it has no CUDA device.")
    print("  After the run, `--report` replaces every figure with the ledger's own")
    print("  `gpu_hours_measured`, which is a measurement.")
    return 0


# ---------------------------------------------------------------------------
# efficiency: the recorded `mfu` field is against the WRONG peak on this hardware
# ---------------------------------------------------------------------------
# nanolab/train.py:_mfu divides achieved FLOP/s by `PEAK_FLOPS`, which defaults to
# 46e12 -- "a 3070 Ti Laptop BF16 dense peak" -- and nothing sets that env var on
# the cluster. Every GH200 run in this repository therefore records an `mfu` that is
# a fraction of a laptop GPU: a field reading 0.97 means 97% of a 3070 Ti, i.e.
# about 4.5% of the H100 die it actually ran on, a factor of 21.5.
#
# PAPER 6.3's MFU table is unaffected -- suite 18 is a 3070 Ti run, where 46e12 is
# the right peak. But PAPER 10 lists the per-job metrics.jsonl as reproduction
# artifacts, and a reader taking `mfu` at face value on the GH200 records would
# misjudge utilization by that factor.
#
# We do not rewrite the field mid-bundle: that would make it mean two different
# things within one matrix, which is worse than meaning one wrong thing
# consistently. We report the corrected figure here instead, naming the peak, and
# derive it from the recorded value rather than re-deriving the model's FLOP count.
TRAIN_PY_DEFAULT_PEAK = 46e12          # nanolab/train.py:_mfu, unset PEAK_FLOPS
DEVICE_PEAK_FLOPS = {                  # bf16 DENSE, not the sparsity-doubled figure
    "gh200": 989.5e12,
    "h100": 989.5e12,
    "a100": 312e12,
    "a10": 125e12,
}


def device_peak_flops(name: str) -> tuple[float, str]:
    low = (name or "").lower()
    for key, peak in DEVICE_PEAK_FLOPS.items():
        if key in low:
            return peak, key
    return TRAIN_PY_DEFAULT_PEAK, "unknown (falling back to train.py's default)"


def efficiency_report(led: dict) -> None:
    """Achieved FLOP/s and TRUE MFU per suite, from the ledger's own records."""
    import statistics
    dev = (led.get("meta") or {}).get("device_name", "")
    peak, label = device_peak_flops(dev)
    rows: dict[str, list[tuple[float, float]]] = {}
    for r in led.get("jobs", []):
        if r.get("status") != "done" or not r.get("mean_tok_s"):
            continue
        d = OUT_ROOT / r["id"] / "metrics.jsonl"
        mfus = []
        if d.exists():
            for line in d.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == "train" and row.get("mfu"):
                    mfus.append(float(row["mfu"]))
        if mfus:
            rows.setdefault(r["suite"], []).append(
                (float(r["mean_tok_s"]), statistics.median(mfus)))
    if not rows:
        return
    workers = (led.get("meta") or {}).get("workers", 1)
    print(f"\n=== efficiency (device {dev or 'unknown'}, bf16 dense peak "
          f"{peak/1e12:.0f} TFLOP/s, matched on {label}) ===")
    print(f"{'suite':<20}{'tok/s/job':>11}{'TFLOP/s':>10}{'true MFU':>10}"
          f"{'as-recorded':>13}")
    for suite in SUITE_ORDER:
        v = rows.get(suite)
        if not v:
            continue
        tok = statistics.median(x for x, _ in v)
        mfu = statistics.median(y for _, y in v)
        ach = mfu * TRAIN_PY_DEFAULT_PEAK
        print(f"  {suite:<18}{tok:>11,.0f}{ach/1e12:>10.1f}"
              f"{100*ach/peak:>9.2f}%{100*mfu:>12.1f}%")
    print(f"  the `as-recorded` column is the raw `mfu` field: a fraction of "
          f"{TRAIN_PY_DEFAULT_PEAK/1e12:.0f} TFLOP/s")
    print(f"  (a 3070 Ti Laptop), not of this device. Ratio {peak/TRAIN_PY_DEFAULT_PEAK:.1f}x.")
    print(f"\n  At {workers} concurrent job(s) the aggregate is roughly "
          f"{workers * 100 * statistics.median(y for v in rows.values() for _, y in v) * TRAIN_PY_DEFAULT_PEAK / peak:.0f}% "
          "of the device.")
    print("  Per-job efficiency here is LOW BY DESIGN and is not a tuning target:")
    print("  compile=False, batch 32 and context 512 are recipe constants inherited")
    print("  from suites 22-26, and changing any of them to chase throughput would")
    print("  break the comparability that is the whole reason for this hardware.")
    print("  Concurrency is the one lever that cannot affect a loss curve.")


# ---------------------------------------------------------------------------
# doc drift: a document that disagrees with the code is a defect, not a typo
# ---------------------------------------------------------------------------
DOCS = (ROOT / "docs/GPU_BUNDLE.md", ROOT / "docs/ISSUES_AND_GAPS_2026-08-22.md")


def check_docs(jobs: list[dict]) -> int:
    """Verify the documented matrix figures against the derived ones.

    This repository has already been bitten twice by a document and the code
    disagreeing: gap D3, and the SP-cells question, where `docs/GPU_BUNDLE.md` said
    the 2x2's SP cells were not re-run while `ISSUES_AND_GAPS` said re-running them
    was "not optional" -- and the runner followed the wrong one. `d7_analyze.py
    --check` exists for the same reason on the D7 record. Both grids in this bundle
    have now been extended mid-run, so these totals move.

    Checked: the headline job total, the per-suite counts in the suite table, and
    the derived GH200-hours to one decimal place.
    """
    import re
    from collections import Counter
    counts = Counter(j["suite"] for j in jobs)
    total = len(jobs)
    rates = measured_rates()
    hours = sum(j["token_budget"] / r / 3600.0
                for j in jobs for r, _ in [job_rate(j, rates)] if r)

    bad: list[str] = []
    for doc in DOCS:
        if not doc.exists():
            bad.append(f"{_rel(doc)}: missing")
            continue
        txt = doc.read_text(encoding="utf-8")
        if f"**{total} jobs**" not in txt and f"**{total}**" not in txt:
            bad.append(f"{_rel(doc)}: does not state the {total}-job total")
        for suite, n in counts.items():
            for mo in re.finditer(rf"\| `{re.escape(suite)}` \| (\d+) \|", txt):
                if int(mo.group(1)) != n:
                    bad.append(f"{_rel(doc)}: `{suite}` listed as {mo.group(1)}, "
                               f"matrix has {n}")
        for mo in re.finditer(r"≈ ([\d.]+) GH200-hours", txt):
            if abs(float(mo.group(1)) - hours) > 0.05:
                bad.append(f"{_rel(doc)}: says {mo.group(1)} GH200-hours, "
                           f"derived is {hours:.2f}")
    print(f"=== doc drift: {total} jobs, {hours:.2f} GH200-hours derived ===")
    if bad:
        for b in bad:
            print(f"  DRIFT  {b}")
        print(f"\n{len(bad)} disagreement(s). Regenerate with --cost and --plan and "
              "correct the document; the code is the source.")
        return 1
    print(f"  ok    {len(DOCS)} document(s) agree with the matrix on every figure "
          "checked")
    return 0


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def corpus_tokens() -> tuple[Path, int, int]:
    cfg = build_config(PRESET, {"run_name": "probe"})
    tag = (cfg.hf_dataset or "").replace("/", "_")
    d = ROOT / cfg.data_dir / tag
    tr, va = d / "train.bin", d / "val.bin"
    n_tr = tr.stat().st_size // 2 if tr.exists() else 0
    n_va = va.stat().st_size // 2 if va.exists() else 0
    return d, n_tr, n_va


def preflight(jobs: list[dict], allow_repeat: bool, sp_cells: str = "rerun",
              allow_cross_hardware: bool = False) -> int:
    print("=== preflight ===")
    bad = 0

    d, n_tr, n_va = corpus_tokens()
    if not n_tr or not n_va:
        print(f"  FAIL  dataset: no train.bin/val.bin under {d}")
        print("        prepare it first: python -m nanolab.prep_fineweb")
        bad += 1
    else:
        print(f"  ok    dataset: {_rel(d)} "
              f"train {n_tr/1e6:.1f}M tok, val {n_va/1e6:.1f}M tok")
        if n_tr != REFERENCE_CORPUS_TOKENS:
            print(f"  {'WARN' if allow_cross_hardware else 'FAIL'}  corpus: "
                  f"{n_tr/1e6:.1f}M tokens, but suites 22-26 trained on "
                  f"{REFERENCE_CORPUS_TOKENS/1e6:.1f}M")
            print("        The corpus is part of the recipe. The Batcher samples windows")
            print("        WITH replacement, so the same token budget over a smaller corpus")
            print("        is a different training distribution -- a second variable moving")
            print("        alongside the parametrization under test, and the drift check")
            print("        against suite 24 and E2's merge into suite 26's board both")
            print("        assume it is held. Copy the reference train.bin/val.bin rather")
            print("        than re-tokenizing; a fresh tokenization of the same nominal")
            print("        size is not guaranteed byte-identical either.")
            if not allow_cross_hardware:
                bad += 1
        else:
            print(f"  ok    corpus: matches the {REFERENCE_CORPUS_TOKENS/1e6:.1f}M-token "
                  "corpus suites 22-26 trained on")
        # The Batcher samples windows uniformly WITH replacement, so a budget
        # above the corpus is not an error -- but it means that job revisits
        # tokens and a job below the corpus does not. In a matched pair that is a
        # second variable moving with the first.
        over = [(j["id"], j["token_budget"] / n_tr) for j in jobs
                if j.get("token_budget", 0) > n_tr]
        if over:
            print(f"  {'WARN' if allow_repeat else 'FAIL'}  "
                  f"{len(over)} job(s) request more tokens than the corpus holds:")
            for jid, x in over:
                print(f"          {jid}: {x:.2f} epochs of a {n_tr/1e6:.0f}M-token corpus")
            need = max(j["token_budget"] for j in jobs)
            print("        These revisit training tokens while shorter jobs in the same")
            print("        comparison do not, which moves a second variable alongside the")
            print("        one under test. For d10_horizon that is fatal: the pair exists to")
            print("        make horizon the ONLY difference between its two arms.")
            print(f"        Enlarge the corpus to >= {need/1e6:.0f}M tokens:")
            print(f"          rm {_rel(d)}/train.bin {_rel(d)}/val.bin")
            print(f"          python -m nanolab.prep_fineweb --config sample-10BT "
                  f"--max_tokens {int(need * 1.05)}")
            print("        or pass --allow-data-repeat to accept it; `data_epochs` is then")
            print("        recorded per job in the ledger so the repeat cannot go unreported.")
            if not allow_repeat:
                bad += 1

    n_gpu = detect_gpus()
    dev = device_name()
    print(f"  {'ok   ' if n_gpu else 'WARN '} cuda: {n_gpu} device(s) visible"
          + (f"  [{dev}]" if dev else "  (jobs will run on CPU and take days)"))

    # The hardware control, enforced rather than documented.
    gated = sorted({j["suite"] for j in jobs if j["suite"] in GH200_REQUIRED})
    if gated and n_gpu and not is_gh200(dev):
        print(f"  {'WARN' if allow_cross_hardware else 'FAIL'}  hardware: "
              f"{', '.join(gated)} require a GH200; this box reports {dev!r}")
        print("        e2_matched32_50m's cells are MERGED INTO suite 26's board")
        print("        (experiment-notes/nanolab/artifacts/26-matched32_lock.json), whose")
        print("        other eight rows were measured on a Lambda GH200. Filling its two")
        print("        `source: suite22` rows from another GPU replaces a same-box caveat")
        print("        with the cross-hardware one PAPER 7.1 refuses -- the same pair")
        print("        differs by ~0.18-0.3 nats at matched token markers across two")
        print("        boxes -- which leaves that board worse than it found it.")
        print("        Rent a GH200, drop the suite (--only ...), or pass")
        print("        --allow-cross-hardware-board to record the caveat and proceed.")
        if not allow_cross_hardware:
            bad += 1
    elif gated and n_gpu:
        print(f"  ok    hardware: GH200 -- {', '.join(gated)} join suite 26's board "
              "on its own hardware")

    if sp_cells == "suite24":
        ok = bool(n_gpu) and is_gh200(dev)
        print(f"  {'ok   ' if ok else 'FAIL '} --sp-cells suite24: "
              + ("GH200 confirmed by model name (the individual machine still cannot "
                 "be)" if ok else f"this box reports {dev!r}, not a GH200"))
        if not ok:
            print("        Without the re-run the 2x2 sets new muP cells against suite")
            print("        24's GH200 cells, which is the comparison PAPER 7.1 refuses.")
            bad += 1

    try:
        free = free_gib()
        need = 2.0 + 0.6 * len(jobs)      # ckpt.pt + best.pt + final.pt per job
        print(f"  {'ok   ' if free > need else 'FAIL '} disk: {free:.0f} GiB free, "
              f"~{need:.0f} GiB needed for {len(jobs)} jobs")
        bad += free <= need
    except OSError as e:
        print(f"  WARN  disk: could not stat ({e})")

    dirty, stale = [], []
    for j in jobs:
        d = OUT_ROOT / j["id"]
        st = inspect_run(d)
        if st["status"] in ("partial", "suspect"):
            dirty.append((j["id"], st["status"], st["starts"], st["dones"]))
        elif st["status"] == "done":
            why = verify_fingerprint(j, d)
            if why:
                stale.append((j["id"], why))
    if stale:
        print(f"  FAIL  {len(stale)} finished run(s) no longer match the requested recipe:")
        for jid, why in stale[:6]:
            print(f"          {jid}: {why}")
        if len(stale) > 6:
            print(f"          ... and {len(stale) - 6} more")
        print("        These would be SKIPPED as done and published under the new")
        print("        recipe's label while holding the old recipe's numbers. Archive")
        print("        them first: python3 scripts/gpu_bundle.py --reset-partial")
        bad += 1
    if dirty:
        print(f"  FAIL  {len(dirty)} run dir(s) are not clean:")
        for jid, s, a, b in dirty:
            print(f"          {jid}: {s} ({a} start / {b} done records)")
        print("        metrics.jsonl is opened in APPEND mode, so re-running these")
        print("        would mix segments in one file -- the run128m_20k defect (D10).")
        print("        Archive them first: python3 scripts/gpu_bundle.py --reset-partial")
        print("        NOTE: a job that is running RIGHT NOW also looks like this, and")
        print("        preflight cannot tell the two apart. If another invocation is")
        print("        live, this is that -- check before archiving, because archiving")
        print("        a running job's directory loses the run. Each --only invocation")
        print("        preflights only its own suite, so a chain is unaffected.")
        bad += 1
    else:
        print("  ok    run dirs: no partial or multi-segment runs")

    t = load_transfer()
    if t:
        print("  ok    proxy transfer: " + ", ".join(
            f"{a} matrix_lr={v['matrix_lr']:g}"
            for a, v in sorted(t.get("arms", {}).items())))
    else:
        print("  info  proxy transfer: none yet -- run --only e1_proxy first "
              "(every muP suite is blocked until it exists)")
    an = load_anchor()
    if an:
        print("  ok    basin anchor: " + ", ".join(
            f"{a} {v['mult']:g}x" for a, v in sorted(an.get("arms", {}).items()))
            + "  (1.0x would mean the muP transfer landed)")
    else:
        print("  info  basin anchor: none yet -- run --only e1_mup_basin "
              f"({', '.join(MUP_ANCHOR_SUITES)} are blocked until it exists)")

    print(f"\npreflight: {'FAILED' if bad else 'clean'}")
    return 1 if bad else 0


def reset_partial(jobs: list[dict], root: Path) -> int:
    moved = 0
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for j in jobs:
        d = root / j["id"]
        st = inspect_run(d)
        if st["status"] == "done" and verify_fingerprint(j, d):
            st = dict(st, status="stale")
        if st["status"] in ("partial", "suspect", "stale"):
            dest = ARCHIVE_ROOT / stamp / j["id"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(dest))
            print(f"  archived {j['id']} ({st['status']}) -> {_rel(dest)}")
            moved += 1
    print(f"{moved} directory(ies) archived" if moved else "nothing to archive")
    return 0


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def report(root: Path) -> int:
    path = ledger_path(root)
    if not path.exists():
        print(f"no ledger at {_rel(path)}")
        return 1
    led = json.loads(path.read_text(encoding="utf-8"))
    print(f"ledger {_rel(path)}  started {led.get('started_at')}  "
          f"updated {led.get('updated_at')}")
    print(f"{led['jobs_done']}/{led['jobs_total']} done, "
          f"{led.get('gpu_hours_measured', 0)} GPU-hours measured")
    print(f"\n{'suite':<20}{'done':>6}{'other':>8}")
    for s in SUITE_ORDER:
        counts = led.get("by_suite", {}).get(s)
        if not counts:
            continue
        other = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()) if k != "done")
        print(f"  {s:<18}{counts.get('done', 0):>6}  {other}")
    bad = [r for r in led["jobs"] if r["status"] not in ("done", "pending")]
    if bad:
        print("\nnot done:")
        for r in bad:
            print(f"  {r['id']:<44} {r['status']:<9} "
                  f"{r.get('failure_reason', '') or ''}")
    report_proxy(analyse_proxy(led["jobs"]))
    efficiency_report(led)
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the matrix and exit")
    ap.add_argument("--dry-run", action="store_true", help="print every job config")
    ap.add_argument("--preflight", action="store_true", help="gate the box, run nothing")
    ap.add_argument("--report", action="store_true", help="read the ledger, run nothing")
    ap.add_argument("--cost", action="store_true",
                    help="derive GPU-hours from the committed suite-22..26 throughput")
    ap.add_argument("--reset-partial", action="store_true",
                    help="archive partial/multi-segment run dirs so they can be re-run")
    ap.add_argument("--smoke", action="store_true",
                    help="40-step check of one job per suite, in an ISOLATED subtree")
    ap.add_argument("--only", default=None, help="run one suite id")
    ap.add_argument("--workers", type=int, default=0,
                    help="concurrent jobs (default: one per visible GPU, else 1)")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated device ids to spread across (default: all)")
    ap.add_argument("--sp-cells", choices=("rerun", "suite24"), default="rerun",
                    help="'suite24' only on the GH200 that ran suite 24; see the docstring")
    ap.add_argument("--check-docs", action="store_true",
                    help="verify the documented job/hour figures against the matrix")
    ap.add_argument("--analyse", "--analyze", dest="analyse", action="store_true",
                    help="derive crossing tokens and the PAPER 8.4 readouts from the ledger")
    ap.add_argument("--with-d10", action="store_true",
                    help="include d10_horizon (3 seeds, ~3x the rest of the bundle)")
    ap.add_argument("--allow-cross-hardware-board", action="store_true",
                    help="run the GH200-gated suites on another GPU and record the caveat")
    ap.add_argument("--allow-data-repeat", action="store_true",
                    help="accept jobs whose token budget exceeds the corpus")
    ap.add_argument("--ignore-vram", action="store_true",
                    help="skip the measured per-job VRAM cap (separate from --oversubscribe)")
    ap.add_argument("--oversubscribe", action="store_true",
                    help="allow more concurrent jobs than visible GPUs (will likely OOM)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="launch without gating the box (not recommended)")
    args = ap.parse_args()

    transfer = load_transfer()
    anchor = load_anchor()
    with_d10 = args.with_d10 or args.only == "d10_horizon"
    jobs = build_matrix(sp_cells=args.sp_cells, transfer=transfer,
                        with_d10=with_d10, anchor=anchor)
    if args.only:
        jobs = [j for j in jobs if j["suite"] == args.only]
        if not jobs:
            sys.exit(f"no jobs for suite {args.only!r}. known: {', '.join(SUITE_ORDER)}")

    root = SMOKE_ROOT if args.smoke else OUT_ROOT
    if args.smoke:
        seen, picked = set(), []
        for j in jobs:
            if j["suite"] not in seen:
                seen.add(j["suite"])
                picked.append(j)
        # A smoke run is a 40-step check in an ISOLATED subtree and is never a
        # measurement, so the muP suites must not be blocked out of it -- the
        # documented order of operations is `--smoke` BEFORE `--only e1_proxy`, and
        # the muP code path is exactly what needs checking before the box bills.
        # They run at the preset's matrix_lr, labelled, and the label is what keeps
        # that placeholder from ever being mistaken for a transferred value: a real
        # run still refuses to launch them until the sweep publishes.
        for j in picked:
            if j.pop("blocked_on", None):
                j["smoke_placeholder_lr"] = True
                j["overrides"].pop("matrix_lr", None)
        jobs = picked

    if args.check_docs:
        return check_docs(jobs)
    if args.cost:
        return cost_report(jobs)
    if args.report:
        return report(root)
    if args.analyse:
        path = ledger_path(root)
        if not path.exists():
            print(f"no ledger at {_rel(path)}", file=sys.stderr)
            return 1
        return report_crossings(json.loads(path.read_text(encoding="utf-8"))["jobs"])
    if args.reset_partial:
        return reset_partial(jobs, root)

    if args.plan:
        from collections import Counter
        c = Counter(j["suite"] for j in jobs)
        blocked = Counter(j["suite"] for j in jobs if j.get("blocked_on"))
        print(f"{'suite':<20}{'jobs':>6}{'blocked':>9}   note")
        for s in SUITE_ORDER:
            if c.get(s):
                print(f"  {s:<18}{c[s]:>6}{blocked.get(s, 0):>9}   {SUITE_DOC[s]}")
        print(f"  {'TOTAL':<18}{len(jobs):>6}{sum(blocked.values()):>9}")
        if c.get("e1_mup_basin"):
            print(f"\n  e1_mup_basin multipliers: "
                  + ", ".join(f"{x:g}x" for x in BASIN_MULTS)
                  + f"  (1x = e1_mup; competing Muon rule predicts "
                  f"{TARGET_WIDTH / BASE_WIDTH:g}x)")
        if blocked:
            for j in jobs:
                if j.get("blocked_on"):
                    print(f"\n  blocked: {j['suite']} -- {j['blocked_on']}")
                    break
        if args.sp_cells == "suite24":
            print("\n  --sp-cells suite24: the 2x2's SP cells come from suite 24 (GH200).")
            print("  Only correct if THIS box is that GH200; otherwise the 2x2 is")
            print("  confounded by hardware (PAPER 7.1).")
        return 0

    if args.dry_run:
        for j in jobs:
            flag = "  [BLOCKED]" if j.get("blocked_on") else ""
            print(f"{j['id']}{flag}: {json.dumps(j['overrides'], sort_keys=True)}")
        return 0

    pf = dict(sp_cells=args.sp_cells,
              allow_cross_hardware=args.allow_cross_hardware_board)
    if args.preflight:
        return preflight(jobs, args.allow_data_repeat, **pf)
    if not args.skip_preflight and not args.smoke:
        if preflight(jobs, args.allow_data_repeat, **pf):
            print("\nrefusing to launch. Fix the above, or pass --skip-preflight.",
                  file=sys.stderr)
            return 1
        print()

    blocked = [j for j in jobs if j.get("blocked_on")]
    if blocked:
        print(f"refusing to launch {len(blocked)} blocked job(s):", file=sys.stderr)
        for j in blocked[:4]:
            print(f"  {j['id']}: {j['blocked_on']}", file=sys.stderr)
        print("  Run `--only e1_proxy` first; a bracketed minimum publishes "
              f"{TRANSFER.name}.", file=sys.stderr)
        return 1

    if args.gpus:
        devices = [d.strip() for d in args.gpus.split(",") if d.strip()]
    else:
        n = detect_gpus()
        devices = [str(i) for i in range(n)] if n else [None]
    n_real = len([d for d in devices if d is not None])
    workers = args.workers or n_real or 1
    workers = max(1, min(workers, len(jobs)))
    if n_real and workers > n_real and not args.oversubscribe:
        print(f"refusing --workers {workers} on {n_real} visible GPU(s): that co-locates "
              f"{workers / n_real:.1f} training jobs per device and OOMs at this model "
              f"size.\n  Use --workers {n_real}, name more devices with --gpus, or pass "
              f"--oversubscribe if you mean it.", file=sys.stderr)
        return 1

    # --oversubscribe says "co-locate them"; it does not say "fill the card". Sizing
    # a worker count against the cheaper arm is how this bundle put four 23.3 GiB
    # minGRU jobs on a 97.9 GiB card at 97.6% full. The cap is computed from the
    # HEAVIEST job actually queued, and it is not what --oversubscribe waives.
    total_vram = device_total_vram_gib()
    safe, per_job = vram_safe_workers(jobs, total_vram)
    if safe and workers > safe and not args.ignore_vram:
        print(f"refusing --workers {workers}: the heaviest job queued measures "
              f"{per_job:.1f} GiB, so {workers} of them need "
              f"{workers * per_job:.1f} GiB of a {total_vram:.1f} GiB device "
              f"({100 * workers * per_job / total_vram:.0f}%).\n"
              f"  At most {safe} fit under the {VRAM_HEADROOM:.0%} headroom. Use "
              f"--workers {safe}, or --ignore-vram if the figure is wrong for your "
              f"shapes.\n  (JOB_VRAM_GIB is measured, not computed -- it includes "
              f"activations and the caching allocator, not just weights.)",
              file=sys.stderr)
        return 1
    if safe and total_vram:
        print(f"vram: {workers} x {per_job:.1f} GiB = {workers * per_job:.1f} GiB "
              f"of {total_vram:.1f} GiB ({100 * workers * per_job / total_vram:.0f}%), "
              f"cap {safe}")

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    _, n_tr, _ = corpus_tokens()
    dev_name = device_name()
    meta = {"sp_cells": args.sp_cells, "workers": workers,
            "devices": devices, "smoke": bool(args.smoke),
            "device_name": dev_name, "is_gh200": is_gh200(dev_name),
            "cross_hardware_board_accepted": bool(args.allow_cross_hardware_board),
            "corpus_train_tokens": n_tr,
            "transfer": (transfer or {}).get("arms", {}),
            "anchor": (anchor or {}).get("arms", {})}

    records = []
    for j in jobs:
        st = inspect_run(root / j["id"])
        rec = {"id": j["id"], "suite": j["suite"], "arm": j["arm"], "seed": j["seed"],
               "tag": j["tag"], "status": st["status"], "best_val": st["best_val"],
               "final_val": st["final_val"], "tokens": st["tokens"],
               "elapsed_s": st["elapsed_s"], "mean_tok_s": st["mean_tok_s"],
               "curve": st["curve"], "matrix_lr": j["overrides"].get("matrix_lr"),
               "data_epochs": (round(j.get("token_budget", 0) / n_tr, 3) if n_tr else None)}
        if j.get("transfer"):
            rec["transfer"] = j["transfer"]
        if st["status"] == "missing":
            rec["status"] = "pending"
        elif st["status"] == "done":
            # A finished run is only a finished run OF THIS RECIPE. Resume matched
            # on the job id alone, and the id does not encode matrix_lr -- so
            # re-anchoring the transferred learning rate and re-running would have
            # SKIPPED every muP cell and published numbers trained at the old rate,
            # under the new rate's label. The fingerprint is verified after a job
            # runs (PAPER 3.3); it has to be verified on resume too, or the check
            # only guards the path that was never the risk.
            stale = verify_fingerprint(j, root / j["id"]) if not args.smoke else None
            if stale:
                rec["status"] = "stale"
                rec["failure_reason"] = f"recipe changed since this run: {stale}"
        records.append(rec)
    write_ledger(root, records, started, meta)

    todo = [(j, r) for j, r in zip(jobs, records) if r["status"] == "pending"]
    for j, r in zip(jobs, records):
        if r["status"] == "done":
            print(f"skip {j['id']} (done, best_val {r['best_val']:.4f})")
        elif r["status"] == "stale":
            print(f"SKIP {j['id']} (stale: {r['failure_reason']})")
        elif r["status"] in ("partial", "suspect"):
            print(f"skip {j['id']} ({r['status']} -- run --reset-partial to redo it)")
    print(f"{len(todo)} job(s) to run, {workers} worker(s) over devices {devices}\n",
          flush=True)

    pool: queue.Queue = queue.Queue()
    for i in range(workers):
        pool.put(devices[i % len(devices)])
    counter = {"n": 0}
    counter_lock = threading.Lock()

    def work(item):
        j, rec = item
        dev = pool.get()
        try:
            with counter_lock:
                counter["n"] += 1
                i = counter["n"]
            print(f"[{i}/{len(todo)}] {j['id']} (gpu {dev}) ...", flush=True)
            rec["status"] = "running"
            rec["device"] = dev
            write_ledger(root, records, started, meta)
            code, elapsed = run_one(j, root, args.smoke, dev)
            st = inspect_run(root / j["id"])
            fp = verify_fingerprint(j, root / j["id"]) if not args.smoke else None
            rec.update(returncode=code, wall_s=round(elapsed, 1),
                       best_val=st["best_val"], final_val=st["final_val"],
                       tokens=st["tokens"], elapsed_s=st["elapsed_s"],
                       mean_tok_s=st["mean_tok_s"], curve=st["curve"],
                       finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            status, reason = classify_result(code, st, fp)
            rec["status"] = status
            if reason:
                rec["failure_reason"] = reason
            write_ledger(root, records, started, meta)
            print(f"    -> {j['id']}: {rec['status']} "
                  f"{st['best_val'] if st['best_val'] is not None else ''} "
                  f"({elapsed/60:.1f} min)"
                  + (f"  [{reason}]" if reason else ""), flush=True)
        finally:
            pool.put(dev)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, todo))

    analysis = analyse_proxy(records)
    report_proxy(analysis)
    if not args.smoke and analysis.get("arms"):
        save_transfer(analysis)
    if not args.smoke:
        save_anchor(analyse_basin(records))
    if not args.smoke:
        report_crossings(records)

    failed = [r for r in records if r["status"] not in ("done",)]
    print(f"\nledger: {_rel(ledger_path(root))}")
    print(f"GPU-hours measured this session: "
          f"{sum(r.get('elapsed_s') or 0 for r in records)/3600:.2f}")
    if failed:
        print(f"{len(failed)} did not complete: "
              + ", ".join(r["id"] for r in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
