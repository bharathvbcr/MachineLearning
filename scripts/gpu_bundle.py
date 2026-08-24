#!/usr/bin/env python3
"""GPU bundle: the outstanding rented-GPU suites, as one resumable runner.

Covers the experiments the paper names but has not run:

  E1  The parametrization arm specified in PAPER section 8.4. Five sub-arms:
        e1_proxy        proxy-width matrix-LR sweep, PER ARM, to tune before transfer
        e1_sp_rerun     the SP cells of the 2x2, re-run on THIS box (see below)
        e1_mup          the muP cells of the 2x2
        e1_perlayer_sp  per-layer standard-parametrization prescription
        e1_embed_lr     embedding-LR-only ablation
  E2  Suite 26 never reran attention/minGRU at 50M matched batch 32; its first and
      eighth board rows are suite 22's sample, capping the combined ranking at
      Medium-High. Ten jobs close it.
  D10 Suite 20's horizon claim is withdrawn: run128m_20k is eight resumed segments
      with a broken token counter, and its LR moves with its horizon. A matched PAIR
      (10k and 20k at the SAME learning rate, uninterrupted) is what the claim needs.
      This is NOT a reproduction of suite 20 -- that would need the original
      RTX 3070 Ti. It answers the science on new hardware; report it that way.

WHY e1_sp_rerun EXISTS. Section 8.4's 2x2 puts muP cells against suite 24's SP
cells. Suite 24 ran on a GH200. On any other box the 2x2 is confounded by hardware
and proves nothing -- PAPER section 7.1 refuses exactly this comparison, because the
same architecture pair differs by ~0.18-0.3 nats at matched token markers across two
GPUs. So unless this box IS the GH200, all four cells must be measured here.
``--sp-cells suite24`` drops the re-run for a GH200 launch; the default re-runs them.

Cadence comes from ``crossover_replicate.scale_to_token_budget`` so eval markers line
up with suites 22-26 and loss-vs-tokens curves remain comparable.

Usage:
  python3 scripts/gpu_bundle.py --plan          # the matrix, with blockers, no work
  python3 scripts/gpu_bundle.py --preflight     # gate the box before it bills
  python3 scripts/gpu_bundle.py --smoke         # 40-step check, ISOLATED from the matrix
  python3 scripts/gpu_bundle.py --only e1_proxy --workers 4
  python3 scripts/gpu_bundle.py --report        # read the ledger, no work
  python3 scripts/gpu_bundle.py --workers 4     # everything
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
    SEEDS, TOKEN_BUDGET, LOCKED20_TOKEN_BUDGET, scale_to_token_budget,
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

# suite-24 recipe: batch 32, 20M stop, 50M cosine horizon, eval_iters 20
S24 = dict(batch=32, budget=LOCKED20_TOKEN_BUDGET, horizon=TOKEN_BUDGET, eval_iters=20)
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
# The grid spans 0.0016..0.05, a 31x range at factor-2 spacing, deliberately
# reaching far below the inherited 0.025 rather than bracketing it symmetrically.
# D7 found both optimizer finalists 6-12x above their true optima and needed four
# grid extensions to find the bottom; a symmetric +/-2x grid around an inherited
# value is how that started.
PROXY_MATRIX_LRS = (0.0016, 0.003125, 0.00625, 0.0125, 0.025, 0.05)

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
    "e1_proxy": "proxy-width matrix-LR sweep at mup_base_width, 1 seed, locates a peak only",
    "e1_sp_rerun": "SP cells of PAPER 8.4's 2x2, re-run on THIS box (hardware control)",
    "e1_mup": "muP cells of PAPER 8.4's 2x2; matrix_lr transferred from e1_proxy",
    "e1_perlayer_sp": "per-layer SP (Everett et al.) -- APPROXIMATION, see optim.py caveats",
    "e1_embed_lr": "embedding-LR-only ablation (Kalra & Barkeshli)",
    "e2_matched32_50m": "suite 26's missing attention/minGRU cells at 50M / batch 32",
    "d10_horizon": "matched 10k vs 20k at ONE learning rate; NOT a suite-20 reproduction",
}
SUITE_ORDER = tuple(SUITE_DOC)

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


def build_matrix(sp_cells: str = "rerun", transfer: dict | None = None) -> list[dict]:
    jobs: list[dict] = []

    # --- E1a: proxy-width matrix-LR sweep, PER ARM. Tune at BASE_WIDTH, transfer
    # to TARGET_WIDTH. Per arm because attention and minGRU may have different
    # optima: D7 found two optimizers with offset basins whose ordering crossed
    # over, and transferring one arm's optimum to both would rebuild the
    # unequal-tuning error that retired the funnel's champion.
    #
    # One seed: this locates a peak, it does not rank arms. Cheap by construction --
    # a 256-wide model is ~1/9 the FLOPs of the 768-wide target.
    #
    # cfg.lr stays at the suite value. Under muP the embedding/scalar LR is
    # width-constant, so transferring it unchanged is correct by the rule; that it
    # is itself inherited rather than tuned is a stated limitation of this arm.
    for arm in ARMS:
        for mlr in PROXY_MATRIX_LRS:
            jobs.append(_job("e1_proxy", arm, SEEDS[0],
                             dict(mup=True, mup_base_width=BASE_WIDTH, matrix_lr=mlr),
                             d_model=BASE_WIDTH, tag=f"mlr{mlr:g}", **S24))

    # --- E1b: the SP cells of the 2x2, on THIS box. See the module docstring.
    if sp_cells == "rerun":
        for arm in ARMS:
            for seed in SEEDS:
                jobs.append(_job("e1_sp_rerun", arm, seed, {}, **S24))

    # --- E1c: the muP cells. matrix_lr comes from e1_proxy, per arm. Without a
    # transfer these jobs inherit the preset's 0.025 -- the exact "arm mis-tuned
    # on the axis under test" that section 8.4's readouts cannot survive -- so
    # they are marked blocked and refused at launch rather than run wrong.
    for arm in ARMS:
        tuned = (transfer or {}).get("arms", {}).get(arm)
        extra = dict(mup=True, mup_base_width=BASE_WIDTH)
        if tuned:
            extra["matrix_lr"] = tuned["matrix_lr"]
        for seed in SEEDS:
            j = _job("e1_mup", arm, seed, extra, **S24)
            if tuned:
                j["transfer"] = {"matrix_lr": tuned["matrix_lr"],
                                 "source": tuned.get("source", "e1_proxy"),
                                 "bracketed": tuned.get("bracketed")}
            else:
                j["blocked_on"] = ("e1_proxy: no tuned matrix_lr for "
                                   f"{arm!r} in {TRANSFER.name}")
            jobs.append(j)

    # --- E1d: per-layer SP (Everett et al.). See the caveats in optim.py: the
    # prescription is stated for pure Adam and our hybrid sends hidden matrices to
    # Muon, and tied embeddings prevent a separate readout rate. This arm is an
    # APPROXIMATION of their prescription and must be reported as one.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_job("e1_perlayer_sp", arm, seed,
                             dict(per_layer_sp=True, mup_base_width=BASE_WIDTH), **S24))

    # --- E1e: embedding-LR-only. Raise ONLY the embedding LR by the width ratio.
    for arm in ARMS:
        for seed in SEEDS:
            jobs.append(_job("e1_embed_lr", arm, seed,
                             dict(embed_lr_mult=TARGET_WIDTH / BASE_WIDTH,
                                  mup_base_width=BASE_WIDTH), **S24))

    # --- E2: suite 26's missing attention/minGRU cells at 50M, matched batch 32.
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
            "token_budget": 32 * 1024 * c["max_steps"],
            "token_budget_requested": budget,
            "overrides": dict(
                run_name=jid, mixer="attention", layer_mixers="", seed=SEEDS[0],
                d_model=TARGET_WIDTH, n_head=heads_for(TARGET_WIDTH), head_dim=HEAD_DIM,
                batch_size=32, grad_accum=1, block_size=1024,
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


# ---------------------------------------------------------------------------
# proxy sweep -> transfer
# ---------------------------------------------------------------------------
def proxy_curves(records: list[dict]) -> dict[str, list[tuple[float, float]]]:
    by_arm: dict[str, list[tuple[float, float]]] = {}
    for r in records:
        if r.get("suite") != "e1_proxy" or r.get("status") != "done":
            continue
        # final_val at a fixed token count, not best_val. PAPER 3.2: "A best_val
        # field is not a paired snapshot. Minimum-over-all-evaluations is not a
        # ranking and is never reported as one." Every proxy cell stops at the
        # same token budget, so the last eval IS the paired snapshot.
        v = r.get("final_val")
        if v is None:
            continue
        by_arm.setdefault(r["arm"], []).append((float(r["tag"].replace("mlr", "")), float(v)))
    for pts in by_arm.values():
        pts.sort()
    return by_arm


def analyse_proxy(records: list[dict]) -> dict:
    """Per-arm minimum plus whether the grid brackets it.

    A boundary minimum measures "lower is better within this range", not an
    optimum. D7 needed four grid extensions to find its bottom and twice reported
    a boundary minimum as an optimum before the next point contradicted it.
    """
    out = {"base_width": BASE_WIDTH, "target_width": TARGET_WIDTH,
           "swept": list(PROXY_MATRIX_LRS), "arms": {}}
    for arm, pts in sorted(proxy_curves(records).items()):
        best_lr, best = min(pts, key=lambda p: p[1])
        interior = len(pts) > 2 and best_lr not in (pts[0][0], pts[-1][0])
        out["arms"][arm] = {
            "matrix_lr": best_lr, "final_val": round(best, 6),
            "points": len(pts), "bracketed": bool(interior),
            "curve": [[lr, round(v, 6)] for lr, v in pts],
            "source": "e1_proxy",
        }
    return out


def report_proxy(analysis: dict) -> None:
    if not analysis.get("arms"):
        return
    print("\n=== e1_proxy: matrix-LR sweep, per arm (val loss at the token budget) ===")
    unbracketed = []
    for arm, a in sorted(analysis["arms"].items()):
        print(f"  {arm}:  ({a['points']}/{len(PROXY_MATRIX_LRS)} points done)")
        for lr, v in a["curve"]:
            mark = "  <- min" if lr == a["matrix_lr"] else ""
            print(f"    matrix_lr {lr:<10g} {v:.6f}{mark}")
        if not a["bracketed"]:
            unbracketed.append((arm, a["matrix_lr"]))
    if unbracketed:
        print("\n  WARNING: boundary minimum -- the grid does not bracket the optimum:")
        for arm, lr in unbracketed:
            print(f"    {arm}: best is matrix_lr {lr:g}, an end of the swept range")
        print("  A boundary minimum measures 'lower/higher is better within this range',")
        print("  not an optimum. Extend PROXY_MATRIX_LRS past that end and re-run before")
        print("  launching e1_mup -- transferring an edge value would mis-tune the very")
        print("  arm section 8.4's readouts are read against.")
    else:
        print("\n  Both arms bracketed: each minimum is interior to the grid.")


def load_transfer() -> dict | None:
    if not TRANSFER.exists():
        return None
    try:
        return json.loads(TRANSFER.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_transfer(analysis: dict) -> None:
    """Publish only bracketed arms. An unbracketed arm has no optimum to transfer."""
    keep = {a: v for a, v in analysis["arms"].items()
            if v["bracketed"] and v["points"] == len(PROXY_MATRIX_LRS)}
    if not keep:
        return
    payload = dict(analysis)
    payload["arms"] = keep
    payload["generated_by"] = "scripts/gpu_bundle.py --only e1_proxy"
    payload["rule"] = ("muP transfers the tuned hidden-layer LR from the base width; "
                       "optim.py divides matrix_lr by d_model/mup_base_width, so the "
                       "value published here is the BASE-width value and is passed "
                       "through unscaled.")
    payload["excluded"] = {a: "minimum not bracketed by the grid"
                           for a in analysis["arms"] if a not in keep}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = TRANSFER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, TRANSFER)
    print(f"  wrote {_rel(TRANSFER)}: "
          + ", ".join(f"{a} matrix_lr={v['matrix_lr']:g}" for a, v in sorted(keep.items())))


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
    ("8x A100 40GB SXM4", 15.92, 8, 40, (0.45, 0.60)),
    ("1x A100 40GB SXM4", 1.99, 1, 40, (0.45, 0.60)),
    ("8x A100 80GB SXM4", 22.32, 8, 80, (0.50, 0.68)),
    ("1x H100 80GB PCIe", 3.29, 1, 80, (0.70, 0.85)),
    ("4x H100 80GB SXM5", 16.36, 4, 80, (0.95, 1.05)),
    ("2x H100 80GB SXM5", 8.38, 2, 80, (0.95, 1.05)),
    ("1x H100 80GB SXM5", 4.29, 1, 80, (0.95, 1.05)),
    ("1x A10 24GB PCIe", 1.29, 1, 24, (0.20, 0.30)),
)
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

    print(f"\n=== projected wall clock and total, incl. {SETUP_HOURS:g} h setup ===")
    print("    (throughput vs GH200 is an ASSUMPTION -- the bracket, not a point)")
    print(f"{'instance':<22}{'$/GPU-h':>9}{'full bundle':>22}{'without d10_horizon':>22}")
    rows = []
    for name, price, gpus, vram, (lo, hi) in INSTANCES:
        def bracket(hours):
            a = makespan([h / hi for h in hours], gpus) + SETUP_HOURS
            b = makespan([h / lo for h in hours], gpus) + SETUP_HOURS
            return a, b
        fa, fb = bracket(all_h)
        na, nb = bracket(no_d10)
        rows.append((name, price / gpus, fa * price, fb * price, fa, fb, na, nb, price))
        print(f"  {name:<20}{price/gpus:>9.2f}"
              f"{f'${fa*price:,.0f}-{fb*price:,.0f} / {fa:.1f}-{fb:.1f} h':>22}"
              f"{f'${na*price:,.0f}-{nb*price:,.0f} / {na:.1f}-{nb:.1f} h':>22}")

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
    print(f"\n  d10_horizon is {d10_h:.1f} of the {tot:.1f} GPU-hours ({d10_h/tot*100:.0f}%),")
    print("  in TWO jobs, one of which no amount of parallelism shortens. It is also the")
    print("  lowest-value item here: suite 20's horizon claim is already withdrawn in")
    print("  either direction, so this pair adds a new measurement rather than settling")
    print("  a live question, and it is the only part needing an enlarged corpus.")
    print("  Running E1+E2 alone -- the arm PAPER 8.4's readouts depend on -- is the")
    print("  right-hand column above. Decide the two separately.")

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


def preflight(jobs: list[dict], allow_repeat: bool) -> int:
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
    print(f"  {'ok   ' if n_gpu else 'WARN '} cuda: {n_gpu} device(s) visible"
          + ("" if n_gpu else "  (jobs will run on CPU and take days)"))

    try:
        free = shutil.disk_usage(ROOT).free / 1024 ** 3
        need = 2.0 + 0.6 * len(jobs)      # ckpt.pt + best.pt + final.pt per job
        print(f"  {'ok   ' if free > need else 'FAIL '} disk: {free:.0f} GiB free, "
              f"~{need:.0f} GiB needed for {len(jobs)} jobs")
        bad += free <= need
    except OSError as e:
        print(f"  WARN  disk: could not stat ({e})")

    dirty = []
    for j in jobs:
        st = inspect_run(OUT_ROOT / j["id"])
        if st["status"] in ("partial", "suspect"):
            dirty.append((j["id"], st["status"], st["starts"], st["dones"]))
    if dirty:
        print(f"  FAIL  {len(dirty)} run dir(s) are not clean:")
        for jid, s, a, b in dirty:
            print(f"          {jid}: {s} ({a} start / {b} done records)")
        print("        metrics.jsonl is opened in APPEND mode, so re-running these")
        print("        would mix segments in one file -- the run128m_20k defect (D10).")
        print("        Archive them first: python3 scripts/gpu_bundle.py --reset-partial")
        bad += 1
    else:
        print("  ok    run dirs: no partial or multi-segment runs")

    for name, path in (("proxy transfer", TRANSFER),):
        t = load_transfer()
        if t:
            arms = ", ".join(f"{a} matrix_lr={v['matrix_lr']:g}"
                             for a, v in sorted(t.get("arms", {}).items()))
            print(f"  ok    {name}: {arms or 'empty'}")
        else:
            print(f"  info  {name}: none yet -- run --only e1_proxy first "
                  f"(e1_mup is blocked until it exists)")

    print(f"\npreflight: {'FAILED' if bad else 'clean'}")
    return 1 if bad else 0


def reset_partial(jobs: list[dict], root: Path) -> int:
    moved = 0
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for j in jobs:
        d = root / j["id"]
        st = inspect_run(d)
        if st["status"] in ("partial", "suspect"):
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
    ap.add_argument("--allow-data-repeat", action="store_true",
                    help="accept jobs whose token budget exceeds the corpus")
    ap.add_argument("--oversubscribe", action="store_true",
                    help="allow more concurrent jobs than visible GPUs (will likely OOM)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="launch without gating the box (not recommended)")
    args = ap.parse_args()

    transfer = load_transfer()
    jobs = build_matrix(sp_cells=args.sp_cells, transfer=transfer)
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
        jobs = picked

    if args.cost:
        return cost_report(jobs)
    if args.report:
        return report(root)
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

    if args.preflight:
        return preflight(jobs, args.allow_data_repeat)
    if not args.skip_preflight and not args.smoke:
        if preflight(jobs, args.allow_data_repeat):
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

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    _, n_tr, _ = corpus_tokens()
    meta = {"sp_cells": args.sp_cells, "workers": workers,
            "devices": devices, "smoke": bool(args.smoke),
            "corpus_train_tokens": n_tr,
            "transfer": (transfer or {}).get("arms", {})}

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
        records.append(rec)
    write_ledger(root, records, started, meta)

    todo = [(j, r) for j, r in zip(jobs, records) if r["status"] == "pending"]
    for j, r in zip(jobs, records):
        if r["status"] == "done":
            print(f"skip {j['id']} (done, best_val {r['best_val']:.4f})")
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
