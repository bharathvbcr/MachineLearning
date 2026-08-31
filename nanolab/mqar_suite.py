"""E8: the MQAR recall probe across the section 4.5 arm families.

Readout is a SUCCESS RATE, not a mean recall, and that is a measured decision
rather than a stylistic one. At a fixed config, identical but for
initialization, recall lands at 0.553 / 0.542 / 0.957 -- the induction head
either forms inside the budget or it does not, and training longer does not
rescue a run that missed it (a low seed sat at 0.566 after 8k steps). With
``set_seed`` pinned the same config reproduces exactly, so this is genuine
initialization sensitivity, not nondeterminism.

Averaging that distribution yields a number describing no model that exists.
``fraction of seeds reaching SOLVED`` is the honest statistic, it is what the
phase-transition literature reports, and it comes with a binomial interval
instead of a t-interval over a bimodal sample.

Runs untied: with ``tie_embeddings=True`` -- the default every section 4 suite
uses -- attention itself caps near 0.55 and the metric cannot separate the arms
at all. See nanolab/mqar.py for that measurement.

BATCH 256, and that is calibrated rather than chosen. At batch 32 the reference
arm does not form the head at all; at 256 it forms it every time, at the same
depth, the same 3000 steps, and essentially the same wall clock (the sequence is
15 tokens long, so a GH200 spends a small batch on kernel launches rather than
arithmetic). Measured 2026-08-28 on the GH200::

    L=4  d=256 bs=32   3000 steps   recall 0.552 / 0.550    55 s
    L=12 d=256 bs=32   3000 steps   recall 0.553 / 0.708   158 s
    L=4  d=256 bs=256  3000 steps   recall 1.000 / 1.000    64 s
    L=12 d=256 bs=256  3000 steps   recall 0.999 / 1.000   169 s

So the probe's answer to "can this architecture do in-context recall" is decided
by batch size before any architecture is compared: 0.55 says no, 1.00 says
perfectly, from the same model. Report the batch alongside any recall number.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import json
import math
import time
from pathlib import Path

import torch

from .config import Config
from .crossover_replicate import ARMS
from .model import build_model
from .mqar import MQARBatcher, recall_accuracy, vocab_for
from .optim import build_optimizers
from .utils import set_seed

# The run name carries the whole recipe -- pairs, batch, steps -- because resume
# keys on it. The first sweep skipped an attention seed as "done, recall 0.542"
# that had been trained at batch 32 for 6000 steps, into a board being built at
# batch 256 for 3000, and batch is the variable that decides whether the head
# forms at all. A name that does not carry the recipe is not an identifier.

# The section 4.5 board's five distinct families (backlog E8).
E8_ARMS = ("attention", "mingru", "gdn", "hybrid_mingru10_attn2",
           "hybrid_gdn_periodic")
E8_SEEDS = tuple(range(1, 16))          # 15: a rate needs more than a mean does
SOLVED = 0.80                           # sits in the gap between the two modes

DEFAULT_OUT = Path("nanolab/out/mqar_e8")

# E16: the same probe across sequence lengths that STRADDLE the SWA window.
# `block_size` is 4*pairs - 1 here (main passes n_queries = n_pairs), so:
#
#   pairs  16 -> seq  63   a 64-wide window spans the whole sequence, so
#                          `swa_w64` is EXACTLY `attention`. This cell is the
#                          control: if the harness separates them here, the
#                          separation at the other cells is an artifact.
#   pairs  64 -> seq 255   most keys sit outside a 64-wide window.
#   pairs 128 -> seq 511   8x the window; only the sinks reach back.
#
# The confound is named rather than hidden: in MQAR, sequence length IS also
# task difficulty (more pairs to store). So the SWA-vs-GDN comparison is only
# valid WITHIN a cell, where every arm faces the same difficulty. Reading a
# trend ACROSS cells cannot separate distance from difficulty -- that is the
# same axis PAPER section 6.8 already reports as difficulty-dependent.
E16_OUT = Path("nanolab/out/mqar_e16")
E16_PAIRS = (16, 64, 128)
E16_ARMS = ("attention", "gdn", "mingru", "swa_w64", "swa_w64_nosink")


def e8_config(arm: str, seed: int, *, n_pairs=4, n_queries=4, n_keys=16,
              n_values=16, steps=3000, d_model=256, n_layer=12,
              batch_size=256) -> Config:
    spec = {a.name: a for a in ARMS}[arm]
    # An arm's own knobs -- e.g. the SWA window -- are what make it that arm.
    # Without this, every `swa_*` arm here would silently train at the Config
    # default window and the whole board would be one architecture under four
    # names. `run_name` carries the arm NAME (not `spec.mixer`), so the ledger
    # keeps `swa_w64` and `swa_w64_nosink` apart.
    knobs = dict(spec.overrides)
    return Config(
        run_name=f"mqar_p{n_pairs}_b{batch_size}_t{steps}_{arm}_s{seed}",
        mixer=spec.mixer, layer_mixers=spec.layer_mixers or "", **knobs,
        seed=seed, batch_size=batch_size,
        mqar_n_pairs=n_pairs, mqar_n_queries=n_queries,
        mqar_n_keys=n_keys, mqar_n_values=n_values,
        block_size=2 * (n_pairs + n_queries) - 1,
        vocab_size=vocab_for(n_keys, n_values),
        d_model=d_model, n_layer=n_layer, n_head=4, head_dim=d_model // 4,
        tie_embeddings=False,           # see module docstring: required, not tuned
        fused_ce=False,                 # recall_accuracy needs logits
        optimizer="adamw", lr=1e-3, matrix_lr=1e-3,
        max_steps=steps,
    )


def run_one(cfg: Config, device: str, arm: str | None = None) -> dict:
    """Train one arm/seed and score exact-match recall. Deterministic per seed."""
    set_seed(cfg.seed)
    train_b = MQARBatcher(cfg, device, "train")
    val_b = MQARBatcher(cfg, device, "val")
    model = build_model(cfg).to(device)
    opts = build_optimizers(model, cfg)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if str(device).startswith("cuda") else contextlib.nullcontext())
    model.train()
    t0 = time.time()
    loss_v = float("nan")
    for _ in range(cfg.max_steps):
        x, y = train_b.batch()
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        loss_v = float(loss.detach())
        if not math.isfinite(loss_v):
            raise RuntimeError(f"{cfg.run_name}: loss went non-finite")
    recall = recall_accuracy(model, val_b, ctx, iters=25)
    # `cfg.mixer` is the FAMILY ("swa"), which would collapse swa_w64,
    # swa_w128 and swa_w64_nosink into one row. The arm name is the identifier.
    return {"run": cfg.run_name, "arm": arm or cfg.mixer,
            "swa_window": cfg.swa_window if cfg.mixer == "swa" else None,
            "swa_sinks": cfg.swa_sinks if cfg.mixer == "swa" else None,
            "layer_mixers": cfg.layer_mixers, "seed": cfg.seed,
            "batch_size": cfg.batch_size, "n_pairs": cfg.mqar_n_pairs,
            "recall": recall, "solved": recall >= SOLVED,
            "final_loss": loss_v, "elapsed_s": time.time() - t0,
            "steps": cfg.max_steps}


def _drop_workers_flag(argv: list[str]) -> list[str]:
    """Strip `--workers N` / `--workers=N` before re-invoking a worker.

    Both forms, and the VALUE of the space-separated form: leaving a bare `3`
    behind made argparse reject it in every worker at once. The suite refused
    the whole grid rather than running a partial one, which is why this is a
    caught bug and not a silently short board.
    """
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--workers":
            skip = True
            continue
        if a.startswith("--workers="):
            continue
        out.append(a)
    return out


def shard(jobs: list, workers: int, index: int) -> list:
    """Round-robin slice of `jobs` for worker `index` of `workers`.

    Round-robin, not contiguous blocks: arms differ several-fold in cost (E8
    measured GDN at 275 s against attention's 159 s on the same cell), so a
    contiguous split hands one worker every slow arm and the suite waits on it.
    """
    if workers < 1 or not 0 <= index < workers:
        raise ValueError(f"bad shard {index}/{workers}")
    return jobs[index::workers]


def spawn_workers(argv: list[str], workers: int) -> int:
    """Re-invoke this module once per worker, each on its own shard.

    The 225-run grid used to execute in one Python loop on a box that fits far
    more: these are 9.5M-parameter models, and E8's own numbers put a seq-15 run
    at ~72K tok/s, which is under 1% MFU -- the GPU is idle between kernel
    launches, not saturated. Workers share one ledger via an O_APPEND write of
    one JSON line per run, which POSIX keeps atomic below PIPE_BUF, so no lock
    is needed for records this size.

    Returns the first non-zero exit code, or 0.
    """
    procs = []
    for i in range(workers):
        env = dict(os.environ, MQAR_SHARD=f"{i}/{workers}")
        procs.append(subprocess.Popen(
            [sys.executable, "-u", "-m", "nanolab.mqar_suite", *argv], env=env))
    rc = 0
    for i, pr in enumerate(procs):
        code = pr.wait()
        if code and not rc:
            rc = code
        if code:
            print(f"  worker {i} exited {code}")
    return rc


def _parse_batch_by_cell(spec: str) -> dict[int, int]:
    """`{"16": 256, ...}` or `16:256,64:128`. Empty -> {}."""
    spec = (spec or "").strip()
    if not spec:
        return {}
    if spec.startswith("{"):
        return {int(k): int(v) for k, v in json.loads(spec).items()}
    out = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition(":")
        if not v:
            raise SystemExit(f"bad --batch-by-cell entry {part!r}; want pairs:batch")
        out[int(k)] = int(v)
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Binomial interval. A rate over 15 seeds is not a t-interval."""
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def board(records: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in records:
        by.setdefault(r["run"].rsplit("_s", 1)[0].replace("mqar_", ""), []).append(r)  # p{N}_{arm}
    rows = []
    for arm, rs in by.items():
        k, n = sum(bool(r["solved"]) for r in rs), len(rs)
        lo, hi = _wilson(k, n)
        vals = sorted(r["recall"] for r in rs)
        rows.append({"arm": arm, "solved": k, "n": n, "rate": k / n if n else 0.0,
                     "lo": lo, "hi": hi,
                     "median_recall": vals[len(vals) // 2] if vals else float("nan"),
                     "min_recall": vals[0] if vals else float("nan"),
                     "max_recall": vals[-1] if vals else float("nan")})
    rows.sort(key=lambda r: (-r["rate"], -r["median_recall"]))
    return rows


def report(rows: list[dict]) -> None:
    print(f"\n=== E8 MQAR: fraction of seeds forming the head "
          f"(recall >= {SOLVED:g}); rows are p<pairs>_<arm> ===")
    print("  %-24s %8s %22s %9s %s" % ("arm", "solved", "rate [95% Wilson]",
                                       "median", "min..max recall"))
    for r in rows:
        print("  %-24s %4d/%-3d %7.2f [%.2f, %.2f] %9.3f   %.3f..%.3f"
              % (r["arm"], r["solved"], r["n"], r["rate"], r["lo"], r["hi"],
                 r["median_recall"], r["min_recall"], r["max_recall"]))
    if rows and all(r["solved"] == 0 for r in rows):
        print("\n  NO arm formed the head on any seed. That is a statement about "
              "the budget,\n  not about the arms: re-check that the reference arm "
              "can saturate before\n  reading this as a recall result.")


def calibrate(cells, batches, device, seeds=3, steps=3000, ref="attention"):
    """Find, per cell, the smallest batch at which the REFERENCE arm saturates.

    The module docstring records why this is not optional: at batch 32 the
    reference arm never forms the induction head and at 256 it always does, on
    the same model. A cell whose reference arm cannot saturate measures the
    budget, not the architectures -- so every arm on it would score ~0 and the
    board would read as "SWA fails at long range" when it means "nothing was
    trainable here". Sequence length changes that threshold, so E8's calibrated
    256 does not transfer to a 511-token cell by assumption.

    Returns {pairs: batch_or_None}. None means NO tested batch saturated, which
    is a refusal to price the cell, not a default to fall back on.
    """
    picked: dict[int, int | None] = {}
    for pairs in cells:
        seq = 4 * pairs - 1
        picked[pairs] = None
        for bs in sorted(batches):
            recalls = []
            for seed in range(1, seeds + 1):
                cfg = e8_config(ref, seed, steps=steps, batch_size=bs,
                                n_pairs=pairs, n_queries=pairs,
                                n_keys=max(16, 4 * pairs),
                                n_values=max(16, 4 * pairs))
                t0 = time.time()
                rec = run_one(cfg, device, arm=ref)
                recalls.append(rec["recall"])
                print(f"  calib seq={seq:4} bs={bs:4} seed={seed} "
                      f"recall={rec['recall']:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
            if all(r >= SOLVED for r in recalls):
                picked[pairs] = bs
                print(f"  => seq={seq}: batch {bs} saturates the reference arm "
                      f"({seeds}/{seeds} seeds)")
                break
        if picked[pairs] is None:
            print(f"  => seq={seq}: NO tested batch saturated {ref}. This cell "
                  f"is not measurable at these batches; raise the batch or the "
                  f"step budget before running arms on it.")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--arms", default=",".join(E8_ARMS))
    ap.add_argument("--seeds", type=int, default=len(E8_SEEDS))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256,
                    help="calibrated: at 32 the reference arm never forms the "
                         "head, at 256 it always does (see module docstring)")
    ap.add_argument("--pairs", type=int, default=4,
                    help="key-value pairs per sequence; the difficulty axis. "
                         "Queries always equal pairs, so the positional shortcut "
                         "can be right for at most one query.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--cells", default="",
                    help="E16: comma-separated `pairs` values to sweep, e.g. "
                         "16,64,128 (block_size = 4*pairs-1 -> 63,255,511). "
                         "Overrides --pairs; each cell is its own board.")
    ap.add_argument("--calibrate", default="",
                    help="E16: comma-separated batch sizes to test per cell. "
                         "Finds the smallest batch at which the reference arm "
                         "saturates, writes calibration.json, and STOPS.")
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many worker processes over the grid. The "
                         "models are ~9.5M params and a single run leaves the "
                         "GPU mostly idle; workers share the ledger by atomic "
                         "line-append. Ignored when MQAR_SHARD is set.")
    ap.add_argument("--batch-by-cell", default="",
                    help="E14: JSON or `pairs:batch,...` from a calibration "
                         "pass. Required when --cells spans more than one cell, "
                         "because one batch does not serve every length.")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "runs.jsonl"
    done = {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[r["run"]] = r

    cells = [int(x) for x in a.cells.split(",") if x.strip()] or [a.pairs]

    if a.calibrate:
        batches = [int(x) for x in a.calibrate.split(",") if x.strip()]
        picked = calibrate(cells, batches, a.device, steps=a.steps)
        path = out / "calibration.json"
        path.write_text(json.dumps({str(k): v for k, v in picked.items()},
                                   indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
        unmeasurable = [k for k, v in picked.items() if v is None]
        if unmeasurable:
            raise SystemExit(
                f"cells {unmeasurable} have no saturating batch; refusing to "
                f"report them as calibrated")
        return

    batch_by_cell = _parse_batch_by_cell(a.batch_by_cell)
    if len(cells) > 1 and not batch_by_cell:
        raise SystemExit(
            "--cells spans several sequence lengths and batch is the variable "
            "that decides whether the head forms at all (see module docstring). "
            "Run --calibrate first and pass --batch-by-cell, or run one cell at "
            "a time with an explicit --batch.")

    shard_env = os.environ.get("MQAR_SHARD", "")
    if a.workers > 1 and not shard_env and not a.report_only:
        argv = _drop_workers_flag(sys.argv[1:])
        rc = spawn_workers(argv, a.workers)
        if rc:
            raise SystemExit(f"a worker failed (exit {rc}); grid incomplete")
        done = {}                      # re-read: the workers wrote it, not us
        if ledger.exists():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[r["run"]] = r
        report(board(list(done.values())))
        return

    widx, wtot = (int(x) for x in shard_env.split("/")) if shard_env else (0, 1)

    if not a.report_only:
        arms = [x for x in a.arms.split(",") if x]
        for pairs in cells:
            bs = batch_by_cell.get(pairs, a.batch)
            seq = 4 * pairs - 1
            jobs = shard([(arm, s) for arm in arms for s in E8_SEEDS[:a.seeds]],
                         wtot, widx)
            tag = f" shard {widx}/{wtot}" if wtot > 1 else ""
            print(f"\n=== cell pairs={pairs} seq={seq} batch={bs}{tag} "
                  f"({len(jobs)} runs, {a.steps} steps, device={a.device}) ===")
            for i, (arm, seed) in enumerate(jobs, 1):
                cfg = e8_config(arm, seed, steps=a.steps, batch_size=bs,
                                n_pairs=pairs,
                                n_queries=pairs,
                                n_keys=max(16, 4 * pairs),
                                n_values=max(16, 4 * pairs))
                if cfg.run_name in done:
                    print(f"[{i}/{len(jobs)}] skip {cfg.run_name} "
                          f"(done, recall {done[cfg.run_name]['recall']:.3f})")
                    continue
                print(f"[{i}/{len(jobs)}] {cfg.run_name} ...", flush=True)
                rec = run_one(cfg, a.device, arm=arm)
                rec["n_pairs"], rec["block_size"] = pairs, cfg.block_size
                with ledger.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                done[rec["run"]] = rec
                print(f"    -> recall {rec['recall']:.3f} "
                      f"{'SOLVED' if rec['solved'] else '-'} "
                      f"({rec['elapsed_s']:.0f}s)")

    report(board(list(done.values())))


if __name__ == "__main__":
    main()
