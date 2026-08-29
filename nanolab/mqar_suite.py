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


def e8_config(arm: str, seed: int, *, n_pairs=4, n_queries=4, n_keys=16,
              n_values=16, steps=3000, d_model=256, n_layer=12,
              batch_size=256) -> Config:
    spec = {a.name: a for a in ARMS}[arm]
    return Config(
        run_name=f"mqar_p{n_pairs}_b{batch_size}_t{steps}_{arm}_s{seed}",
        mixer=spec.mixer, layer_mixers=spec.layer_mixers or "",
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


def run_one(cfg: Config, device: str) -> dict:
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
    return {"run": cfg.run_name, "arm": cfg.mixer,
            "layer_mixers": cfg.layer_mixers, "seed": cfg.seed,
            "batch_size": cfg.batch_size, "n_pairs": cfg.mqar_n_pairs,
            "recall": recall, "solved": recall >= SOLVED,
            "final_loss": loss_v, "elapsed_s": time.time() - t0,
            "steps": cfg.max_steps}


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

    if not a.report_only:
        arms = [x for x in a.arms.split(",") if x]
        jobs = [(arm, s) for arm in arms for s in E8_SEEDS[:a.seeds]]
        print(f"{len(jobs)} run(s), {a.steps} steps each, {a.pairs} pairs, "
              f"device={a.device}")
        for i, (arm, seed) in enumerate(jobs, 1):
            cfg = e8_config(arm, seed, steps=a.steps, batch_size=a.batch,
                            n_pairs=a.pairs,
                            n_queries=a.pairs,
                            n_keys=max(16, 4 * a.pairs),
                            n_values=max(16, 4 * a.pairs))
            if cfg.run_name in done:
                print(f"[{i}/{len(jobs)}] skip {cfg.run_name} "
                      f"(done, recall {done[cfg.run_name]['recall']:.3f})")
                continue
            print(f"[{i}/{len(jobs)}] {cfg.run_name} ...", flush=True)
            rec = run_one(cfg, a.device)
            with ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            done[rec["run"]] = rec
            print(f"    -> recall {rec['recall']:.3f} "
                  f"{'SOLVED' if rec['solved'] else '-'} ({rec['elapsed_s']:.0f}s)")

    report(board(list(done.values())))


if __name__ == "__main__":
    main()
