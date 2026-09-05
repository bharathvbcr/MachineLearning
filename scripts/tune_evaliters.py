"""What does `eval_iters = 20` buy, and what would a smaller value cost?

Evaluation is ~15% of a 50M run (61 evals x 20 iterations x a forward pass), and
`eval_iters` is a recipe field, so cutting it is a scientific decision rather
than a tuning one. The decision needs the number nobody has measured: how much
of a board's REPORTED quantity is eval noise.

The subtlety that makes this worth measuring rather than assuming: `Batcher`
seeds the val split with `cfg.seed + 1`, so two arms at the same seed evaluate on
IDENTICAL batches. Eval noise is therefore largely common-mode and cancels in the
paired per-seed difference the boards report -- but only to the extent the two
models respond alike to the same batch, which is an empirical question.

Loads two trained checkpoints at the same seed, scores both on the same K val
batches, and reports the standard error of (a) each arm's mean loss and (b) the
paired difference, as a function of eval_iters.

    python scripts/tune_evaliters.py --suite nanolab/out/crossover50m_loop32 \
        --a cx32loop_attention_s1337 --b cx32loop_attn6_s1337 --batches 200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402


def per_batch_losses(run_dir: Path, n_batches: int):
    from nanolab.config import Config
    from nanolab.data import Batcher, get_dataset
    from nanolab.model import build_model
    cfg_d = json.loads((run_dir / "config.json").read_text())
    cfg = Config(**{k: v for k, v in cfg_d.items() if k in Config.__dataclass_fields__})
    state = torch.load(run_dir / "ckpt.pt", map_location="cuda", weights_only=False)
    model = build_model(cfg).to("cuda")
    model.load_state_dict(state["model"])
    model.eval()
    data_dir, _, _ = get_dataset(cfg)
    # Same construction as train.evaluate: the val generator is seeded cfg.seed+1,
    # so two arms at one seed walk the SAME batch sequence.
    val = Batcher(Path(data_dir), "val", cfg, "cuda")
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    out = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = val.batch()
            with ac:
                _, loss = model(x, y)
            out.append(float(loss))
    return torch.tensor(out), cfg.seed, state.get("step")


def se_at(sd: float, n: int) -> float:
    return sd / math.sqrt(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--out", default="nanolab/out/_tune/evaliters.json")
    args = ap.parse_args()

    suite = REPO / args.suite
    la, seed_a, step_a = per_batch_losses(suite / args.a, args.batches)
    lb, seed_b, step_b = per_batch_losses(suite / args.b, args.batches)
    if seed_a != seed_b:
        raise SystemExit(f"seeds differ ({seed_a} vs {seed_b}); pairing is meaningless")

    d = la - lb
    res = {
        "a": args.a, "b": args.b, "seed": seed_a, "steps": [step_a, step_b],
        "batches": args.batches,
        "mean_a": float(la.mean()), "sd_a": float(la.std()),
        "mean_b": float(lb.mean()), "sd_b": float(lb.std()),
        "mean_diff": float(d.mean()), "sd_diff": float(d.std()),
        "corr": float(torch.corrcoef(torch.stack([la, lb]))[0, 1]),
    }
    print(f"  {args.a}: mean {res['mean_a']:.4f}  sd {res['sd_a']:.4f}")
    print(f"  {args.b}: mean {res['mean_b']:.4f}  sd {res['sd_b']:.4f}")
    print(f"  per-batch correlation between the two arms: {res['corr']:.4f}")
    print(f"  paired difference: mean {res['mean_diff']:+.4f}  sd {res['sd_diff']:.4f}")
    print()
    print(f"  {'eval_iters':>10} {'SE(arm mean)':>14} {'SE(paired diff)':>17}   "
          f"{'eval cost/run':>14}")
    res["by_n"] = {}
    for n in (5, 10, 20, 40):
        sa, sd_ = se_at(res["sd_a"], n), se_at(res["sd_diff"], n)
        res["by_n"][n] = {"se_arm": sa, "se_diff": sd_}
        print(f"  {n:>10} {sa:>14.4f} {sd_:>17.4f}   {n / 20 * 100:>13.0f}%")
    print()
    print("  reference effect size: the 8+4 hybrid's paired gap at 50M is "
          "-0.0176 nats (docs/architecture-review-2026-09-04)")
    Path(REPO / args.out).write_text(json.dumps(res, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
