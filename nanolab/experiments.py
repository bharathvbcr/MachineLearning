"""
nanolab.experiments — the deliberate experiments the guide says to run (§6.3).

Each "bake-off" holds seed/data/tokens fixed and changes exactly ONE variable,
then prints a ranked table of final val loss. This is the whole pedagogy of the
guide (§0, §6.3): the curves are the curriculum.

    python -m nanolab.experiments optimizer --preset phase2     # SGD/AdamW/Muon/...
    python -m nanolab.experiments schedule  --preset phase2     # constant/cosine/wsd/plateau
    python -m nanolab.experiments mixer     --preset phase2     # attention/mingru/mamba2/gdn
    python -m nanolab.experiments warmup    --preset cpu_smoke  # warmup ablation (§5.2)
    python -m nanolab.experiments overfit   --preset cpu_smoke  # overfitting demo (§6.3)
    python -m nanolab.experiments lrfind    --preset cpu_smoke  # LR finder (§5.1)
"""

from __future__ import annotations

import argparse

from .config import build_config, MIXERS, OPTIMIZERS, SCHEDULES
from .train import train


def _variants(base_cfg, field, values):
    out = []
    for v in values:
        d = base_cfg.to_dict()
        d[field] = v
        d["run_name"] = f"{base_cfg.run_name}_{field}_{v}"
        out.append((v, build_config(overrides=d)))
    return out


def _run_table(title, variants, overfit=0):
    print(f"\n########## BAKE-OFF: {title} ##########")
    print("Fixed: seed/data/tokens. Changing one variable per run (guide §6.3).\n")
    results = []
    for label, cfg in variants:
        print(f"\n----- variant: {label} -----")
        try:
            val = train(cfg, overfit=overfit)
        except Exception as e:
            print(f"  [variant {label} failed: {e}]")
            val = float("nan")
        results.append((label, val))
    results.sort(key=lambda r: (r[1] != r[1], r[1]))   # NaNs last
    print(f"\n========== {title}: ranked by val loss ==========")
    for rank, (label, val) in enumerate(results, 1):
        print(f"  {rank}. {label:<16} val_loss={val:.4f}")
    return results


def optimizer_bakeoff(base):
    # SGD's LR ceiling is far lower; give each its sensible peak so the contrast
    # is "optimizer", not "someone forgot to tune SGD".
    # each optimizer gets its own sensible peak LR so the contrast is the
    # optimizer, not a mistuned LR. Prodigy is LR-free (multiplier = 1).
    lr_for = {opt: base.lr for opt in OPTIMIZERS}
    lr_for.update({"sgd_momentum": 0.1, "lion": base.lr / 5,
                   "cautious_lion": base.lr / 5, "prodigy": 1.0})
    variants = []
    for opt in OPTIMIZERS:
        d = base.to_dict()
        d["optimizer"] = opt
        d["lr"] = lr_for[opt]
        d["run_name"] = f"{base.run_name}_opt_{opt}"
        variants.append((opt, build_config(overrides=d)))
    return _run_table(
        "broad optimizer funnel", variants)


def schedule_bakeoff(base):
    return _run_table("schedule (constant vs cosine vs WSD vs plateau)",
                      _variants(base, "schedule", list(SCHEDULES)))


def mixer_bakeoff(base):
    return _run_table("mixer (attention vs mingru vs mamba2 vs gdn)",
                      _variants(base, "mixer", list(MIXERS)))


def warmup_ablation(base):
    """Run once WITHOUT warmup and watch the early loss spike, then with warmup
    and watch it stabilize (guide §5.2)."""
    variants = []
    for w in (0, base.warmup_steps or 100):
        d = base.to_dict()
        d["warmup_steps"] = w
        d["run_name"] = f"{base.run_name}_warmup_{w}"
        variants.append((f"warmup={w}", build_config(overrides=d)))
    return _run_table("warmup ablation (0 vs warmup) — watch the early spike", variants)


def overfit_demo(base):
    """Train on a tiny data slice and watch val peel away from train — the
    clearest generalization lesson (guide §6.3)."""
    d = base.to_dict()
    d["run_name"] = f"{base.run_name}_overfit"
    d["max_steps"] = max(base.max_steps, 300)
    cfg = build_config(overrides=d)
    print("\n########## OVERFITTING DEMO (tiny data slice) ##########")
    print("Watch val_loss rise while train_loss keeps falling (guide §6.3).\n")
    return train(cfg, overfit=cfg.block_size * cfg.batch_size * 2)


def lrfind(base):
    """LR finder (guide §5.1): sweep LR up, plot loss vs LR, pick a notch below
    the knee."""
    from .data import Batcher, get_dataset
    from .model import build_model
    from .optim import build_optimizers
    from .schedules import lr_finder
    from .utils import pick_device, set_seed

    set_seed(base.seed)
    device = pick_device(base.device)
    data_dir, vocab, _ = get_dataset(base)
    if base.vocab_size == 0:
        base.vocab_size = vocab
    model = build_model(base).to(device)
    opts = build_optimizers(model, base)
    it = Batcher(data_dir, "train", base, device).iterator()
    hist = lr_finder(model, opts, it, base, device, num_iters=120)
    print("\n########## LR FINDER (guide §5.1) ##########")
    print(f"{'lr':>12} {'loss':>10}")
    knee_lr, best = None, float("inf")
    for lr, loss in hist:
        marker = ""
        if loss < best:
            best = loss
            knee_lr = lr
        print(f"{lr:12.2e} {loss:10.4f}{marker}")
    print(f"\nUsable ceiling ~ {hist[-1][0]:.1e}; lowest-loss LR ~ {knee_lr:.1e}.")
    print("Pick your peak LR a notch BELOW the knee (where loss starts to climb).")
    return hist


COMMANDS = {
    "optimizer": optimizer_bakeoff,
    "schedule": schedule_bakeoff,
    "mixer": mixer_bakeoff,
    "warmup": warmup_ablation,
    "overfit": overfit_demo,
    "lrfind": lrfind,
}


def main():
    p = argparse.ArgumentParser(description="nanolab experiment bake-offs (guide §6.3)")
    p.add_argument("command", choices=list(COMMANDS))
    p.add_argument("--preset", default="phase2")
    args, _ = p.parse_known_args()
    base = build_config(args.preset, {"run_name": f"exp_{args.command}"})
    COMMANDS[args.command](base)


if __name__ == "__main__":
    main()
