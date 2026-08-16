"""
Live loss plot for a nanolab training run.

Usage:
    python -m nanolab.plot_live                        # default: run128m_20k
    python -m nanolab.plot_live run128m_10k           # any run name
    python -m nanolab.plot_live nanolab/out/my_run    # full path
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---- resolve run directory ----
if len(sys.argv) > 1:
    arg = sys.argv[1]
    run_dir = Path(arg) if Path(arg).exists() else Path("nanolab/out") / arg
else:
    run_dir = Path("nanolab/out/run128m_20k")

metrics_file = run_dir / "metrics.jsonl"
PREV_BEST = 3.621   # run128m_10k baseline


def read_metrics():
    train_steps, train_loss = [], []
    val_steps, val_loss = [], []
    if not metrics_file.exists():
        return (train_steps, train_loss), (val_steps, val_loss)
    with open(metrics_file, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("event") == "train" and r.get("loss", 0) > 0:
                    train_steps.append(r["step"])
                    train_loss.append(r["loss"])
                elif r.get("event") == "eval" and r.get("val_loss", 0) > 0:
                    val_steps.append(r["step"])
                    val_loss.append(r["val_loss"])
            except Exception:
                pass
    return (train_steps, train_loss), (val_steps, val_loss)


def smooth(values, k=50):
    if len(values) < k:
        return list(range(len(values))), values
    out = []
    for i in range(len(values) - k + 1):
        out.append(sum(values[i:i + k]) / k)
    return list(range(k - 1, len(values))), out


fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Training: {run_dir.name}  (refreshes every 10s)", fontsize=12)


def update(_frame):
    (ts, tl), (vs, vl) = read_metrics()

    ax_train.cla()
    ax_train.set_title("Train loss (step log)")
    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Loss")
    ax_train.grid(True, alpha=0.3)
    if tl:
        ax_train.plot(ts, tl, color="steelblue", alpha=0.25, linewidth=0.6, label="raw")
        si, sl = smooth(tl, k=min(50, max(1, len(tl) // 10)))
        ax_train.plot([ts[i] for i in si], sl, color="steelblue", linewidth=2, label="smoothed")
        ax_train.axhline(PREV_BEST, color="green", linestyle="--", linewidth=1, label=f"prev best {PREV_BEST}")
        ax_train.legend(fontsize=8)
        ax_train.set_ylim(bottom=min(min(tl) * 0.97, PREV_BEST * 0.97))

    ax_val.cla()
    ax_val.set_title("Val loss (eval checkpoints)")
    ax_val.set_xlabel("Step")
    ax_val.set_ylabel("Loss")
    ax_val.grid(True, alpha=0.3)
    if vl:
        ax_val.plot(vs, vl, "o-", color="tomato", linewidth=2, markersize=6, label="val loss")
        ax_val.axhline(PREV_BEST, color="green", linestyle="--", linewidth=1.5, label=f"prev best {PREV_BEST}")
        ax_val.legend(fontsize=8)
        ax_val.set_ylim(bottom=min(min(vl) * 0.97, PREV_BEST * 0.97))
        # annotate latest val
        ax_val.annotate(f"{vl[-1]:.4f}", (vs[-1], vl[-1]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
    elif not metrics_file.exists():
        ax_val.text(0.5, 0.5, "Waiting for metrics.jsonl…",
                    ha="center", va="center", transform=ax_val.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.94])


ani = animation.FuncAnimation(fig, update, interval=10_000, cache_frame_data=False)
update(None)   # draw immediately on launch
plt.show()
