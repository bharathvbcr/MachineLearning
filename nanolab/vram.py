"""VRAM sizing for the job runners — one owner, used by both.

`scripts/gpu_bundle.py` has refused an over-subscribed plan since it was
written; `crossover_replicate launch` never did, and on 2026-09-05 that cost two
minGRU jobs at d_model 1536 to CUDA OOM. The board that consumed their absence
then picked a learning rate from the one survivor, which sat on the edge of the
swept range, and ran ten 50M-token jobs at it.

The deeper defect was in the sizing itself. `JOB_VRAM_DEFAULT_GIB = 23.3` was
documented as sizing an unknown shape "pessimistically rather than
optimistically" -- but 23.3 is the worst cell measured *at d_model 768*, so for
1536 it is optimistic by more than a factor of three. An unmeasured shape was
handed a number indistinguishable from a measurement. A check that could not run
must never report the same result as a check that ran and passed, so an
unmeasured shape is now SCALED and flagged, never defaulted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Measured on a GH200, batch 32 / ctx 512 / 12 layers. Measured, not computed.
MEASURED_JOB_VRAM_GIB: dict[tuple[str, int, int], float] = {
    ("attention", 768, 32): 17.0,
    ("mingru", 768, 32): 23.3,
}
# Anchor width and shape for the scaling below: what every measured cell is.
_ANCHOR_D_MODEL = 768
_ANCHOR_BATCH = 32
_ANCHOR_CTX = 512
# fp32 weights + grads + two Adam moments. At d_model 768 that is 1.84 GiB of a
# 23.3 GiB job, i.e. ACTIVATIONS DOMINATE and a parameter-ratio scaling is the
# wrong model -- it over-sized d_model 1152 badly enough to refuse a plan that
# demonstrably ran. Splitting the two terms is what makes the estimate fit both
# observations this repo actually has.
_OPT_STATE_BYTES_PER_PARAM = 16
# Applied on top of the scaled figure for a shape nobody has measured. Kept at
# 1.10 rather than something rounder because the two known points bracket it:
# minGRU d_model 1152 ran two-to-a-device on a 94.5 GiB card and must stay
# allowed, and minGRU 1536 OOMed two-to-a-device and must be refused. A larger
# margin refuses the first; a smaller one is thinner than the extrapolation
# deserves.
UNMEASURED_SAFETY = 1.10
VRAM_HEADROOM = 0.85            # refuse a plan that would fill more than this


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


def _params_at(mixer: str, d_model: int) -> float:
    from .config import Config
    return float(Config(mixer=mixer, d_model=d_model,
                        n_head=max(1, d_model // 64), head_dim=64
                        ).estimate_params())


def job_vram_gib(mixer: str, d_model: int, batch_size: int,
                 block_size: int = _ANCHOR_CTX) -> tuple[float, bool]:
    """(GiB for one job, whether that figure was measured rather than scaled).

    Two terms, because they scale differently and the anchor is dominated by the
    one that does NOT follow parameter count:

      optimizer + weights + grads   ~ params        (d_model^2)
      activations                   ~ d_model * batch * ctx

    The activation share is backed out of the measured anchor rather than
    modelled from first principles, so the fit is exact at the anchor by
    construction and the extrapolation is in the scaling law alone.
    """
    key = (mixer, d_model, batch_size)
    if key in MEASURED_JOB_VRAM_GIB and block_size == _ANCHOR_CTX:
        return MEASURED_JOB_VRAM_GIB[key], True
    anchor = MEASURED_JOB_VRAM_GIB.get((mixer, _ANCHOR_D_MODEL, _ANCHOR_BATCH))
    if anchor is None:
        # Unknown MIXER, not just an unknown width: size on the worst cell there
        # is and still scale it. Refusing a plan that would have fit costs a
        # worker slot; admitting one that will not costs the run.
        anchor = max(MEASURED_JOB_VRAM_GIB.values())
    state_anchor = (_params_at(mixer, _ANCHOR_D_MODEL)
                    * _OPT_STATE_BYTES_PER_PARAM / 1024 ** 3)
    act_anchor = max(0.0, anchor - state_anchor)
    state = _params_at(mixer, d_model) * _OPT_STATE_BYTES_PER_PARAM / 1024 ** 3
    act = act_anchor * (d_model / _ANCHOR_D_MODEL) \
        * (batch_size / _ANCHOR_BATCH) * (block_size / _ANCHOR_CTX)
    return (state + act) * UNMEASURED_SAFETY, False


def safe_workers(per_job_gib: float, total_gib: float,
                 headroom: float = VRAM_HEADROOM) -> int:
    """How many jobs of that size fit under the headroom. 0 if unknown."""
    if per_job_gib <= 0 or total_gib <= 0:
        return 0
    return max(1, int(total_gib * headroom / per_job_gib))


def check_plan(shapes: list[tuple[str, int, int]], workers: int,
               total_gib: float) -> tuple[bool, str]:
    """(ok, message). Sized on the HEAVIEST shape in the set, not the average:
    the scheduler is free to put the expensive ones together, and it does."""
    if not shapes or total_gib <= 0:
        return True, ""
    sized = [(job_vram_gib(*s), s) for s in shapes]
    (per, measured), worst = max(sized, key=lambda x: x[0][0])
    safe = safe_workers(per, total_gib)
    if workers <= safe:
        return True, ""
    how = "measured" if measured else "SCALED from the 768 cell, not measured"
    return False, (
        f"refusing --workers {workers}: the heaviest job queued is "
        f"{worst[0]} d_model={worst[1]} batch={worst[2]} at {per:.1f} GiB "
        f"({how}), so {workers} of them need {workers * per:.1f} GiB of a "
        f"{total_gib:.1f} GiB device ({100 * workers * per / total_gib:.0f}%).\n"
        f"  Use --workers {safe}, or --ignore-vram if the figure is wrong for "
        f"your shapes.")
