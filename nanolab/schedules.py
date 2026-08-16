"""
nanolab.schedules — learning-rate schedules (guide §5.3) + LR finder (§5.1).

Schedules to compare (the §5.3 table):
  constant  — flat after warmup; for debugging / LR-free optimizers
  cosine    — warmup -> cosine to floor; default for fixed-length runs
  wsd       — warmup-stable-decay; flexible run length, resume-friendly
  plateau   — ReduceLROnPlateau; the simplest reactive/self-correcting schedule

All non-reactive schedules are pure functions of ``step`` so the train loop can
just call ``lr = sched(step)``. ReduceLROnPlateau is stateful (it needs the val
loss), so it is a small class with the same call signature plus ``.observe()``.
"""

from __future__ import annotations

import math


def make_schedule(cfg):
    """Return a callable ``sched(step) -> lr`` (and, for plateau, a stateful
    object exposing ``.observe(val_loss)``)."""
    if cfg.schedule == "constant":
        return ConstantSchedule(cfg)
    if cfg.schedule == "cosine":
        return CosineSchedule(cfg)
    if cfg.schedule == "wsd":
        return WSDSchedule(cfg)
    if cfg.schedule == "plateau":
        return PlateauSchedule(cfg)
    raise ValueError(cfg.schedule)


def _peak_lr(cfg) -> float:
    # Muon's "lr" is the matrix LR; for scheduling the *relative* shape we use
    # the AdamW lr as the reference peak (the loop applies the multiplier to all
    # groups proportionally).
    return cfg.matrix_lr if cfg.optimizer in {
        "muon_ns5_adamw", "muon_ns3_adamw", "muon_polar_adamw",
        "normuon_adamw", "muown_adamw", "mona_adamw", "mimuon_adamw",
    } else cfg.lr


class _Base:
    reactive = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.peak = _peak_lr(cfg)
        self.warmup = cfg.warmup_steps
        self.total = cfg.lr_max_steps if cfg.lr_max_steps > 0 else cfg.max_steps

    def _warmup_mult(self, step):
        if self.warmup <= 0:
            return 1.0
        return min(1.0, (step + 1) / self.warmup)


class ConstantSchedule(_Base):
    def __call__(self, step):
        return self.peak * self._warmup_mult(step)


class CosineSchedule(_Base):
    """Warmup -> cosine decay to ``lr_floor_frac * peak`` (guide §5.3 code)."""

    def __call__(self, step):
        if step < self.warmup:
            return self.peak * self._warmup_mult(step)
        t = (step - self.warmup) / max(1, self.total - self.warmup)
        t = min(1.0, t)
        floor = self.cfg.lr_floor_frac
        return self.peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


class WSDSchedule(_Base):
    """Warmup-Stable-Decay: warmup -> long flat -> short linear decay. Flexible
    run length & resume-friendly (you can stop the flat phase anywhere)."""

    def __call__(self, step):
        if step < self.warmup:
            return self.peak * self._warmup_mult(step)
        decay_steps = int(self.cfg.wsd_decay_frac * self.total)
        decay_start = self.total - decay_steps
        if step < decay_start:
            return self.peak
        frac = (step - decay_start) / max(1, decay_steps)
        floor = self.cfg.lr_floor_frac
        return self.peak * (floor + (1 - floor) * (1 - frac))


class PlateauSchedule(_Base):
    """ReduceLROnPlateau — drop LR by ``factor`` when val loss stalls for
    ``patience`` evals. The simplest *reactive* schedule (guide §5.3)."""

    reactive = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.factor = cfg.plateau_factor
        self.patience = cfg.plateau_patience
        self.mult = 1.0
        self.best = math.inf
        self.bad = 0

    def __call__(self, step):
        return self.peak * self._warmup_mult(step) * self.mult

    def observe(self, val_loss):
        if val_loss < self.best - 1e-4:
            self.best = val_loss
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.mult *= self.factor
                self.bad = 0


def apply_lr(optimizers, lr, cfg):
    """Apply the schedule as a MULTIPLIER on each param group's *initial* LR, so
    per-group scaling (the Muon/Adam ratio AND μP's hidden-LR/width scaling, §10)
    is preserved across the schedule. The schedule shape is ``lr / peak``."""
    frac = lr / max(_peak_lr(cfg), 1e-12)
    for opt in optimizers:
        for g in opt.param_groups:
            if "initial_lr" not in g:
                g["initial_lr"] = g["lr"]      # snapshot the builder's per-group LR
            g["lr"] = g["initial_lr"] * frac


# ---------------------------------------------------------------------------
# LR finder (guide §5.1): sweep LR up exponentially, plot loss vs LR, the knee
# just before loss explodes marks the usable ceiling — pick the peak a notch
# below it.
# ---------------------------------------------------------------------------
def lr_finder(model, optimizers, data_iter, cfg, device,
              start_lr=1e-6, end_lr=1.0, num_iters=200):

    mult = (end_lr / start_lr) ** (1 / num_iters)
    lr = start_lr
    history = []
    best = math.inf
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = lr
    model.train()
    for i in range(num_iters):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
        loss_v = loss.item()
        history.append((lr, loss_v))
        best = min(best, loss_v)
        if loss_v > 4 * best and i > 10:        # exploded — stop early
            break
        lr *= mult
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = lr
    return history
