#!/usr/bin/env python3
"""Where each cell's learning-rate curve bottoms out, from committed run records.

    python3 scripts/lr_argmin.py SUITE [--compare OTHER_SUITE]

A *cell* is (d_model, mixer shape) -- the thing whose learning rate is being
chosen. Shape comes from ``layer_mixers`` in each run's own config, never from
the arm name: `periodic` carries three attention layers and `bookend` two, and a
board once compared them as though placement were the only difference.

The multiplier is derived as ``lr / 6e-4`` rather than parsed out of the arm
name, so a name and an override that disagree cannot both be believed.

**Every comparison here is paired by seed.** An earlier version of this file took
the plain mean of whatever seeds each point happened to have, and G11 showed what
that costs: it added seeds 42 and 100 to one point per curve and two argmins
appeared to move. Neither had. Seed 100 is the easiest seed at every cell on
these boards and 777 the hardest -- 4.1229 against 4.1803 at one of them, a
0.0574 spread, eighteen times the rerun floor. A mean over an unbalanced seed set
therefore says more about which seeds were run than about the learning rate. So
the grid argmin is read on the seeds common to *every* point, and the margin over
the runner-up on the seeds those two share.

Three ways this refuses to name an argmin, each of which has happened:
  * fewer than three points, or a minimum at either edge -- a grid that has not
    yet found a minimum (E21's first sweep picked its top multiplier in all six
    cells; every 50M grid so far has picked its bottom one);
  * no seed common to every point, so the curve cannot be read paired at all;
  * a margin over the runner-up at or under the 0.0031 rerun floor -- a basin
    flat at this grid spacing. d1152 attention sits here: 1x beats 0.5x by 0.0064
    paired on one seed, and the two straddle the 1/width prediction.

``--compare`` prints the same table for a second suite and the shift between the
two argmins per cell, which is the horizon-transfer question when the two suites
differ only in token budget.

No GPU. Reads ``metrics.jsonl`` + ``config.json`` only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nanolab.run_identity import group_key, identity_fields, unrecorded  # noqa: E402

BASE_LR = 6e-4

# Fields that describe how long a run trained rather than what it was. They are
# part of a run's identity -- two budgets are two experiments -- but they are
# held out of a cell's LABEL so that ``--compare`` can still line up the same
# architecture across a 50M and a 200M suite, which is the whole point of that
# flag. Holding them out of the label never merges two cells: grouping is on the
# full identity, and a label shared by two cells is reported as ambiguous rather
# than matched.
BUDGET_FIELDS = ("max_steps", "lr_max_steps", "warmup_steps", "eval_interval",
                 "eval_iters", "lr_floor_frac", "wsd_decay_frac")

# Rerun-to-rerun spread at a fixed seed on this harness, measured over the
# determinism suite: 0.0014 mean, 0.0031 max. A paired difference at or under the
# max is not a difference this rig can resolve.
RERUN_FLOOR = 0.0031


# Filled in by the last cells() call: why runs were left out. A reader that
# silently drops every run on a board and reports "no finished runs" is
# indistinguishable from one that looked and found nothing.
SKIPPED: dict = {}

# Filled in by the last cells() call: cells whose identity is ambiguous because
# two runs in the suite share a full identity AND a seed. Those cells are
# withheld from the argmin entirely -- see `cells()`.
COLLISIONS: dict = {}

# Filled in by the last cells() call: cells whose runs do not hold `matrix_lr`
# in a fixed ratio to `lr`, so "the multiplier" does not name a point on a
# one-dimensional curve. Also withheld -- see `cells()`.
NOT_ONE_D: dict = {}

# Filled in by the last cells() call: {field: how many runs predate it}. Those
# runs were grouped using `Config`'s default for the field. The substitution is
# safe on this corpus and is argued for in `run_identity.project`, but it is an
# assumption, so it is printed rather than kept quiet.
FILLED: dict = {}


class Cell:
    """One (architecture, recipe) whose learning rate is being chosen.

    The cell used to be ``(d_model, layer_mixers or mixer)``. That key was
    already the second attempt -- an earlier one read the shape off the arm name
    -- and it was still too coarse, because it names only the fields someone
    remembered. On the committed corpus it collapses:

      * ``crossover50m_loop32``: ``attention``, ``attn3``, ``attn6``,
        ``looped_attn3x4`` and ``looped_attn6x2`` all record ``mixer=attention``
        with an empty ``layer_mixers`` and differ on ``n_layer``/``n_loops``.
        Five architectures, one cell, and whichever run sorted last won.
      * ``crossover50m_swa2k``: ``swa_w512``, ``swa_w256``, ``swa_w64`` and
        ``swa_w64_nosink`` all record ``mixer=swa`` and differ on
        ``swa_window``/``swa_sinks``. Four cells reported as one.
      * ``gdn`` and ``gdn_pub`` differ only in ``gdn_rule``, so the published and
        repository recurrences share a cell wherever both were run.

    So the identity is the whole config (``run_identity``), minus the learning
    rate, which is the axis being swept. The LABEL is built from just the fields
    that actually vary inside this suite, so the printout stays readable and says
    what distinguishes the cell rather than what it happens to be called.
    """

    __slots__ = ("ident", "d_model", "shape", "varying", "_label")

    def __init__(self, ident, d_model, shape, varying):
        self.ident = ident
        self.d_model = d_model
        self.shape = shape
        self.varying = varying          # {field: value} that differ within suite
        self._label = None

    def __eq__(self, other):
        return isinstance(other, Cell) and self.ident == other.ident

    def __hash__(self):
        return hash(self.ident)

    @property
    def label(self) -> str:
        if self._label is None:
            extra = " ".join(f"{k}={v}" for k, v in sorted(self.varying.items()))
            self._label = f"{self.shape}{' ' + extra if extra else ''}"
        return self._label

    @property
    def sort_key(self) -> tuple:
        return (self.d_model or 0, self.label)

    def __repr__(self) -> str:
        return f"d{self.d_model} {self.label}"


def _read_final(metrics: Path) -> tuple[float | None, bool]:
    """(final_val, saw_done) from one metrics.jsonl."""
    final, saw_done = None, False
    for line in metrics.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "done":
            saw_done = True
            final = r.get("final_val")
    return final, saw_done


def cells(suite: str) -> dict:
    """{Cell: {mult: {seed: final_val}}} for one suite.

    Keyed by seed rather than appended to a list, so that no caller can average
    two points over different seed sets without first noticing they differ.

    Runs that cannot be read are counted into `SKIPPED` rather than dropped in
    silence. `crossover50m` is the case that forced this: 50 finished runs, none
    of them carrying `final_val` because the board predates the field, and this
    function reported the suite empty. Its end-of-schedule value is the last
    `eval` record -- which is what the published rows were computed from -- so
    "empty" was not just unhelpful, it was wrong.

    Two runs that share a full identity AND a seed are a genuine ambiguity: one
    of them has to be chosen and nothing in the record says which. That cell goes
    into `COLLISIONS` and is withheld, rather than resolved by sort order.
    """
    root = ROOT / "nanolab" / "out" / suite
    if not root.exists():
        raise SystemExit(f"no such suite: {root}")
    SKIPPED.clear()
    COLLISIONS.clear()
    NOT_ONE_D.clear()
    FILLED.clear()

    # First pass: read every readable run, so that the fields which vary across
    # the suite -- and therefore belong in a cell's label -- are known before any
    # label is built.
    runs = []
    for d in sorted(root.iterdir()):
        m, c = d / "metrics.jsonl", d / "config.json"
        if not (d.is_dir() and m.exists() and c.exists()):
            continue
        final, saw_done = _read_final(m)
        if final is None:
            key = ("finished but no final_val (board predates the field?)"
                   if saw_done else "no done record (still running, or died)")
            SKIPPED[key] = SKIPPED.get(key, 0) + 1
            continue
        runs.append((d, json.loads(c.read_text()), float(final)))
    if not runs:
        return {}

    extra_keys = {k for _, cfg, _ in runs for k in cfg}
    for _, cfg, _ in runs:
        for f in unrecorded(cfg, exclude=('lr', 'matrix_lr'), extra_keys=extra_keys):
            FILLED[f] = FILLED.get(f, 0) + 1
    candidates = [f for f in identity_fields(
        extra_keys=extra_keys, exclude=("lr", "matrix_lr") + BUDGET_FIELDS)
        if f not in ("d_model", "mixer", "layer_mixers")]

    def shape_of(cfg) -> str:
        return cfg.get("layer_mixers") or cfg.get("mixer", "?")

    # Scope the label to fields that vary among runs of the SAME shape. A suite
    # holding both `attention` and `swa` arms varies on `swa_window` overall, but
    # tagging the attention cells with a window they never read says nothing
    # about them -- the distinguishing field for a cell is one its own siblings
    # disagree about.
    # A field earns a place in the label only if peers disagree about a value
    # they both RECORDED. A field half the suite predates is a real unknown --
    # it splits the identity, deliberately -- but "mqar_n_keys=None" in a label
    # describes the age of a config file, not the experiment, and printing the
    # absence as `None` would also spell it the same way as a recorded null.
    label_fields: dict[str, list[str]] = {}
    for shape in {shape_of(cfg) for _, cfg, _ in runs}:
        peers = [cfg for _, cfg, _ in runs if shape_of(cfg) == shape]
        label_fields[shape] = [
            f for f in candidates
            if len({repr(c[f]) for c in peers if f in c}) > 1]

    # The swept axis is the JOINT multiplier: every committed sweep scales `lr`
    # and `matrix_lr` together (which is exactly why an argmin here cannot be
    # read as an Adam-group result -- the Muon group moved too). So both are held
    # out of the cell identity and the point is named by `lr / BASE_LR`.
    #
    # That naming is only valid while `matrix_lr / lr` is fixed inside the cell.
    # `gpu_bundle`'s `e1_proxy_attention_mlr*` runs sweep `matrix_lr` at a FIXED
    # `lr`, so they would all land on one multiplier and overwrite each other.
    # A cell like that is not a one-dimensional learning-rate curve, and this
    # reader is a one-dimensional reader: it withholds it and says so, rather
    # than reporting an argmin over an axis that is not an axis.
    by_cell: dict = {}
    for d, cfg, final in runs:
        shape = shape_of(cfg)
        cell = Cell(
            ident=group_key(cfg, exclude=("lr", "matrix_lr"), extra_keys=extra_keys),
            d_model=cfg.get("d_model"),
            shape=shape,
            varying={f: cfg[f] for f in label_fields[shape] if f in cfg},
        )
        by_cell.setdefault(cell, []).append((d, cfg, final))

    out: dict = {}
    for cell, members in by_cell.items():
        ratios = {round(float(cfg.get("matrix_lr") or 0.0) / cfg["lr"], 9)
                  for _, cfg, _ in members if cfg.get("lr")}
        if len(ratios) > 1:
            NOT_ONE_D[cell] = sorted(ratios)
            continue
        seen: dict = {}
        curve: dict = {}
        for d, cfg, final in members:
            mult = round(cfg["lr"] / BASE_LR, 4)
            seed = cfg.get("seed")
            prev = seen.get((mult, seed))
            if prev is not None:
                COLLISIONS.setdefault(cell, []).append((prev, d.name))
                continue
            seen[(mult, seed)] = d.name
            curve.setdefault(mult, {})[seed] = final
        if cell not in COLLISIONS:
            out[cell] = curve
    return out


def shared_seeds(curve: dict, mults=None) -> set:
    """Seeds present at every one of `mults` (default: every point on the curve)."""
    mults = sorted(curve) if mults is None else list(mults)
    if not mults:
        return set()
    common = set(curve[mults[0]])
    for m in mults[1:]:
        common &= set(curve[m])
    return common


def paired_means(curve: dict, seeds) -> dict:
    """{mult: mean over exactly `seeds`}. Caller guarantees each mult has them."""
    return {m: statistics.mean(curve[m][s] for s in seeds) for m in curve}


def argmin(curve: dict) -> tuple[float, str]:
    """(multiplier at the lowest paired mean, verdict).

    `verdict` is the string "interior" only when the argmin is one this rig can
    defend: read on a seed set common to the whole grid, off at least three
    points, not at either edge, and ahead of the runner-up by more than the
    rerun floor. Every other outcome returns a sentence saying what is missing.
    Callers gate on ``verdict == "interior"``, so each of these fails closed.
    """
    if not curve:
        return float("nan"), "no finished runs"
    common = shared_seeds(curve)
    if not common:
        per = ", ".join(f"{m:g}x:{sorted(curve[m])}" for m in sorted(curve))
        return float("nan"), (
            "no seed is common to every point, so the grid cannot be read "
            f"paired ({per}); rerun the odd points onto the shared seeds")
    means = paired_means(curve, common)
    order = sorted(means, key=lambda k: means[k])
    best = order[0]
    n = len(common)
    tag = f"paired on {n} seed{'s' if n != 1 else ''} {sorted(common)}"
    if len(means) < 3:
        return best, f"only {len(means)} point(s); no argmin from fewer than 3 [{tag}]"
    if best == min(means):                       # lowest multiplier on the grid
        return best, f"EDGE (bottom) -- extend the grid DOWN; not an argmin [{tag}]"
    if best == max(means):
        return best, f"EDGE (top) -- extend the grid UP; not an argmin [{tag}]"

    # The runner-up decides whether the argmin is located or the basin is flat.
    # Read that one comparison on every seed those two points share, which is
    # usually more than the whole grid has in common.
    up = order[1]
    pair = shared_seeds(curve, (best, up))
    deltas = [curve[up][s] - curve[best][s] for s in sorted(pair)]
    margin = statistics.mean(deltas)
    wins = sum(1 for d in deltas if d > 0)
    pair_tag = (f"{best:g}x beats {up:g}x by {margin:.4f} on {wins}/{len(deltas)} "
                f"of the {len(deltas)} seeds they share")
    if margin <= RERUN_FLOOR:
        return best, (f"FLAT -- {pair_tag}, at or under the {RERUN_FLOOR} rerun "
                      f"floor; the argmin is not resolved at this grid spacing [{tag}]")
    return best, "interior"


def show(suite: str) -> dict:
    data = cells(suite)
    skipped, collided, flat = dict(SKIPPED), dict(COLLISIONS), dict(NOT_ONE_D)
    print(f"\n=== {suite} ===")
    for why, n in sorted(skipped.items()):
        print(f"  NOTE: {n} run(s) not read -- {why}")
    if FILLED:
        fields = ", ".join(f"{f} ({c})" for f, c in sorted(FILLED.items()))
        print(f"  NOTE: grouped using Config defaults for field(s) some configs "
              f"predate -- {fields}. Each default was chosen to match the "
              f"behaviour already on disk; see run_identity.project.")
    for cell, pairs in sorted(collided.items(), key=lambda kv: kv[0].sort_key):
        print(f"  WITHHELD d{cell.d_model} {cell.label}: two runs share this cell's "
              f"full identity AND a seed, so which one the point takes is decided "
              f"by sort order and nothing else.")
        for a, b in pairs:
            print(f"      {a}  vs  {b}")
    for cell, ratios in sorted(flat.items(), key=lambda kv: kv[0].sort_key):
        print(f"  WITHHELD d{cell.d_model} {cell.label}: matrix_lr/lr takes "
              f"{len(ratios)} values here ({', '.join(f'{r:g}' for r in ratios)}), "
              f"so this is a two-dimensional grid and 'the multiplier' does not "
              f"name a point on it. Read the two axes separately.")
    if not data:
        if skipped or collided or flat:
            print("  REFUSING to report this suite as empty: every run was skipped "
                  "or withheld for the reason(s) above, which is not the same as "
                  "finding nothing.")
        else:
            print("  (no finished runs)")
        return {}
    picks = {}
    for cell in sorted(data, key=lambda c: c.sort_key):
        curve = data[cell]
        best, verdict = argmin(curve)
        picks[cell] = (best, verdict)
        common = shared_seeds(curve)
        pts = "  ".join(
            f"{m:g}x:"
            + (f"{statistics.mean(curve[m][s] for s in common):.4f}"
               if common else "----")
            + ("*" if m == best else "")
            + f"(n{len(curve[m])})"
            for m in sorted(curve))
        print(f"  d{cell.d_model} {cell.label[:60]:62s}")
        print(f"      {pts}")
        dropped = sorted({s for m in curve for s in curve[m]} - common)
        if dropped:
            print(f"      means paired on {sorted(common)}; "
                  f"{len(dropped)} seed(s) {dropped} not at every point, held out")
        print(f"      argmin {best:g}x -- {verdict}")
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite")
    ap.add_argument("--compare", help="second suite to read the same way")
    a = ap.parse_args()
    picks = show(a.suite)
    if not a.compare:
        return
    other = show(a.compare)
    print(f"\n=== argmin shift: {a.suite} vs {a.compare} ===")

    # Cells are grouped on the full identity, which includes the token budget --
    # so the same architecture at 50M and at 200M is deliberately two identities
    # and cannot be matched on that key. Match on (d_model, label), which holds
    # the budget fields out. A label that is not unique inside its own suite is
    # ambiguous and is reported rather than paired off arbitrarily.
    def by_label(picked: dict) -> tuple[dict, set]:
        seen, dupes = {}, set()
        for cell in picked:
            k = (cell.d_model, cell.label)
            if k in seen:
                dupes.add(k)
            seen[k] = cell
        return seen, dupes

    left, ldup = by_label(picks)
    right, rdup = by_label(other)
    for k in sorted(ldup | rdup):
        print(f"  AMBIGUOUS d{k[0]} {k[1]}: more than one cell in a suite carries "
              f"this label once the budget fields are held out; not matched.")
    shared = sorted(set(left) & set(right) - ldup - rdup)
    if not shared:
        print("  no cell appears in both suites")
        return
    for k in shared:
        b1, v1 = picks[left[k]]
        b2, v2 = other[right[k]]
        note = "" if v1 == v2 == "interior" else \
            "   [at least one side is not a located argmin; shift is a lower bound]"
        print(f"  d{k[0]} {k[1][:38]:40s} {b2:g}x -> {b1:g}x"
              f"  ({'no move' if b1 == b2 else f'x{b1 / b2:g}'}){note}")


if __name__ == "__main__":
    main()
