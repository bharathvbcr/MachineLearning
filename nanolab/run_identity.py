"""What makes two runs the same experiment. The canonical owner of that question.

Every reader in this tree groups runs: a board pairs an arm against a reference
by seed, an LR sweep groups a curve by cell, an inventory groups cells. Each of
them used to carry its own idea of which config fields matter, and each was
wrong in the same direction -- they named the fields someone remembered, so a
field nobody remembered could differ between two runs without anyone being told.

The concrete failures this replaces
-----------------------------------
``paired_board.RECIPE_KEYS`` listed ten fields out of ``Config``'s hundred-odd:
batch, block, eval_iters, the two step counts, the two learning rates, warmup,
schedule and optimizer. It therefore waved through ``compile``, ``dtype``,
``dataset``, ``grad_accum``, ``fused_ce``, ``weight_decay``, ``tf32``,
``eval_train`` and every architecture flag. The manuscript's 50M-to-200M budget
comparison says "only the budget and cosine horizon change"; the old width-384
configs record ``compile: false`` and their 200M counterparts ``compile: true``,
and the guard could not see it.

``lr_argmin.cells`` keyed a cell on ``(d_model, layer_mixers)``. ``gdn`` and
``gdn_pub`` produce the *same* ``layer_mixers`` string -- the recurrence differs
in ``gdn_rule``, which is not in the key -- so the two arms landed in one cell
and the last one read off disk silently won. The archived ``recompute.py`` hits
the same collision from the other side and now dies on the current corpus with
``duplicate MQAR cell seed ... mqar_e28_p8``.

The fix is the direction of the list
------------------------------------
An allowlist of fields that matter will always be missing the field someone adds
tomorrow, and its failure mode is silence. So identity is the DENYLIST below: a
``Config`` field is part of a run's identity unless it is named here with a
reason. Add a field to ``Config`` and it is identity-relevant by default; the
guard tightens on its own rather than quietly loosening.

Three states, not two
---------------------
Comparing a 2026-08 config against a 2026-09 one is not a two-way question. A
field the older run never recorded is **unknown**, and unknown is not equal.
Filling it with today's default would assert something about that run that
nobody measured -- exactly the ``compile`` case above, where assuming the modern
default would have claimed the old runs compiled when they did not. So
``compare`` returns ``differ``, ``unknown`` and ``by_design`` separately, and a
caller that cannot tell the difference between "checked and matched" and "could
not check" is a caller that reports an unexamined pairing as a verified one.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping

from .config import Config

__all__ = [
    "MISSING",
    "NON_IDENTITY",
    "PAIRING_FIELD",
    "Comparison",
    "compare",
    "identity_fields",
    "group_key",
    "project",
]


class _Missing:
    """A field the run never recorded. Not a value, and never equal to one."""

    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __repr__(self) -> str:
        return "<not recorded>"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()

# Runs are paired BY seed, so the seed identifies which run within an arm rather
# than which experiment. Every comparison holds it out; every grouping uses it as
# the key runs are collected under.
PAIRING_FIELD = "seed"

# The only ``Config`` fields that do not enter a run's identity. Each one is here
# because it cannot change the numbers, and the reason is recorded so that the
# next person to widen this set has to argue with it.
NON_IDENTITY: dict[str, str] = {
    "run_name": "the run's name; the thing being compared, not a recipe field",
    "out_dir": "where artifacts are written",
    "data_dir": "where the corpus is read from; the corpus itself is `dataset`",
    "log_interval": "how often a step line is printed; no effect on the trajectory",
    "ckpt_interval": "how often a checkpoint is written; no effect on the trajectory",
    "mem_fraction": "allocator cap -- turns an over-budget step into a clean OOM "
                    "rather than host-RAM spill; changes whether a run survives, "
                    "never what it computes",
    "optimizer_state_limit_gib": "a pre-flight guard rail, not a hyperparameter",
    "copy_probe": "the repeated-span probe draws from its own fixed generator "
                  "(train.copy_probe_loss, seed 0xC0DE) under no_grad and restores "
                  "training mode, so a run with it on trains identically to one "
                  "without -- verified, not assumed",
}


def identity_fields(extra_keys: Iterable[str] = (),
                    exclude: Iterable[str] = ()) -> tuple[str, ...]:
    """Every field whose value is part of a run's identity, sorted.

    ``extra_keys`` lets a caller fold in keys a recorded config carries that
    today's ``Config`` no longer declares. A field that was removed from the
    dataclass but still sits in two configs with different values is still a
    difference between those runs, and dropping it because the code moved on
    would be the same silence this module exists to remove.
    """
    drop = set(NON_IDENTITY) | {PAIRING_FIELD} | set(exclude)
    names = {f.name for f in dataclasses.fields(Config)} | set(extra_keys)
    return tuple(sorted(names - drop))


def _defaults() -> dict[str, Any]:
    out = {}
    for f in dataclasses.fields(Config):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:   # type: ignore[misc]
            out[f.name] = f.default_factory()                # type: ignore[misc]
    return out


def unrecorded(cfg: Mapping[str, Any], exclude: Iterable[str] = (),
               extra_keys: Iterable[str] = ()) -> tuple[str, ...]:
    """Identity fields ``Config`` declares that this config never wrote down."""
    src = normalize(cfg)
    return tuple(k for k in identity_fields(extra_keys=extra_keys, exclude=exclude)
                 if k not in src and k in _defaults())


# Fields ``Config`` computes from another field when they are left at 0, listed
# with what they are computed from. They are still identity -- the resolved value
# is what the run used -- but they are not INDEPENDENT of their source, so an arm
# that declares the source declares these too. Without this, pairing a width-384
# arm against a width-768 one refuses on `n_kv_head` (6 vs 12) rather than on the
# width, which names a consequence instead of the cause.
DERIVED_FROM: dict[str, tuple[str, ...]] = {
    "n_kv_head": ("n_head",),
    "head_dim": ("d_model", "n_head"),
    "kv_lora_rank": ("d_model",),
    "q_lora_rank": ("d_model",),
    "rope_head_dim": ("head_dim", "d_model", "n_head"),
    "curriculum_start_len": ("block_size",),
    "lr_max_steps": ("max_steps",),
}


def expand_declared(declared: Iterable[str]) -> set[str]:
    """Add the fields that follow mechanically from the ones an arm declares.

    An arm registered at ``n_head=6`` has ``n_kv_head=6`` by derivation, not by
    drift, so refusing the pairing on it reports a consequence as if it were an
    independent mismatch. Applied to a fixed point, since a derived field can
    itself be a source (``head_dim`` feeds ``rope_head_dim``).
    """
    out = set(declared)
    while True:
        grown = out | {d for d, srcs in DERIVED_FROM.items()
                       if out & set(srcs)}
        if grown == out:
            return out
        out = grown


def normalize(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """One spelling per meaning, before anything is compared or grouped.

    ``lr_max_steps = 0`` means "decay over ``max_steps``"; newer suites write the
    number itself. 76 configs on this corpus use the old spelling and 1,263 the
    new one, so without this the same schedule reads as two, and the board's own
    documented example refuses itself.

    Nothing else on this corpus needs it: ``head_dim`` and ``n_kv_head`` are
    resolved by ``Config.__post_init__`` before a config is written (0 recorded
    zero times out of 1,339), and the MLA and curriculum "0 means derived" fields
    are uniformly 0 because they are unused here, so they cannot split a group.
    """
    out = dict(cfg)
    if out.get("lr_max_steps") in (0, None) and out.get("max_steps") is not None:
        out["lr_max_steps"] = out["max_steps"]
    return out


def project(cfg: Mapping[str, Any], exclude: Iterable[str] = (),
            extra_keys: Iterable[str] = (),
            fill_defaults: bool = False) -> dict[str, Any]:
    """The identity fields of one recorded config, ``MISSING`` where unrecorded.

    ``fill_defaults`` substitutes ``Config``'s default for a field the config
    predates. Use it for GROUPING and not for the pairing guard, because the two
    want opposite errors:

      * A guard that fills defaults asserts something about a run nobody
        measured, which is how the ``compile`` confound stayed invisible.
      * A grouper that does not fill defaults shatters a curve whenever half its
        points were run before a flag existed. On this corpus that splits the
        width-1152 ladder into "the 8x point" and "everything else", purely
        because the later run's config carries ``flash_cuda`` and the earlier
        ones do not.

    Filling is safe HERE for a specific, checkable reason rather than by hope:
    every flag this corpus is missing was added with the default set to the
    behaviour already on disk (``gdn_rule='repo'``, ``mingru_expand=2``,
    ``moe_router_weight='renorm'``, ``n_loops=1`` and so on). So a filled field
    re-merges runs that did the same thing while still separating the ones that
    did not -- ``gdn_pub`` records ``gdn_rule='published'`` explicitly and stays
    its own group. ``compile``, the one flag whose default does NOT describe the
    older runs, is recorded explicitly by all 1,339 configs here and is never
    filled. Callers should report ``unrecorded()`` alongside any grouping so the
    substitution is visible rather than assumed.
    """
    fill = _defaults() if fill_defaults else {}
    src = normalize(cfg)
    return {k: src.get(k, fill.get(k, MISSING))
            for k in identity_fields(extra_keys=extra_keys, exclude=exclude)}


@dataclasses.dataclass(frozen=True)
class Comparison:
    """The result of asking whether two runs are the same experiment.

    ``differ``    -- both recorded the field and the values disagree.
    ``by_design`` -- ditto, but the arm registry declares that field as part of
                     what distinguishes these two arms (an arm registered with an
                     ``lr`` override IS "that shape at that rate").
    ``unknown``   -- at least one run never recorded it. Not a difference, and
                     not a match: it is a check that could not run.
    ``checked``   -- how many fields were compared with both values present.
    """

    differ: dict[str, tuple[Any, Any]]
    by_design: dict[str, tuple[Any, Any]]
    unknown: dict[str, tuple[Any, Any]]
    checked: int

    @property
    def ok(self) -> bool:
        """True when nothing undeclared differs. Says nothing about ``unknown``."""
        return not self.differ

    def summary(self) -> str:
        parts = [f"{self.checked} field(s) compared"]
        if self.differ:
            parts.append(f"{len(self.differ)} differ: "
                         + ", ".join(f"{k}={a!r} vs {b!r}"
                                     for k, (a, b) in sorted(self.differ.items())))
        if self.by_design:
            parts.append("differ BY DESIGN on " + ", ".join(sorted(self.by_design)))
        if self.unknown:
            parts.append(f"{len(self.unknown)} not recorded by both: "
                         + ", ".join(sorted(self.unknown)))
        return "; ".join(parts)


def compare(a: Mapping[str, Any], b: Mapping[str, Any],
            declared: Iterable[str] = (), exclude: Iterable[str] = ()) -> Comparison:
    """Compare two recorded configs on identity, holding out ``declared`` fields.

    ``declared`` comes from the arm registry -- the fields the registry sets as
    part of these two arms' identity. A config cannot say whether a value was
    chosen for the arm or drifted into it, which is the whole reason the
    exemption is read from the registry and not from the runs.
    """
    declared = set(declared)
    a, b = normalize(a), normalize(b)
    keys = identity_fields(extra_keys=set(a) | set(b), exclude=exclude)
    differ: dict[str, tuple[Any, Any]] = {}
    by_design: dict[str, tuple[Any, Any]] = {}
    unknown: dict[str, tuple[Any, Any]] = {}
    checked = 0
    for k in keys:
        va, vb = a.get(k, MISSING), b.get(k, MISSING)
        if va is MISSING or vb is MISSING:
            if not (va is MISSING and vb is MISSING):
                unknown[k] = (va, vb)
            continue
        checked += 1
        if va == vb:
            continue
        (by_design if k in declared else differ)[k] = (va, vb)
    return Comparison(differ=differ, by_design=by_design, unknown=unknown,
                      checked=checked)


def group_key(cfg: Mapping[str, Any], exclude: Iterable[str] = (),
              extra_keys: Iterable[str] = (),
              fill_defaults: bool = True) -> tuple:
    """A hashable identity for grouping runs, with ``exclude`` held out.

    An LR sweep holds out ``lr``/``matrix_lr`` because that is the axis; what it
    must NOT hold out is everything else, which is how ``gdn`` and ``gdn_pub``
    came to share a cell. Values are stringified so that two runs recording
    ``1`` and ``1.0`` for the same field do not open two groups.

    ``fill_defaults`` is on by default here and off in ``compare``: see
    ``project`` for why grouping and guarding want opposite treatment of a field
    a config predates.
    """
    return tuple((k, repr(v)) for k, v in
                 sorted(project(cfg, exclude=exclude, extra_keys=extra_keys,
                                fill_defaults=fill_defaults).items()))
