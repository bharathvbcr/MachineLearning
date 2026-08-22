#!/usr/bin/env python3
"""Single source of truth for pairing a Qwen3.8-27B target with its drafter.

The rule this module owns: **a Qwen3.8-27B target never loads without a drafter
unless someone said so out loud.** Bare AR decoding of this target is 15-17 tok/s
against 36.91 with the MTP drafter — a 2.18x regression that is invisible at the
call site, because a drafter-less load is not an error. It answers every request,
just slowly, for as long as the process stays up.

Two failure shapes are closed here:

1. *Omission.* An entry point that forgets to pass `--draft` used to load bare.
   Now the default pairing for (engine, target) is attached automatically and the
   attachment is announced.
2. *Silent reinterpretation.* `serve_qwen.py --draft none` documented itself as
   "serve plain AR" while `draft or DEFAULT_MTP_DRAFT` quietly substituted the
   default drafter back in — so the one documented way to ask for a bare load did
   the opposite. `none` now means bare, and bare requires `--allow-bare`.

The pairing table is engine-keyed because the drafter is only valid for the engine
that can execute it: the MTP head runs under mlx-vlm, DSpark under mlx-dspark,
DFlash under dflash-mlx. `mlx-lm` is absent from the table on purpose — it has no
drafter path at all, so every guarded load through it is a deliberate baseline and
must carry `--allow-bare`.

Also enforces the oMLX side, where the pairing lives in a config file rather than
in an argv:

    python3 scripts/qwen_draft_policy.py --check-omlx    # exit 3 if a target would load bare
    python3 scripts/qwen_draft_policy.py --apply-omlx    # write the measured pairing
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# The models this policy governs
# --------------------------------------------------------------------------- #

TARGET_4BIT = "mlx-community/Qwen3.8-27B-4bit"
TARGET_8BIT = "mlx-community/Qwen3.8-27B-8bit"
TARGET_NVFP4 = "mlx-community/Qwen3.8-27B-nvfp4"

MTP_4BIT = "mlx-community/Qwen3.8-27B-MTP-4bit"
MTP_NVFP4 = "mlx-community/Qwen3.8-27B-MTP-nvfp4"
DFLASH_36 = "z-lab/Qwen3.6-27B-DFlash"
DFLASH2_38 = "incoai/Qwen3.8-27B-DFlash2"
DSPARK_4BIT = "DimInfer/Qwen3.8-27B-Dspark-v1"
DSPARK_8BIT = "RadixArk/Qwen3.8-27B-DSpark"

DEFAULT_TARGET_MODEL = TARGET_4BIT

# Any Qwen3.8-27B checkpoint, however it is spelled: repo id, local snapshot path,
# or an oMLX model id (which replaces "/" with "--"). Quantization suffix is
# deliberately not part of the match — an unrecognized 3.8-27B variant must still
# trip the guard rather than slip through as "not my problem".
GUARDED_TARGET = re.compile(r"qwen-?3\.8-27b", re.IGNORECASE)

# (engine, target) -> drafter. An engine missing from this table has no drafter
# path; a target missing from an engine's table has no *measured* pairing for it.
# Both cases refuse rather than guess: pairing an unmeasured drafter with a target
# is exactly how an unverified speedup ships as a default.
PAIRINGS = {
    # 36.91 tok/s, 2.18x, 91.2% acceptance — the measured default.
    "mlx-vlm": {TARGET_4BIT: MTP_4BIT, TARGET_NVFP4: MTP_NVFP4},
    # 31.20 tok/s, 1.84x, byte-identical to AR on a greedy spot-check.
    "dspark": {TARGET_4BIT: DSPARK_4BIT, TARGET_8BIT: DSPARK_8BIT},
    # 3.6 cross-applied: 1.17x at 53.5% acceptance. Superseded, still available.
    "dflash": {TARGET_4BIT: DFLASH_36},
    # oMLX 0.6.2-dflash2 only. Throughput and losslessness both UNVERIFIED on this
    # machine (measured under GPU contention; greedy output was not bit-identical
    # to AR). Kept out of every argv-driven engine above on purpose — this entry
    # exists so --check-omlx can tell "configured" from "bare", not to make DFlash2
    # anything's default.
    "omlx": {TARGET_4BIT: DFLASH2_38},
}

# mlx-lm decodes autoregressively with no drafter hook, so it cannot satisfy the
# policy — naming it here keeps the refusal message specific instead of generic.
DRAFTERLESS_ENGINES = {"mlx-lm"}


class BareLoadRefused(SystemExit):
    """A guarded target was about to load with no drafter and no opt-out.

    The message is printed here rather than handed to SystemExit, because
    `SystemExit(2)` prints nothing and `SystemExit(message)` would exit 1. A guard
    that stops the process without saying why just looks like a broken script.
    """

    def __init__(self, message, stream=None):
        print(message, file=stream if stream is not None else sys.stderr)
        super().__init__(2)
        self.message = message


def is_guarded_target(target) -> bool:
    """True if `target` names a Qwen3.8-27B checkpoint in any spelling."""
    return bool(target) and bool(GUARDED_TARGET.search(str(target)))


def default_drafter_for(target, engine):
    """The measured drafter for (engine, target), or None if there is no pairing."""
    return PAIRINGS.get(engine, {}).get(target)


def resolve_draft(target, draft, *, engine, context, allow_bare=False,
                  bare_reason=None, stream=None):
    """Return the drafter to load with, or None for an approved bare load.

    Args:
        target: target checkpoint id or path.
        draft: drafter the caller asked for. None means "unspecified" (the policy
            fills it in); the literal "none" means "explicitly bare".
        engine: which runtime will execute the pair — a key of PAIRINGS, or
            "mlx-lm" for the drafterless baseline engine.
        context: human-readable description of the call site, used in messages.
        allow_bare: operator opt-out, from `--allow-bare` or a benchmark baseline.
        bare_reason: why bare is legitimate here. Required when allow_bare is set,
            so the exemption lands in the log instead of being inferred later.
        stream: where announcements go. Resolved at call time, not bound as a
            default — a default would capture the interpreter's original stderr and
            write past any redirect a caller (or a test) has installed.

    Raises:
        BareLoadRefused: guarded target, no drafter, no opt-out.
    """
    stream = stream if stream is not None else sys.stderr
    asked_bare = (draft == "none")
    if asked_bare:
        draft = None

    if not is_guarded_target(target):
        return draft

    if draft:
        return draft

    if asked_bare or engine in DRAFTERLESS_ENGINES:
        if not allow_bare:
            raise BareLoadRefused(_refusal(target, engine, context))
        if not bare_reason:
            raise BareLoadRefused(
                f"{context}: allow_bare set with no reason. An unexplained bare "
                "load is indistinguishable from a forgotten drafter."
            )
        print(f"\n{'!'*70}\n[draft-policy] BARE LOAD — {target} will decode with NO "
              f"drafter.\n  Context: {context}\n  Reason : {bare_reason}\n"
              f"  Expect ~15-17 tok/s, not the 36.91 of the MTP arm.\n{'!'*70}\n",
              file=stream)
        return None

    pick = default_drafter_for(target, engine)
    if not pick:
        raise BareLoadRefused(_refusal(target, engine, context))

    print(f"[draft-policy] {context}: no --draft given; attaching the measured "
          f"drafter for {engine}: {pick}", file=stream)
    return pick


def _refusal(target, engine, context):
    known = sorted(PAIRINGS.get(engine, {}))
    if engine in DRAFTERLESS_ENGINES:
        why = (f"engine '{engine}' has no drafter path — every load through it is "
               f"bare AR (~15-17 tok/s vs 36.91 with the MTP drafter).")
        fix = ("Use an engine that drafts (mlx-vlm/dspark/dflash), or pass "
               "--allow-bare if a bare baseline is what you actually want.")
    elif known:
        why = (f"no measured drafter is paired with {target} on engine '{engine}'. "
               f"Paired targets: {', '.join(known)}.")
        fix = ("Pass --draft <repo> to name one explicitly, or --allow-bare to "
               "accept AR throughput.")
    else:
        why = f"engine '{engine}' has no pairings recorded."
        fix = "Pass --draft <repo> explicitly, or --allow-bare."
    return (f"\n{'='*70}\n[draft-policy] REFUSING to load {target} with no drafter."
            f"\n  Context: {context}\n  Why    : {why}\n  Fix    : {fix}\n"
            f"  Owner  : scripts/qwen_draft_policy.py\n{'='*70}")


def add_allow_bare_flag(parser):
    """Register the one documented way to ask for a drafter-less guarded load."""
    parser.add_argument(
        "--allow-bare", action="store_true",
        help="Permit loading a Qwen3.8-27B target with no drafter (plain AR, "
             "~15-17 tok/s). Required for baseline measurements; without it, "
             "drafter-less loads are refused by scripts/qwen_draft_policy.py.")


# --------------------------------------------------------------------------- #
# oMLX: the pairing lives in a config file, so the guard is a config check
# --------------------------------------------------------------------------- #

OMLX_SETTINGS = Path(os.path.expanduser("~/.omlx/model_settings.json"))

# Written by --apply-omlx. Block size 5 and verify mode "dflash" are z-lab's own
# oMLX recommendation for this drafter; the 4-bit draft quantization keeps the
# 3.85 GB bf16 drafter from eating the headroom the 15.7 GB target needs.
OMLX_DFLASH2_SETTINGS = {
    "dflash_enabled": True,
    "dflash_draft_model": DFLASH2_38,
    "dflash_draft_quant_enabled": True,
    "dflash_draft_quant_weight_bits": 4,
    "dflash_draft_quant_group_size": 64,
    "dflash_block_size": 5,
    "dflash_verify_mode": "dflash",
}


def omlx_model_id(repo: str) -> str:
    """oMLX addresses HF-cache models by repo id with '/' replaced by '--'."""
    return repo.replace("/", "--")


def audit_omlx(settings: dict):
    """Return (ok, findings) for every guarded target oMLX would serve.

    `settings` is the parsed model_settings.json. A guarded target with no entry,
    with dflash disabled, or with dflash enabled but no draft model would load
    bare — all three are reported as the same finding class, because all three
    produce the same 2.18x regression at runtime.
    """
    models = (settings or {}).get("models", {})
    findings, ok = [], True
    guarded = {mid: cfg for mid, cfg in models.items() if is_guarded_target(mid)}

    expected = omlx_model_id(TARGET_4BIT)
    if expected not in guarded:
        ok = False
        findings.append(f"MISSING: {expected} has no settings entry — it would load bare.")

    for mid, cfg in sorted(guarded.items()):
        if not cfg.get("dflash_enabled"):
            ok = False
            findings.append(f"BARE: {mid} has dflash_enabled={cfg.get('dflash_enabled')!r}.")
        elif not cfg.get("dflash_draft_model"):
            ok = False
            findings.append(f"BARE: {mid} enables dflash but names no dflash_draft_model.")
        else:
            findings.append(f"OK: {mid} -> {cfg['dflash_draft_model']}")
    return ok, findings


def load_omlx_settings(path=None):
    path = Path(path or OMLX_SETTINGS)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_omlx_settings(path=None) -> str:
    """Write the DFlash2 pairing for the 4-bit target, preserving other models."""
    path = Path(path or OMLX_SETTINGS)
    data = load_omlx_settings(path) or {"version": 1, "models": {}}
    data.setdefault("version", 1)
    data.setdefault("models", {})
    mid = omlx_model_id(TARGET_4BIT)
    entry = dict(data["models"].get(mid, {}))
    entry.update(OMLX_DFLASH2_SETTINGS)
    data["models"][mid] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)
    return mid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-omlx", action="store_true",
                   help="Exit 3 if any Qwen3.8-27B target in oMLX would load bare.")
    g.add_argument("--apply-omlx", action="store_true",
                   help="Write the DFlash2 pairing for the 4-bit target into "
                        "~/.omlx/model_settings.json (restart oMLX to pick it up).")
    args = p.parse_args()

    if args.apply_omlx:
        mid = apply_omlx_settings()
        print(f"[draft-policy] wrote {mid} -> {DFLASH2_38} in {OMLX_SETTINGS}")
        print("[draft-policy] restart oMLX for it to take effect.")

    settings = load_omlx_settings()
    if settings is None:
        print(f"[draft-policy] {OMLX_SETTINGS} does not exist — oMLX would load "
              f"every model bare. Run --apply-omlx.", file=sys.stderr)
        return 3

    ok, findings = audit_omlx(settings)
    for line in findings:
        print(f"  {line}")
    if not ok:
        print("\n[draft-policy] FAIL: a Qwen3.8-27B target would load with no "
              "drafter. Run --apply-omlx.", file=sys.stderr)
        return 3
    print("\n[draft-policy] OK: every Qwen3.8-27B target oMLX knows about is "
          "paired with a drafter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
