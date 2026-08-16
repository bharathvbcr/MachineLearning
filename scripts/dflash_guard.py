#!/usr/bin/env python3
"""Single source of truth for whether the installed dflash-mlx decodes losslessly.

Every entry point that shells out to `dflash` (generate, serve, benchmark) must
consult this module. DFlash's selling point is that speculative decoding is
*lossless* — the target verifies every drafted token, so output is identical to
plain AR decoding from the same checkpoint. That guarantee is version-dependent,
and when it does not hold the failure is silent: tokens still stream, throughput
still looks good, and nothing in the output says the verifier read the wrong head.

Below 0.1.9, `QwenGdnTargetOps.text_wrapper` returns the OUTER wrapper for any
checkpoint exposing both `.model` and `.language_model`. Qwen3.8 is exactly that
shape (`Qwen3_5ForConditionalGeneration`). The outer wrapper carries no `args`, so

    getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True)

falls through to its `True` default and the verifier computes logits via
`wrapper.model.embed_tokens.as_linear(...)` — the tied-embedding path — even
though the real text model sets `tie_word_embeddings=False` and owns a separate
`lm_head`. Verified against the installed package: the wrong path yields a
different argmax, i.e. a different token.

Fixed in v0.1.9. Note v0.1.9/v0.1.10 exist only as GitHub tags; PyPI still serves
0.1.8, so `pip install -U dflash-mlx` will NOT pick the fix up.
"""

import re
import sys

DFLASH_MIN_LOSSLESS = (0, 1, 9)

DFLASH_UPGRADE_HINT = (
    "v0.1.9 fixes Qwen target-wrapper selection for checkpoints exposing both "
    ".model and .language_model — Qwen3.8 is exactly that shape "
    "(Qwen3_5ForConditionalGeneration), so verifier logits may take the wrong "
    "tied-embedding path. v0.1.10 adds full-context draft features and "
    "`benchmark --sustained-minutes`. Both are GitHub-only (PyPI still serves "
    "0.1.8): pip install 'git+https://github.com/bstnxbt/dflash-mlx@v0.1.10'"
)


def dflash_version():
    """Return (version_string, parsed_tuple) for the installed dflash-mlx.

    `dflash --version` is NOT a valid invocation — argparse exits 2 with empty
    stdout — so package metadata is the only reliable source. Returns
    (None, None) when the version cannot be determined; callers must treat that
    as unknown, never as 'new enough'.
    """
    try:
        import importlib.metadata as md
        ver = md.version("dflash-mlx")
    except Exception:
        return None, None
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", ver)
    return ver, (tuple(int(x) for x in m.groups()) if m else None)


def dflash_lossless_blocker():
    """Reason the installed dflash-mlx cannot be trusted to verify, else None.

    An undetermined version is a blocker, not a pass: a check that could not run
    must never report the same result as a check that ran and succeeded.
    """
    ver, parsed = dflash_version()

    if parsed is None:
        detail = f"metadata reports {ver!r}" if ver else "package metadata unavailable"
        return (f"dflash-mlx version UNDETERMINED ({detail}) — cannot confirm the "
                "verifier fix is present, so losslessness is unverified.")

    if parsed < DFLASH_MIN_LOSSLESS:
        return (f"dflash-mlx {ver} < 0.1.9 returns the OUTER wrapper for checkpoints "
                "exposing both .model and .language_model. Qwen3.8 is exactly that "
                "shape (Qwen3_5ForConditionalGeneration); the outer wrapper has no "
                "`args`, so tie_word_embeddings defaults to True and the verifier "
                "reads embed_tokens.as_linear instead of the text model's lm_head. "
                "Emitted tokens are NOT guaranteed to match the target.")

    return None


def warn_unless_lossless(context: str, stream=None):
    """Print a loud banner when the installed dflash cannot verify. Returns the
    blocker string, or None when the installed build is trustworthy.

    Callers that must not proceed silently should check the return value.
    """
    blocker = dflash_lossless_blocker()
    if blocker is None:
        return None

    stream = stream or sys.stderr
    bar = "!" * 74
    print(f"\n{bar}", file=stream)
    print(f"!! LOSSLESSNESS NOT ESTABLISHED — {context}", file=stream)
    print(f"!! {blocker}", file=stream)
    print(f"!! {DFLASH_UPGRADE_HINT}", file=stream)
    print(f"{bar}\n", file=stream)
    return blocker
