#!/usr/bin/env python3
"""
OpenAI-Compatible Local Server Runner for Qwen Models on Apple Silicon (MLX & DFlash).
Provides OpenAI `/v1/chat/completions` endpoints with prefix caching and streaming support.
"""

import argparse
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Optional

# scripts/ is not a package; make sibling modules importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflash_guard import warn_unless_lossless  # noqa: E402
# Per-backend drafter defaults used to live here as four constants that each call
# site re-applied with `draft or DEFAULT_...`. That idiom is what let `--draft none`
# mean "use the default" while the help text promised plain AR. The pairing table
# and the bare-load rule now have one owner.
from qwen_draft_policy import (  # noqa: E402
    DEFAULT_TARGET_MODEL,
    add_allow_bare_flag,
    apply_omlx_settings,
    audit_omlx,
    load_omlx_settings,
    omlx_model_id,
    resolve_draft,
)

# oMLX ships as a macOS app; this is the CLI inside the bundle. Module-level so a
# test can point it somewhere harmless.
OMLX_CLI = "/Applications/oMLX.app/Contents/MacOS/omlx-cli"


def resolve_drafter(repo: str) -> str:
    """Resolve a drafter repo to a local directory, weight files only.

    Canonical owner for this: `scripts/bench_qwen38.py` imports it rather than
    keeping a second copy. mlx-dspark resolves a bare `--drafter` repo id with a
    full snapshot_download, which on DimInfer drags down 5.7 GB of GGUFs beside the
    3.7 GB safetensors — files nothing in this repo loads, re-attempted on every
    start until they complete. Already-cached weights make this a no-op.
    """
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.md"])

# mlx-vlm's server defaults to 2048 output tokens and truncates silently — no
# error, no finish_reason=length surfaced to the caller. A coding agent asked to
# rewrite a file longer than ~2048 tokens gets a cut-off response, retries, and
# grows its context each round until it burns through its turn limit. Observed
# on a 245-line file: context climbed 2,845 -> 45,140 tokens across 5 requests
# and the task never completed. kon itself asks for 16384.
DEFAULT_MAX_TOKENS = 16384


def resolve_mlx_vlm_cli(subcommand: str = "server") -> list[str]:
    """Console script is `mlx_vlm.server`; some installs expose `mlx_vlm server`."""
    dotted = f"mlx_vlm.{subcommand}"
    if shutil.which(dotted):
        return [dotted]
    if shutil.which("mlx_vlm"):
        return ["mlx_vlm", subcommand]
    return [sys.executable, "-m", dotted]


def serve_mlx_vlm(model: str, draft: Optional[str] = None, host: str = "127.0.0.1", port: int = 8000,
                  max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
    print(f"\n{'='*70}")
    print(f"[Starting mlx-vlm Server — MTP Speculative Decoding]")
    print(f"  Target Model: {model}")
    print(f"  Draft Model : {draft or 'none (plain AR)'}")
    print(f"  Host:Port   : http://{host}:{port}")
    print(f"{'='*70}\n")

    cmd = resolve_mlx_vlm_cli("server") + ["--model", model, "--host", host, "--port", str(port),
                                           "--max-tokens", str(max_tokens)]
    if draft:
        cmd.extend(["--draft-model", draft, "--draft-kind", "mtp"])

    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)


def serve_dflash(model: str, draft: Optional[str] = None, host: str = "127.0.0.1", port: int = 8000,
                 prefix_cache: bool = True, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
    print(f"\n{'='*70}")
    print(f"[Starting DFlash Speculative Decoding Server]")
    print(f"  Target Model: {model}")
    print(f"  Draft Model : {draft or 'Auto-resolved from DFlash registry'}")
    print(f"  Host:Port   : http://{host}:{port}")
    print(f"  Prefix Cache: {'Enabled (L1/L2)' if prefix_cache else 'Disabled'}")
    print(f"{'='*70}\n")

    # A server is the worst place for this to fail quietly: it would answer every
    # request with unverified tokens for as long as it stays up.
    warn_unless_lossless(f"`dflash serve` on {model}")

    cmd = [
        "dflash", "serve",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--max-tokens", str(max_tokens),
    ]
    if draft:
        cmd.extend(["--draft", draft])
    if not prefix_cache:
        cmd.append("--no-prefix-cache")

    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)

def serve_dspark(model: str, draft: Optional[str] = None, host: str = "127.0.0.1", port: int = 8000,
                 prefix_cache: bool = True, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
    print(f"\n{'='*70}")
    print(f"[Starting DSpark Speculative Decoding Server]")
    print(f"  Target Model: {model}")
    print(f"  Draft Model : {draft}")
    print(f"  Host:Port   : http://{host}:{port}")
    print(f"  Prefix Cache: {'Enabled' if prefix_cache else 'Disabled'}")
    print(f"{'='*70}\n")

    # No losslessness guard here, unlike the dflash path: that guard exists for a
    # specific dflash-mlx verifier bug, not as a general speculative-decoding gate.
    # mlx-dspark 0.12.2 was checked directly on this target — greedy output matched
    # the AR baseline character for character (docs/qwen_mlx_dflash_guide.md 4b-iii).
    cmd = [
        "mlx-dspark", "serve",
        "--model", model,
        "--mode", "dspark",
        "--drafter", resolve_drafter(draft),
        "--host", host,
        "--port", str(port),
        "--default-max-tokens", str(max_tokens),
    ]
    if not prefix_cache:
        cmd.append("--no-prefix-cache")

    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)


def serve_omlx(model: str, draft: Optional[str] = None, host: str = "127.0.0.1",
               port: int = 8891, settings_path=None) -> None:
    """Start oMLX, but only after its config pairs the target with a drafter.

    oMLX takes the drafter from `~/.omlx/model_settings.json`, not from argv, so
    the failure mode is different from every other backend here: a config that
    lost its `dflash_enabled` entry produces a server that loads bare and never
    mentions it. Audit first, repair if needed, and refuse if the audit still
    fails — the same rule the argv backends get, applied to the config file.
    """
    if not Path(OMLX_CLI).exists():
        print(f"[serve_qwen] oMLX not found at {OMLX_CLI}. Install the DFlash2 build "
              f"(z-lab/omlx-fork release 0.6.2-dflash2) first.", file=sys.stderr)
        raise SystemExit(2)

    ok, findings = audit_omlx(load_omlx_settings(settings_path) or {})
    if not ok:
        print("[serve_qwen] oMLX config would load a Qwen3.8-27B target bare:")
        for line in findings:
            print(f"    {line}")
        mid = apply_omlx_settings(settings_path)
        print(f"[serve_qwen] repaired: {mid} -> {draft}")
        ok, findings = audit_omlx(load_omlx_settings(settings_path) or {})
        if not ok:
            print("[serve_qwen] config still fails its audit after repair:", file=sys.stderr)
            for line in findings:
                print(f"    {line}", file=sys.stderr)
            raise SystemExit(2)

    print(f"\n{'='*70}")
    print(f"[Starting oMLX Server — DFlash 2 Speculative Decoding]")
    print(f"  Target Model: {omlx_model_id(model)}")
    print(f"  Draft Model : {draft}")
    print(f"  Host:Port   : http://{host}:{port}")
    print(f"{'='*70}\n")
    # Unlike the MTP and DSpark arms, this pairing has no verified losslessness
    # result on this machine: greedy output was not bit-identical to AR, and the
    # throughput comparison was taken under GPU contention. Say so every start.
    print(f"{'!'*70}\n[serve_qwen] DFlash 2 losslessness is UNVERIFIED here — greedy "
          f"output diverged\n  from AR, and drafter-dependent divergence was observed. "
          f"Do not treat this\n  arm's output as interchangeable with AR until that is "
          f"settled.\n{'!'*70}\n", file=sys.stderr)

    cmd = [OMLX_CLI, "serve", "--host", host, "--port", str(port), "--hf-cache"]
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)


def serve_mlx_lm(model: str, host: str = "127.0.0.1", port: int = 8000,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
    print(f"\n{'='*70}")
    print(f"[Starting Native MLX-LM Server]")
    print(f"  Model    : {model}")
    print(f"  Host:Port: http://{host}:{port}")
    print(f"{'='*70}\n")

    cmd = [
        "python3", "-m", "mlx_lm.server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--max-tokens", str(max_tokens),
    ]
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)

def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Qwen models with OpenAI-compatible API on Apple Silicon.")
    parser.add_argument("--backend", type=str,
                        choices=["mlx-vlm", "dspark", "dflash", "omlx", "mlx-lm"],
                        default="mlx-vlm",
                        help="Serving backend. mlx-vlm = Qwen3.8 + official MTP drafter, the measured "
                             "fastest (36.91 tok/s, 2.18x) and the default; dspark = 3.8-native DSpark "
                             "drafter (31.20 tok/s, 1.84x, lossless); dflash = 3.6 drafter cross-applied "
                             "(1.17x, superseded by dspark); omlx = DFlash 2 via the oMLX app "
                             "(throughput and losslessness both UNVERIFIED on this machine); "
                             "mlx-lm = plain AR baseline.")
    parser.add_argument("--model", type=str, default=DEFAULT_TARGET_MODEL,
                        help=f"Model repository or local path (default: {DEFAULT_TARGET_MODEL}).")
    parser.add_argument("--draft", type=str, default=None,
                        help="Draft model repository. Defaults per backend; pass 'none' to serve "
                             "plain AR, which a Qwen3.8-27B target accepts only with --allow-bare.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address.")
    parser.add_argument("--port", type=int, default=8000, help="Port number.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max output tokens per response (default: {DEFAULT_MAX_TOKENS}). "
                             "The mlx-vlm default of 2048 truncates long agent edits silently.")
    parser.add_argument("--no-prefix-cache", action="store_true", help="Disable prefix cache (dflash and dspark backends).")
    add_allow_bare_flag(parser)
    args = parser.parse_args()

    # One resolution for every backend, before any of them starts a server. A
    # server is the worst place to discover a missing drafter: it would answer at
    # AR speed for as long as it stays up, and nothing in its output says so.
    draft = resolve_draft(
        args.model, args.draft, engine=args.backend,
        context=f"serve_qwen.py --backend {args.backend}",
        allow_bare=args.allow_bare,
        bare_reason="operator passed --allow-bare" if args.allow_bare else None,
    )

    # mlx-dspark and dflash take the drafter as a required argument; there is no
    # drafterless mode to fall back to, so an approved bare load still cannot be
    # served by them. Say which backend does it instead of failing inside the CLI.
    if draft is None and args.backend in ("dspark", "dflash", "omlx"):
        print(f"[serve_qwen] --backend {args.backend} has no drafterless mode. For a "
              f"bare AR server use: --backend mlx-lm --allow-bare", file=sys.stderr)
        raise SystemExit(2)

    if args.backend == "mlx-vlm":
        serve_mlx_vlm(args.model, draft=draft, host=args.host, port=args.port,
                      max_tokens=args.max_tokens)
    elif args.backend == "dspark":
        serve_dspark(args.model, draft=draft, host=args.host, port=args.port,
                     prefix_cache=not args.no_prefix_cache, max_tokens=args.max_tokens)
    elif args.backend == "dflash":
        # Explicit draft required: the dflash registry has no Qwen3.8 entry.
        serve_dflash(args.model, draft=draft, host=args.host,
                     port=args.port, prefix_cache=not args.no_prefix_cache,
                     max_tokens=args.max_tokens)
    elif args.backend == "omlx":
        serve_omlx(args.model, draft=draft, host=args.host, port=args.port)
    else:
        serve_mlx_lm(args.model, host=args.host, port=args.port, max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
