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

DEFAULT_TARGET_MODEL = "mlx-community/Qwen3.8-27B-4bit"
DEFAULT_MTP_DRAFT = "mlx-community/Qwen3.8-27B-MTP-4bit"
DEFAULT_DFLASH_DRAFT = "z-lab/Qwen3.6-27B-DFlash"

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
    parser.add_argument("--backend", type=str, choices=["mlx-vlm", "dflash", "mlx-lm"], default="mlx-vlm",
                        help="Serving backend. mlx-vlm = Qwen3.8 + official MTP drafter (also serves vision); "
                             "dflash = block-diffusion drafter; mlx-lm = plain AR baseline.")
    parser.add_argument("--model", type=str, default=DEFAULT_TARGET_MODEL,
                        help=f"Model repository or local path (default: {DEFAULT_TARGET_MODEL}).")
    parser.add_argument("--draft", type=str, default=None,
                        help="Draft model repository. Defaults per backend; pass 'none' to serve plain AR.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address.")
    parser.add_argument("--port", type=int, default=8000, help="Port number.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max output tokens per response (default: {DEFAULT_MAX_TOKENS}). "
                             "The mlx-vlm default of 2048 truncates long agent edits silently.")
    parser.add_argument("--no-prefix-cache", action="store_true", help="Disable prefix cache in DFlash.")
    args = parser.parse_args()

    draft = None if args.draft == "none" else args.draft

    if args.backend == "mlx-vlm":
        serve_mlx_vlm(args.model, draft=draft or DEFAULT_MTP_DRAFT, host=args.host, port=args.port,
                      max_tokens=args.max_tokens)
    elif args.backend == "dflash":
        # Explicit draft required: the dflash registry has no Qwen3.8 entry.
        serve_dflash(args.model, draft=draft or DEFAULT_DFLASH_DRAFT, host=args.host,
                     port=args.port, prefix_cache=not args.no_prefix_cache,
                     max_tokens=args.max_tokens)
    else:
        serve_mlx_lm(args.model, host=args.host, port=args.port, max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
