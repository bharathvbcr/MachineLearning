#!/usr/bin/env python3
"""
Unified Apple Silicon inference and benchmarking harness for Qwen models on MLX & DFlash.
Supports:
  1. Direct MLX generation (Qwen 3.8 27B 4bit/8bit)
  2. DFlash speculative block-diffusion generation (Qwen 3.6 27B + z-lab Drafter)
  3. Comparative throughput, latency, and acceptance rate benchmarking
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

from mlx_lm.sample_utils import make_sampler

# scripts/ is not a package; make sibling modules importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflash_guard import warn_unless_lossless  # noqa: E402

DEFAULT_TARGET_MODEL = "mlx-community/Qwen3.8-27B-4bit"

# Official MTP drafter split from the same Qwen3.8-27B checkpoint as the target.
# Runs under mlx-vlm (--draft-kind mtp). Same training run as the target, so there is
# no cross-model acceptance risk.
DEFAULT_MTP_DRAFT = "mlx-community/Qwen3.8-27B-MTP-4bit"

# Cross-applied 3.6 drafter. There is NO z-lab Qwen3.8 DFlash checkpoint as of
# 2026-08-16, and dflash-mlx's registry will not auto-resolve a draft for a 3.8
# target, so this must always be passed explicitly. Architecturally compatible
# (same hidden size / layer count / layer_types) but trained on 3.6 hidden states —
# acceptance is an open question. See scripts/bench_qwen38.py.
DEFAULT_DFLASH_DRAFT = "z-lab/Qwen3.6-27B-DFlash"

def run_mlx_direct(model_id: str, prompt: str, max_tokens: int = 512, temp: float = 0.7, enable_thinking: bool = False) -> dict[str, object]:
    print(f"\n{'='*70}")
    print(f"[MLX Direct Inference] Model: {model_id}")
    print(f"{'='*70}\n")

    print(f"Loading {model_id} into unified memory...", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_id)
    load_time = time.time() - t0
    print(f"[✓] Model loaded in {load_time:.2f}s\n", flush=True)

    # Format chat prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        chat_prompt = prompt

    print(f"Prompt: {prompt}\n")
    print(f"--- Response ---")
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    full_response = ""
    sampler = make_sampler(temp=temp)

    for response in stream_generate(
        model,
        tokenizer,
        prompt=chat_prompt,
        max_tokens=max_tokens,
        sampler=sampler
    ):
        if first_token_time is None:
            first_token_time = time.time()
        sys.stdout.write(response.text)
        sys.stdout.flush()
        full_response += response.text
        token_count += 1

    total_time = time.time() - start_time
    ttft = (first_token_time - start_time) if first_token_time else 0.0
    decode_time = (time.time() - first_token_time) if first_token_time else total_time
    tok_per_sec = (token_count / decode_time) if decode_time > 0 else 0.0

    print(f"\n\n{'='*70}")
    print(f"Generation Stats:")
    print(f"  • Tokens generated: {token_count}")
    print(f"  • Time to First Token (TTFT): {ttft*1000:.1f} ms")
    print(f"  • Total decode time: {decode_time:.2f} s")
    print(f"  • Throughput: {tok_per_sec:.2f} tokens/sec")
    print(f"  • Peak memory: {mx.get_peak_memory() / (1024**3):.2f} GB")
    print(f"{'='*70}\n")
    return {
        "mode": "mlx-direct",
        "model": model_id,
        "tokens": token_count,
        "ttft_ms": ttft * 1000,
        "tok_s": tok_per_sec,
        "response": full_response
    }

def run_dflash_speculative(target_model: str, draft_model: str, prompt: str, max_tokens: int = 512) -> dict[str, object]:
    print(f"\n{'='*70}")
    print(f"[DFlash Speculative Block-Diffusion Inference]")
    print(f"  Target: {target_model}")
    print(f"  Draft : {draft_model or 'Auto-resolved'}")
    print(f"{'='*70}\n")

    # DFlash's whole premise is that speculation is lossless. On a build where the
    # verifier reads the wrong head that premise fails silently — tokens still
    # stream and throughput still looks fine. Say so before generating.
    warn_unless_lossless(f"`dflash generate` against {target_model}")

    cmd = [
        "dflash", "generate",
        "--model", target_model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--verify-mode", "adaptive"
    ]
    if draft_model:
        cmd.extend(["--draft", draft_model])

    t0 = time.time()
    proc = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    print(f"\nCompleted DFlash run in {elapsed:.2f}s")
    return {"mode": "dflash-speculative", "elapsed": elapsed}

def resolve_mlx_vlm_cli(subcommand: str = "generate") -> list[str]:
    """mlx-vlm ships console scripts named `mlx_vlm.generate` / `mlx_vlm.server`.
    Some installs also expose a unified `mlx_vlm <subcommand>`. Probe, then fall
    back to the module form so this works across mlx-vlm versions."""
    import shutil
    dotted = f"mlx_vlm.{subcommand}"
    if shutil.which(dotted):
        return [dotted]
    if shutil.which("mlx_vlm"):
        return ["mlx_vlm", subcommand]
    return [sys.executable, "-m", dotted]


def run_mtp_speculative(target_model: str, draft_model: str, prompt: str,
                        max_tokens: int = 512, temp: float = 0.0,
                        enable_thinking: bool = False) -> dict[str, object]:
    """Speculative decoding with the official Qwen3.8 MTP drafter via mlx-vlm.

    Lossless: the target verifies every proposed token, so output matches plain
    AR decoding from the same target checkpoint.
    """
    print(f"\n{'='*70}")
    print(f"[MTP Speculative Inference — mlx-vlm]")
    print(f"  Target: {target_model}")
    print(f"  Draft : {draft_model}")
    print(f"{'='*70}\n")

    cmd = resolve_mlx_vlm_cli("generate") + [
        "--model", target_model,
        "--draft-model", draft_model,
        "--draft-kind", "mtp",
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", str(temp),
    ]
    if enable_thinking:
        cmd.append("--enable-thinking")

    print(f"Command: {' '.join(cmd)}\n")
    t0 = time.time()
    proc = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"\n[!] mlx-vlm exited {proc.returncode}. Is mlx-vlm installed? "
              f"`pip install -U mlx-vlm` (speculative decoding needs >= 0.6.13).",
              file=sys.stderr)

    print(f"\nCompleted MTP run in {elapsed:.2f}s")
    return {"mode": "mtp-speculative", "elapsed": elapsed, "returncode": proc.returncode}


def run_benchmark(target_model: str, draft_model: str, prompt: str, max_tokens: int = 256) -> None:
    print(f"\n{'='*70}")
    print(f"[DFlash vs Baseline MLX Benchmark]")
    print(f"  Model: {target_model}")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"{'='*70}\n")

    # A speedup measured against a broken verifier is not a speedup.
    warn_unless_lossless(f"`dflash benchmark` against {target_model}")

    cmd = [
        "dflash", "benchmark",
        "--model", target_model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--repeat", "2",
        "--cooldown", "3"
    ]
    if draft_model:
        cmd.extend(["--draft", draft_model])

    subprocess.run(cmd)

def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen 27B DFlash Speculative Engine Runner (MLX on Apple Silicon)")
    parser.add_argument("--mode", type=str,
                        choices=["mtp-speculative", "dflash-speculative", "mlx-direct", "benchmark"],
                        default="mtp-speculative", help="Inference mode to run.")
    parser.add_argument("--model", type=str, default=DEFAULT_TARGET_MODEL,
                        help=f"Target model ID or path (default: {DEFAULT_TARGET_MODEL}).")
    parser.add_argument("--draft", type=str, default=None,
                        help="Draft model ID. Defaults to the MTP drafter in mtp-speculative mode "
                             "and the 3.6 DFlash drafter in dflash-speculative mode.")
    parser.add_argument("--prompt", type=str,
                        default="Explain why Apple Silicon unified memory architecture enables ultra-fast large language model inference.",
                        help="Text prompt for generation.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable the Qwen thinking template (mtp-speculative mode).")
    args = parser.parse_args()

    if args.mode == "mlx-direct":
        run_mlx_direct(args.model, args.prompt, max_tokens=args.max_tokens, temp=args.temp)
    elif args.mode == "mtp-speculative":
        run_mtp_speculative(args.model, args.draft or DEFAULT_MTP_DRAFT, args.prompt,
                            max_tokens=args.max_tokens, temp=args.temp,
                            enable_thinking=args.enable_thinking)
    elif args.mode == "dflash-speculative":
        # Always explicit: the dflash registry has no Qwen3.8 entry and will reject
        # the target rather than auto-resolving a drafter.
        run_dflash_speculative(args.model, args.draft or DEFAULT_DFLASH_DRAFT,
                               args.prompt, max_tokens=args.max_tokens)
    elif args.mode == "benchmark":
        run_benchmark(args.model, args.draft or DEFAULT_DFLASH_DRAFT,
                      args.prompt, max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
