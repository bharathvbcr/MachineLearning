"""
nanolab.sample — generate text from a trained AR checkpoint (guide §6).

    python -m nanolab.sample --run run128m_fineweb_2k --prompt "The mitochondria" --tokens 120

Loads ``<out_dir>/<run>/config.json`` + ``best.pt`` (weights-only), rebuilds the
model, and autoregressively samples with temperature / top-k. Uses the same GPT-2
BPE tokenizer the data was tokenized with (tiktoken), or a char tokenizer if the
run used one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import build_config
from .model import build_model
from .utils import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run name under --out_dir")
    p.add_argument("--out_dir", default="nanolab/out")
    p.add_argument("--ckpt", default="best.pt", help="best.pt | final.pt | ckpt.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--n", type=int, default=1, help="number of samples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    run_dir = Path(args.out_dir) / args.run
    cfg = build_config(overrides=json.loads((run_dir / "config.json").read_text()))
    device = pick_device(cfg.device)
    model = build_model(cfg).to(device).eval()
    state = torch.load(run_dir / args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    if cfg.tokenizer == "char":
        from .data import CharTokenizer
        tok = CharTokenizer.load(Path(cfg.data_dir) / cfg.dataset)
        encode, decode = tok.encode, tok.decode
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)            # noqa: E731
        decode = enc.decode

    ids = encode(args.prompt) or [enc.eot_token if cfg.tokenizer != "char" else 0]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    print(f"=== {args.run}/{args.ckpt}  (T={args.temperature} top_k={args.top_k}) ===")
    for i in range(args.n):
        out = model.generate(x, args.tokens, temperature=args.temperature, top_k=args.top_k)
        text = decode(out[0].tolist())
        print(f"\n--- sample {i + 1} ---\n{text}")


if __name__ == "__main__":
    main()
