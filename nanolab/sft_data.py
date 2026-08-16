"""
nanolab.sft_data — Phase-2 SFT data prep (torch-free). Turns a public CoT dataset
into the think/answer template and writes prompt-masked token streams.

    python -m nanolab.sft_data --dataset gsm8k --max_examples 8000

Default source is GSM8K (``openai/gsm8k``, config ``main``): each row has a
question and a chain-of-thought answer ending in ``#### <number>``. We map the
reasoning into the ``<think>`` span and the final number into a JSON answer
``{"answer": N}`` — so a single dataset teaches *both* "reason first" and
"emit structured output". Falls back to a synthetic arithmetic corpus when
offline, so the pipeline (and the smoke test) always runs.

Outputs into ``nanolab/data/sft/``:
    {train,val}_tokens.bin   uint16 packed token ids
    {train,val}_loss.bin     uint8  per-token loss mask (1 = train on it)

IMPORTANT: this module is intentionally torch-free. ``datasets`` pulls pyarrow,
and torch + pyarrow in one Windows process segfault (see nanolab/data.py). Run
this standalone to create the bins; nanolab/sft.py then only reads them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .special_tokens import build_example

_GSM8K_ANS = re.compile(r"####\s*([-\d,\.]+)")


def _gpt2():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def _synthetic(n: int):
    """Deterministic arithmetic CoT — offline fallback so the pipeline always
    runs. Mirrors GSM8K shape: a word problem, a short reasoning, a number."""
    out = []
    for i in range(n):
        a, b = 3 + (i % 17), 2 + (i * 7 % 13)
        q = f"A shop had {a} apples and bought {b} more. How many apples are there now?"
        reasoning = f"Start with {a} apples. Add {b} more. {a} + {b} = {a + b}."
        out.append((q, reasoning, a + b))
    return out


def _load_gsm8k(max_examples: int):
    """Yield (question, reasoning, answer_number) from GSM8K, or synthetic if the
    download fails."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        rows = []
        for ex in ds:
            m = _GSM8K_ANS.search(ex["answer"])
            if not m:
                continue
            num = m.group(1).replace(",", "")
            reasoning = _GSM8K_ANS.sub("", ex["answer"]).strip()
            rows.append((ex["question"], reasoning, num))
            if len(rows) >= max_examples:
                break
        if rows:
            return rows, "gsm8k"
    except Exception as e:                       # offline / no datasets installed
        print(f"[sft_data] GSM8K unavailable ({type(e).__name__}: {e}); using synthetic")
    return _synthetic(min(max_examples, 2000)), "synthetic"


def _answer_json(num: str) -> str:
    """Render the final answer as a JSON object so SFT ties reasoning to a
    structured payload the constrained decoder also targets."""
    try:
        val = int(num)
    except ValueError:
        try:
            val = float(num)
        except ValueError:
            val = num                            # keep as string if non-numeric
    return json.dumps({"answer": val})


def build(dataset: str, max_examples: int, out_dir: Path, val_frac: float = 0.02):
    enc = _gpt2()
    eot = enc.eot_token
    if dataset == "gsm8k":
        rows, tag = _load_gsm8k(max_examples)
    elif dataset == "synthetic":
        rows, tag = _synthetic(max_examples), "synthetic"
    else:
        raise SystemExit(f"unknown --dataset {dataset!r} (use gsm8k|synthetic)")

    out_dir.mkdir(parents=True, exist_ok=True)
    tokens: list[int] = []
    mask: list[int] = []
    n_examples = 0
    # also dump raw (question, reasoning, answer) so STaR (nanolab.star) can read
    # gold answers without importing `datasets` in a torch process (segfaults on
    # Windows). One JSON object per line.
    with open(out_dir / "examples.jsonl", "w", encoding="utf-8") as jf:
        for q, reasoning, num in rows:
            tk, mk = build_example(enc, q, reasoning, _answer_json(str(num)), eot)
            tokens.extend(tk)
            mask.extend(mk)
            n_examples += 1
            jf.write(json.dumps({"question": q, "reasoning": reasoning, "answer": str(num)}) + "\n")

    tokens_np = np.asarray(tokens, dtype=np.uint16)
    mask_np = np.asarray(mask, dtype=np.uint8)
    n = len(tokens_np)
    split = int((1.0 - val_frac) * n)

    tokens_np[:split].tofile(out_dir / "train_tokens.bin")
    mask_np[:split].tofile(out_dir / "train_loss.bin")
    tokens_np[split:].tofile(out_dir / "val_tokens.bin")
    mask_np[split:].tofile(out_dir / "val_loss.bin")

    meta = {"source": tag, "examples": n_examples, "tokens": int(n),
            "train_tokens": int(split), "val_tokens": int(n - split),
            "trained_frac": float(mask_np.mean())}
    (out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[sft_data] source={tag} examples={n_examples} tokens={n} "
          f"train={split} val={n - split} trained_frac={mask_np.mean():.2f}")
    print(f"[sft_data] wrote -> {out_dir}")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gsm8k", help="gsm8k | synthetic")
    p.add_argument("--max_examples", type=int, default=8000)
    p.add_argument("--out_dir", default="nanolab/data/sft")
    args = p.parse_args()
    build(args.dataset, args.max_examples, Path(args.out_dir))


if __name__ == "__main__":
    main()
