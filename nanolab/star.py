"""
nanolab.star — Phase-3 STaR (Self-Taught Reasoner) bootstrap: improve reasoning
*correctness* (not just format) by training the model on its OWN correct traces.

    python -m nanolab.star --base_run sft128m_gsm8k --dataset gsm8k \
        --questions nanolab/data/sft/examples.jsonl --rounds 3 --samples 4

Loop (Zelikman et al. 2022):
  1. Sample ``--samples`` think-then-answer traces per question from the current
     model (the constrained decoder makes every answer a valid JSON int).
  2. Keep a trace iff its decoded answer equals the gold answer (rejection).
  3. ``--rationalize``: for still-unsolved questions, reveal the answer as a hint,
     sample again, and keep a correct trace — but train on the ORIGINAL question
     (no hint), so the model learns to reach it unaided.
  4. Fine-tune the model on the kept traces (reuses nanolab.sft).
  5. Repeat from the improved checkpoint.

Each piece is already built: generation = nanolab.reason (KV-cached, special-token
think/answer), the template + masking = nanolab.special_tokens, the fine-tune =
nanolab.sft. STaR just closes the loop.

Requires an SFT'd starting checkpoint (``--base_run``) that already emits the
``<think>``/``<|answer|>`` tokens — a raw base model never will. Build one with
nanolab.sft first. Gold answers are read from a JSONL (see nanolab.sft_data's
``examples.jsonl``) to avoid importing ``datasets`` in a torch process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import sft
from .reason import load_harness
from .sft_data import _synthetic
from .special_tokens import build_example

ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}}


def _num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def is_correct(pred, gold) -> bool:
    p, g = _num(pred), _num(gold)
    return p is not None and g is not None and abs(p - g) < 1e-6


def _load_rows(dataset: str, questions: str, max_q: int):
    """Return [(question, gold_reasoning, gold_answer), ...]."""
    if dataset == "synthetic":
        return _synthetic(max_q)
    rows = []
    for line in Path(questions).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append((d["question"], d.get("reasoning", ""), d["answer"]))
        if len(rows) >= max_q:
            break
    return rows


def collect(h, rows, samples, temperature, top_k, think_tokens, rationalize):
    """Sample traces, keep the correct ones. Returns (triples, stats)."""
    kept, n_solved, n_rationalized = [], 0, 0
    for q, _gold_reasoning, gold in rows:
        solved = False
        for _ in range(samples):
            r = h.reason_then_answer(q, ANSWER_SCHEMA, special=True,
                                     think_tokens=think_tokens,
                                     temperature=temperature, top_k=top_k)
            if is_correct(r["parsed"].get("answer"), gold):
                kept.append((q, r["reasoning"], r["json"]))
                solved = True
                n_solved += 1
                break
        if solved or not rationalize:
            continue
        # rationalization: hint the answer, keep a correct trace, train on bare q
        hq = f"{q}\nHint: the answer is {gold}."
        for _ in range(samples):
            r = h.reason_then_answer(hq, ANSWER_SCHEMA, special=True,
                                     think_tokens=think_tokens,
                                     temperature=temperature, top_k=top_k)
            if is_correct(r["parsed"].get("answer"), gold):
                kept.append((q, r["reasoning"], r["json"]))
                n_rationalized += 1
                break
    stats = {"questions": len(rows), "solved": n_solved,
             "rationalized": n_rationalized, "kept": len(kept)}
    return kept, stats


def write_bins(triples, out_dir: Path, val_frac=0.05):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    tokens, mask = [], []
    for q, reasoning, answer_text in triples:
        tk, mk = build_example(enc, q, reasoning, answer_text, eot)
        tokens.extend(tk)
        mask.extend(mk)
    t = np.asarray(tokens, dtype=np.uint16)
    m = np.asarray(mask, dtype=np.uint8)
    n = len(t)
    split = max(1, int((1.0 - val_frac) * n))
    out_dir.mkdir(parents=True, exist_ok=True)
    t[:split].tofile(out_dir / "train_tokens.bin")
    m[:split].tofile(out_dir / "train_loss.bin")
    t[split:].tofile(out_dir / "val_tokens.bin")
    m[split:].tofile(out_dir / "val_loss.bin")
    return n


def run_star(base_run, star_run, out_dir, dataset, questions, rounds, samples,
             temperature, top_k, think_tokens, rationalize, max_q,
             sft_steps, lr, batch_size, block_size, seed):
    rows = _load_rows(dataset, questions, max_q)
    print(f"[star] {len(rows)} questions, {rounds} rounds, {samples} samples/q, "
          f"rationalize={rationalize}")
    cur = base_run
    for rd in range(rounds):
        h = load_harness(cur, out_dir, "best.pt")
        kept, stats = collect(h, rows, samples, temperature, top_k, think_tokens, rationalize)
        acc = stats["solved"] / max(stats["questions"], 1)
        print(f"[star] round {rd}: model={cur}  solved={stats['solved']}/{stats['questions']} "
              f"({acc:.0%})  +rationalized={stats['rationalized']}  kept={stats['kept']}")
        del h
        if not kept:
            print("[star] no correct traces collected — stopping (model too weak / raise --samples)")
            break
        ddir = Path("nanolab/data/star") / f"{star_run}_r{rd}"
        ntok = write_bins(kept, ddir)
        run = f"{star_run}_r{rd}"
        print(f"[star] round {rd}: fine-tuning {run} on {len(kept)} traces ({ntok} tokens)")
        sft.run_sft(base_run=cur, run=run, out_dir=out_dir, data_dir=str(ddir),
                    steps=sft_steps, lr=lr, batch_size=batch_size, block_size=block_size,
                    eval_interval=max(50, sft_steps // 4), weight_decay=0.0, seed=seed)
        cur = run
    print(f"[star] done. final checkpoint: {out_dir}/{cur}")
    return cur


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run", required=True, help="SFT'd starting checkpoint (emits <think>/<|answer|>)")
    p.add_argument("--star_run", default="star128m")
    p.add_argument("--out_dir", default="nanolab/out")
    p.add_argument("--dataset", default="gsm8k", help="gsm8k | synthetic")
    p.add_argument("--questions", default="nanolab/data/sft/examples.jsonl",
                   help="JSONL with question/answer (for --dataset gsm8k)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--think_tokens", type=int, default=96)
    p.add_argument("--rationalize", action="store_true")
    p.add_argument("--max_q", type=int, default=500)
    p.add_argument("--sft_steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_size", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    run_star(args.base_run, args.star_run, args.out_dir, args.dataset, args.questions,
             args.rounds, args.samples, args.temperature, args.top_k, args.think_tokens,
             args.rationalize, args.max_q, args.sft_steps, args.lr, args.batch_size,
             args.block_size, args.seed)


if __name__ == "__main__":
    main()
