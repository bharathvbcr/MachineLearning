"""
nanolab.special_tokens — reasoning/answer control tokens (Phase 2), torch-free.

The GPT-2 BPE tokenizer has 50257 real tokens, but the 128M model embeds a
*padded* vocab of 50304 (rounded to a multiple of 64 for matmul speed). That
leaves ids 50257..50303 — 47 unused embedding rows that never appeared in
pretraining data. We repurpose three of them as reasoning control tokens, so the
think-then-answer format needs **no embedding-table surgery**: the rows already
exist, SFT simply learns them.

Template a trained model emits:

    <prompt>\n <think> reasoning… </think> <|answer|> {"answer": …}<eot>

Both the (torch-free) data prep and the (torch) inference harness import these
constants, so the format lives in exactly one place.
"""

from __future__ import annotations

GPT2_REAL_VOCAB = 50257          # tiktoken "gpt2" real token count

THINK = 50257                    # <think>
THINK_END = 50258                # </think>
ANSWER = 50259                   # <|answer|>

# id -> display string, for decoding/debugging
SPECIAL_STR = {THINK: "<think>", THINK_END: "</think>", ANSWER: "<|answer|>"}


def build_example(enc, question: str, reasoning: str, answer_text: str,
                  eot: int) -> tuple[list[int], list[int]]:
    """Tokenize one (question, reasoning, answer) triple into the think/answer
    template. Returns (token_ids, loss_mask) where loss_mask is 0 on the prompt
    (we don't train the model to reproduce the question) and 1 on the completion
    (think span + answer + eot — what the model must learn to generate)."""
    prompt = enc.encode_ordinary(question.strip() + "\n")
    think = [THINK] + enc.encode_ordinary(" " + reasoning.strip() + " ") + [THINK_END]
    answer = [ANSWER] + enc.encode_ordinary(answer_text.strip()) + [eot]
    tokens = prompt + think + answer
    mask = [0] * len(prompt) + [1] * (len(think) + len(answer))
    return tokens, mask
