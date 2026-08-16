"""
nanolab.reason — Phase-1 inference harness: think-then-answer + schema-guided
(constrained) JSON decoding, running on an EXISTING base checkpoint (no training).

Two capabilities, one module:

1. **Structured output, valid by construction.** ``Harness.generate_json`` walks a
   JSON schema and emits the structural scaffolding (braces, keys, quotes, commas)
   *deterministically*, letting the model fill only the **values** under per-type
   token masks. A 50304-way softmax physically cannot place a stray ``"`` inside a
   string or a non-digit inside a number, so ``json.loads`` on the output never
   fails — regardless of how small or lightly-trained the model is. The model's
   tiny capacity shows up as weak *content*, never as broken *structure*.

2. **Reason before answering.** ``Harness.reason_then_answer`` first generates a
   free-form thinking span, then feeds that scratchpad back in before the
   constrained answer. The phase markers live in ONE place (``THINK_OPEN`` /
   ``ANSWER_OPEN``) so a Phase-2 SFT model that uses real ``<think>`` /
   ``<|answer|>`` special tokens drops in by changing those two constants.

CLI (mirrors nanolab.sample):

    python -m nanolab.reason --run run128m_fineweb_2k \
        --question "Is the sky blue? Give your confidence." --schema demo

Honest ceiling: this is a ~128M GPT-2-small base model. Expect reliably-formatted
output with shallow, frequently-wrong reasoning. The harness guarantees *form*;
Phase-2 SFT is what improves *content*.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import build_config
from .model import build_model
from .special_tokens import ANSWER, GPT2_REAL_VOCAB, THINK, THINK_END
from .utils import pick_device

# Phase markers. Today they are plain text a base model can pattern-match; in
# Phase 2 (SFT) swap these for dedicated special tokens, e.g. "<think>\n" and
# "</think>\n<|answer|>\n", after extending the embedding table by those rows.
THINK_OPEN = "Let's think step by step.\n"
ANSWER_OPEN = "Final answer (JSON):\n"

NEG_INF = float("-inf")


class BPE:
    """GPT-2 BPE wrapper that also precomputes per-token byte-class masks used by
    the constrained decoder. ``vocab_size`` is the model's (padded) vocab; ids at
    or above tiktoken's real ``n_vocab`` are padding and are masked off always."""

    def __init__(self, vocab_size: int):
        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token
        self.vocab_size = vocab_size
        self.n_real = self.enc.n_vocab            # 50257 real BPE tokens

        tok_bytes = [self.enc.decode_single_token_bytes(i) for i in range(self.n_real)]
        pad = vocab_size - self.n_real

        digits = set(b"0123456789")
        ctrl = set(range(0x20))                   # JSON forbids raw control chars

        def flag(pred) -> list[bool]:
            return [pred(b) for b in tok_bytes] + [False] * pad

        # tokens that must NOT appear inside a JSON string (would break it)
        self._string_block = flag(
            lambda b: any((c == 0x22 or c == 0x5C or c in ctrl) for c in b)
        )
        self._is_digits = flag(lambda b: len(b) > 0 and all(c in digits for c in b))
        self._has_quote = flag(lambda b: 0x22 in b)
        self._is_real = [True] * self.n_real + [False] * pad

        # single-character structural tokens we steer numbers/arrays with
        self.dot_id = self._single(".")
        self.comma_id = self._single(",")
        self.close_arr_id = self._single("]")
        # single-digit token ids; restricting a number's FIRST token to these
        # lets us forbid JSON-illegal leading zeros ("0123") by construction.
        self.digit_ids = [self._single(str(d)) for d in range(10)]
        self.zero_id = self.digit_ids[0]
        single_digit = set(self.digit_ids)
        self._single_digit = [i in single_digit for i in range(self.n_real)] + [False] * pad

    def _single(self, ch: str) -> int:
        ids = self.enc.encode_ordinary(ch)
        return ids[0] if ids else self.eot

    def encode(self, s: str) -> list[int]:
        return self.enc.encode_ordinary(s)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


class Harness:
    """Bundles a model + tokenizer + device and the constrained-decoding logic."""

    def __init__(self, model, bpe: BPE, device, block_size: int):
        self.model = model
        self.bpe = bpe
        self.device = device
        self.block_size = block_size
        b = lambda lst: torch.tensor(lst, dtype=torch.bool, device=device)  # noqa: E731
        self.m_string_block = b(bpe._string_block)
        self.m_is_digits = b(bpe._is_digits)
        self.m_has_quote = b(bpe._has_quote)
        self.m_is_real = b(bpe._is_real)
        self.m_single_digit = b(bpe._single_digit)
        # KV-cached decode only works for the attention mixer — recurrent/MLA
        # mixers have no forward_cached, so they fall back to recompute.
        self.cache_ok = getattr(model, "cfg", None) is not None and model.cfg.mixer == "attention"

    # -- low-level token plumbing --------------------------------------------
    def empty(self) -> torch.Tensor:
        return torch.zeros((1, 0), dtype=torch.long, device=self.device)

    def feed_text(self, idx: torch.Tensor, text: str) -> torch.Tensor:
        ids = self.bpe.encode(text)
        if not ids:
            return idx
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        return torch.cat([idx, t], dim=1)

    def feed_id(self, idx: torch.Tensor, tok: int) -> torch.Tensor:
        t = torch.tensor([[tok]], dtype=torch.long, device=self.device)
        return torch.cat([idx, t], dim=1)

    @torch.no_grad()
    def next_logits(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx[:, -self.block_size:])
        return logits[0, -1].float()

    def _masked(self, logits: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        out = torch.where(allow, logits, torch.full_like(logits, NEG_INF))
        return out

    def _sample(self, logits, temperature, top_k, greedy=False) -> int:
        if greedy:
            return int(torch.argmax(logits))
        logits = logits / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.numel()))
            logits = torch.where(logits < v[-1], torch.full_like(logits, NEG_INF), logits)
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1))

    @torch.no_grad()
    def _score(self, ctx: "_Ctx", text: str) -> float:
        """Total log-prob the model assigns to ``text`` continuing ``ctx``. One
        full forward (all-position logits) over ctx.ids+text — used only for the
        rare bool/enum field choices, so it's cheap relative to the decode."""
        tids = self.bpe.encode(text)
        if not tids:
            return 0.0
        prefix = ctx.ids if ctx.ids else [self.bpe.eot]
        seq = prefix + tids
        x = torch.tensor([seq], dtype=torch.long, device=self.device)
        hidden = self.model.forward_hidden(x)
        if self.model.output_mult != 1.0:
            hidden = hidden * self.model.output_mult
        logp = F.log_softmax(self.model.lm_head(hidden)[0].float(), dim=-1)
        start = len(prefix)
        return sum(float(logp[start + j - 1, t]) for j, t in enumerate(tids))

    # -- per-type constrained value generators (mutate ctx in place) ----------
    def _gen_string(self, ctx, temperature, top_k, max_tokens, min_tokens=1):
        ctx.feed_text('"')
        n = 0
        while n < max_tokens:
            logits = ctx.logits()
            raw_top = int(torch.argmax(logits))
            # let the model end the string naturally: if its *unconstrained* top
            # choice would close the string (a quote) or stop (eot), we stop.
            if n >= min_tokens and (raw_top == self.bpe.eot or bool(self.m_has_quote[raw_top])):
                break
            allow = self.m_is_real & ~self.m_string_block
            allow[self.bpe.eot] = False
            ctx.feed_id(self._sample(self._masked(logits, allow), temperature, top_k))
            n += 1
        ctx.feed_text('"')

    def _gen_number(self, ctx, integer: bool, max_tokens=8):
        """Emit a JSON-legal number by FSM: a non-zero-led integer part (or a lone
        "0"), then for floats an optional ".<digits>" fraction. The first token is
        drawn from single-digit tokens so we can forbid leading zeros; both the
        integer and fraction parts are guaranteed at least one digit."""
        digits = self.m_is_digits & self.m_is_real

        def run_digits(n):
            while n < max_tokens:
                logits = ctx.logits()
                if not bool(digits[int(torch.argmax(logits))]):   # model wants to stop
                    break
                ctx.feed_id(int(torch.argmax(self._masked(logits, digits))))
                n += 1
            return n

        # integer part: first token is a single digit (controls leading zeros)
        tok = int(torch.argmax(self._masked(ctx.logits(), self.m_single_digit & self.m_is_real)))
        ctx.feed_id(tok)
        n = 1
        if tok != self.bpe.zero_id:                  # "0" must stand alone in JSON
            n = run_digits(n)
        if integer:
            return
        # optional fraction: only if the model actually wants a decimal point
        if int(torch.argmax(ctx.logits())) != self.bpe.dot_id:
            return
        ctx.feed_id(self.bpe.dot_id)
        ctx.feed_id(int(torch.argmax(self._masked(ctx.logits(), digits))))   # >=1 frac digit
        run_digits(n + 1)

    def _gen_bool(self, ctx):
        t, f = self._score(ctx, "true"), self._score(ctx, "false")
        ctx.feed_text("true" if t >= f else "false")

    def _gen_enum(self, ctx, options):
        best, best_lp = None, NEG_INF
        for opt in options:
            literal = json.dumps(opt)            # quotes strings, leaves nums bare
            lp = self._score(ctx, literal)
            if lp > best_lp:
                best, best_lp = literal, lp
        ctx.feed_text(best if best is not None else "null")

    def _gen_array(self, ctx, item_schema, opts, min_items, max_items):
        ctx.feed_text("[")
        count = 0
        while count < max_items:
            if count > 0:
                ctx.feed_text(", ")
            self._gen_value(ctx, item_schema, opts)
            count += 1
            if count >= max_items:
                break
            if count >= min_items:
                # let the model vote: continue (",") vs close ("]")
                logits = ctx.logits()
                if float(logits[self.bpe.close_arr_id]) >= float(logits[self.bpe.comma_id]):
                    break
        ctx.feed_text("]")

    def _gen_object(self, ctx, schema, opts):
        props = schema.get("properties", {})
        ctx.feed_text("{")
        keys = list(props)
        for i, k in enumerate(keys):
            ctx.feed_text(f"{json.dumps(k)}: ")
            self._gen_value(ctx, props[k], opts)
            if i < len(keys) - 1:
                ctx.feed_text(", ")
        ctx.feed_text("}")

    def _gen_value(self, ctx, schema, opts):
        if "enum" in schema:
            return self._gen_enum(ctx, schema["enum"])
        t = schema.get("type", "string")
        if t == "object":
            return self._gen_object(ctx, schema, opts)
        if t == "array":
            return self._gen_array(
                ctx, schema.get("items", {"type": "string"}), opts,
                schema.get("minItems", 1), schema.get("maxItems", opts["max_items"]))
        if t == "string":
            return self._gen_string(ctx, opts["temperature"], opts["top_k"], opts["max_str_tokens"])
        if t == "integer":
            return self._gen_number(ctx, integer=True)
        if t == "number":
            return self._gen_number(ctx, integer=False)
        if t == "boolean":
            return self._gen_bool(ctx)
        if t == "null":
            return ctx.feed_text("null")
        raise ValueError(f"unsupported schema type: {t!r}")

    # -- public entry points -------------------------------------------------
    @torch.no_grad()
    def generate_json(self, schema: dict, prefix: str = "", cached: bool = True,
                      ctx: "_Ctx | None" = None, temperature=0.7, top_k=40,
                      max_str_tokens=24, max_items=4):
        """Schema-guided decode. ``json.loads(<returned text>)`` is guaranteed to
        succeed and match the schema's shape. Pass ``prefix`` to condition on text,
        or an existing ``ctx`` to continue a session (used by reason_then_answer)."""
        if ctx is None:
            ctx = _Ctx(self, cached)
            if prefix:
                ctx.feed_text(prefix)
        opts = dict(temperature=temperature, top_k=top_k,
                    max_str_tokens=max_str_tokens, max_items=max_items)
        start = len(ctx.ids)
        self._gen_value(ctx, schema, opts)
        return self.bpe.decode(ctx.ids[start:]), ctx

    @torch.no_grad()
    def reason_then_answer(self, question: str, schema: dict, think_tokens=128,
                           temperature=0.8, top_k=50, special=False, cached=True, **json_kw):
        """Free-form think span, then a constrained JSON answer conditioned on it.

        ``special=False`` (base model): scaffold the phases with plain-text markers
        a base model can pattern-match. ``special=True`` (Phase-2 SFT model): drive
        the phases with the learned ``<think>`` / ``</think>`` / ``<|answer|>``
        control tokens — the thinking span ends when the model emits ``</think>``."""
        ctx = _Ctx(self, cached)
        if special:
            ctx.feed_text(question.strip() + "\n")
            ctx.feed_id(THINK)
        else:
            ctx.feed_text(f"{question}\n\n{THINK_OPEN}")

        think_ids: list[int] = []
        for _ in range(think_tokens):
            tok = self._sample(ctx.logits(), temperature, top_k)
            # stop on eot, or (SFT path) when the model closes the think span / emits
            # any control token — those ids aren't decodable as ordinary text.
            if tok == self.bpe.eot or tok >= GPT2_REAL_VOCAB:
                break
            ctx.feed_id(tok)
            think_ids.append(tok)
        reasoning = self.bpe.decode(think_ids).strip()

        if special:
            ctx.feed_id(THINK_END)
            ctx.feed_id(ANSWER)
        else:
            ctx.feed_text(f"\n{ANSWER_OPEN}")
        json_text, ctx = self.generate_json(schema, ctx=ctx, **json_kw)
        return {"reasoning": reasoning, "json": json_text, "parsed": json.loads(json_text)}


class _Ctx:
    """Decoding context shared by both speed paths.

    ``cached`` threads per-layer KV caches through the model's
    ``forward_hidden_window`` — O(T) incremental decode, attention mixer only.
    Otherwise it recomputes from the full token tensor each step (O(T^2), but
    works for any mixer). ``ids`` always holds every committed token so output
    spans can be decoded and bool/enum scoring can run a one-off full forward.
    Committing tokens returns the next-token logits for free in the cached path."""

    def __init__(self, h: "Harness", cached: bool):
        self.h = h
        self.cached = cached and h.cache_ok
        self.ids: list[int] = []
        if self.cached:
            self.caches = [dict() for _ in h.model.blocks]
            self._last: torch.Tensor | None = None
        else:
            self.idx = h.empty()

    def feed_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        if self.cached:
            abs_start = len(self.ids)
            x = torch.tensor([ids], dtype=torch.long, device=self.h.device)
            hid = self.h.model.forward_hidden_window(x, abs_start, self.caches, True, True)
            hl = hid[:, -1, :]
            if self.h.model.output_mult != 1.0:
                hl = hl * self.h.model.output_mult
            self._last = self.h.model.lm_head(hl)[0].float()
        else:
            t = torch.tensor([ids], dtype=torch.long, device=self.h.device)
            self.idx = torch.cat([self.idx, t], dim=1)
        self.ids.extend(ids)

    def feed_text(self, text: str) -> None:
        self.feed_ids(self.h.bpe.encode(text))

    def feed_id(self, tok: int) -> None:
        self.feed_ids([tok])

    def logits(self) -> torch.Tensor:
        if self.cached:
            if self._last is None:               # empty context: seed with eot (BOS)
                self.feed_id(self.h.bpe.eot)
            return self._last
        return self.h.next_logits(self.idx)


DEMO_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "is_certain": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
}


def load_harness(run: str, out_dir: str, ckpt: str) -> Harness:
    run_dir = Path(out_dir) / run
    cfg = build_config(overrides=json.loads((run_dir / "config.json").read_text()))
    if cfg.tokenizer != "gpt2":
        raise SystemExit("nanolab.reason currently requires the gpt2 BPE tokenizer")
    device = pick_device(cfg.device)
    model = build_model(cfg).to(device).eval()
    state = torch.load(run_dir / ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return Harness(model, BPE(cfg.vocab_size), device, cfg.block_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out_dir", default="nanolab/out")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--question", default="Is the sky blue? Give your confidence.")
    p.add_argument("--schema", default="demo", help='"demo" | path to a JSON schema file')
    p.add_argument("--think_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--no_think", action="store_true", help="structured output only, skip reasoning")
    p.add_argument("--special", action="store_true",
                   help="use learned <think>/<|answer|> tokens (for a Phase-2 SFT checkpoint)")
    p.add_argument("--no_cache", action="store_true", help="force recompute decode (debug)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # the model can emit any UTF-8 byte; don't let the Windows cp1252 console
    # crash the run on an unprintable char.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    torch.manual_seed(args.seed)
    schema = DEMO_SCHEMA if args.schema == "demo" else json.loads(Path(args.schema).read_text())
    h = load_harness(args.run, args.out_dir, args.ckpt)

    cached = not args.no_cache
    if args.no_think:
        text, _ = h.generate_json(schema, prefix=f"{args.question}\n{ANSWER_OPEN}",
                                  cached=cached, temperature=args.temperature, top_k=args.top_k)
        result = {"reasoning": None, "json": text, "parsed": json.loads(text)}
    else:
        result = h.reason_then_answer(
            args.question, schema, think_tokens=args.think_tokens,
            temperature=args.temperature, top_k=args.top_k, special=args.special, cached=cached)

    print(f"=== {args.run}/{args.ckpt} ===")
    if result["reasoning"] is not None:
        print(f"\n--- reasoning ---\n{result['reasoning']}")
    print(f"\n--- structured answer ---\n{result['json']}")
    print(f"\nvalid JSON: {isinstance(result['parsed'], dict) or isinstance(result['parsed'], list)}")
    print(f"parsed: {result['parsed']}")


if __name__ == "__main__":
    main()
