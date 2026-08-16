# 11 — Tokenization Deep Dive

File 01 said "a tokenizer maps text to integers." This file opens that box: **how BPE actually
builds its vocabulary, byte-level vs character vs subword, why padding to 50304, the bits-per-byte
conversion, and why the vocab size is the single biggest Parameter Golf lever.**

---

## 11.1 Why tokenize at all?

A model needs a fixed, finite set of input symbols, each with its own embedding vector. Two naive
extremes:

- **One token per word** — vocab is huge (millions), can't handle new/misspelled words ("out of
  vocabulary"), wastes embedding params.
- **One token per character** — tiny vocab (~100), never OOV, but sequences become very long (the
  model spends its O(T²) attention budget on spelling, not meaning).

**Subword tokenization** (BPE, SentencePiece, WordPiece) is the compromise: common words become
one token, rare words split into a few subword pieces. "tokenization" → `token`, `ization`.

---

## 11.2 BPE (Byte-Pair Encoding) — the actual algorithm

BPE *learns* its vocabulary from a corpus by greedily merging the most frequent adjacent pair,
repeatedly:

```
  start:   every character is its own token
  corpus:  "low low lower newest widest"  →  l o w   l o w   l o w e r   ...

  count adjacent pairs → most frequent is ("l","o")  → MERGE into "lo"
  recount             → most frequent is ("lo","w")  → MERGE into "low"
  recount             → ("e","s") → "es" ; ("es","t") → "est" ; ...
  repeat until vocab reaches the target size (e.g. 50,000 merges)
```

Each merge adds one token to the vocabulary. The final vocab = the base characters/bytes + all
learned merges. **Encoding** new text = apply the learned merges in order. The result: frequent
words are single tokens, rare words are a handful of subwords, nothing is ever OOV.

GPT-2's tokenizer (`tiktoken`, used by nanolab's `tokenizer="gpt2"`) is **byte-level BPE**: it
starts from the 256 raw bytes, so *any* byte sequence (emoji, code, other languages) is
representable. That's the `vocab_size=50257` your notes reference.

---

## 11.3 Byte-level vs character vs subword vs SentencePiece

| Scheme | Base units | Vocab | Seq length | Used in your work |
|---|---|---|---|---|
| **Char** | characters | ~27–256 | longest | `cpu_smoke`, text8/enwik8 (bits-per-char) |
| **Byte-level BPE** | 256 bytes + merges | 50,257 | medium | GPT-2, all nanolab `gpt2` runs |
| **SentencePiece** | learned subwords (lang-agnostic) | **1,024** | medium | Parameter Golf champion (slim vocab) |
| **Raw bytes** | bytes, no merges | 256 | longest | APRDH adaptive byte model |

- **SentencePiece** treats text as a raw stream (spaces are a normal symbol, `▁`), so it needs no
  pre-tokenization and works for any language. The champion trains a **1024-token** SP vocab — a
  deliberate Parameter Golf move (see §11.6).
- **Raw bytes** (APRDH) skips merges entirely: 256 symbols, but the model must learn *everything*,
  including word formation, from bytes — which is why APRDH adds **span-mixer patching** to group
  bytes into learned chunks (a *learned*, in-model tokenizer).

---

## 11.4 The padded vocab: why 50304 not 50257

Your `phase1` config uses `vocab_size=50304`, not GPT-2's actual `50257`. That's **50257 rounded
up to the next multiple of 64**. Reason: GPU tensor cores process matmuls in tiles (multiples of
8/64); a vocab that's a clean multiple makes the final `[*, 50304]` logits matmul and the
embedding lookup hit aligned, efficient kernels. The 47 extra tokens are unused padding rows —
their embeddings just never get trained. A free ~speedup; a classic "minute thing."

(In `diffusion.py` you reused id **50257** — the first padding slot — as the special `[MASK]`
token precisely because it was a spare id in the padded range. Tidy.)

---

## 11.5 From loss to bits-per-byte (the exact conversion)

The model's cross-entropy loss is in **nats per token**. The competition wants **bits per byte**.
The conversion (file 01) in full:

```
  bits_per_token = loss_nats / ln(2)                         # nats → bits
  BPB            = bits_per_token × (tokens / bytes)          # per-token → per-byte
                 = bits_per_token / (bytes_per_token)
```

The `tokens/bytes` ratio is where the tokenizer matters: a tokenizer that packs **more bytes per
token** (fewer tokens for the same text) divides the per-token bits across more bytes → lower BPB,
*if the model can still predict those denser tokens well.* This is the tension:

- Bigger vocab → more bytes/token → fewer tokens to predict, but each prediction is *harder*
  (1-of-50304) and the embeddings cost params.
- Smaller vocab → easier per-token prediction, but more tokens/byte → the per-token bits get
  spread over fewer bytes.

BPB being **tokenizer-agnostic** is the whole point: you can't win by choosing a vocab that makes
the *loss number* look small — it normalizes back to bytes of real text. It's an honest
compression metric.

---

## 11.6 Vocab size as the Parameter Golf lever (the trade, quantified)

From file 10: at `d=512`, an embedding table is `vocab × 512` params, and it's **tied** so you pay
for it once but it's ~⅓ of a small model.

```
  vocab 50257, d=512:  embedding ≈ 25.7M params  → dominates a tight budget, blows past 16 MB
  vocab  1024, d=512:  embedding ≈  0.52M params  → ~50× smaller; budget freed for depth/width
```

Embedding params vs vocab size, at d=512 (each `█` ≈ 1M params; this is *tied*, so paid once):

```
  vocab  50257  ██████████████████████████  25.7M   ← blows the 16 MB budget on embeddings alone
  vocab  16000  ████████                      8.2M
  vocab   4096  ██                            2.1M
  vocab   1024  ▌                             0.5M   ★ champion — ~50× smaller than GPT-2's
                └ every M of params saved here is a M you can spend on depth/width/aux heads
```

So the champion's 1024-token SentencePiece vocab isn't a quality choice — it's a **budget
reallocation**: spend the saved embedding params on more layers, aux heads, and the gated-attn +
value-residual machinery that actually moved BPB. The cost (more tokens per sentence, so you need
more sequence length / steps to see the same text) is acceptable under the compute budget.

This is the cleanest example of the competition's core skill: **every parameter is a budget line
item; move them to where they buy the most BPB.**

---

## 11.7 Practical tokenizer mechanics in your pipeline

- **Pre-tokenize once, store as `.bin`.** nanolab's `prep_fineweb.py` tokenizes in a *separate,
  torch-free process* (the Windows pyarrow+torch segfault, file: memory) and writes a flat
  `uint16`/`uint32` array of ids. Training just memory-maps it — no tokenizer in the hot loop.
- **Packing.** Documents are concatenated and chopped into `block_size` chunks with no padding
  waste — every position trains. A document-boundary token separates them.
- **`uint16` vs `uint32`.** A 50304 vocab fits in 16 bits (max 65535) → store ids as `uint16`,
  halving the dataset's disk/RAM vs `uint32`. A 1024 vocab still uses uint16 (no uint10 type), but
  compresses far better.

**Next:** [`12-generation-and-sampling.md`](12-generation-and-sampling.md) — now that the model
predicts a distribution, how do you actually turn it into text?
