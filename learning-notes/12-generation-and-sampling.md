# 12 — Generation and Sampling

Training teaches the model to output a probability distribution over the next token (file 01).
**Generation** is turning that distribution into actual text, one token at a time. This file
covers greedy/temperature/top-k/top-p sampling (what `nanolab/sample.py` does), the **KV cache**
that makes generation fast, and why diffusion decoding (file 15) is fundamentally different.

---

## 12.1 The autoregressive generation loop

```
  prompt = "Once upon a"
  repeat until done:
     logits = model(prompt)            # [.., T, vocab]  — take the LAST position's row
     probs  = softmax(logits[-1] / temperature)
     next   = sample_from(probs)       # pick one token id
     prompt = prompt + next            # append, and loop
```

Every generated token feeds back in as input — that's "autoregressive." `model.generate(x, n,
temperature, top_k)` in your `sample.py` is exactly this loop.

---

## 12.2 The decoding strategies (how to pick `next`)

Given the probability distribution over the vocab, you choose a token. The spectrum from
deterministic to random:

### Greedy — always the argmax
```
  next = argmax(probs)        # always the single most likely token
```
Deterministic and "safe," but **repetitive and dull** — it gets stuck in loops ("the the the"),
because the locally-most-likely token isn't globally best. Used when you want reproducibility
(and it's what a pure loss-minimizer would do).

### Temperature — sharpen or flatten the distribution
```
  probs = softmax(logits / T)
```
- `T → 0`: distribution becomes a spike → approaches greedy (conservative, repetitive).
- `T = 1`: the model's raw distribution.
- `T > 1`: flattens it → more random, more "creative," more mistakes.

Your `sample.py` defaults to **T=0.8** — slightly sharpened, a common sweet spot for coherent-but-
not-robotic text. (Note: this is the *same* temperature knob as the **logit calibration** in file
07 — there it's tuned to minimize BPB, here it's tuned for generation quality.)

```
  logits [2.0, 1.0, 0.1, 3.0]:
    T=0.5 → probs [0.12, 0.04, 0.01, 0.83]   ← sharp, almost greedy
    T=1.0 → probs [0.24, 0.09, 0.04, 0.64]
    T=2.0 → probs [0.27, 0.16, 0.10, 0.46]   ← flat, adventurous
```

The same distribution, drawn (each row is one temperature; bars are the 4 token probs):

```
  token:        A     B     C     D
  T=0.5  A ██▌   B █    C ▏    D ████████████████▌   ← spiky: D dominates (conservative)
  T=1.0  A ████▊ B █▊   C ▊    D ████████████▊       ← the model's raw belief
  T=2.0  A █████▍B ███▏ C ██   D █████████▏          ← flattened: C/B now plausible (creative/risky)
         └ raising T transfers probability mass from the peak toward the tail.
```

### Top-k — only sample from the k most likely
```
  keep the k highest-prob tokens, zero the rest, renormalize, sample
```
`sample.py` defaults to **top_k=50**: even at high temperature, the model can never pick a token
outside the top 50 — kills the long tail of absurd tokens while keeping variety. Simple, effective.

### Top-p (nucleus) — smallest set covering probability p
```
  sort tokens by prob; keep the smallest set whose cumulative prob ≥ p (e.g. 0.9); sample from it
```
Adaptive: when the model is confident (one token at 0.95), the nucleus is tiny → nearly greedy;
when uncertain (many similar options), the nucleus is large → more diversity. Usually preferred
over a fixed k. (nanolab ships top-k; top-p is a two-line addition to the same loop.)

**In practice:** temperature ~0.7–0.9 + top-k 40–50 or top-p 0.9 is the standard "coherent
sampling" recipe. Greedy/low-T for factual tasks, higher-T for creative ones.

---

## 12.3 The KV cache — why generation isn't O(T²) per token

Naively, generating token T means re-running attention over all T-1 previous tokens — and you do
that for *every* new token, so generating an N-token passage costs O(N³). Absurd. The fix:

> **KV cache** — the K and V vectors of past tokens never change, so compute them once and
> **store** them. For each new token you only compute *its* Q, K, V and attend against the cached
> K/V.

```
  without cache:  token t recomputes K,V for all 0..t   → O(t) work per token, O(T²) total
  with cache:     token t computes only its own K,V, reuses cached 0..t-1 → O(1) extra per token
```

This is the whole reason **GQA / MQA / MLA** (file 02) exist: the KV cache **grows with sequence
length and dominates memory** during long generation. GQA's 4 KV heads instead of 12 shrinks the
cache 3×; MLA compresses it into a latent. Your champion's 8Q/4KV is a direct KV-cache decision.

> **The diffusion contrast (file 15):** a diffusion LM does **parallel bidirectional** denoising —
> there's no left-to-right growing cache in the same sense, which is *why the guide warns that
> GQA/MLA's motivation weakens for diffusion.* The decoding model determines whether the
> cache-shrinking tricks even apply.

---

## 12.4 Sampling and the BPB metric — a subtle point

Parameter Golf scores **BPB**, which is computed from the model's *probabilities on the true text*
(teacher-forced) — **it does not involve sampling at all.** Sampling quality (does it write good
stories?) and BPB (does it assign high probability to real text?) are *correlated but different*.
You can have:
- Low BPB but boring greedy output (a good compressor, conservative generator).
- A model that samples beautifully but has mediocre BPB (mode-seeking).

This is why your evaluation pipeline (file 07) tunes **temperature/softcap as a calibration** on
BPB, separately from any generation temperature. **Know which metric you're optimizing.**

---

## 12.5 Speed tricks worth knowing (context)

- **Speculative decoding** — a small "draft" model proposes several tokens, the big model verifies
  them in one parallel pass; accepted tokens are free. 2–3× faster generation, same distribution.
- **Continuous batching / paged KV cache (vLLM)** — production serving tricks to pack many
  generation requests and manage KV-cache memory in pages.
- **Why diffusion is interesting here:** it generates *many tokens in parallel per step* (file 15),
  trading more forward passes for fewer sequential steps — a different point on the
  latency/throughput curve than AR + speculative decoding.

These aren't in your trainers (you're optimizing training + BPB, not serving), but they're the
reason GQA/MLA/quantization matter commercially — the same techniques, pointed at inference cost.

**Next:** [`13-regularization-init-and-stability.md`](13-regularization-init-and-stability.md) —
the unglamorous tricks that decide whether training converges at all.
