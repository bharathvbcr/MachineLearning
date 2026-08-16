# 07 — Quantization and the 16 MB Budget

Parameter Golf scores you on a model whose **weights + code must compress under 16 MB**. Training
a good model is only half the job; the other half is **shrinking it without wrecking it**. This
file covers quantization (int8/int6), quantization-*aware* training, GPTQ-style clipping, and the
packaging pipeline — the techniques in `train_gpt_sprint_native.py`.

Distinguish this from file 06: **precision (bf16/fp8) is about training speed; quantization (int8/
int6) is about the final artifact size.** Different goal, related idea (fewer bits).

---

## 7.1 Quantization: floats → small integers

A trained weight is a 32-bit (or 16-bit) float. Most of those bits are wasted — the weights in a
given matrix row cluster in a narrow range. **Quantization** maps them to small integers:

```
  for each row (or block) of weights:
     scale = max(|w|) / (2^(bits-1) - 1)     # find the range
     q = round(w / scale)                    # store small ints  (e.g. int8: -128..127)
     # at use time:  w ≈ q × scale           # dequantize
```

- **int8** = 8 bits/weight → **4× smaller** than fp32, ~negligible quality loss. The safe default.
- **int6** = 6 bits/weight → **5.3× smaller**, more aggressive. Your sprint trainer uses **per-row
  int6** as the main format.
- **Per-row (per-channel) scaling** — a *separate* scale for every row, so a row of small weights
  isn't forced to share a coarse scale with a row of large ones. Much better than one global scale.
- **Mixed precision categories** — keep the *sensitive* tensors (embeddings, the LM head, norms)
  at higher precision (int8) and quantize the bulk matrices harder (int6). Spend bits where they
  matter.

---

## 7.2 Quantization-Aware Training (QAT) vs post-training

- **Post-training quantization (PTQ):** train in float, quantize at the end. Simple, but the model
  never "knew" it would be quantized, so accuracy drops.
- **Quantization-Aware Training (QAT):** **simulate** the int6 rounding *during* training (the
  forward pass rounds the weights; the backward pass uses a "straight-through estimator" to pass
  gradients through the non-differentiable rounding). The model **learns weights that survive
  quantization** — it adapts to the rounding noise. Much less quality loss at int6.

Your `train_gpt_sprint_native.py` does QAT, so the int6 export is close to the float quality
rather than a cliff.

---

## 7.3 GPTQ-lite clip search — choosing the clipping range

When you quantize, outlier weights blow up the `scale` (because `scale = max(|w|)`), making *every
other* weight coarse. **GPTQ-style clip search** sweeps candidate clip thresholds, *clips* the
outliers, and picks the threshold that minimizes the resulting output error — trading a tiny error
on a few outliers for much finer resolution on the bulk. Your trainer does a "GPTQ-lite clip
search." Net effect: better int6 quality for free.

---

## 7.4 EMA weights — a free quality bump

**Exponential Moving Average** of the weights: keep a slowly-updated running average
`w_ema = 0.999·w_ema + 0.001·w` alongside the live weights, and **evaluate/ship the EMA copy**.
The EMA sits in a flatter, more central spot of the loss landscape than the jittery live weights —
it generalizes slightly better, basically for free. Standard trick; your sprint trainer ships EMA.

---

## 7.5 Compression: int-quantize *then* entropy-code

Quantization gets you to int6; a general compressor squeezes further. The pipeline matches the
competition's:

```
  trained weights ──QAT int6/int8 (per-row)──▶ small ints ──zlib / zstd / lzma──▶ .ptz artifact
```

The size cascade, as a bar (relative to the fp32 model = 100%):

```
  fp32 weights        ████████████████████████████████  100%   (4 bytes/weight)
  bf16                ████████████████                    50%   (2 bytes)
  int8 per-row        ████████                            25%   (4× smaller)
  int6 per-row        ██████                              ~19%  (5.3× — champion's main format)
  int6 + zstd/lzma    ███                                 ~10%  (entropy-code the redundant ints)
                      └ stacking quantization × entropy-coding is how 16 MB becomes plenty
```

The int weights still have redundancy (many repeated values, low entropy), so **zstd/lzma**
entropy-coding shrinks them again. The result is written as `.ptz` next to the raw `.pt`. Your
champion's full artifact came in at **~1.34 MB** — comfortably under the 16 MB budget, which means
there was *headroom to spend on more parameters* (a strategic lever: the budget is a constraint to
fill, not just satisfy).

---

## 7.6 Evaluation-time tricks (legal ways to lower BPB without bigger weights)

The sprint trainer squeezes extra BPB at **eval** time — these don't change the artifact size:

- **Sliding-window evaluation** (file 02) — evaluate with a local attention window for efficiency
  and sometimes better calibration on long sequences.
- **Test-Time Training (TTT)** — a *legal* form where the model adapts its state to the eval text
  on the fly (within the rules). Squeezes out extra bits by specializing to the local distribution.
- **Logit calibration** — temperature scaling and softcap sweeps: divide the logits by a tuned
  temperature `T` before softmax so the predicted probabilities are better calibrated. A small,
  free BPB win found by sweeping `T` on a held-out slice.

These are the "last mile" — once the architecture and weights are fixed, calibration + TTT +
windowed eval shave the final fraction of a bit.

---

## 7.7 The full submission pipeline

```
  champion.json (gated attn + value resid)
        │
        ▼
  train_gpt_sprint_native.py     ← QAT, int6 per-row, EMA, GPTQ-clip, TTT, calibration
        │
        ▼
  submission_packaging.py        ← flatten to ONE file: build/train_gpt_sprint_submit.py
        │                           (+ FlashAttention fallback prelude so it runs anywhere)
        ▼
  preflight_h100.py              ← validate data paths, artifact, and CODE SIZE before paying
        │                           for an 8×H100 run
        ▼
  8×H100 sprint run              ← 16 MB artifact, ≤ 10 minutes
```

`preflight_h100.py` matters because an 8×H100 run costs real money — you validate the packaged
single-file trainer, the data paths, and the **code-size accounting** (the *code* counts toward
the 16 MB too, not just the weights) before launching. Measure twice, cut once.

**Next:** [`08-experiments-and-results.md`](08-experiments-and-results.md) — every result in one
place, plus the experiments not yet covered (APRDH, diffusion).
