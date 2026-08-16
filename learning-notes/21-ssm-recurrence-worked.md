# 21 — The SSM/Recurrence Math, Worked

File 04 introduced minGRU, Mamba-2, and Gated DeltaNet as "running summaries" and showed they beat
attention at low token budgets. This file works the recurrences with actual numbers, derives *why*
chunk-parallel scanning gives the same answer faster, and connects it to your verified kernels.

---

## 21.1 The shared shape: a linear recurrence

All three mixers are variants of one update — a state that **decays** and **accumulates**:

```
  state_t = decay_t ⊙ state_{t-1} + input_write_t
  output_t = read(state_t)
```

The differences are just *what* decay and write are:

| Mixer | decay | write | read |
|---|---|---|---|
| **minGRU** | `1 − z_t` (input-gated) | `z_t ⊙ h̃_t` | `h_t` directly |
| **Mamba-2** | `exp(dt·A)` (selective) | `dt·B_t·x_t` | `C_t · state_t` |
| **Gated DeltaNet** | `α_t` (gate) | delta rule (overwrite) | `q_t · state_t` |

Because the decay depends only on the input (not on `state_{t-1}` in a nonlinear way), the whole
thing is a **linear scan** — and linear scans can be parallelized (the key to file 04's kernels).

---

## 21.2 minGRU, worked (scalar, 4 steps)

`h_t = (1 − z_t)·h_{t-1} + z_t·h̃_t`. Let inputs give these gates/candidates:

```
  t:   z_t   h̃_t        h_t = (1−z)·h_{t-1} + z·h̃
  ─────────────────────────────────────────────────
  0:   0.9   1.0    h_0 = 0.1·0   + 0.9·1.0 = 0.90
  1:   0.2   0.0    h_1 = 0.8·0.90 + 0.2·0.0 = 0.72
  2:   0.5   2.0    h_2 = 0.5·0.72 + 0.5·2.0 = 1.36
  3:   0.1   0.0    h_3 = 0.9·1.36 + 0.1·0.0 = 1.22
```

Read it: `z` near 1 means "overwrite the state with the new candidate" (t=0); `z` near 0 means
"keep the old state, ignore input" (t=1, t=3). The state is a **gated running memory**. Notice the
information from `h̃_0 = 1.0` is still influencing `h_3 = 1.22` — that's how a fixed-size state
carries long-range context *without* attention's all-pairs comparison. This is the "inductive
bias" that wins when data is scarce (file 04): the model gets "recent stuff matters, decay the
past" for free instead of learning it.

---

## 21.3 Why it can be parallelized (the associative-scan insight)

That `for t` loop looks inherently sequential. But unroll it:

```
  h_2 = (1−z_2)·h_1 + z_2·h̃_2
      = (1−z_2)·[(1−z_1)·h_0 + z_1·h̃_1] + z_2·h̃_2
      = (1−z_2)(1−z_1)·h_0  +  (1−z_2)·z_1·h̃_1  +  z_2·h̃_2
```

Each `h_t` is a sum of input terms weighted by **products of decays**. Those cumulative products
(`(1−z_2)(1−z_1)…`) can be computed with a **parallel prefix scan** (a.k.a. associative scan) in
`O(log T)` depth instead of `O(T)` sequential steps. That's why minGRU was already fast in your GPU
sweep (6.7K tok/s, file 04) — it uses a parallel scan, no Python loop. Mamba-2 and GDN needed *you*
to write the chunk-parallel version (below) to get there.

---

## 21.4 Mamba-2 (SSD), and the chunk-parallel derivation

Mamba-2's recurrence in your nanolab mapping (mixers.py, ported from `train_hypercascade.py`):

```
  state_t = exp(dt_t · A) ⊙ state_{t-1} + dt_t · B_t · x_t      (state is a [d_state × d] matrix)
  y_t     = C_t · state_t
```

- `A` < 0, so `exp(dt·A)` ∈ (0,1) is the decay (ZOH discretization, file 04). `log_dA = dt·A`.
- `B` writes the (scaled) input `x_scaled = dt·x` into the state; `C` reads it out.
- "Selective": `dt, B, C` are functions of the input, so the model *chooses* per token.

**The chunk-parallel trick** (your `ssd_chunk_parallel`, 9.7× speedup): split T into chunks of size
C. Within a chunk, expand the recurrence (like §21.3) — it becomes a **decay-weighted attention**
between the chunk's tokens:

```
  pass 1 (intra-chunk, parallel):  for tokens i ≥ j in the same chunk,
        y_i += (decay from j to i) · (C_i · B_j) · x_j      ← a C×C matrix, computed as one matmul
  pass 2 (inter-chunk, sequential over CHUNKS only):
        carry each chunk's final state to the next chunk    ← T/C steps, not T
```

So instead of T sequential steps you do **T/C chunks**, each an internal matmul — at T=1024, C=32
that's ~32 chunk-steps, **fully vectorized** (the 9.7× in file 04). The math is *identical* to the
sequential version — which is exactly what `verify_scan.py` checks to **1e-5** tolerance. The
intra-chunk "decay-weighted C·Bᵀ attention" is why Mamba-2 is sometimes called a linear-attention
in disguise.

---

## 21.5 Gated DeltaNet — the delta rule

GDN's write isn't just "add the input" — it's "**remove the old association, then write the new
one**," like a cleanly-overwriting memory (file 04). Schematically, for a key/value write:

```
  vanilla linear attn:  state += k ⊗ v                    (just accumulates — old k/v associations pile up)
  delta rule:           state += β · k ⊗ (v − stateᵀk)    (subtract what's already stored for k first)
```

That `(v − stateᵀk)` is the **prediction error** for key k — write only the *correction*. This is a
fast-weight / Hebbian-with-forgetting update, and it's why GDN has **stronger recall** than vanilla
SSMs (it doesn't let stale associations accumulate). The `β` and `α` gates control write strength
and decay. Your `gdn_chunked` (mixers.py:433) does this chunkwise in the "WY" form, verified vs the
sequential reference (`verify_gdn_wy.py`).

The three recurrent designs, ranked by their 2M-token bake-off result (real, file 08) —
the delta rule's cleaner memory shows up as the best of the three:

```
  recurrent mixers @ 2M tokens (shorter = better):
  minGRU          5.837  ##########                 simplest gated memory, wins at this budget
  Gated DeltaNet  5.994  #######################     delta-rule recall — best of Mamba/GDN
  Mamba-2         6.040  ###########################  selective SSM
  ── for reference, the attention baselines: ──
  attention       6.073  #############################
  MLA             6.156  ####################################
```

GDN beat Mamba-2 in your 2M-token bake-off (5.994 vs 6.040) — the delta rule's cleaner memory paid
off even at small scale.

---

## 21.6 The fp32 requirement, now derivable

You can now *see* why the scan needs fp32 (file 06, 16). The state is a **product of many decays**:
`state_t` includes terms like `exp(dt_1·A)·exp(dt_2·A)·…·exp(dt_t·A)`. Multiplying hundreds of
numbers in (0,1):
- In **bf16** (7 mantissa bits, ~2 digits), each multiply rounds; over T steps the errors compound
  into real drift — the state is wrong by the end.
- In **fp32** (23 mantissa bits), the per-step rounding is ~10⁵× smaller, so the accumulated product
  stays accurate.

This is the general principle from file 06 made concrete: **a long chain of dependent multiplies
(accumulation) needs precision; a one-shot matmul or a self-correcting iteration (file 19) doesn't.**
The recurrence is the worst case for low precision, so autocast is disabled inside it.

---

## 21.7 The unifying picture

```
  Attention:   O(T²)   compares every pair directly         — exact recall, expensive, capacity-rich
  minGRU:      O(T)    gated running scalar/vector memory    — cheapest, parallel scan
  Mamba-2:     O(T)    selective decaying state matrix        — chunk-parallel = decay-weighted attn
  GatedDeltaN: O(T)    delta-rule fast-weight memory          — best recall of the recurrent three
```

All four mix the sequence; they trade exactness for cost. The chunk-parallel kernels (files 04, 21)
are what made the O(T) ones *practical* on your hardware, which is what made the bake-off and the
crossover (file 08) measurable at all. Without §21.4's derivation turned into code, mamba2/gdn stay
at 333/238 tok/s and the experiment never runs.

**Next:** [`22-distributed-and-scaling-systems.md`](22-distributed-and-scaling-systems.md) — how the
single-GPU picture changes when the competition's 8×H100 enters.
