# 20 — RoPE: How Rotation Encodes Position

File 02 said RoPE "rotates Q and K so attention scores depend on relative position." This file
proves it with the actual math and numbers, grounded in your `nanolab/mixers.py` `build_rope_cache`
/ `apply_rope`. RoPE is the cleanest piece of math in the whole stack — worth understanding fully.

---

## 20.1 The problem (one more time, precisely)

Attention's score is `q · k` (file 02, 10). It's **position-blind**: if token "cat" is at slot 3
or slot 50, its q and k vectors are identical, so the scores are identical. But "the cat" and "cat
the" mean different things — position matters. We need to inject *where each token is* into q and k
**without** breaking the dot-product structure.

GPT-2 added a learned vector per position (absolute). RoPE does something better: it makes `q · k`
depend only on the **relative** offset `(m − n)` between positions m and n.

---

## 20.2 The idea: rotate by an angle proportional to position

Take a 2-D slice of the query vector at position `m`. RoPE **rotates** it by angle `m·θ`:

```
  rotation by angle φ:   [x']   [cos φ  −sin φ] [x]
                         [y'] = [sin φ   cos φ] [y]
```

A query at position m is rotated by `m·θ`; a key at position n is rotated by `n·θ`. Now the dot
product of the rotated query and rotated key:

```
  R(mθ)q · R(nθ)k  =  q · R((n−m)θ) k        ← KEY IDENTITY
```

The two rotations *combine into a single rotation by the difference* `(n−m)θ`. So the score depends
only on **how far apart** the tokens are, not their absolute slots. That's relative position, for
free, from a rotation. (This is the rotation-matrix property `R(a)ᵀR(b) = R(b−a)`.)

---

## 20.3 Worked example (head_dim 2, θ = 1 radian)

```
  q = [1, 0]  at position m = 2  →  rotate by 2·1 = 2 rad
  k = [1, 0]  at position n = 5  →  rotate by 5·1 = 5 rad

  R(2)q = [cos2, sin2]   = [−0.416,  0.909]
  R(5)k = [cos5, sin5]   = [ 0.284, −0.959]

  score = R(2)q · R(5)k = (−0.416)(0.284) + (0.909)(−0.959) = −0.118 − 0.872 = −0.990
```

Check the identity — it should equal `q · R(5−2)k = q · R(3)k`:
```
  R(3)k = [cos3, sin3] = [−0.990, 0.141]
  q · R(3)k = [1,0]·[−0.990, 0.141] = −0.990   ✓  matches
```

Whether the two tokens sit at (2,5), (10,13), or (100,103), the score is the **same −0.990**,
because the offset is always 3. Position is encoded purely as *relative distance*. That's why RoPE
generalizes to sequences longer than training (a distance of 3 means the same thing everywhere)
far better than learned absolute positions.

---

## 20.4 Many frequencies (the full head)

One angle θ only captures one "wavelength" of position. RoPE uses **head_dim/2 different
frequencies**, geometrically spaced — exactly your `build_rope_cache` (mixers.py:48):

```python
  inv_freq = 1.0 / (base ** (arange(0, head_dim, 2) / head_dim))   # base default 10000
  # pairs of dimensions rotate at frequencies 1, 1/base^(2/d), 1/base^(4/d), ...
  emb = positions[:, None] * inv_freq[None, :]
  cos, sin = emb.cos(), emb.sin()                                  # cached once, reused every step
```

- **Low dimensions rotate fast** (high frequency) → encode *fine* position (adjacent tokens).
- **High dimensions rotate slowly** (low frequency, period ~`base`=10000) → encode *coarse* position
  (far-apart tokens).

The frequency bands across a head's dimension pairs (like a clock with many hands at different speeds):

```
  dim pair 0   ●∿∿∿∿∿∿∿∿  fastest   period ~6 tokens     → "is the previous token X?"
  dim pair 1   ●∿∿∿∿       fast       period ~40
  dim pair 2   ●∿∿         medium     period ~270
   ...                                   (geometric: ×base^(2/d) each step)
  dim pair d/2 ●           slowest    period ~10000 tokens → "roughly where in the document?"
               └ a token's full position = its readout across ALL these clocks at once
```

This multi-scale design is why one rotation captures both "the immediately preceding word" and
"early vs late in the passage" — different dimension pairs specialize at different distances.

It's the same multi-scale trick as the original sinusoidal embeddings, but applied as a *rotation*
so the relative-position identity holds. `apply_rope` (mixers.py:57) just does `x·cos + rotate_half(x)·sin`
— the efficient form of the 2×2 rotations across all pairs at once.

---

## 20.5 Where it sits in the forward pass

From file 10's shape walk, and your `Attention.forward` (mixers.py:112):

```
  q = q_proj(x)                  [B, T, H, D]
  k = k_proj(x)                  [B, T, n_kv, D]
  q = apply_rope(q, cos, sin)    ← rotate queries by their positions
  k = apply_rope(k, cos, sin)    ← rotate keys by their positions
  (QK-norm is applied around here too — file 03)
  y = scaled_dot_product_attention(q, k, v)   ← scores now carry relative position
```

Crucially, **only q and k are rotated, not v.** Position should influence *which* tokens attend to
each other (the q·k scores), but the *content* delivered (v) shouldn't be spun around. A subtle,
correct design choice you can now see in the code.

---

## 20.6 Why "strictly better, cheap" (the guide's verdict)

- **Cheap:** cos/sin are precomputed once (`build_rope_cache`) and reused every step; `apply_rope`
  is one elementwise multiply-add. No learned parameters at all (unlike GPT-2's position table,
  which *costs* `block_size × d_model` params — more budget saved, file 11).
- **Better:** relative encoding generalizes to longer contexts; YaRN can extend it further by
  rescaling the frequencies (the `rope_base`/`rope_head_dim` knobs your MLA path exposes,
  mixers.py:534).
- **Composes:** works identically for attention, and the SSM mixers don't need it (their recurrence
  is inherently positional) — which is part of why mixers are swappable in your model (file 04).

RoPE, Newton–Schulz (file 19), and the attention dot product (file 10) are the three places where a
small piece of linear algebra does something that *looks* like it should need a learned component
but doesn't. That elegance is why the modern stack is both better and cheaper than GPT-2.

**Next:** [`21-ssm-recurrence-worked.md`](21-ssm-recurrence-worked.md) — the recurrence math behind
the mixers that beat attention at low token counts.
